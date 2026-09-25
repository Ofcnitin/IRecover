/**
 * Azusa -- White Balance stage.
 * -----------------------------
 * NOTE ON NAMING: "Azusa" was requested as a named white-balance library.
 * No such browser-compatible JavaScript/TypeScript package exists (verified
 * by search -- see project notes). Per the task's own fallback instruction
 * (section 42), this module implements the *functional role* -- white
 * balance -- as a deterministic, dependency-free algorithm, named `Azusa`
 * so call sites read the way the target architecture describes them.
 *
 * Algorithm: a robust, confidence-gated variant of gray-world white
 * balance, operating on linear-light RGB.
 *
 * IR-recovered images are NOT ordinary photographs: they usually have no
 * true photographic white/gray reference, and the color in the frame is
 * itself an artistic/deterministic reinterpretation of intensity, not a
 * captured scene color. Aggressive white balance would fight the
 * IR->RGB color mapping instead of refining it. So this stage:
 *
 *   1. Estimates a channel-balance correction from the image's own
 *      robust (percentile-trimmed) channel means -- classic gray-world,
 *      but outlier-resistant.
 *   2. Computes a "confidence" for that estimate from how *chromatic*
 *      the image is overall (a very saturated, intentionally-colored
 *      image is less likely to actually be a neutral scene, so we trust
 *      the gray-world assumption less).
 *   3. Scales the correction by confidence AND by a hard user-controlled
 *      strength ceiling, so the stage can only ever nudge color balance,
 *      never invent or dramatically shift it.
 */

import { linearLuminance } from './colorSpace';

export interface WhiteBalanceOptions {
  /** 0..100 user strength. 0 disables the stage entirely. */
  strength: number;
  /**
   * Hard ceiling on the per-channel gain this stage may apply, regardless
   * of strength/confidence. Keeps the stage "conservative" per spec even
   * at strength=100. Default +/-12%.
   */
  maxGain?: number;
}

export interface WhiteBalanceResult {
  gainR: number;
  gainG: number;
  gainB: number;
  confidence: number; // 0..1, how much of the estimated correction was actually trusted
}

export const DEFAULT_MAX_GAIN = 0.12;

/**
 * Computes the white-balance gains for a planar linear-RGB buffer WITHOUT
 * mutating it. This is the single source of truth for the Azusa gain
 * estimate -- both the CPU apply path below and the WebGL2 GPU path
 * (see processing/gpu/webgl2Backend.ts) call this same function, so the
 * two backends can never estimate different gains, only apply the same
 * gains differently (JS loop vs. GPU shader).
 */
export function computeWhiteBalanceGains(
  r: Float32Array,
  g: Float32Array,
  b: Float32Array,
  opts: WhiteBalanceOptions
): WhiteBalanceResult {
  const n = r.length;
  const identity: WhiteBalanceResult = { gainR: 1, gainG: 1, gainB: 1, confidence: 0 };
  if (opts.strength <= 0 || n === 0) return identity;

  const maxGain = opts.maxGain ?? DEFAULT_MAX_GAIN;

  // Robust means: trim the extreme 2% at each end per channel so a few
  // strongly color-mapped highlight/shadow pixels (sky, deep shadow) don't
  // dominate the gray-world estimate.
  const meanR = trimmedMean(r);
  const meanG = trimmedMean(g);
  const meanB = trimmedMean(b);
  const grayMean = (meanR + meanG + meanB) / 3;

  if (grayMean < 1e-5) return identity;

  // Ideal (uncapped, unweighted) gray-world gains.
  const idealGainR = grayMean / Math.max(meanR, 1e-5);
  const idealGainG = grayMean / Math.max(meanG, 1e-5);
  const idealGainB = grayMean / Math.max(meanB, 1e-5);

  // Confidence: derived from average chroma. Highly saturated / strongly
  // colored images are more likely to be *intentionally* colorful (this is
  // a recolored IR image, not a photo with a color cast), so trust the
  // neutral-world assumption less as saturation rises.
  const avgChroma = averageChroma(r, g, b, meanR, meanG, meanB);
  const confidence = clamp01(1 - avgChroma * 3.5);

  const userStrength = clamp01(opts.strength / 100);
  const appliedFraction = userStrength * confidence;

  const gainR = clamp(1 + (idealGainR - 1) * appliedFraction, 1 - maxGain, 1 + maxGain);
  const gainG = clamp(1 + (idealGainG - 1) * appliedFraction, 1 - maxGain, 1 + maxGain);
  const gainB = clamp(1 + (idealGainB - 1) * appliedFraction, 1 - maxGain, 1 + maxGain);

  return { gainR, gainG, gainB, confidence };
}

/**
 * Estimates and applies white balance to a planar linear-RGB buffer, in
 * place. Returns the gains actually applied and the confidence used, so
 * the caller can surface diagnostics.
 */
export function applyAzusaWhiteBalance(
  r: Float32Array,
  g: Float32Array,
  b: Float32Array,
  opts: WhiteBalanceOptions
): WhiteBalanceResult {
  const result = computeWhiteBalanceGains(r, g, b, opts);
  const n = r.length;
  for (let i = 0; i < n; i++) {
    r[i] = r[i] * result.gainR;
    g[i] = g[i] * result.gainG;
    b[i] = b[i] * result.gainB;
  }
  return result;
}

/** Mean of a channel after discarding the darkest/brightest 2% of samples. */
function trimmedMean(channel: Float32Array): number {
  const n = channel.length;
  if (n === 0) return 0;
  // For very large images, sort a random-ish stride subsample instead of
  // the full buffer -- percentile trimming doesn't need every pixel.
  const maxSamples = 65536;
  let samples: number[];
  if (n <= maxSamples) {
    samples = Array.from(channel);
  } else {
    const stride = Math.floor(n / maxSamples);
    samples = [];
    for (let i = 0; i < n; i += stride) samples.push(channel[i]);
  }
  samples.sort((a, b) => a - b);
  const trim = Math.floor(samples.length * 0.02);
  const lo = trim;
  const hi = samples.length - trim;
  let sum = 0;
  let count = 0;
  for (let i = lo; i < hi; i++) {
    sum += samples[i];
    count++;
  }
  return count > 0 ? sum / count : samples[Math.floor(samples.length / 2)] ?? 0;
}

/** Average normalized chroma (max-min channel spread relative to luminance) across a subsample. */
function averageChroma(
  r: Float32Array,
  g: Float32Array,
  b: Float32Array,
  meanR: number,
  meanG: number,
  meanB: number
): number {
  const n = r.length;
  const maxSamples = 32768;
  const stride = Math.max(1, Math.floor(n / maxSamples));
  let sum = 0;
  let count = 0;
  for (let i = 0; i < n; i += stride) {
    const lum = Math.max(linearLuminance(r[i], g[i], b[i]), 1e-4);
    const spread = Math.max(r[i], g[i], b[i]) - Math.min(r[i], g[i], b[i]);
    sum += spread / lum;
    count++;
  }
  // Reference the sample-set average chroma against the global mean too,
  // so a single-hue tinted image (mean itself is chromatic) also reduces
  // confidence, not just per-pixel spread.
  const meanSpread = Math.max(meanR, meanG, meanB) - Math.min(meanR, meanG, meanB);
  const meanLum = Math.max((meanR + meanG + meanB) / 3, 1e-4);
  const globalChroma = meanSpread / meanLum;
  const localChroma = count > 0 ? sum / count : 0;
  return Math.max(localChroma, globalChroma);
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}
