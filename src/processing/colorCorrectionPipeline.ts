/**
 * ColorCorrectionPipeline -- Precision Color Correction stage.
 * --------------------------------------------------------------
 * NOTE ON NAMING: as with Azusa and AutoTone, no browser-suitable
 * "ColorCorrectionPipeline" package exists. This module implements the
 * functional role -- fine, restrained color correction -- deterministically,
 * per the task's section 42 fallback instruction.
 *
 * Runs AFTER AutoTone, on linear-light RGB, and BEFORE local refinement /
 * noise reduction / sharpening / gamut protection.
 *
 * Deliberately restrained: the goal is a natural-looking RGB
 * approximation, not maximum saturation. Two operations, both computed in
 * OKLCH (the polar form of OKLab) so lightness is untouched and hue is
 * preserved by construction:
 *
 *   1. Channel balance -- a very small global per-channel gain, capped
 *      tightly, that nudges out any lingering mean color cast left after
 *      white balance (which is itself confidence-gated and may have
 *      applied little or no correction).
 *   2. Perceptual chroma shaping ("vibrance") -- boosts chroma more in
 *      the midtones than in the shadows/highlights, and rolls it off for
 *      already-saturated pixels, mimicking how a careful colorist adds
 *      "punch" without oversaturating skin- or sky-like tones.
 */

import { linearRgbToOklab, oklabToLinearRgb } from './colorSpace';

export interface ColorCorrectionOptions {
  /** 0..100 user strength for perceptual chroma shaping. 0 disables both operations. */
  strength: number;
  /** Hard ceiling on channel-balance gain. Default +/-4% -- intentionally tighter than white balance. */
  maxChannelGain?: number;
  /** Hard ceiling on chroma multiplier at the sweet spot (midtones, moderate existing chroma). Default 1.18. */
  maxChromaBoost?: number;
}

export interface ColorCorrectionResult {
  gainR: number;
  gainG: number;
  gainB: number;
}

export const DEFAULT_MAX_CHANNEL_GAIN = 0.04;
export const DEFAULT_MAX_CHROMA_BOOST = 1.18;

/**
 * Computes only the small global channel-balance gains (step 1) WITHOUT
 * mutating the buffer and without running the per-pixel chroma-shaping
 * pass (step 2). Single source of truth shared by the CPU apply path
 * below and the WebGL2 GPU path (processing/gpu/webgl2Backend.ts), which
 * runs step 2 itself as a per-pixel fragment shader using these gains as
 * uniforms.
 */
export function computeChannelBalanceGains(
  r: Float32Array,
  g: Float32Array,
  b: Float32Array,
  opts: ColorCorrectionOptions
): ColorCorrectionResult {
  const n = r.length;
  const identity: ColorCorrectionResult = { gainR: 1, gainG: 1, gainB: 1 };
  if (opts.strength <= 0 || n === 0) return identity;

  const maxChannelGain = opts.maxChannelGain ?? DEFAULT_MAX_CHANNEL_GAIN;
  const strengthFrac = clamp01(opts.strength / 100);

  let sumR = 0;
  let sumG = 0;
  let sumB = 0;
  for (let i = 0; i < n; i++) {
    sumR += r[i];
    sumG += g[i];
    sumB += b[i];
  }
  const meanR = sumR / n;
  const meanG = sumG / n;
  const meanB = sumB / n;
  const grayMean = (meanR + meanG + meanB) / 3;

  const gainR = grayMean > 1e-5 ? clamp(1 + (grayMean / meanR - 1) * strengthFrac * 0.3, 1 - maxChannelGain, 1 + maxChannelGain) : 1;
  const gainG = grayMean > 1e-5 ? clamp(1 + (grayMean / meanG - 1) * strengthFrac * 0.3, 1 - maxChannelGain, 1 + maxChannelGain) : 1;
  const gainB = grayMean > 1e-5 ? clamp(1 + (grayMean / meanB - 1) * strengthFrac * 0.3, 1 - maxChannelGain, 1 + maxChannelGain) : 1;

  return { gainR, gainG, gainB };
}

export function applyColorCorrectionPipeline(
  r: Float32Array,
  g: Float32Array,
  b: Float32Array,
  opts: ColorCorrectionOptions
): ColorCorrectionResult {
  const n = r.length;
  const identity: ColorCorrectionResult = { gainR: 1, gainG: 1, gainB: 1 };
  if (opts.strength <= 0 || n === 0) return identity;

  const maxChromaBoost = opts.maxChromaBoost ?? DEFAULT_MAX_CHROMA_BOOST;
  const strengthFrac = clamp01(opts.strength / 100);

  const { gainR, gainG, gainB } = computeChannelBalanceGains(r, g, b, opts);

  // ---- 2. Perceptual chroma shaping, per pixel, in OKLCH. ----
  for (let i = 0; i < n; i++) {
    const rr = r[i] * gainR;
    const gg = g[i] * gainG;
    const bb = b[i] * gainB;

    const [L, a, bLab] = linearRgbToOklab(rr, gg, bb);
    const C = Math.sqrt(a * a + bLab * bLab);

    if (C < 1e-5) {
      r[i] = rr;
      g[i] = gg;
      b[i] = bb;
      continue;
    }

    // Midtone weight: peaks at L=0.5, tapers toward shadows/highlights so
    // we don't add chroma to near-black/near-white pixels (which usually
    // means noise or intentional highlight/shadow detail, not color).
    const midtoneWeight = 1 - Math.min(1, Math.abs(L - 0.5) / 0.5) ** 1.5;

    // Roll-off for already-saturated pixels: diminishing boost as chroma
    // rises, so strongly color-mapped areas (deliberate preset color)
    // aren't pushed toward oversaturation/clipping.
    const saturationRolloff = 1 / (1 + C * 3);

    const boost = 1 + (maxChromaBoost - 1) * strengthFrac * midtoneWeight * saturationRolloff;
    const newC = C * boost;
    const scale = newC / C;

    const [nr, ng, nb] = oklabToLinearRgb(L, a * scale, bLab * scale);
    r[i] = nr;
    g[i] = ng;
    b[i] = nb;
  }

  return { gainR, gainG, gainB };
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}
