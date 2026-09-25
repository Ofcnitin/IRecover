/**
 * AutoTone -- Tonal Correction stage.
 * -----------------------------------
 * NOTE ON NAMING: as with Azusa (see whiteBalance.ts), no browser-suitable
 * "AutoTone" package exists. This module implements the functional role --
 * histogram-driven tonal correction -- as a deterministic algorithm, per
 * the task's section 42 fallback instruction.
 *
 * Runs AFTER white balance, on linear-light RGB, and BEFORE
 * ColorCorrectionPipeline. It reuses the project's existing 256-bucket
 * histogram implementation (computed on a derived luminance channel)
 * rather than re-deriving percentile logic from scratch.
 *
 * This is a light *refinement* pass, not a replacement for the main tone
 * curve already applied earlier in the pipeline (exposure/contrast/gamma
 * on the source intensity, before IR->RGB color mapping). Its job is to
 * clean up whatever the color mapping and white balance introduced --
 * e.g. a slightly elevated black point from gain adjustments -- using
 * robust percentile statistics, with highlight/shadow protection so it
 * never clips or crushes aggressively.
 */

import { computeHistogram } from './histogram';
import { linearLuminance, clamp01 } from './colorSpace';

export interface AutoToneOptions {
  /** 0..100 user strength. 0 disables the stage entirely. */
  strength: number;
}

export interface AutoToneResult {
  blackPoint: number; // 0..1, linear
  whitePoint: number; // 0..1, linear
  appliedFraction: number;
  blackClipPercent: number;
  whiteClipPercent: number;
}

/** kneeStart is shared with the GPU shader (see processing/gpu/webgl2Backend.ts) as a named constant, not a magic number. */
export const AUTOTONE_KNEE_START = 0.92;

/**
 * Computes AutoTone's black/white points from a planar linear-RGB buffer
 * WITHOUT mutating it or computing clip percentages (which require the
 * per-pixel apply pass). Single source of truth for both the CPU apply
 * path below and the WebGL2 GPU path.
 */
export function computeAutoToneParams(
  r: Float32Array,
  g: Float32Array,
  b: Float32Array,
  opts: AutoToneOptions
): { blackPoint: number; whitePoint: number; appliedFraction: number; degenerate: boolean } {
  const n = r.length;
  const none = { blackPoint: 0, whitePoint: 1, appliedFraction: 0, degenerate: true };
  if (opts.strength <= 0 || n === 0) return none;

  // Build an 8-bit luminance snapshot to reuse the existing histogram/
  // percentile machinery rather than duplicating it.
  const lum8 = new Uint8ClampedArray(n);
  for (let i = 0; i < n; i++) {
    lum8[i] = clamp01(linearLuminance(r[i], g[i], b[i])) * 255;
  }
  const hist = computeHistogram(lum8);

  // Robust percentile-based black/white points -- conservative percentiles
  // (0.5 / 99.5) so we only trim genuine outliers, never meaningful content.
  const rawBlack = hist.percentiles[0.5] / 255;
  const rawWhite = hist.percentiles[99.5] / 255;

  // Guard against a degenerate (near-flat) histogram.
  if (rawWhite - rawBlack < 0.02) return none;

  const strengthFrac = clampNum(opts.strength / 100, 0, 1);
  // Only ever move the black/white points a fraction of the way toward the
  // percentile-ideal ones -- "correction", not a hard re-levels.
  const appliedFraction = strengthFrac * 0.6;
  const blackPoint = 0 + (rawBlack - 0) * appliedFraction;
  const whitePoint = 1 + (rawWhite - 1) * appliedFraction;

  return { blackPoint, whitePoint, appliedFraction, degenerate: false };
}

/**
 * Applies AutoTone to a planar linear-RGB buffer in place.
 */
export function applyAutoTone(
  r: Float32Array,
  g: Float32Array,
  b: Float32Array,
  opts: AutoToneOptions
): AutoToneResult {
  const n = r.length;
  const none: AutoToneResult = {
    blackPoint: 0,
    whitePoint: 1,
    appliedFraction: 0,
    blackClipPercent: 0,
    whiteClipPercent: 0,
  };
  if (opts.strength <= 0 || n === 0) return none;

  const params = computeAutoToneParams(r, g, b, opts);
  if (params.degenerate) return none;

  const { blackPoint, whitePoint, appliedFraction } = params;
  const range = Math.max(whitePoint - blackPoint, 1e-4);

  let blackClipped = 0;
  let whiteClipped = 0;

  for (let i = 0; i < n; i++) {
    r[i] = toneMap(r[i], blackPoint, range, AUTOTONE_KNEE_START);
    g[i] = toneMap(g[i], blackPoint, range, AUTOTONE_KNEE_START);
    b[i] = toneMap(b[i], blackPoint, range, AUTOTONE_KNEE_START);
    if (r[i] <= 0 && g[i] <= 0 && b[i] <= 0) blackClipped++;
    if (r[i] >= 1 && g[i] >= 1 && b[i] >= 1) whiteClipped++;
  }

  return {
    blackPoint,
    whitePoint,
    appliedFraction,
    blackClipPercent: (blackClipped / n) * 100,
    whiteClipPercent: (whiteClipped / n) * 100,
  };
}

function toneMap(v: number, blackPoint: number, range: number, kneeStart: number): number {
  let x = (v - blackPoint) / range;
  if (x > kneeStart) {
    // Soft knee: compress values above kneeStart with a smooth roll-off
    // toward 1.0 rather than a hard clip.
    const over = (x - kneeStart) / (1 - kneeStart); // 0..(unbounded)
    const compressed = kneeStart + (1 - kneeStart) * (1 - Math.exp(-over));
    x = compressed;
  }
  return clamp01(x);
}

function clampNum(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}
