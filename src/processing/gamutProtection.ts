/**
 * Gamut Protection
 * -----------------
 * Final safety stage before sRGB 8-bit encoding. Prevents the "neon"
 * artifacts that classical color-correction math (channel matrices,
 * saturation boosts, OKLab re-composition) can produce when a computed
 * color falls outside the displayable sRGB gamut.
 *
 * Instead of naive `clamp(0, 1)` per channel -- which shifts hue and can
 * produce visible banding/posterization at gamut edges -- this stage
 * works in OKLCH (the polar form of OKLab: Lightness, Chroma, hue) and
 * reduces Chroma only, along a line of constant Lightness and hue, until
 * the color re-projects inside the sRGB cube. This is the same family of
 * technique used by the CSS Color 4 gamut-mapping algorithm.
 *
 * Input/output: linear-light RGB, each channel roughly 0..1 but the
 * caller may pass values that overshoot slightly (e.g. 1.04) as a result
 * of upstream float-precision processing -- that's exactly what this
 * stage exists to resolve gracefully.
 */

import { linearRgbToOklab, oklabToLinearRgb } from './colorSpace';

export interface GamutProtectionOptions {
  /**
   * How many bisection steps to use when searching for the largest
   * in-gamut chroma. 8 steps is already sub-1% precision; more is rarely
   * worth the cost.
   */
  steps?: number;
  /** Small epsilon so we don't chase numerically-perfect gamut boundary. */
  epsilon?: number;
}

export const DEFAULT_STEPS = 10;
export const DEFAULT_EPS = 1e-4;

function inGamut(r: number, g: number, b: number, eps: number): boolean {
  return r >= -eps && r <= 1 + eps && g >= -eps && g <= 1 + eps && b >= -eps && b <= 1 + eps;
}

/**
 * Maps a single linear-light RGB triple back into the sRGB gamut,
 * preserving Lightness and hue and reducing Chroma only as much as
 * necessary. Values already in-gamut are returned unchanged (aside from
 * float rounding), so this is safe to run unconditionally.
 */
export function gamutMapLinearRgb(
  r: number,
  g: number,
  b: number,
  opts: GamutProtectionOptions = {}
): [number, number, number] {
  const steps = opts.steps ?? DEFAULT_STEPS;
  const eps = opts.epsilon ?? DEFAULT_EPS;

  if (inGamut(r, g, b, eps)) {
    return [clamp01(r), clamp01(g), clamp01(b)];
  }

  const [L, a, bb] = linearRgbToOklab(r, g, b);
  const C = Math.sqrt(a * a + bb * bb);
  if (C < 1e-6) {
    // Achromatic (gray) but out of range purely on lightness -- just clamp.
    return [clamp01(r), clamp01(g), clamp01(b)];
  }
  const hueA = a / C;
  const hueB = bb / C;

  // Binary search the largest chroma (0..C) that stays in-gamut at this L/hue.
  let lo = 0;
  let hi = C;
  let best: [number, number, number] = [clamp01(r), clamp01(g), clamp01(b)];

  for (let i = 0; i < steps; i++) {
    const mid = (lo + hi) / 2;
    const candidate = oklabToLinearRgb(L, hueA * mid, hueB * mid);
    if (inGamut(candidate[0], candidate[1], candidate[2], eps)) {
      best = candidate;
      lo = mid;
    } else {
      hi = mid;
    }
  }

  return [clamp01(best[0]), clamp01(best[1]), clamp01(best[2])];
}

/**
 * Applies gamut protection across a full planar linear-RGB image buffer
 * in place. Only touches pixels that are actually out-of-gamut (the vast
 * majority of a well-processed image), so the average-case cost is low.
 */
export function protectGamut(
  r: Float32Array,
  g: Float32Array,
  b: Float32Array,
  opts: GamutProtectionOptions = {}
): { outOfGamutCount: number } {
  const eps = opts.epsilon ?? DEFAULT_EPS;
  let outOfGamutCount = 0;
  for (let i = 0; i < r.length; i++) {
    if (!inGamut(r[i], g[i], b[i], eps)) {
      outOfGamutCount++;
      const [nr, ng, nb] = gamutMapLinearRgb(r[i], g[i], b[i], opts);
      r[i] = nr;
      g[i] = ng;
      b[i] = nb;
    } else {
      r[i] = clamp01(r[i]);
      g[i] = clamp01(g[i]);
      b[i] = clamp01(b[i]);
    }
  }
  return { outOfGamutCount };
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}
