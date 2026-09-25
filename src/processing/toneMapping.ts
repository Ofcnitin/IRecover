import { clamp01 } from './colorSpace';

export interface ToneParams {
  exposure: number; // stops
  brightness: number; // -100..100
  contrast: number; // -100..100
  gamma: number; // 0.2..3
  shadowLift: number; // 0..100
  highlightRecovery: number; // 0..100
}

/**
 * A smooth, deterministic filmic-ish tone curve. Applied per-pixel to a
 * normalized [0,1] intensity value. Designed to avoid crushed blacks and
 * clipped highlights while still giving punch in the midtones.
 */
export function buildToneCurve(params: ToneParams): (i: number) => number {
  const exposureMul = Math.pow(2, params.exposure);
  const brightnessAdd = params.brightness / 255;
  const contrastAmt = clamp01((params.contrast + 100) / 200); // 0..1, 0.5 = neutral
  const gamma = Math.max(0.05, params.gamma);
  const shadowLift = clamp01(params.shadowLift / 100);
  const highlightRecovery = clamp01(params.highlightRecovery / 100);

  return (iRaw: number): number => {
    let i = iRaw * exposureMul + brightnessAdd;
    i = clamp01(i);

    // Shadow lift: raises near-black values without flattening the whole range.
    if (shadowLift > 0) {
      i = i + shadowLift * 0.35 * (1 - i) * Math.exp(-i * 6);
    }

    // Highlight recovery: gentle roll-off near 1.0 instead of hard clipping.
    if (highlightRecovery > 0) {
      const knee = 1 - highlightRecovery * 0.5;
      if (i > knee) {
        const over = (i - knee) / Math.max(0.0001, 1 - knee);
        i = knee + (1 - knee) * (1 - Math.exp(-over * 2));
      }
    }

    // Gamma.
    i = Math.pow(clamp01(i), 1 / gamma);

    // Contrast: an S-curve pivoted at mid-gray (0.5), strength from contrastAmt.
    const s = (contrastAmt - 0.5) * 2; // -1..1
    if (s !== 0) {
      const k = s > 0 ? 1 + s * 3 : 1 / (1 - s * 3);
      i = sCurve(i, k);
    }

    return clamp01(i);
  };
}

/** Pivoted power-curve S-shape around 0.5, monotonic and smooth. */
function sCurve(x: number, k: number): number {
  if (x <= 0.5) {
    return 0.5 * Math.pow(x / 0.5, k);
  }
  return 1 - 0.5 * Math.pow((1 - x) / 0.5, k);
}
