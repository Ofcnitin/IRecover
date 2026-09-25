import type { InputInterpretation } from '../types/image';
import { computeHistogram } from './histogram';
import type { HistogramData } from '../types/image';

export interface IntensityExtraction {
  /** One 8-bit intensity value per pixel, source dynamic range, not yet leveled. */
  intensity: Uint8ClampedArray;
  histogram: HistogramData;
}

/**
 * Reduces the source RGBA buffer down to a single per-pixel intensity value
 * according to how the user says the input should be interpreted.
 */
export function extractIntensity(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  interpretation: InputInterpretation
): IntensityExtraction {
  const n = width * height;
  const intensity = new Uint8ClampedArray(n);

  if (interpretation === 'grayscale-ir') {
    // Straightforward: whichever channel(s) carry the signal, average them.
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      intensity[i] = (rgba[p] + rgba[p + 1] + rgba[p + 2]) / 3;
    }
  } else if (interpretation === 'false-color-ir') {
    // The RGB channels are a false-color encoding of a single physical quantity.
    // Recover a single intensity signal via channel statistics: weight each
    // channel by its contribution to the between-pixel variance, favoring the
    // channel that actually carries information (rather than assuming standard
    // luma weights, which are tuned for natural photographs, not false color).
    const means = [0, 0, 0];
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      means[0] += rgba[p];
      means[1] += rgba[p + 1];
      means[2] += rgba[p + 2];
    }
    means[0] /= n;
    means[1] /= n;
    means[2] /= n;

    const variances = [0, 0, 0];
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      variances[0] += (rgba[p] - means[0]) ** 2;
      variances[1] += (rgba[p + 1] - means[1]) ** 2;
      variances[2] += (rgba[p + 2] - means[2]) ** 2;
    }
    const total = variances[0] + variances[1] + variances[2] || 1;
    const w = [variances[0] / total, variances[1] / total, variances[2] / total];

    for (let i = 0, p = 0; i < n; i++, p += 4) {
      intensity[i] = w[0] * rgba[p] + w[1] * rgba[p + 1] + w[2] * rgba[p + 2];
    }
  } else {
    // native-rgb: treat as an ordinary photograph's luma. This mode mostly
    // exists so users can compare "do nothing" output, or feed in RGB NIR
    // composites that already look reasonable.
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      intensity[i] = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
    }
  }

  return { intensity, histogram: computeHistogram(intensity) };
}

/**
 * Stretches intensity using a black/white point (either manual or
 * percentile-derived) into a normalized Float32Array in [0, 1].
 */
export function applyLevels(
  intensity: Uint8ClampedArray,
  blackPoint: number,
  whitePoint: number
): Float32Array {
  const n = intensity.length;
  const out = new Float32Array(n);
  const range = Math.max(1, whitePoint - blackPoint);
  for (let i = 0; i < n; i++) {
    let v = (intensity[i] - blackPoint) / range;
    if (v < 0) v = 0;
    else if (v > 1) v = 1;
    out[i] = v;
  }
  return out;
}

export function resolveLevelPoints(
  histogram: HistogramData,
  autoLevels: boolean,
  manualBlack: number,
  manualWhite: number,
  blackPercentile: number,
  whitePercentile: number
): { black: number; white: number } {
  if (!autoLevels) {
    return { black: manualBlack, white: Math.max(manualBlack + 1, manualWhite) };
  }
  const black = percentileFromHistogram(histogram, blackPercentile);
  const white = Math.max(black + 1, percentileFromHistogram(histogram, whitePercentile));
  return { black, white };
}

function percentileFromHistogram(histogram: HistogramData, p: number): number {
  const key = Object.keys(histogram.percentiles)
    .map(Number)
    .reduce((closest, cur) => (Math.abs(cur - p) < Math.abs(closest - p) ? cur : closest));
  return histogram.percentiles[key];
}
