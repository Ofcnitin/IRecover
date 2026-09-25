import type { HistogramData } from '../types/image';

/**
 * Computes a 256-bucket histogram plus summary statistics from an 8-bit
 * intensity buffer (one value per pixel, 0..255).
 */
export function computeHistogram(intensity: Uint8ClampedArray | Uint8Array): HistogramData {
  const buckets = new Array<number>(256).fill(0);
  const n = intensity.length;
  let sum = 0;
  let min = 255;
  let max = 0;

  for (let i = 0; i < n; i++) {
    const v = intensity[i];
    buckets[v]++;
    sum += v;
    if (v < min) min = v;
    if (v > max) max = v;
  }

  const mean = n > 0 ? sum / n : 0;

  // Cumulative distribution for percentile lookups.
  const cumulative = new Array<number>(256);
  let running = 0;
  for (let i = 0; i < 256; i++) {
    running += buckets[i];
    cumulative[i] = running;
  }

  const percentileValue = (p: number): number => {
    const target = (p / 100) * n;
    for (let i = 0; i < 256; i++) {
      if (cumulative[i] >= target) return i;
    }
    return 255;
  };

  const percentiles: Record<number, number> = {
    0.5: percentileValue(0.5),
    1: percentileValue(1),
    2: percentileValue(2),
    50: percentileValue(50),
    98: percentileValue(98),
    99: percentileValue(99),
    99.5: percentileValue(99.5),
  };

  return {
    luminance: buckets,
    min,
    max,
    mean,
    median: percentiles[50],
    percentiles,
  };
}
