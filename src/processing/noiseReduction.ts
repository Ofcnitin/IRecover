import { boxBlur } from './contrast';
import { clamp01 } from './colorSpace';

export type NoiseMethod = 'gaussian' | 'median' | 'bilateral';

/**
 * Applies the chosen noise-reduction method to a normalized [0,1] intensity
 * buffer. strength is 0..100 and maps to filter radius/sigma. `gaussianPasses`
 * controls how many box-blur passes approximate the Gaussian (more passes =
 * smoother/more accurate approximation, at a real compute cost) -- driven by
 * the Processing Quality setting.
 */
export function reduceNoise(
  intensity: Float32Array,
  width: number,
  height: number,
  method: NoiseMethod,
  strength: number,
  gaussianPasses = 3
): Float32Array {
  if (strength <= 0) return intensity;
  const amt = strength / 100;

  switch (method) {
    case 'gaussian': {
      const radius = Math.max(1, Math.round(amt * 4));
      let blurred = intensity;
      const passes = Math.max(1, gaussianPasses);
      for (let i = 0; i < passes; i++) {
        blurred = boxBlur(blurred, width, height, radius);
      }
      return blend(intensity, blurred, amt);
    }
    case 'median': {
      const radius = amt > 0.66 ? 2 : 1;
      const med = medianFilter(intensity, width, height, radius);
      return blend(intensity, med, amt);
    }
    case 'bilateral': {
      const filtered = bilateralLite(intensity, width, height, Math.max(1, Math.round(amt * 3)), 0.05 + amt * 0.2);
      return blend(intensity, filtered, amt);
    }
  }
}

function blend(a: Float32Array, b: Float32Array, t: number): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] * (1 - t) + b[i] * t;
  return out;
}

function medianFilter(
  intensity: Float32Array,
  width: number,
  height: number,
  radius: number
): Float32Array {
  const out = new Float32Array(intensity.length);
  const windowSize = (2 * radius + 1) * (2 * radius + 1);
  const buf = new Float32Array(windowSize);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let k = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        const yy = clampIdx(y + dy, height);
        for (let dx = -radius; dx <= radius; dx++) {
          const xx = clampIdx(x + dx, width);
          buf[k++] = intensity[yy * width + xx];
        }
      }
      const sorted = buf.slice(0, k).sort();
      out[y * width + x] = sorted[Math.floor(k / 2)];
    }
  }
  return out;
}

/**
 * A lightweight edge-preserving smoothing filter inspired by bilateral
 * filtering: weights neighbors by both spatial and intensity distance,
 * over a small window (kept small for performance in-browser).
 */
function bilateralLite(
  intensity: Float32Array,
  width: number,
  height: number,
  radius: number,
  rangeSigma: number
): Float32Array {
  const out = new Float32Array(intensity.length);
  const spatialSigma = radius / 2 + 0.5;
  const twoSpatialSigma2 = 2 * spatialSigma * spatialSigma;
  const twoRangeSigma2 = 2 * rangeSigma * rangeSigma;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const centerIdx = y * width + x;
      const centerVal = intensity[centerIdx];
      let sum = 0;
      let weightSum = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        const yy = clampIdx(y + dy, height);
        for (let dx = -radius; dx <= radius; dx++) {
          const xx = clampIdx(x + dx, width);
          const val = intensity[yy * width + xx];
          const spatialDist2 = dx * dx + dy * dy;
          const rangeDist2 = (val - centerVal) * (val - centerVal);
          const weight = Math.exp(-spatialDist2 / twoSpatialSigma2 - rangeDist2 / twoRangeSigma2);
          sum += val * weight;
          weightSum += weight;
        }
      }
      out[centerIdx] = weightSum > 0 ? clamp01(sum / weightSum) : centerVal;
    }
  }
  return out;
}

function clampIdx(i: number, size: number): number {
  if (i < 0) return 0;
  if (i >= size) return size - 1;
  return i;
}
