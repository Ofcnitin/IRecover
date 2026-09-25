import { clamp01 } from './colorSpace';

/**
 * Separable box blur, used as a cheap approximation of a Gaussian blur for
 * local-mean computations. Operates in-place-safe (reads src, writes dst).
 * radius is in pixels.
 */
export function boxBlur(
  src: Float32Array,
  width: number,
  height: number,
  radius: number
): Float32Array {
  if (radius < 1) return src.slice();
  const tmp = new Float32Array(src.length);
  const dst = new Float32Array(src.length);
  const r = Math.round(radius);

  // Horizontal pass.
  for (let y = 0; y < height; y++) {
    const rowOff = y * width;
    let acc = 0;
    for (let x = -r; x <= r; x++) {
      acc += src[rowOff + clampIdx(x, width)];
    }
    for (let x = 0; x < width; x++) {
      tmp[rowOff + x] = acc / (2 * r + 1);
      const addX = x + r + 1;
      const subX = x - r;
      acc += src[rowOff + clampIdx(addX, width)] - src[rowOff + clampIdx(subX, width)];
    }
  }

  // Vertical pass.
  for (let x = 0; x < width; x++) {
    let acc = 0;
    for (let y = -r; y <= r; y++) {
      acc += tmp[clampIdx(y, height) * width + x];
    }
    for (let y = 0; y < height; y++) {
      dst[y * width + x] = acc / (2 * r + 1);
      const addY = y + r + 1;
      const subY = y - r;
      acc += tmp[clampIdx(addY, height) * width + x] - tmp[clampIdx(subY, height) * width + x];
    }
  }

  return dst;
}

function clampIdx(i: number, size: number): number {
  if (i < 0) return 0;
  if (i >= size) return size - 1;
  return i;
}

/**
 * Local contrast enhancement: boosts contrast relative to a local mean
 * (large-radius blur), which approximates the effect of CLAHE without the
 * cost of per-tile histogram equalization. strength 0..100.
 */
export function applyLocalContrast(
  intensity: Float32Array,
  width: number,
  height: number,
  strength: number,
  radius = 24
): Float32Array {
  if (strength <= 0) return intensity;
  const localMean = boxBlur(intensity, width, height, radius);
  const amt = strength / 100;
  const out = new Float32Array(intensity.length);
  for (let i = 0; i < intensity.length; i++) {
    const detail = intensity[i] - localMean[i];
    out[i] = clamp01(intensity[i] + detail * amt * 1.5);
  }
  return out;
}

/** Local variance (texture) map, used by the optional scene heuristics. */
export function localVariance(
  intensity: Float32Array,
  width: number,
  height: number,
  radius = 4
): Float32Array {
  const mean = boxBlur(intensity, width, height, radius);
  const sq = new Float32Array(intensity.length);
  for (let i = 0; i < intensity.length; i++) sq[i] = intensity[i] * intensity[i];
  const meanSq = boxBlur(sq, width, height, radius);
  const out = new Float32Array(intensity.length);
  for (let i = 0; i < intensity.length; i++) {
    out[i] = Math.max(0, meanSq[i] - mean[i] * mean[i]);
  }
  return out;
}
