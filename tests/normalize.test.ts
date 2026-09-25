import { describe, it, expect } from 'vitest';
import { extractIntensity, applyLevels, resolveLevelPoints } from '../src/processing/normalize';
import { computeHistogram } from '../src/processing/histogram';

function makeRgba(pixels: [number, number, number][]): Uint8ClampedArray {
  const out = new Uint8ClampedArray(pixels.length * 4);
  pixels.forEach(([r, g, b], i) => {
    out[i * 4] = r;
    out[i * 4 + 1] = g;
    out[i * 4 + 2] = b;
    out[i * 4 + 3] = 255;
  });
  return out;
}

describe('extractIntensity', () => {
  it('averages channels for grayscale-ir interpretation', () => {
    const rgba = makeRgba([[10, 20, 30]]);
    const { intensity } = extractIntensity(rgba, 1, 1, 'grayscale-ir');
    expect(intensity[0]).toBeCloseTo(20, 0);
  });

  it('uses standard luma weights for native-rgb interpretation', () => {
    const rgba = makeRgba([[255, 0, 0]]);
    const { intensity } = extractIntensity(rgba, 1, 1, 'native-rgb');
    expect(intensity[0]).toBeCloseTo(0.299 * 255, 0);
  });

  it('weights the highest-variance channel more for false-color-ir', () => {
    // Red channel varies a lot, green/blue are constant -> red should dominate.
    const rgba = makeRgba([
      [0, 128, 128],
      [255, 128, 128],
    ]);
    const { intensity } = extractIntensity(rgba, 2, 1, 'false-color-ir');
    expect(intensity[0]).toBeLessThan(intensity[1]);
    // With all variance in red, intensity should track the red channel closely.
    expect(intensity[0]).toBeCloseTo(0, 0);
    expect(intensity[1]).toBeCloseTo(255, 0);
  });

  it('produces a histogram matching the extracted intensity', () => {
    const rgba = makeRgba([
      [0, 0, 0],
      [255, 255, 255],
    ]);
    const { intensity, histogram } = extractIntensity(rgba, 2, 1, 'grayscale-ir');
    expect(histogram).toEqual(computeHistogram(intensity));
  });
});

describe('applyLevels', () => {
  it('maps black point to 0 and white point to 1', () => {
    const intensity = Uint8ClampedArray.from([50, 100, 150, 200]);
    const out = applyLevels(intensity, 50, 200);
    expect(out[0]).toBeCloseTo(0, 4);
    expect(out[3]).toBeCloseTo(1, 4);
  });

  it('clamps values outside the black/white range', () => {
    const intensity = Uint8ClampedArray.from([0, 255]);
    const out = applyLevels(intensity, 50, 200);
    expect(out[0]).toBe(0);
    expect(out[1]).toBe(1);
  });

  it('never divides by zero when black and white points collide', () => {
    const intensity = Uint8ClampedArray.from([100]);
    expect(() => applyLevels(intensity, 100, 100)).not.toThrow();
  });
});

describe('resolveLevelPoints', () => {
  it('returns manual points when autoLevels is false', () => {
    const histogram = computeHistogram(Uint8ClampedArray.from([0, 128, 255]));
    const { black, white } = resolveLevelPoints(histogram, false, 10, 240, 1, 99);
    expect(black).toBe(10);
    expect(white).toBe(240);
  });

  it('derives points from percentiles when autoLevels is true', () => {
    const data = new Uint8ClampedArray(1000);
    for (let i = 0; i < data.length; i++) data[i] = Math.min(255, i / 4);
    const histogram = computeHistogram(data);
    const { black, white } = resolveLevelPoints(histogram, true, 0, 255, 1, 99);
    expect(black).toBeGreaterThanOrEqual(0);
    expect(white).toBeLessThanOrEqual(255);
    expect(white).toBeGreaterThan(black);
  });
});
