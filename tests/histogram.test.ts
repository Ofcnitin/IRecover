import { describe, it, expect } from 'vitest';
import { computeHistogram } from '../src/processing/histogram';

describe('computeHistogram', () => {
  it('handles a completely black image', () => {
    const data = new Uint8ClampedArray(1000).fill(0);
    const h = computeHistogram(data);
    expect(h.min).toBe(0);
    expect(h.max).toBe(0);
    expect(h.mean).toBe(0);
    expect(h.luminance[0]).toBe(1000);
  });

  it('handles a completely white image', () => {
    const data = new Uint8ClampedArray(1000).fill(255);
    const h = computeHistogram(data);
    expect(h.min).toBe(255);
    expect(h.max).toBe(255);
    expect(h.mean).toBe(255);
    expect(h.luminance[255]).toBe(1000);
  });

  it('handles a uniform gray image', () => {
    const data = new Uint8ClampedArray(500).fill(128);
    const h = computeHistogram(data);
    expect(h.min).toBe(128);
    expect(h.max).toBe(128);
    expect(h.mean).toBe(128);
    expect(h.median).toBe(128);
  });

  it('computes correct min/max/mean for a known distribution', () => {
    const data = Uint8ClampedArray.from([0, 50, 100, 150, 200, 250]);
    const h = computeHistogram(data);
    expect(h.min).toBe(0);
    expect(h.max).toBe(250);
    expect(h.mean).toBeCloseTo(125, 5);
  });

  it('percentiles are monotonically non-decreasing', () => {
    const data = new Uint8ClampedArray(10000);
    for (let i = 0; i < data.length; i++) data[i] = i % 256;
    const h = computeHistogram(data);
    const keys = Object.keys(h.percentiles)
      .map(Number)
      .sort((a, b) => a - b);
    for (let i = 1; i < keys.length; i++) {
      expect(h.percentiles[keys[i]]).toBeGreaterThanOrEqual(h.percentiles[keys[i - 1]]);
    }
  });
});
