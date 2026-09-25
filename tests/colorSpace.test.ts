import { describe, it, expect } from 'vitest';
import {
  srgbToLinear,
  linearToSrgb,
  rgbToHsl,
  hslToRgb,
  srgbToOklab,
  oklabToSrgb,
} from '../src/processing/colorSpace';

describe('sRGB <-> linear RGB', () => {
  it('round-trips values without significant drift', () => {
    for (const v of [0, 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 1]) {
      const roundTripped = linearToSrgb(srgbToLinear(v));
      expect(roundTripped).toBeCloseTo(v, 4);
    }
  });

  it('maps 0 to 0 and 1 to 1', () => {
    expect(srgbToLinear(0)).toBe(0);
    expect(srgbToLinear(1)).toBeCloseTo(1, 6);
    expect(linearToSrgb(0)).toBe(0);
    expect(linearToSrgb(1)).toBeCloseTo(1, 6);
  });

  it('is monotonically increasing', () => {
    const samples = Array.from({ length: 20 }, (_, i) => i / 19);
    for (let i = 1; i < samples.length; i++) {
      expect(srgbToLinear(samples[i])).toBeGreaterThanOrEqual(srgbToLinear(samples[i - 1]));
    }
  });
});

describe('RGB <-> HSL', () => {
  it('round-trips grayscale values', () => {
    for (const v of [0, 0.3, 0.5, 0.8, 1]) {
      const [h, s, l] = rgbToHsl(v, v, v);
      expect(s).toBeCloseTo(0, 5);
      const [r2, g2, b2] = hslToRgb(h, s, l);
      expect(r2).toBeCloseTo(v, 4);
      expect(g2).toBeCloseTo(v, 4);
      expect(b2).toBeCloseTo(v, 4);
    }
  });

  it('round-trips a saturated color', () => {
    const original: [number, number, number] = [0.8, 0.3, 0.1];
    const [h, s, l] = rgbToHsl(...original);
    const [r2, g2, b2] = hslToRgb(h, s, l);
    expect(r2).toBeCloseTo(original[0], 4);
    expect(g2).toBeCloseTo(original[1], 4);
    expect(b2).toBeCloseTo(original[2], 4);
  });
});

describe('sRGB <-> OKLab', () => {
  it('round-trips colors within valid RGB range', () => {
    const samples: [number, number, number][] = [
      [0, 0, 0],
      [1, 1, 1],
      [0.5, 0.5, 0.5],
      [0.8, 0.2, 0.1],
      [0.1, 0.6, 0.3],
    ];
    for (const [r, g, b] of samples) {
      const [L, a, bb] = srgbToOklab(r, g, b);
      const [r2, g2, b2] = oklabToSrgb(L, a, bb);
      expect(r2).toBeCloseTo(r, 3);
      expect(g2).toBeCloseTo(g, 3);
      expect(b2).toBeCloseTo(b, 3);
    }
  });

  it('keeps grayscale colors on the neutral axis (a ~ 0, b ~ 0)', () => {
    const [, a, b] = srgbToOklab(0.5, 0.5, 0.5);
    expect(Math.abs(a)).toBeLessThan(1e-4);
    expect(Math.abs(b)).toBeLessThan(1e-4);
  });
});
