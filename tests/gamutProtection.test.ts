import { describe, it, expect } from 'vitest';
import { gamutMapLinearRgb, protectGamut } from '../src/processing/gamutProtection';

describe('gamutMapLinearRgb', () => {
  it('leaves already in-gamut colors unchanged (within float tolerance)', () => {
    const [r, g, b] = gamutMapLinearRgb(0.4, 0.5, 0.6);
    expect(r).toBeCloseTo(0.4, 5);
    expect(g).toBeCloseTo(0.5, 5);
    expect(b).toBeCloseTo(0.6, 5);
  });

  it('maps an out-of-gamut overshoot back into [0,1] on every channel', () => {
    const [r, g, b] = gamutMapLinearRgb(1.4, -0.2, 0.5);
    for (const v of [r, g, b]) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
      expect(Number.isNaN(v)).toBe(false);
    }
  });

  it('handles achromatic out-of-range gray by clamping', () => {
    const [r, g, b] = gamutMapLinearRgb(1.5, 1.5, 1.5);
    expect(r).toBeLessThanOrEqual(1);
    expect(g).toBeLessThanOrEqual(1);
    expect(b).toBeLessThanOrEqual(1);
  });

  it('preserves hue direction reasonably well when compressing chroma', () => {
    // A strongly oversaturated red-ish out-of-gamut color.
    const [r, g, b] = gamutMapLinearRgb(1.8, -0.3, -0.3);
    // Should remain red-dominant after mapping.
    expect(r).toBeGreaterThan(g);
    expect(r).toBeGreaterThan(b);
  });
});

describe('protectGamut', () => {
  it('reports zero out-of-gamut pixels for an already-valid image', () => {
    const n = 16;
    const r = new Float32Array(n).fill(0.5);
    const g = new Float32Array(n).fill(0.5);
    const b = new Float32Array(n).fill(0.5);
    const { outOfGamutCount } = protectGamut(r, g, b);
    expect(outOfGamutCount).toBe(0);
  });

  it('fixes out-of-gamut pixels in place without introducing NaN', () => {
    const n = 8;
    const r = new Float32Array(n).fill(1.3);
    const g = new Float32Array(n).fill(-0.1);
    const b = new Float32Array(n).fill(0.4);
    const { outOfGamutCount } = protectGamut(r, g, b);
    expect(outOfGamutCount).toBe(n);
    for (let i = 0; i < n; i++) {
      expect(r[i]).toBeGreaterThanOrEqual(0);
      expect(r[i]).toBeLessThanOrEqual(1);
      expect(g[i]).toBeGreaterThanOrEqual(0);
      expect(g[i]).toBeLessThanOrEqual(1);
      expect(Number.isNaN(b[i])).toBe(false);
    }
  });
});
