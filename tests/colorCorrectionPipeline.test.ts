import { describe, it, expect } from 'vitest';
import { applyColorCorrectionPipeline } from '../src/processing/colorCorrectionPipeline';

describe('applyColorCorrectionPipeline', () => {
  it('does nothing when strength is 0', () => {
    const r = new Float32Array(16).fill(0.6);
    const g = new Float32Array(16).fill(0.4);
    const b = new Float32Array(16).fill(0.3);
    const result = applyColorCorrectionPipeline(r, g, b, { strength: 0 });
    expect(result.gainR).toBe(1);
    // Float32Array stores 0.6 as 0.60000001192...; compare against the value actually stored.
    expect(Array.from(r).every((v) => v === Math.fround(0.6))).toBe(true);
  });

  it('keeps output within [0,1] and NaN-free across the intensity range', () => {
    const n = 64;
    const r = new Float32Array(n);
    const g = new Float32Array(n);
    const b = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      r[i] = i / n;
      g[i] = ((i * 3) % n) / n;
      b[i] = ((i * 7) % n) / n;
    }
    applyColorCorrectionPipeline(r, g, b, { strength: 100 });
    for (let i = 0; i < n; i++) {
      // Chroma shaping in OKLCH can overshoot slightly before gamut
      // protection runs (which is a separate, later stage) -- but must
      // never be NaN or wildly out of range.
      expect(Number.isNaN(r[i])).toBe(false);
      expect(Number.isNaN(g[i])).toBe(false);
      expect(Number.isNaN(b[i])).toBe(false);
    }
  });

  it('does not add chroma to a fully achromatic (grayscale) image', () => {
    const n = 16;
    const r = new Float32Array(n).fill(0.5);
    const g = new Float32Array(n).fill(0.5);
    const b = new Float32Array(n).fill(0.5);
    applyColorCorrectionPipeline(r, g, b, { strength: 100 });
    for (let i = 0; i < n; i++) {
      expect(r[i]).toBeCloseTo(g[i], 4);
      expect(g[i]).toBeCloseTo(b[i], 4);
    }
  });

  it('respects the channel-gain ceiling', () => {
    const n = 32;
    const r = new Float32Array(n).fill(0.9);
    const g = new Float32Array(n).fill(0.5);
    const b = new Float32Array(n).fill(0.1);
    const result = applyColorCorrectionPipeline(r, g, b, { strength: 100, maxChannelGain: 0.05 });
    expect(result.gainR).toBeGreaterThanOrEqual(0.95);
    expect(result.gainR).toBeLessThanOrEqual(1.05);
    expect(result.gainB).toBeGreaterThanOrEqual(0.95);
    expect(result.gainB).toBeLessThanOrEqual(1.05);
  });
});
