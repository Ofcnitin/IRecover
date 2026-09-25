import { describe, it, expect } from 'vitest';
import '../tests/setup';
import { applyAutoTone } from '../src/processing/autoTone';

function makeChannel(n: number, value: number): Float32Array {
  return new Float32Array(n).fill(value);
}

describe('applyAutoTone', () => {
  it('does nothing when strength is 0', () => {
    const r = makeChannel(32, 0.3);
    const g = makeChannel(32, 0.3);
    const b = makeChannel(32, 0.3);
    const result = applyAutoTone(r, g, b, { strength: 0 });
    expect(result.appliedFraction).toBe(0);
    // Float32Array stores 0.3 as 0.30000001192...; compare against the value actually stored.
    expect(Array.from(r).every((v) => v === Math.fround(0.3))).toBe(true);
  });

  it('leaves a degenerate (near-flat) histogram untouched to avoid divide-by-near-zero artifacts', () => {
    const r = makeChannel(32, 0.5);
    const g = makeChannel(32, 0.5);
    const b = makeChannel(32, 0.5);
    const result = applyAutoTone(r, g, b, { strength: 100 });
    expect(result.appliedFraction).toBe(0);
  });

  it('stretches a low-contrast (compressed dynamic range) image without producing NaN or out-of-range values', () => {
    const n = 64;
    const r = new Float32Array(n);
    const g = new Float32Array(n);
    const b = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const v = 0.3 + (i / n) * 0.3; // compressed into [0.3, 0.6]
      r[i] = v;
      g[i] = v;
      b[i] = v;
    }
    const result = applyAutoTone(r, g, b, { strength: 80 });
    expect(result.appliedFraction).toBeGreaterThan(0);
    for (let i = 0; i < n; i++) {
      expect(r[i]).toBeGreaterThanOrEqual(0);
      expect(r[i]).toBeLessThanOrEqual(1);
      expect(Number.isNaN(r[i])).toBe(false);
    }
  });

  it('applies a soft knee rather than hard-clipping bright highlights', () => {
    const n = 32;
    const r = new Float32Array(n);
    const g = new Float32Array(n);
    const b = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const v = i / n; // full range 0..~1
      r[i] = v;
      g[i] = v;
      b[i] = v;
    }
    applyAutoTone(r, g, b, { strength: 100 });
    // No channel should have jumped to exactly 1.0 for values that weren't
    // already at/near the top of the range (soft knee, not hard clip).
    const midHighlight = r[Math.floor(n * 0.85)];
    expect(midHighlight).toBeLessThan(1);
  });
});
