import { describe, it, expect } from 'vitest';
import { applyAzusaWhiteBalance } from '../src/processing/whiteBalance';

function makeChannel(n: number, value: number): Float32Array {
  return new Float32Array(n).fill(value);
}

describe('applyAzusaWhiteBalance', () => {
  it('does nothing when strength is 0', () => {
    const r = makeChannel(16, 0.6);
    const g = makeChannel(16, 0.4);
    const b = makeChannel(16, 0.3);
    const result = applyAzusaWhiteBalance(r, g, b, { strength: 0 });
    expect(result.gainR).toBe(1);
    expect(result.gainG).toBe(1);
    expect(result.gainB).toBe(1);
    // Float32Array stores 0.6 as 0.60000001192...; compare against the value actually stored.
    expect(Array.from(r).every((v) => v === Math.fround(0.6))).toBe(true);
  });

  it('nudges a near-neutral image with a mild color cast toward balance with high confidence', () => {
    // A mild cast on a near-neutral image is the classic gray-world target:
    // low chroma (spread/luminance ~0.09), so confidence is high (~0.7) and
    // the gains are non-trivial. (A STRONGLY tinted uniform image, e.g.
    // 0.55/0.45/0.35, is deliberately given ~0 confidence -- see the next test.)
    const r = makeChannel(64, 0.47);
    const g = makeChannel(64, 0.45);
    const b = makeChannel(64, 0.43);
    const result = applyAzusaWhiteBalance(r, g, b, { strength: 100 });
    expect(result.confidence).toBeGreaterThan(0.5);
    // Red channel (brightest) should be pulled down, blue (dimmest) pulled up.
    expect(result.gainR).toBeLessThan(1);
    expect(result.gainB).toBeGreaterThan(1);
  });

  it('distrusts a strongly single-hue tinted image (confidence ~0, gains stay at 1)', () => {
    // Documented design (whiteBalance.ts): a very saturated/tinted image is
    // more likely intentionally colored than a neutral scene with a cast, so
    // the gray-world correction is not trusted.
    const r = makeChannel(64, 0.55);
    const g = makeChannel(64, 0.45);
    const b = makeChannel(64, 0.35);
    const result = applyAzusaWhiteBalance(r, g, b, { strength: 100 });
    expect(result.confidence).toBe(0);
    expect(result.gainR).toBe(1);
    expect(result.gainB).toBe(1);
  });

  it('never exceeds the max gain ceiling regardless of strength', () => {
    const r = makeChannel(64, 0.9);
    const g = makeChannel(64, 0.5);
    const b = makeChannel(64, 0.1);
    const result = applyAzusaWhiteBalance(r, g, b, { strength: 100, maxGain: 0.1 });
    expect(result.gainR).toBeGreaterThanOrEqual(0.9);
    expect(result.gainR).toBeLessThanOrEqual(1.1);
    expect(result.gainB).toBeGreaterThanOrEqual(0.9);
    expect(result.gainB).toBeLessThanOrEqual(1.1);
  });

  it('reduces correction confidence for a strongly chromatic (intentionally colorful) image', () => {
    // Highly saturated, non-neutral image -- confidence should be low so the
    // stage doesn't fight an intentional color mapping.
    const n = 64;
    const r = makeChannel(n, 0.95);
    const g = makeChannel(n, 0.15);
    const b = makeChannel(n, 0.05);
    const result = applyAzusaWhiteBalance(r, g, b, { strength: 100 });
    expect(result.confidence).toBeLessThan(0.3);
  });

  it('produces no NaN or out-of-range-in-a-broken-way values', () => {
    const r = makeChannel(16, 0);
    const g = makeChannel(16, 0);
    const b = makeChannel(16, 0);
    const result = applyAzusaWhiteBalance(r, g, b, { strength: 100 });
    expect(Number.isNaN(result.gainR)).toBe(false);
    expect(Array.from(r).some((v) => Number.isNaN(v))).toBe(false);
  });
});
