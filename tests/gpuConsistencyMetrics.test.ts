import { describe, it, expect } from 'vitest';
import { compareArrays, compareInterleavedChannels } from '../tools/gpu-consistency/metrics';

describe('compareArrays', () => {
  it('reports zero error for identical arrays', () => {
    const a = [0.1, 0.5, 0.9, 0];
    const m = compareArrays(a, a, 0.01);
    expect(m.mae).toBe(0);
    expect(m.maxError).toBe(0);
    expect(m.fractionOverTolerance).toBe(0);
    expect(m.countOverTolerance).toBe(0);
  });

  it('computes MAE and max error correctly', () => {
    const ref = [0, 0, 0, 0];
    const actual = [0.1, 0.2, 0, 0.3];
    const m = compareArrays(ref, actual, 1);
    expect(m.mae).toBeCloseTo((0.1 + 0.2 + 0 + 0.3) / 4, 10);
    expect(m.maxError).toBeCloseTo(0.3, 10);
    expect(m.maxErrorIndex).toBe(3);
  });

  it('counts elements over tolerance correctly', () => {
    const ref = [0, 0, 0, 0, 0];
    const actual = [0.01, 0.02, 0.2, 0.005, 0.5];
    const m = compareArrays(ref, actual, 0.05);
    // 0.2 and 0.5 exceed 0.05
    expect(m.countOverTolerance).toBe(2);
    expect(m.fractionOverTolerance).toBeCloseTo(2 / 5, 10);
  });

  it('throws on length mismatch', () => {
    expect(() => compareArrays([1, 2], [1], 0.1)).toThrow(/length mismatch/);
  });

  it('handles empty arrays without dividing by zero', () => {
    const m = compareArrays([], [], 0.1);
    expect(m.mae).toBe(0);
    expect(m.fractionOverTolerance).toBe(0);
    expect(m.count).toBe(0);
  });
});

describe('compareInterleavedChannels', () => {
  it('splits RGBA data into per-channel metrics and an alpha-excluded combined metric', () => {
    // Two RGBA pixels; alpha is deliberately way off but should not count
    // when channelsToCompare=3.
    const ref = [10, 20, 30, 255, 40, 50, 60, 255];
    const actual = [11, 20, 33, 0 /* alpha off, excluded */, 40, 53, 60, 0];
    const result = compareInterleavedChannels(ref, actual, 4, 5, 3);

    expect(result.perChannel).toHaveLength(3);
    // R channel errors: |11-10|=1, |40-40|=0 -> MAE = 0.5
    expect(result.perChannel[0].mae).toBeCloseTo(0.5, 10);
    // G channel errors: |20-20|=0, |53-50|=3 -> MAE = 1.5
    expect(result.perChannel[1].mae).toBeCloseTo(1.5, 10);
    // B channel errors: |33-30|=3, |60-60|=0 -> MAE = 1.5
    expect(result.perChannel[2].mae).toBeCloseTo(1.5, 10);
    // Combined MAE over R,G,B only (6 values): R (1+0) + G (0+3) + B (3+0) = 7 -> 7/6
    expect(result.combined.mae).toBeCloseTo((1 + 0 + 0 + 3 + 3 + 0) / 6, 10);
    // None of R/G/B errors exceed tolerance=5
    expect(result.combined.countOverTolerance).toBe(0);
  });

  it('throws when length is not a multiple of channel count', () => {
    expect(() => compareInterleavedChannels([1, 2, 3], [1, 2, 3], 4, 1)).toThrow(/not a multiple/);
  });

  it('throws on length mismatch between reference and actual', () => {
    expect(() => compareInterleavedChannels([1, 2, 3, 4], [1, 2, 3], 4, 1)).toThrow(/length mismatch/);
  });
});
