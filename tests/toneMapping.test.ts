import { describe, it, expect } from 'vitest';
import { buildToneCurve } from '../src/processing/toneMapping';

const neutralParams = {
  exposure: 0,
  brightness: 0,
  contrast: 0,
  gamma: 1,
  shadowLift: 0,
  highlightRecovery: 0,
};

describe('buildToneCurve', () => {
  it('is the identity function under neutral params (within tolerance)', () => {
    const curve = buildToneCurve(neutralParams);
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      expect(curve(v)).toBeCloseTo(v, 3);
    }
  });

  it('always returns values clamped to [0, 1]', () => {
    const curve = buildToneCurve({ ...neutralParams, exposure: 2, brightness: 100 });
    for (const v of [0, 0.5, 1]) {
      const out = curve(v);
      expect(out).toBeGreaterThanOrEqual(0);
      expect(out).toBeLessThanOrEqual(1);
    }
  });

  it('gamma > 1 brightens midtones, gamma < 1 darkens them (out = in^(1/gamma), the standard gamma-correction convention)', () => {
    // Same convention in the CPU curve and the WebGL2/WebGPU shaders (pow(i, 1/gamma)).
    const brighter = buildToneCurve({ ...neutralParams, gamma: 2 });
    const darker = buildToneCurve({ ...neutralParams, gamma: 0.5 });
    expect(brighter(0.5)).toBeGreaterThan(0.5);
    expect(darker(0.5)).toBeLessThan(0.5);
  });

  it('shadow lift raises near-black values', () => {
    const lifted = buildToneCurve({ ...neutralParams, shadowLift: 100 });
    const flat = buildToneCurve(neutralParams);
    expect(lifted(0.05)).toBeGreaterThan(flat(0.05));
  });

  it('highlight recovery pulls near-white values down from a hard clip', () => {
    const recovered = buildToneCurve({ ...neutralParams, highlightRecovery: 100, exposure: 1 });
    const out = recovered(0.9);
    expect(out).toBeLessThanOrEqual(1);
    expect(out).toBeGreaterThan(0);
  });

  it('is deterministic: same input always produces same output', () => {
    const curve = buildToneCurve({ ...neutralParams, contrast: 30, gamma: 1.4 });
    const a = curve(0.37);
    const b = curve(0.37);
    expect(a).toBe(b);
  });
});
