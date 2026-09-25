import { describe, it, expect } from 'vitest';
import { PRESETS, PRESET_LIST } from '../src/processing/presets';
import { sampleRamp } from '../src/processing/colorMapping';

describe('preset definitions', () => {
  it('every preset has at least two color stops, sorted by t, within [0,1]', () => {
    for (const preset of PRESET_LIST) {
      expect(preset.colorStops.length).toBeGreaterThanOrEqual(2);
      for (let i = 1; i < preset.colorStops.length; i++) {
        expect(preset.colorStops[i].t).toBeGreaterThanOrEqual(preset.colorStops[i - 1].t);
      }
      for (const stop of preset.colorStops) {
        expect(stop.t).toBeGreaterThanOrEqual(0);
        expect(stop.t).toBeLessThanOrEqual(1);
        for (const channel of stop.rgb) {
          expect(channel).toBeGreaterThanOrEqual(0);
          expect(channel).toBeLessThanOrEqual(1);
        }
      }
    }
  });

  it('the monochrome preset ramp is a pure black-to-white gradient', () => {
    const mono = PRESETS.monochrome;
    expect(mono.colorStops[0].rgb).toEqual([0, 0, 0]);
    expect(mono.colorStops[mono.colorStops.length - 1].rgb).toEqual([1, 1, 1]);
  });

  it('preset ids are unique and match their key', () => {
    const ids = new Set<string>();
    for (const [key, preset] of Object.entries(PRESETS)) {
      expect(preset.id).toBe(key);
      expect(ids.has(preset.id)).toBe(false);
      ids.add(preset.id);
    }
  });
});

describe('sampleRamp', () => {
  const stops = [
    { t: 0, rgb: [0, 0, 0] as [number, number, number] },
    { t: 0.5, rgb: [0.5, 0.5, 0.5] as [number, number, number] },
    { t: 1, rgb: [1, 1, 1] as [number, number, number] },
  ];

  it('returns the first stop color at t=0 and last stop color at t=1', () => {
    const start = sampleRamp(stops, 0);
    const end = sampleRamp(stops, 1);
    expect(start[0]).toBeCloseTo(0, 3);
    expect(end[0]).toBeCloseTo(1, 3);
  });

  it('clamps out-of-range t values', () => {
    const below = sampleRamp(stops, -0.5);
    const above = sampleRamp(stops, 1.5);
    expect(below).toEqual(stops[0].rgb);
    expect(above).toEqual(stops[2].rgb);
  });

  it('produces monotonically increasing luminance across the ramp (no reversal/banding)', () => {
    let prevLum = -1;
    for (let t = 0; t <= 1; t += 0.05) {
      const [r, g, b] = sampleRamp(stops, t);
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      expect(lum).toBeGreaterThanOrEqual(prevLum - 1e-6);
      prevLum = lum;
    }
  });
});
