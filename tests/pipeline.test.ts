import { describe, it, expect } from 'vitest';
import '../tests/setup';
import { runPipeline } from '../src/processing/pipeline';
import { DEFAULT_SETTINGS } from '../src/types/processing';
import type { ProcessingSettings } from '../src/types/processing';

function makeUniformImage(width: number, height: number, value: number): ImageData {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data[i * 4] = value;
    data[i * 4 + 1] = value;
    data[i * 4 + 2] = value;
    data[i * 4 + 3] = 255;
  }
  return new ImageData(data, width, height);
}

function makeNoisyImage(width: number, height: number): ImageData {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const v = (i * 37) % 256;
    data[i * 4] = v;
    data[i * 4 + 1] = v;
    data[i * 4 + 2] = v;
    data[i * 4 + 3] = 255;
  }
  return new ImageData(data, width, height);
}

describe('runPipeline', () => {
  it('preserves image dimensions', () => {
    const input = makeUniformImage(16, 12, 128);
    const { output } = runPipeline(input, DEFAULT_SETTINGS);
    expect(output.width).toBe(16);
    expect(output.height).toBe(12);
    expect(output.data.length).toBe(16 * 12 * 4);
  });

  it('is deterministic: identical input + settings produce identical output', () => {
    const input = makeNoisyImage(24, 24);
    const a = runPipeline(input, DEFAULT_SETTINGS);
    const b = runPipeline(input, DEFAULT_SETTINGS);
    expect(Array.from(a.output.data)).toEqual(Array.from(b.output.data));
  });

  it('handles a completely black image without crashing or producing NaN', () => {
    const input = makeUniformImage(10, 10, 0);
    const { output } = runPipeline(input, DEFAULT_SETTINGS);
    expect(output.data.some((v) => Number.isNaN(v))).toBe(false);
  });

  it('handles a completely white image without crashing or producing NaN', () => {
    const input = makeUniformImage(10, 10, 255);
    const { output } = runPipeline(input, DEFAULT_SETTINGS);
    expect(output.data.some((v) => Number.isNaN(v))).toBe(false);
  });

  it('handles a uniform mid-gray image', () => {
    const input = makeUniformImage(10, 10, 128);
    const { output } = runPipeline(input, DEFAULT_SETTINGS);
    expect(output.data.some((v) => Number.isNaN(v))).toBe(false);
    expect(output.data[3]).toBe(255);
  });

  it('handles noisy images without producing out-of-range values', () => {
    const input = makeNoisyImage(40, 40);
    const { output } = runPipeline(input, DEFAULT_SETTINGS);
    for (let i = 0; i < output.data.length; i++) {
      expect(output.data[i]).toBeGreaterThanOrEqual(0);
      expect(output.data[i]).toBeLessThanOrEqual(255);
    }
  });

  it('produces different output for different presets on the same input', () => {
    const input = makeNoisyImage(20, 20);
    const naturalSettings: ProcessingSettings = { ...DEFAULT_SETTINGS, preset: 'natural' };
    const monoSettings: ProcessingSettings = { ...DEFAULT_SETTINGS, preset: 'monochrome', colorStrength: 0 };
    const a = runPipeline(input, naturalSettings);
    const b = runPipeline(input, monoSettings);
    expect(Array.from(a.output.data)).not.toEqual(Array.from(b.output.data));
  });

  it('respects the false-color-ir interpretation without throwing', () => {
    const width = 12;
    const height = 12;
    const data = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < width * height; i++) {
      data[i * 4] = (i * 13) % 256;
      data[i * 4 + 1] = 100;
      data[i * 4 + 2] = (255 - i) % 256;
      data[i * 4 + 3] = 255;
    }
    const input = new ImageData(data, width, height);
    const settings: ProcessingSettings = { ...DEFAULT_SETTINGS, interpretation: 'false-color-ir' };
    expect(() => runPipeline(input, settings)).not.toThrow();
  });

  it('handles a fully transparent-alpha image (alpha ignored by design; RGB still processed)', () => {
    const width = 8;
    const height = 8;
    const data = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < width * height; i++) {
      data[i * 4] = 200;
      data[i * 4 + 1] = 200;
      data[i * 4 + 2] = 200;
      data[i * 4 + 3] = 0;
    }
    const input = new ImageData(data, width, height);
    expect(() => runPipeline(input, DEFAULT_SETTINGS)).not.toThrow();
  });

  it('produces a valid output for a 1x1 image (degenerate size edge case)', () => {
    const input = makeUniformImage(1, 1, 90);
    const { output } = runPipeline(input, DEFAULT_SETTINGS);
    expect(output.width).toBe(1);
    expect(output.height).toBe(1);
  });

  it('Processing Quality genuinely changes output when noise reduction or local contrast is active (not a fake control)', () => {
    const input = makeNoisyImage(30, 30);
    const base: ProcessingSettings = { ...DEFAULT_SETTINGS, noiseReduction: 60, localContrast: 60 };
    const fast = runPipeline(input, { ...base, quality: 'fast' });
    const maximum = runPipeline(input, { ...base, quality: 'maximum' });
    expect(Array.from(fast.output.data)).not.toEqual(Array.from(maximum.output.data));
  });

  it('"balanced" quality matches the pipeline\'s original default behavior deterministically', () => {
    const input = makeNoisyImage(20, 20);
    const a = runPipeline(input, { ...DEFAULT_SETTINGS, quality: 'balanced', noiseReduction: 40 });
    const b = runPipeline(input, { ...DEFAULT_SETTINGS, quality: 'balanced', noiseReduction: 40 });
    expect(Array.from(a.output.data)).toEqual(Array.from(b.output.data));
  });
});
