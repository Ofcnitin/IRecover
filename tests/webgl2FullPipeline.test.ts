import { describe, it, expect } from 'vitest';
import { assertFullPipelineWebGL2Supported, runFullPipelineWebGL2 } from '../src/processing/gpu/webgl2FullPipeline';
import { runPipeline } from '../src/processing/pipeline';
import { DEFAULT_SETTINGS } from '../src/types/processing';
import { PRESETS } from '../src/processing/presets';

function makeIntensity(n: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (i % 5) / 4;
  return out;
}

describe('WebGL2 full pipeline: environment safety', () => {
  // Same environment limitation as webgl2PrecisionPipeline.test.ts: no
  // real GPU/DOM under Vitest/Node (see tests/setup.ts), so these tests
  // exercise the fallback contract, not actual shader execution.

  it('assertFullPipelineWebGL2Supported throws (not crashes) with no GPU/canvas available', () => {
    expect(() => assertFullPipelineWebGL2Supported()).toThrow();
  });

  it('runFullPipelineWebGL2 throws cleanly when unsupported', () => {
    expect(() =>
      runFullPipelineWebGL2(makeIntensity(64), 8, 8, DEFAULT_SETTINGS, PRESETS.natural.colorStops, {
        gaussianPasses: 3,
        localContrastRadius: 24,
      })
    ).toThrow();
  });

  it("rejects the 'median' noise method with its own explicit, honest error before touching the GPU", () => {
    const settings = { ...DEFAULT_SETTINGS, noiseReduction: 40, noiseMethod: 'median' as const };
    expect(() =>
      runFullPipelineWebGL2(makeIntensity(16), 4, 4, settings, PRESETS.natural.colorStops, {
        gaussianPasses: 3,
        localContrastRadius: 24,
      })
    ).toThrow(/median/i);
  });

  it('rejects buffer/size mismatches before any GPU work', () => {
    const wrongSize = new Float32Array(10);
    expect(() =>
      runFullPipelineWebGL2(wrongSize, 4, 4, DEFAULT_SETTINGS, PRESETS.natural.colorStops, {
        gaussianPasses: 3,
        localContrastRadius: 24,
      })
    ).toThrow(/buffer\/size mismatch/);
  });

  it('rejects a preset with more color stops than the shader supports', () => {
    const tooManyStops = Array.from({ length: 20 }, (_, i) => ({
      t: i / 19,
      rgb: [0.5, 0.5, 0.5] as [number, number, number],
    }));
    expect(() =>
      runFullPipelineWebGL2(makeIntensity(16), 4, 4, DEFAULT_SETTINGS, tooManyStops, {
        gaussianPasses: 3,
        localContrastRadius: 24,
      })
    ).toThrow(/MAX_STOPS/);
  });
});

describe('runPipeline: end-to-end regression coverage for the GPU-attempt restructuring', () => {
  // These run the CPU fallback path in full (the only path this
  // environment can execute), which is exactly what real browsers
  // without WebGL2 support take too -- and it's the path every browser
  // takes for the 'median' noise method regardless of GPU support. This
  // guards against the pipeline.ts control-flow rewrite (wrapping the
  // original steps 4-10 in an `if (fullGpuResult) {...} else {...}`)
  // having silently broken the CPU reference implementation.

  function makeInput(width: number, height: number): ImageData {
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

  it('produces a correctly-shaped result with default settings', () => {
    const input = makeInput(12, 12);
    const result = runPipeline(input, DEFAULT_SETTINGS);
    expect(result.output.width).toBe(12);
    expect(result.output.height).toBe(12);
    expect(result.output.data.length).toBe(12 * 12 * 4);
    expect(result.inputHistogram).toBeTruthy();
    expect(result.outputHistogram).toBeTruthy();
  });

  it('reports honest (non-GPU) precision diagnostics when no GPU is available', () => {
    const input = makeInput(8, 8);
    const result = runPipeline(input, DEFAULT_SETTINGS);
    expect(result.precision).toBeTruthy();
    expect(result.precision!.backend.resolved).toBe('cpu');
    expect(result.precision!.gpuAccelerated).toBe(false);
  });

  it('handles noiseReduction + localContrast + sharpen + a non-default preset together', () => {
    const input = makeInput(10, 10);
    const settings = {
      ...DEFAULT_SETTINGS,
      preset: 'landscape' as const,
      noiseReduction: 40,
      noiseMethod: 'bilateral' as const,
      localContrast: 30,
      sharpenAmount: 60,
    };
    const result = runPipeline(input, settings);
    expect(result.output.data.length).toBe(10 * 10 * 4);
  });

  it("still produces valid output when noiseMethod is 'median' (GPU-unsupported, CPU-only)", () => {
    const input = makeInput(8, 8);
    const settings = { ...DEFAULT_SETTINGS, noiseReduction: 40, noiseMethod: 'median' as const };
    const result = runPipeline(input, settings);
    expect(result.output.data.length).toBe(8 * 8 * 4);
  });

  it('produces valid output with precisionPipeline disabled', () => {
    const input = makeInput(8, 8);
    const settings = { ...DEFAULT_SETTINGS, precisionPipeline: false };
    const result = runPipeline(input, settings);
    expect(result.output.data.length).toBe(8 * 8 * 4);
    expect(result.precision).toBeUndefined();
  });
});
