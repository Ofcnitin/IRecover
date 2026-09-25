import { describe, it, expect, afterEach, vi } from 'vitest';
import { runPipelineAsync } from '../src/processing/pipelineAsync';
import { runPipeline, type PipelineResult } from '../src/processing/pipeline';
import { DEFAULT_SETTINGS, type ProcessingSettings } from '../src/types/processing';
import type { runFullPipelineWebGPU } from '../src/processing/gpu/webgpu/webgpuFullPipeline';

/**
 * runPipelineAsync's honesty contract, driven by an injected WebGPU
 * implementation (Node has no WebGPU). A fake that succeeds lets us assert
 * what a genuine WebGPU run reports; fakes that fail in every way a real
 * one can let us assert that NOTHING ever reports 'webgpu' unless the
 * WebGPU implementation itself produced the pixels. The same invariants
 * are checked against real WebGPU, real failure injection, in a real
 * browser by tools/gpu-consistency/webgpu/run-webgpu.ts.
 */

afterEach(() => vi.unstubAllGlobals());

function image(w: number, h: number, seed = 7): ImageData {
  const data = new Uint8ClampedArray(w * h * 4);
  let s = seed;
  for (let i = 0; i < w * h; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    const v = (s >>> 24) & 255;
    data.set([v, v, v, 255], i * 4);
  }
  return new ImageData(data, w, h);
}

type Impl = typeof runFullPipelineWebGPU;

function fakeSuccess(): { impl: Impl; calls: any[][] } {
  const calls: any[][] = [];
  const impl = (async (intensity: Float32Array, w: number, h: number, ...rest: any[]) => {
    calls.push([intensity, w, h, ...rest]);
    const rgba = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < w * h; i++) rgba.set([i % 251, (i * 3) % 251, (i * 7) % 251, 255], i * 4);
    return {
      rgba,
      precision: {
        whiteBalance: { gainR: 1, gainG: 1, gainB: 1, confidence: 0 },
        autoTone: { blackPoint: 0, whitePoint: 1, blackClipPercent: 0, whiteClipPercent: 0 },
        colorCorrection: { gainR: 1, gainG: 1, gainB: 1 },
        diagnosticsApproximate: true,
      },
      tiling: { maxTileDim: 2048, colorMap: { cols: 1, rows: 1, tileCount: 1, halo: 0 }, precision: { cols: 1, rows: 1, tileCount: 1 }, sharpen: null },
      adapter: { vendor: 'fake', architecture: 'fake', device: '', description: '', isFallbackAdapter: false, software: false },
      resources: { passesDispatched: 3, submits: 1, texturesCreated: 2, textureReuses: 1, peakTexturesLive: 2 },
    };
  }) as unknown as Impl;
  return { impl, calls };
}

const failing = (thrown: unknown): Impl => (async () => { throw thrown; }) as unknown as Impl;

/** The invariants that make "WebGPU active" trustworthy. */
function assertTruthful(r: PipelineResult, expectExecuted: 'webgpu' | 'webgl2' | 'cpu'): void {
  const ex = r.execution!;
  expect(ex.executed).toBe(expectExecuted);
  expect(ex.gpuAccelerated).toBe(expectExecuted !== 'cpu');
  const last = ex.attempts[ex.attempts.length - 1];
  expect(last.backend).toBe(ex.executed);
  expect(last.ok).toBe(true);
  expect(ex.attempts.filter((a) => a.ok)).toHaveLength(1); // exactly one attempt succeeded
  expect(ex.adapter !== undefined).toBe(expectExecuted === 'webgpu');
  if (r.precision) {
    expect(r.precision.backend.resolved === 'webgpu').toBe(expectExecuted === 'webgpu');
    if (expectExecuted === 'cpu') expect(r.precision.gpuAccelerated).toBe(false);
  }
}

describe('runPipelineAsync: when WebGPU succeeds', () => {
  it("reports executed='webgpu' with adapter, resources and tiling, and no fall back", async () => {
    const { impl } = fakeSuccess();
    const r = await runPipelineAsync(image(24, 20), { ...DEFAULT_SETTINGS, processingEngine: 'webgpu' }, { runWebGPU: impl });
    assertTruthful(r, 'webgpu');
    expect(r.execution!.fellBack).toBe(false);
    expect(r.execution!.attempts).toEqual([{ backend: 'webgpu', ok: true }]);
    expect(r.execution!.adapter?.vendor).toBe('fake');
    expect(r.execution!.resources?.passesDispatched).toBe(3);
    expect(r.execution!.tiling?.colorMap.tileCount).toBe(1);
    expect(r.precision?.backend).toEqual({ requested: 'webgpu', resolved: 'webgpu', fellBack: false });
    expect(r.precision?.gpuAccelerated).toBe(true);
  });

  it('returns the pixels the WebGPU implementation produced, with a matching output histogram', async () => {
    const { impl } = fakeSuccess();
    const r = await runPipelineAsync(image(24, 20), { ...DEFAULT_SETTINGS, processingEngine: 'webgpu' }, { runWebGPU: impl });
    expect(r.output.width).toBe(24);
    expect(r.output.height).toBe(20);
    expect(Array.from(r.output.data.slice(0, 8))).toEqual([0, 0, 0, 255, 1, 3, 7, 255]);
    expect(r.outputHistogram.luminance.reduce((a, b) => a + b, 0)).toBe(24 * 20);
  });

  it('feeds the WebGPU stage exactly what the synchronous pipeline computes in steps 1-3 (no drift)', async () => {
    const { impl, calls } = fakeSuccess();
    const input = image(30, 22, 3);
    const settings: ProcessingSettings = { ...DEFAULT_SETTINGS, processingEngine: 'webgpu', autoLevels: true };
    const r = await runPipelineAsync(input, settings, { runWebGPU: impl });
    const sync = runPipeline(input, { ...settings, processingEngine: 'cpu' });
    expect(r.levelsUsed).toEqual(sync.levelsUsed);
    expect(r.inputHistogram).toEqual(sync.inputHistogram);
    const [intensity, w, h, passedSettings] = calls[0];
    expect(intensity).toBeInstanceOf(Float32Array);
    expect(intensity.length).toBe(30 * 22);
    expect([w, h]).toEqual([30, 22]);
    expect(passedSettings.processingEngine).toBe('webgpu');
    expect(Math.min(...intensity)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...intensity)).toBeLessThanOrEqual(1);
  });

  it('omits precision diagnostics when the precision pipeline is off (like the other backends)', async () => {
    const { impl } = fakeSuccess();
    const r = await runPipelineAsync(image(16, 16), { ...DEFAULT_SETTINGS, precisionPipeline: false, processingEngine: 'webgpu' }, { runWebGPU: impl });
    expect(r.precision).toBeUndefined();
    assertTruthful(r, 'webgpu');
  });
});

describe('runPipelineAsync: WebGPU fails -> the EXISTING WebGL2 -> CPU chain, reported honestly', () => {
  const failures: [string, unknown][] = [
    ['an Error', new Error('injected: device lost')],
    ['a named error', Object.assign(new Error('no adapter'), { name: 'WebGPUUnavailableError' })],
    ['a non-Error rejection', 'string rejection'],
    ['undefined', undefined],
  ];

  it.each(failures)('%s => never reports webgpu; records why; pixels equal the synchronous pipeline', async (_n: string, thrown: unknown) => {
    const input = image(24, 20);
    const settings: ProcessingSettings = { ...DEFAULT_SETTINGS, sceneHeuristics: true, noiseReduction: 25, noiseMethod: 'gaussian', sharpenAmount: 40, processingEngine: 'webgpu' };
    const r = await runPipelineAsync(input, settings, { runWebGPU: failing(thrown) });

    assertTruthful(r, 'cpu'); // Node has no GL, so the sync chain bottoms out on CPU
    expect(r.execution!.fellBack).toBe(true);
    expect(r.execution!.attempts[0]).toMatchObject({ backend: 'webgpu', ok: false });
    expect(typeof r.execution!.attempts[0].reason).toBe('string');
    expect(r.execution!.attempts[0].reason!.length).toBeGreaterThan(0);
    expect(r.execution!.attempts.map((a) => a.backend)).toEqual(['webgpu', 'webgl2', 'cpu']); // the chain was walked in order
    expect(r.precision?.backend.requested).toBe('webgpu');
    expect(r.precision?.backend.fellBack).toBe(true);
    expect(r.precision?.backend.reason).toMatch(/WebGPU did not run/);

    const cpu = runPipeline(input, { ...settings, processingEngine: 'cpu' });
    expect(Array.from(r.output.data)).toEqual(Array.from(cpu.output.data));
  });

  it('names the failure reason from the thrown error', async () => {
    const r = await runPipelineAsync(image(16, 16), { ...DEFAULT_SETTINGS, processingEngine: 'webgpu' }, { runWebGPU: failing(new Error('queue.submit exploded')) });
    expect(r.execution!.attempts[0].reason).toContain('queue.submit exploded');
  });

  it("with no navigator.gpu at all, an explicit 'webgpu' request falls back and says WebGPU is unavailable", async () => {
    const r = await runPipelineAsync(image(16, 16), { ...DEFAULT_SETTINGS, processingEngine: 'webgpu' });
    assertTruthful(r, 'cpu');
    expect(r.execution!.fellBack).toBe(true);
    expect(r.execution!.attempts[0].reason).toMatch(/not available/);
  });

  it('a WebGPU failure never leaves executed=webgpu no matter how the run failed', async () => {
    for (const [, thrown] of failures) {
      const r = await runPipelineAsync(image(16, 16), { ...DEFAULT_SETTINGS, processingEngine: 'webgpu' }, { runWebGPU: failing(thrown) });
      expect(r.execution!.executed).not.toBe('webgpu');
      expect(r.execution!.gpuAccelerated).toBe(false);
      expect(r.precision?.backend.resolved).not.toBe('webgpu');
    }
  });
});

describe('runPipelineAsync: when WebGPU is NOT attempted', () => {
  it.each(['cpu', 'webgl2'] as const)("engine=%s never calls the WebGPU implementation", async (engine: 'cpu' | 'webgl2') => {
    const { impl, calls } = fakeSuccess();
    const r = await runPipelineAsync(image(16, 16), { ...DEFAULT_SETTINGS, processingEngine: engine }, { runWebGPU: impl });
    expect(calls).toHaveLength(0);
    expect(r.execution!.attempts.some((a) => a.backend === 'webgpu')).toBe(false);
    assertTruthful(r, 'cpu');
  });

  it("engine=auto on a small image does not attempt WebGPU even when navigator.gpu exists (not worthwhile)", async () => {
    vi.stubGlobal('navigator', { gpu: {} });
    const { impl, calls } = fakeSuccess();
    const r = await runPipelineAsync(image(32, 32), { ...DEFAULT_SETTINGS, processingEngine: 'auto' }, { runWebGPU: impl });
    expect(calls).toHaveLength(0);
    expect(r.execution!.attempts.some((a) => a.backend === 'webgpu')).toBe(false);
  });

  it('engine=auto on a large image with navigator.gpu present DOES attempt WebGPU', async () => {
    vi.stubGlobal('navigator', { gpu: {} });
    const { impl, calls } = fakeSuccess();
    const r = await runPipelineAsync(image(600, 600), { ...DEFAULT_SETTINGS, processingEngine: 'auto' }, { runWebGPU: impl });
    expect(calls).toHaveLength(1);
    assertTruthful(r, 'webgpu');
    expect(r.execution!.requested).toBe('auto');
  });

  it("engine=auto on a large image, WebGPU present but failing -> falls back and reports it", async () => {
    vi.stubGlobal('navigator', { gpu: {} });
    const r = await runPipelineAsync(image(600, 600), { ...DEFAULT_SETTINGS, processingEngine: 'auto', precisionPipeline: false }, { runWebGPU: failing(new Error('no device')) });
    expect(r.execution!.executed).not.toBe('webgpu');
    expect(r.execution!.fellBack).toBe(true);
    expect(r.execution!.attempts[0]).toMatchObject({ backend: 'webgpu', ok: false });
  });
});
