import { describe, it, expect, afterEach, vi } from 'vitest';
import { resolveSyncBackend, resolveSyncBackendEx, buildExecutionReport } from '../src/processing/executionReport';
import { resolveBackend, type BackendCapabilities } from '../src/processing/backend';
import { runPipeline } from '../src/processing/pipeline';
import { DEFAULT_SETTINGS } from '../src/types/processing';

const caps = (over: Partial<BackendCapabilities> = {}): BackendCapabilities => ({ webgpu: true, webgl2: true, offscreenCanvas: true, ...over });
const BIG = 1024 * 1024;

describe('resolveSyncBackend: the synchronous pipeline cannot run WebGPU, so it must not claim it', () => {
  it("re-maps an explicit 'webgpu' request to WebGL2 when available, reported as a fall back", () => {
    const { backend, webgpuSkipped } = resolveSyncBackendEx('webgpu', caps(), BIG);
    expect(webgpuSkipped).toBe(true);
    expect(backend.requested).toBe('webgpu');
    expect(backend.resolved).toBe('webgl2');
    expect(backend.fellBack).toBe(true);
    expect(backend.reason).toMatch(/runPipelineAsync/);
  });

  it("re-maps to CPU (still a fall back, still never 'webgpu') when WebGL2 is unavailable", () => {
    const b = resolveSyncBackend('webgpu', caps({ webgl2: false }), BIG);
    expect(b.resolved).toBe('cpu');
    expect(b.fellBack).toBe(true);
  });

  it("'auto' that resolves to WebGPU (large image, navigator.gpu present) is re-mapped too", () => {
    expect(resolveBackend('auto', caps(), BIG).resolved).toBe('webgpu'); // what the pure resolver says
    const b = resolveSyncBackend('auto', caps(), BIG);
    expect(b.resolved).toBe('webgl2');
    expect(b.fellBack).toBe(true);
  });

  it('never returns webgpu for any request/capability combination', () => {
    for (const requested of ['auto', 'cpu', 'webgl2', 'webgpu'] as const) {
      for (const webgpu of [true, false]) {
        for (const webgl2 of [true, false]) {
          for (const px of [16, 512 * 512, BIG]) {
            expect(resolveSyncBackend(requested, caps({ webgpu, webgl2 }), px).resolved).not.toBe('webgpu');
          }
        }
      }
    }
  });

  it('leaves every non-WebGPU resolution exactly as the pure resolver answers (WebGL2 -> CPU chain preserved)', () => {
    for (const requested of ['auto', 'cpu', 'webgl2'] as const) {
      for (const webgl2 of [true, false]) {
        const c = caps({ webgpu: false, webgl2 });
        for (const px of [16, BIG]) {
          expect(resolveSyncBackend(requested, c, px)).toEqual(resolveBackend(requested, c, px));
          expect(resolveSyncBackendEx(requested, c, px).webgpuSkipped).toBe(false);
        }
      }
    }
  });
});

describe('buildExecutionReport', () => {
  it('gpuAccelerated is exactly executed !== cpu', () => {
    for (const executed of ['cpu', 'webgl2', 'webgpu'] as const) {
      const r = buildExecutionReport({ requested: 'auto', executed, attempts: [{ backend: executed, ok: true }] });
      expect(r.gpuAccelerated).toBe(executed !== 'cpu');
    }
  });

  it('fellBack is true when any attempt failed, or a GPU engine was requested but something else ran', () => {
    expect(buildExecutionReport({ requested: 'webgpu', executed: 'webgl2', attempts: [{ backend: 'webgpu', ok: false, reason: 'x' }, { backend: 'webgl2', ok: true }] }).fellBack).toBe(true);
    expect(buildExecutionReport({ requested: 'webgpu', executed: 'cpu', attempts: [{ backend: 'cpu', ok: true }] }).fellBack).toBe(true);
    expect(buildExecutionReport({ requested: 'webgl2', executed: 'webgl2', attempts: [{ backend: 'webgl2', ok: true }] }).fellBack).toBe(false);
    expect(buildExecutionReport({ requested: 'auto', executed: 'cpu', attempts: [{ backend: 'cpu', ok: true }] }).fellBack).toBe(false);
  });

  it('adapter and resources are attached only when WebGPU really executed', () => {
    const adapter = { vendor: 'v', architecture: 'a', device: '', description: '', isFallbackAdapter: false, software: false };
    const resources = { passesDispatched: 1, submits: 1, texturesCreated: 1, textureReuses: 0, peakTexturesLive: 1 };
    const ok = buildExecutionReport({ requested: 'webgpu', executed: 'webgpu', attempts: [{ backend: 'webgpu', ok: true }], adapter, resources });
    expect(ok.adapter).toEqual(adapter);
    expect(ok.resources).toEqual(resources);
    const fell = buildExecutionReport({ requested: 'webgpu', executed: 'webgl2', attempts: [{ backend: 'webgl2', ok: true }], adapter, resources });
    expect(fell.adapter).toBeUndefined();
    expect(fell.resources).toBeUndefined();
  });
});

describe('REGRESSION: the synchronous pipeline must never report WebGPU as active', () => {
  // Before the fix, with navigator.gpu present (every current desktop Chrome) the sync pipeline skipped WebGL2, ran the
  // CPU implementation, and reported `resolved: 'webgpu', fellBack: false` -- for 'webgpu' AND for 'auto' on >= 512x512.
  afterEach(() => vi.unstubAllGlobals());

  function image(w: number, h: number): ImageData {
    const data = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < w * h; i++) data.set([(i * 37) % 256, (i * 37) % 256, (i * 37) % 256, 255], i * 4);
    return new ImageData(data, w, h);
  }

  it.each([
    ['webgpu', 64, 64],
    ['auto', 600, 600],
  ] as const)("engine=%s (%dx%d) with navigator.gpu present but no GPU: reports what really ran (CPU)", (engine: 'webgpu' | 'auto', w: number, h: number) => {
    vi.stubGlobal('navigator', { gpu: {} });
    const r = runPipeline(image(w, h), { ...DEFAULT_SETTINGS, processingEngine: engine });
    expect(r.execution?.executed).toBe('cpu');
    expect(r.execution?.gpuAccelerated).toBe(false);
    expect(r.execution?.fellBack).toBe(true);
    expect(r.execution?.attempts.some((a) => a.backend === 'webgpu' && !a.ok)).toBe(true);
    expect(r.precision?.backend.resolved).not.toBe('webgpu');
    expect(r.precision?.backend.fellBack).toBe(true);
    expect(r.precision?.gpuAccelerated).toBe(false);
  });

  it("engine='cpu' and engine='webgl2' (no GL in Node) report truthfully and never mention webgpu", () => {
    for (const engine of ['cpu', 'webgl2'] as const) {
      const r = runPipeline(image(32, 32), { ...DEFAULT_SETTINGS, processingEngine: engine });
      expect(r.execution?.executed).toBe('cpu');
      expect(r.execution?.attempts.some((a) => a.backend === 'webgpu')).toBe(false);
      expect(r.precision?.backend.resolved).toBe('cpu');
    }
  });

  it('the audit trail lists the executed backend exactly once, last', () => {
    const r = runPipeline(image(32, 32), { ...DEFAULT_SETTINGS, processingEngine: 'cpu' });
    const a = r.execution!.attempts;
    expect(a.filter((x) => x.backend === 'cpu' && x.ok)).toHaveLength(1);
    expect(a[a.length - 1]).toEqual({ backend: 'cpu', ok: true });
  });
});
