import { describe, it, expect } from 'vitest';
import { executesOnCpu, isGpuCapableBackend, resolveBackend, type BackendCapabilities } from '../src/processing/backend';
import { executedOnGpu, buildExecutionReport } from '../src/processing/executionReport';

// Two different questions that used to be conflated:
//   * STATIC capability  -- does this backend KIND have a GPU implementation?
//       isGpuCapableBackend() / executesOnCpu() (its complement)
//   * RUNTIME truth      -- what did THIS run actually execute on?
//       PipelineResult.execution / executedOnGpu()
describe('static capability: isGpuCapableBackend / executesOnCpu', () => {
  it("identifies 'webgl2' and 'webgpu' as GPU-capable, 'cpu' as not", () => {
    expect(isGpuCapableBackend('cpu')).toBe(false);
    expect(isGpuCapableBackend('webgl2')).toBe(true);
    expect(isGpuCapableBackend('webgpu')).toBe(true);
  });

  it('executesOnCpu is the exact complement (CPU is the ONLY implementation only for cpu)', () => {
    for (const kind of ['cpu', 'webgl2', 'webgpu'] as const) {
      expect(executesOnCpu(kind)).toBe(!isGpuCapableBackend(kind));
    }
    expect(executesOnCpu('cpu')).toBe(true);
    expect(executesOnCpu('webgl2')).toBe(false);
    expect(executesOnCpu('webgpu')).toBe(false);
  });

  it("every kind resolveBackend can return is classified, and 'cpu' is the only one that is not GPU-capable", () => {
    const caps: BackendCapabilities = { webgl2: true, webgpu: true, offscreenCanvas: true };
    const kinds = new Set<string>();
    for (const engine of ['auto', 'cpu', 'webgl2', 'webgpu'] as const) {
      for (const px of [16, 1_000_000]) kinds.add(resolveBackend(engine, caps, px).resolved);
    }
    for (const k of kinds) expect(isGpuCapableBackend(k as 'cpu' | 'webgl2' | 'webgpu')).toBe(k !== 'cpu');
  });
});

describe('runtime truth is a separate question: executedOnGpu(report)', () => {
  const base = { requested: 'webgpu' as const, attempts: [] };

  it("a GPU-capable kind that fell back to CPU is capable but NOT executed on GPU", () => {
    const report = buildExecutionReport({
      ...base,
      executed: 'cpu',
      attempts: [{ backend: 'webgpu', ok: false, reason: 'no adapter' }],
    });
    expect(isGpuCapableBackend('webgpu')).toBe(true); // capability unchanged...
    expect(executedOnGpu(report)).toBe(false); // ...but it did not run there
    expect(report.executed).toBe('cpu');
    expect(report.gpuAccelerated).toBe(false);
  });

  it("'webgl2' and 'webgpu' executions are GPU executions; 'cpu' is not", () => {
    expect(executedOnGpu({ executed: 'webgpu' })).toBe(true);
    expect(executedOnGpu({ executed: 'webgl2' })).toBe(true);
    expect(executedOnGpu({ executed: 'cpu' })).toBe(false);
  });

  it('executedOnGpu agrees with the report\'s own gpuAccelerated flag', () => {
    for (const executed of ['cpu', 'webgl2', 'webgpu'] as const) {
      const report = buildExecutionReport({ ...base, executed, attempts: [{ backend: executed, ok: true }] });
      expect(executedOnGpu(report)).toBe(report.gpuAccelerated);
    }
  });
});
