/**
 * Truthful execution reporting.
 * -----------------------------
 * `settings.processingEngine` is what the user ASKED for; `resolveBackend`
 * maps that to what the environment CLAIMS to support (e.g. `navigator.gpu`
 * exists). Neither says what actually ran. `ExecutionReport.executed` does:
 * it is derived only from which implementation produced the output pixels,
 * and `'webgpu'` can only appear when every stage ran as WebGPU compute.
 *
 * `attempts` is the audit trail of the fallback chain
 * (WebGPU -> WebGL2 -> CPU): every backend that was tried and either
 * produced the result (ok) or failed (with the reason).
 */

import type { BackendCapabilities, BackendResolution, ProcessingBackendKind, ProcessingEngine } from './backend';
import { resolveBackend } from './backend';
import type { TilingDiagnostics } from './gpu/tilePlanner';
import type { WebGPUAdapterSummary } from './gpu/webgpu/webgpuDevice';
import type { WebGPUResourceStats } from './gpu/webgpu/webgpuFullPipeline';

export interface ExecutionAttempt {
  backend: ProcessingBackendKind;
  ok: boolean;
  /** Why it failed / was skipped (absent for the backend that succeeded). */
  reason?: string;
}

export interface ExecutionReport {
  /** What settings.processingEngine asked for. */
  requested: ProcessingEngine;
  /** The backend that ACTUALLY produced the output pixels. The single source of truth. */
  executed: ProcessingBackendKind;
  /** executed !== 'cpu'. */
  gpuAccelerated: boolean;
  /** True when a GPU backend was requested/attempted and something else produced the result. */
  fellBack: boolean;
  /** Backends tried, in order, ending with the one that produced the result. */
  attempts: ExecutionAttempt[];
  /**
   * The spatial stages ran on the CPU but the precision stage ran as a
   * WebGL2 shader (the pre-existing middle tier of the fallback chain).
   * `executed` is 'cpu' in that case; this flag says the GPU still helped.
   */
  precisionStageOnGpu: boolean;
  /**
   * Present whenever the GPU path went through the tile planner. On WebGL2
   * this only happens when it actually tiled; on WebGPU the planner always
   * runs (even for a single tile), so object presence alone does NOT mean
   * "was tiled" here -- check `tiling.wasTiled` (or any phase's
   * `tileCount > 1`) instead of `!!execution.tiling`.
   */
  tiling?: TilingDiagnostics;
  /** Present only when executed === 'webgpu': which adapter, incl. whether it is a software renderer. */
  adapter?: WebGPUAdapterSummary;
  /** Present only when executed === 'webgpu'. */
  resources?: WebGPUResourceStats;
}

/**
 * `resolveBackend` may answer `'webgpu'`, but the SYNCHRONOUS pipeline
 * (runPipeline) has no WebGPU implementation -- WebGPU readback is
 * inherently async and lives in runPipelineAsync. The synchronous
 * pipeline is the fallback tier beneath WebGPU, so a `'webgpu'`
 * resolution there must be re-resolved to what it can really execute:
 * WebGL2 if available, otherwise CPU, and reported as a fall back.
 *
 * (Previously such a resolution was passed through unchanged: the sync
 * pipeline skipped WebGL2, ran the CPU implementation, and still
 * reported `resolved: 'webgpu', fellBack: false`.)
 */
export function resolveSyncBackendEx(
  requested: ProcessingEngine,
  caps: BackendCapabilities,
  pixelCount: number
): { backend: BackendResolution; /** True when a WebGPU resolution was re-mapped because this pipeline cannot run it. */ webgpuSkipped: boolean } {
  const r = resolveBackend(requested, caps, pixelCount);
  if (r.resolved !== 'webgpu') return { backend: r, webgpuSkipped: false };
  const next = resolveBackend('webgl2', caps, pixelCount);
  return {
    webgpuSkipped: true,
    backend: {
      requested: r.requested,
      resolved: next.resolved,
      fellBack: true,
      reason:
        `WebGPU runs only through runPipelineAsync(); this synchronous pipeline executed on ${next.resolved === 'webgl2' ? 'WebGL2 (with CPU fallback)' : 'the CPU'}` +
        (next.resolved === 'cpu' && next.reason ? ` (${next.reason})` : '') +
        '.',
    },
  };
}

export function resolveSyncBackend(requested: ProcessingEngine, caps: BackendCapabilities, pixelCount: number): BackendResolution {
  return resolveSyncBackendEx(requested, caps, pixelCount).backend;
}

/**
 * RUNTIME truth: did the pixel pipeline of this run actually execute on a GPU
 * backend? Read from the report, never inferred from the requested engine or
 * from `isGpuCapableBackend()` (a static capability): a request for 'webgpu'
 * that fell back is `executed: 'webgl2'` or `'cpu'` here.
 */
export function executedOnGpu(report: Pick<ExecutionReport, 'executed'>): boolean {
  return report.executed === 'webgl2' || report.executed === 'webgpu';
}

export function buildExecutionReport(p: {
  requested: ProcessingEngine;
  executed: ProcessingBackendKind;
  attempts: ExecutionAttempt[];
  precisionStageOnGpu?: boolean;
  tiling?: TilingDiagnostics;
  adapter?: WebGPUAdapterSummary;
  resources?: WebGPUResourceStats;
}): ExecutionReport {
  const requestedGpu = p.requested === 'webgl2' || p.requested === 'webgpu';
  const report: ExecutionReport = {
    requested: p.requested,
    executed: p.executed,
    gpuAccelerated: p.executed !== 'cpu',
    fellBack: p.attempts.some((a) => !a.ok) || (requestedGpu && p.executed !== p.requested),
    attempts: p.attempts,
    precisionStageOnGpu: p.precisionStageOnGpu === true,
  };
  if (p.tiling) report.tiling = p.tiling;
  if (p.executed === 'webgpu') {
    if (p.adapter) report.adapter = p.adapter;
    if (p.resources) report.resources = p.resources;
  }
  return report;
}
