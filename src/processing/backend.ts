/**
 * Processing Backend abstraction.
 * --------------------------------
 * The high-level pipeline should not scatter `if (webgpu) ... else if
 * (webgl2) ... else ...` throughout its code. This module is the single
 * place that:
 *
 *   1. Detects what's actually available in the current browser context
 *      (main thread OR worker -- `navigator`/`OffscreenCanvas` exist in
 *      both).
 *   2. Resolves a user's requested `ProcessingEngine` ('auto' | 'cpu' |
 *      'webgl2' | 'webgpu') down to a concrete backend, applying sensible
 *      Auto heuristics and safe fallback when a requested backend isn't
 *      available.
 *
 * IMPORTANT / HONEST SCOPING (updated): WebGL2 now has a REAL GLSL ES
 * 3.00 shader implementation of the ENTIRE pipeline from post-levels
 * intensity through sharpening -- noise reduction ('gaussian' and
 * 'bilateral' methods), local contrast, the global tone curve, detail
 * preservation, optional scene heuristics, IR->RGB color-ramp
 * reconstruction, the precision pipeline (Azusa white balance / AutoTone
 * / ColorCorrectionPipeline / gamut protection), and sharpening --
 * across a chain of ping-pong render passes (see
 * processing/gpu/webgl2FullPipeline.ts + fullPipelineShaders.ts, which
 * builds on the original, still-unmodified precision-only shader in
 * webgl2Backend.ts + shaders.ts). `runPipeline()` tries this full GPU
 * path first and falls back to the CPU implementation -- automatically,
 * and honestly reflected in `BackendResolution.resolved` / `.reason` and
 * `PrecisionDiagnostics.gpuAccelerated` -- on any unsupported
 * environment or GPU error, when `noiseMethod` is `'median'` (needs a GPU
 * sorting network, deliberately not rushed), or when tiling itself is
 * impossible. Images exceeding the GPU's texture-size limit are NOT a
 * fallback reason for the full pipeline: they are processed on the GPU
 * in overlapping tiles (gpu/webgl2TiledPipeline.ts) with results
 * identical to the untiled run. (The narrower precision-only path used
 * after a 'median' fallback still does not tile an oversize image.)
 * The app never claims GPU acceleration that didn't happen.
 *
 * WebGPU (WGSL) is implemented (gpu/webgpu/, entered through the ASYNC
 * `runPipelineAsync()` in pipelineAsync.ts -- WebGPU readback is
 * inherently async, so the synchronous `runPipeline()` cannot run it).
 * `resolveBackend()` below only answers what the ENVIRONMENT claims
 * (`navigator.gpu` exists); it cannot know whether an adapter/device can
 * actually be created or whether a run will fall back. The truth about
 * what ran lives in `PipelineResult.execution` (executionReport.ts):
 * `executed` is `'webgpu'` only when WebGPU itself produced the pixels.
 * `executesOnCpu()` below is a static, settings-independent answer
 * ("could this kind ever run off CPU") and is NOT the place to ask
 * whether a *specific* call ran on the GPU -- a `'webgl2'` or `'webgpu'`
 * attempt can still fall back to CPU at runtime (unsupported
 * environment, no adapter, device loss, an untileable request, a shader
 * error). For that per-call, stage-level truth read
 * `PipelineResult.execution` (and `PrecisionDiagnostics`) instead.
 */

export type ProcessingEngine = 'auto' | 'cpu' | 'webgl2' | 'webgpu';
export type ProcessingBackendKind = 'cpu' | 'webgl2' | 'webgpu';

export interface BackendCapabilities {
  webgpu: boolean;
  webgl2: boolean;
  offscreenCanvas: boolean;
}

/**
 * Detects real, currently-available capabilities. Safe to call from the
 * main thread or a Web Worker. Never throws -- any detection failure
 * (e.g. a GPU driver that reports the API but fails context creation)
 * is treated as "unavailable".
 */
export function detectCapabilities(): BackendCapabilities {
  const g = globalThis as unknown as {
    navigator?: { gpu?: unknown };
    OffscreenCanvas?: unknown;
    document?: Document;
  };

  let webgpu = false;
  try {
    webgpu = typeof g.navigator?.gpu !== 'undefined';
  } catch {
    webgpu = false;
  }

  let webgl2 = false;
  try {
    if (typeof g.OffscreenCanvas !== 'undefined') {
      const canvas = new (g.OffscreenCanvas as new (w: number, h: number) => OffscreenCanvas)(2, 2);
      webgl2 = !!canvas.getContext('webgl2');
    } else if (g.document) {
      const canvas = g.document.createElement('canvas');
      webgl2 = !!canvas.getContext('webgl2');
    }
  } catch {
    webgl2 = false;
  }

  return {
    webgpu,
    webgl2,
    offscreenCanvas: typeof g.OffscreenCanvas !== 'undefined',
  };
}

export interface BackendResolution {
  requested: ProcessingEngine;
  resolved: ProcessingBackendKind;
  /** True if the requested engine was unavailable and Auto/CPU fallback was used. */
  fellBack: boolean;
  reason?: string;
}

/** Below this pixel count, CPU is typically faster than paying GPU upload/readback cost. */
const GPU_WORTHWHILE_PIXEL_THRESHOLD = 512 * 512;

/**
 * Resolves a requested processing engine to a concrete, currently-usable
 * backend kind, given real detected capabilities and the image size being
 * processed (used only by the 'auto' heuristic).
 */
export function resolveBackend(
  requested: ProcessingEngine,
  caps: BackendCapabilities,
  pixelCount: number
): BackendResolution {
  if (requested === 'webgpu') {
    if (caps.webgpu) return { requested, resolved: 'webgpu', fellBack: false };
    return { requested, resolved: 'cpu', fellBack: true, reason: 'WebGPU is not available in this browser.' };
  }

  if (requested === 'webgl2') {
    if (caps.webgl2) return { requested, resolved: 'webgl2', fellBack: false };
    return { requested, resolved: 'cpu', fellBack: true, reason: 'WebGL2 is not available in this browser.' };
  }

  if (requested === 'cpu') {
    return { requested, resolved: 'cpu', fellBack: false };
  }

  // 'auto': prefer GPU only when it's actually likely to help.
  const worthwhile = pixelCount >= GPU_WORTHWHILE_PIXEL_THRESHOLD;
  if (worthwhile && caps.webgpu) return { requested, resolved: 'webgpu', fellBack: false };
  if (worthwhile && caps.webgl2) return { requested, resolved: 'webgl2', fellBack: false };
  return { requested, resolved: 'cpu', fellBack: false };
}

/**
 * STATIC capability: does this backend *kind* have a real GPU implementation
 * to attempt? True for 'webgl2' (full-pipeline GLSL + tiling) and 'webgpu'
 * (WGSL compute + tiling); false for 'cpu'.
 *
 * This is a statement about what the code CAN do, never about what a
 * particular call DID do: a GPU-capable kind can still run on the CPU at
 * runtime (no adapter, unsupported environment, a shader/validation error,
 * device loss, an untileable request...). What actually executed is reported
 * by `PipelineResult.execution` -- see `executedOnGpu()` in
 * executionReport.ts -- and must be read from there.
 */
export function isGpuCapableBackend(kind: ProcessingBackendKind): boolean {
  return kind === 'webgl2' || kind === 'webgpu';
}

/**
 * STATIC capability, the exact complement of `isGpuCapableBackend`: true only
 * when CPU is the sole implementation of this kind (i.e. 'cpu'). Kept under
 * its original name for API compatibility. It used to answer `true` for
 * 'webgl2' as well -- a stale answer from before WebGL2 had a GPU
 * implementation, later rationalized by conflating "can run on GPU" with
 * "did run on GPU". Those are different questions; this one is only the
 * former. For what actually ran, read `execution` on the pipeline result.
 */
export function executesOnCpu(kind: ProcessingBackendKind): boolean {
  return !isGpuCapableBackend(kind);
}
