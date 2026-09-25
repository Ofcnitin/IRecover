/**
 * WebGPU device + compiled-pipeline management.
 * ----------------------------------------------
 * Responsibilities:
 *  - Acquire a real adapter and device through `navigator.gpu` (works on
 *    the main thread and in workers; no canvas involved -- the pipeline is
 *    pure compute).
 *  - Cache ONE context per device (compiled pipelines are expensive) and
 *    drop it if the device is lost, so the next run re-acquires cleanly
 *    instead of using a dead device.
 *  - Compile each WGSL kernel lazily, check `getCompilationInfo()` and
 *    throw on any error message (a silently-broken shader must become a
 *    fallback, never a wrong image).
 *  - Describe the adapter honestly, including whether it is a software
 *    renderer, so "WebGPU: active" can never hide that the "GPU" is a CPU
 *    emulation.
 *
 * `navigator.gpu` merely EXISTING (what processing/backend.ts's
 * detectCapabilities() checks) does not mean a usable adapter exists;
 * only requestAdapter()+requestDevice() succeeding does. Every failure
 * path here throws WebGPUUnavailableError with a human-readable reason,
 * which the async pipeline records in its fallback audit trail.
 */

import { KERNELS, KERNEL_NAMES, type KernelName, type KernelSpec } from './wgslShaders';
import {
  SHADER_STAGE,
  type WAdapter,
  type WBindGroupLayout,
  type WBindGroupLayoutEntry,
  type WComputePipeline,
  type WDevice,
  type WGpu,
  type WLimits,
} from './wgpuTypes';

export class WebGPUUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'WebGPUUnavailableError';
  }
}

export interface WebGPUAdapterSummary {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  /** Adapter reported itself as a fallback (software) adapter. */
  isFallbackAdapter: boolean;
  /**
   * True when the adapter is (or looks like) a CPU/software renderer, e.g.
   * SwiftShader or llvmpipe. Reported so callers/UI never present a
   * software-emulated adapter as a hardware GPU.
   */
  software: boolean;
}

export interface CompiledKernel {
  spec: KernelSpec;
  pipeline: WComputePipeline;
  bindGroupLayout: WBindGroupLayout;
}

export interface WebGPUContext {
  device: WDevice;
  limits: WLimits;
  adapter: WebGPUAdapterSummary;
  /** Set (and the context evicted from the cache) if the device is lost. */
  lost: { reason: string; message: string } | null;
  /** Lazily compiles (once per device) and returns a kernel. */
  getKernel(name: KernelName): Promise<CompiledKernel>;
}

export interface AcquireOptions {
  /** Refuse a software/fallback adapter (default false: report it, don't refuse it). */
  requireHardwareAdapter?: boolean;
  /** Test seam: supply a `gpu` object instead of reading navigator.gpu. */
  gpu?: WGpu;
  /** Bypass the cache (tests). */
  fresh?: boolean;
}

const SOFTWARE_RE = /swiftshader|llvmpipe|lavapipe|softpipe|software rasterizer|software renderer|basic render/i;

function summarizeAdapter(adapter: WAdapter): WebGPUAdapterSummary {
  const info = adapter.info ?? {};
  const vendor = info.vendor ?? '';
  const architecture = info.architecture ?? '';
  const device = info.device ?? '';
  const description = info.description ?? '';
  const isFallbackAdapter = info.isFallbackAdapter === true || adapter.isFallbackAdapter === true;
  return {
    vendor,
    architecture,
    device,
    description,
    isFallbackAdapter,
    software: isFallbackAdapter || SOFTWARE_RE.test(`${vendor} ${architecture} ${device} ${description}`),
  };
}

function getGpu(override?: WGpu): WGpu {
  if (override) return override;
  const nav = (globalThis as unknown as { navigator?: { gpu?: WGpu } }).navigator;
  if (!nav || !nav.gpu) throw new WebGPUUnavailableError('WebGPU is not available in this environment (navigator.gpu is undefined).');
  return nav.gpu;
}

function buildBindGroupLayout(device: WDevice, spec: KernelSpec, label: string): WBindGroupLayout {
  const entries: WBindGroupLayoutEntry[] = [{ binding: 0, visibility: SHADER_STAGE.COMPUTE, buffer: { type: 'uniform' } }];
  for (let i = 0; i < spec.inputs; i++) {
    // float32 textures are "unfilterable-float" unless the optional
    // float32-filterable feature is enabled; we only ever textureLoad them.
    entries.push({ binding: 1 + i, visibility: SHADER_STAGE.COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } });
  }
  spec.outputs.forEach((format, k) => {
    entries.push({
      binding: 1 + spec.inputs + k,
      visibility: SHADER_STAGE.COMPUTE,
      storageTexture: { access: 'write-only', format, viewDimension: '2d' },
    });
  });
  return device.createBindGroupLayout({ entries, label });
}

async function compileKernel(device: WDevice, name: KernelName): Promise<CompiledKernel> {
  const spec = KERNELS[name];
  const module = device.createShaderModule({ code: spec.wgsl, label: `ir2vis/${name}` });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length > 0) {
    const first = errors[0];
    throw new WebGPUUnavailableError(`WGSL kernel '${name}' failed to compile: ${first.message} (line ${first.lineNum}:${first.linePos}).`);
  }
  const bindGroupLayout = buildBindGroupLayout(device, spec, `ir2vis/${name}/bgl`);
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout], label: `ir2vis/${name}/layout` });
  const pipeline = await device.createComputePipelineAsync({
    layout,
    compute: { module, entryPoint: 'main' },
    label: `ir2vis/${name}`,
  });
  return { spec, pipeline, bindGroupLayout };
}

function makeContext(device: WDevice, adapter: WebGPUAdapterSummary, onLost: () => void): WebGPUContext {
  const kernels = new Map<KernelName, Promise<CompiledKernel>>();
  const ctx: WebGPUContext = {
    device,
    limits: device.limits,
    adapter,
    lost: null,
    getKernel(name) {
      let p = kernels.get(name);
      if (!p) {
        p = compileKernel(device, name);
        kernels.set(name, p);
      }
      return p;
    },
  };
  device.lost.then((info) => {
    ctx.lost = { reason: info.reason, message: info.message };
    onLost();
  });
  return ctx;
}

let cached: Promise<WebGPUContext> | null = null;

/**
 * Returns a ready WebGPU context, or throws WebGPUUnavailableError. The
 * successful context is cached per page/worker; failures are not cached
 * (they are cheap to retry and may be transient).
 */
export function acquireWebGPU(options: AcquireOptions = {}): Promise<WebGPUContext> {
  if (!options.fresh && !options.gpu && cached) {
    return cached.then((c) => {
      if (c.lost) {
        cached = null;
        return acquireWebGPU(options);
      }
      return c;
    });
  }

  const attempt = (async (): Promise<WebGPUContext> => {
    const gpu = getGpu(options.gpu);
    let adapter: WAdapter | null;
    try {
      adapter = await gpu.requestAdapter();
    } catch (e) {
      throw new WebGPUUnavailableError(`navigator.gpu.requestAdapter() threw: ${e instanceof Error ? e.message : String(e)}`);
    }
    if (!adapter) throw new WebGPUUnavailableError('No WebGPU adapter is available (requestAdapter() returned null).');

    const summary = summarizeAdapter(adapter);
    if (options.requireHardwareAdapter && summary.software) {
      throw new WebGPUUnavailableError(`Only a software WebGPU adapter is available (${summary.architecture || summary.vendor || 'unknown'}); a hardware adapter was required.`);
    }

    let device: WDevice;
    try {
      device = await adapter.requestDevice({ label: 'ir-to-visible' });
    } catch (e) {
      throw new WebGPUUnavailableError(`adapter.requestDevice() failed: ${e instanceof Error ? e.message : String(e)}`);
    }
    if (!device) throw new WebGPUUnavailableError('adapter.requestDevice() returned no device.');

    // The kernels need 8x8 compute workgroups and up to two storage-texture outputs; refuse (=> fall back)
    // rather than fail later with an opaque validation error on an exotic device.
    const l = device.limits;
    const problem =
      l.maxComputeInvocationsPerWorkgroup < 64 ? `maxComputeInvocationsPerWorkgroup=${l.maxComputeInvocationsPerWorkgroup} < 64`
      : l.maxStorageTexturesPerShaderStage < 2 ? `maxStorageTexturesPerShaderStage=${l.maxStorageTexturesPerShaderStage} < 2`
      : l.maxTextureDimension2D < 512 ? `maxTextureDimension2D=${l.maxTextureDimension2D} < 512`
      : null;
    if (problem) {
      try { device.destroy(); } catch { /* ignore */ }
      throw new WebGPUUnavailableError(`WebGPU device limits are insufficient for this pipeline: ${problem}.`);
    }

    return makeContext(device, summary, () => {
      if (cached === attemptRef.p) cached = null;
    });
  })();

  const attemptRef = { p: attempt };
  if (!options.fresh && !options.gpu) {
    cached = attempt;
    attempt.catch(() => {
      if (cached === attempt) cached = null; // never cache a failure
    });
  }
  return attempt;
}

/** Eagerly compiles every kernel (used by the harness to validate all WGSL). */
export async function compileAllKernels(ctx: WebGPUContext): Promise<void> {
  await Promise.all(KERNEL_NAMES.map((n) => ctx.getKernel(n)));
}

/** Drops the cached context (tests). Does not destroy the device. */
export function resetWebGPUCache(): void {
  cached = null;
}

export type WebGPUProbe = { available: true; adapter: WebGPUAdapterSummary } | { available: false; reason: string };

/** Non-throwing availability probe: does a REAL adapter + device exist? */
export async function probeWebGPU(options: AcquireOptions = {}): Promise<WebGPUProbe> {
  try {
    const ctx = await acquireWebGPU(options);
    return { available: true, adapter: ctx.adapter };
  } catch (e) {
    return { available: false, reason: e instanceof Error ? e.message : String(e) };
  }
}
