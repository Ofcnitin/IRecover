/**
 * WebGPU full pipeline (WGSL compute), with large-image tiling.
 * --------------------------------------------------------------
 * Runs the same pipeline as the CPU reference and the WebGL2 path:
 *   intensity -> noise reduction (gaussian | median | bilateral) ->
 *   local contrast -> tone curve -> detail preservation -> scene
 *   heuristics -> IR->RGB color ramp -> [whole-image statistics] ->
 *   white balance -> AutoTone -> color correction -> gamut protection ->
 *   sRGB 8-bit -> sharpening.
 *
 * Tiling model
 * ------------
 * IDENTICAL in structure to the validated WebGL2 tiler
 * (webgl2TiledPipeline.ts; its halo argument in gpu/tilePlanner.ts applies
 * unchanged and the pure planner is reused as-is): every image goes
 * through the planner. An image that fits in one tile is simply a
 * one-tile plan, so there is a single code path and "tiled vs untiled" is
 * the same code with a different `maxTileDim`.
 *
 *   Phase A (per tile, halo = chained spatial reach): spatial stages +
 *     color ramp; only each tile's CORE is read back into full-image
 *     planar linear RGB.
 *   Sync: whole-image WB / AutoTone / CC statistics are estimated ONCE by
 *     the same CPU estimators the CPU and WebGL2 paths use, so no tile
 *     ever gets its own statistics (=> no seams).
 *   Phase C (per tile, no halo -- pointwise): precision shader.
 *   Linear -> 8-bit sRGB on the full image (same expression as WebGL2).
 *   Phase E (per tile, halo = sharpen radius): unsharp mask reading an
 *     immutable copy and writing a separate output.
 *
 * WebGPU-specific engineering
 * ---------------------------
 *  - Textures are pooled by (format, size) and ping-ponged: a texture
 *    released mid-batch is immediately reusable as a later PASS OUTPUT
 *    (passes execute in encoder order), but never as a CPU-side upload
 *    target within the same batch (queue.writeTexture executes before the
 *    batch), so ordering hazards cannot arise. Everything is destroyed at
 *    the end of the run.
 *  - All uniform blocks for a submit live in one arena buffer at
 *    alignment-rounded offsets: one writeBuffer per pass, no per-pass
 *    buffer allocation.
 *  - One submit + one mapAsync per tile per phase; padded (256 B-aligned)
 *    readback rows are de-padded on the CPU.
 *  - Per tile: validation and out-of-memory error scopes are pushed and
 *    popped, and device loss is checked after every await. Any error
 *    throws; this module never returns a partially processed image.
 *
 * Failure model: every failure (no adapter, shader error, validation /
 * OOM error, device loss, TilingUnsupportedError, unsupported option)
 * throws. Callers (processing/pipelineAsync.ts) fall back to the existing
 * WebGL2 -> CPU chain and REPORT that WebGPU did not run.
 */

import type { ProcessingSettings, ColorStop } from '../../../types/processing';
import type { PrecisionPipelineGpuResult } from '../webgl2Backend';
import type { FullPipelineQualityParams } from '../webgl2FullPipeline';
import { MAX_STOPS } from '../fullPipelineShaders';
import { computeWhiteBalanceGains } from '../../whiteBalance';
import { computeAutoToneParams } from '../../autoTone';
import { computeChannelBalanceGains } from '../../colorCorrectionPipeline';
import {
  computeWasTiled,
  DEFAULT_MAX_TILE_DIM,
  planTiles,
  resolveStageRadii,
  SCENE_VARIANCE_RADIUS,
  type Tile,
  type TilePlan,
  type TileRect,
  type TilingDiagnostics,
} from '../tilePlanner';
import { KERNELS, WORKGROUP_SIZE, type KernelName } from './wgslShaders';
import { acquireWebGPU, type AcquireOptions, type WebGPUAdapterSummary, type WebGPUContext } from './webgpuDevice';
import {
  BUFFER_USAGE,
  MAP_MODE,
  TEXTURE_USAGE,
  type WBindGroupEntry,
  type WBuffer,
  type WDevice,
  type WTexture,
  type WTextureFormat,
  type WTextureView,
} from './wgpuTypes';

export interface WebGPUPipelineOptions {
  /** Max tile edge in pixels, halo included. Default min(2048, device maxTextureDimension2D). */
  maxTileDim?: number;
  /**
   * DIAGNOSTICS/TESTS ONLY. Replaces the computed halo; under-sized values
   * produce seams by design (negative controls in the consistency harness).
   */
  unsafeHaloOverride?: { colorMap?: number; sharpen?: number };
  /** Forwarded to acquireWebGPU (requireHardwareAdapter / test `gpu` seam). */
  acquire?: AcquireOptions;
  /**
   * VRAM budget (bytes) for keeping tile cores resident on the GPU between stages. Images needing more
   * (16 B/px) use streaming mode; 0 forces streaming (tests). Default DEFAULT_RESIDENT_BUDGET_BYTES.
   */
  residentBudgetBytes?: number;
}

/** Payload bytes moved (rows are 256 B-padded on the wire for readbacks; padding is not counted). */
export interface TransferCounter {
  count: number;
  bytes: number;
}

/**
 * Where data lived between stages:
 *  - 'direct'    no stage needs another tile's or another phase's output (no precision, no sharpen): each tile goes
 *                colour ramp -> 8-bit quantise -> pack -> ONE packed readback.
 *  - 'resident'  tile cores stay on the GPU across the whole-image statistics sync; precision, quantisation and
 *                sharpening run on them, sharpen halos are assembled GPU-to-GPU. The only float readback is the one
 *                the CPU statistics estimators require.
 *  - 'streaming' the image does not fit the resident budget (or streaming was forced): stage results are staged
 *                through CPU memory between phases (packed 4 B/px on the way back, floats only where a CPU
 *                estimator or a halo upload needs them).
 */
export type TransferMode = 'direct' | 'resident' | 'streaming';

export interface TransferStats {
  mode: TransferMode;
  uploads: {
    /** The intensity tiles: the pipeline's actual input. */
    input: TransferCounter;
    /** Data that was on the CPU only because a previous phase left the GPU (streaming mode). */
    restaged: TransferCounter;
  };
  readbacks: {
    /** Linear float RGB the CPU WB/AutoTone/CC estimators need (they read every pixel). */
    statistics: TransferCounter;
    /** Final 8-bit pixels (4 B/px). */
    output: TransferCounter;
  };
  /** GPU-to-GPU texture copies (core capture / halo assembly): no CPU involvement. */
  gpuCopies: number;
  /** VRAM held across the statistics sync in resident mode (0 otherwise). */
  residentBytes: number;
}

export interface WebGPUResourceStats {
  passesDispatched: number;
  submits: number;
  texturesCreated: number;
  /** Times a pooled texture was reused instead of allocating (ping-pong + cross-tile reuse). */
  textureReuses: number;
  peakTexturesLive: number;
  transfers?: TransferStats;
}

/** Default VRAM budget for keeping tile cores resident (16 B/px => 16 Mpx, e.g. 4096x4096). */
export const DEFAULT_RESIDENT_BUDGET_BYTES = 256 * 1024 * 1024;

export interface WebGPUPipelineResult {
  rgba: Uint8ClampedArray;
  precision?: PrecisionPipelineGpuResult;
  tiling: TilingDiagnostics;
  adapter: WebGPUAdapterSummary;
  resources: WebGPUResourceStats;
}

/** Stage reach incl. median (the WebGL2 planner predates median; this only adds it). */
export function resolveWebGPUStageRadii(settings: ProcessingSettings, quality: FullPipelineQualityParams) {
  const base = resolveStageRadii(settings, quality);
  let medianRadius = 0;
  if (settings.noiseReduction > 0 && settings.noiseMethod === 'median') {
    medianRadius = settings.noiseReduction / 100 > 0.66 ? 2 : 1;
  }
  return { ...base, medianRadius, colorMapReach: base.colorMapReach + medianRadius };
}

// ---------------------------------------------------------------------
// Uniform packing
// ---------------------------------------------------------------------

class Packer {
  readonly bytes: Uint8Array;
  private readonly view: DataView;
  private o = 0;
  constructor(size: number) {
    this.bytes = new Uint8Array(size);
    this.view = new DataView(this.bytes.buffer);
  }
  f32(v: number): this {
    this.view.setFloat32(this.o, v, true);
    this.o += 4;
    return this;
  }
  i32(v: number): this {
    this.view.setInt32(this.o, v, true);
    this.o += 4;
    return this;
  }
  u32(v: number): this {
    this.view.setUint32(this.o, v >>> 0, true);
    this.o += 4;
    return this;
  }
  skip(n: number): this {
    this.o += n;
    return this;
  }
  done(size: number): Uint8Array {
    if (this.o > size) throw new Error(`uniform overflow: wrote ${this.o} of ${size} bytes`);
    return this.bytes;
  }
}

// ---------------------------------------------------------------------
// Texture pool (ping-pong + cross-tile reuse)
// ---------------------------------------------------------------------

class TexturePool {
  private stable = new Map<string, WTexture[]>();
  private recent = new Map<string, WTexture[]>();
  private all: WTexture[] = [];
  private live = 0;
  readonly stats = { created: 0, reuses: 0, peakLive: 0 };

  constructor(private readonly device: WDevice) {}

  private static key(format: WTextureFormat, w: number, h: number): string {
    return `${format}:${w}x${h}`;
  }

  private take(map: Map<string, WTexture[]>, key: string): WTexture | undefined {
    return map.get(key)?.pop();
  }

  private create(format: WTextureFormat, w: number, h: number): WTexture {
    const t = this.device.createTexture({
      size: { width: w, height: h },
      format,
      usage: TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.STORAGE_BINDING | TEXTURE_USAGE.COPY_DST | TEXTURE_USAGE.COPY_SRC,
      label: `ir2vis/${format}/${w}x${h}`,
    });
    this.all.push(t);
    this.stats.created++;
    return t;
  }

  private markLive(): void {
    this.live++;
    if (this.live > this.stats.peakLive) this.stats.peakLive = this.live;
  }

  /** For pass outputs: may reuse a texture released earlier in THIS batch (ordered by the encoder). */
  acquireOutput(format: WTextureFormat, w: number, h: number): WTexture {
    const key = TexturePool.key(format, w, h);
    let t = this.take(this.recent, key);
    if (!t) t = this.take(this.stable, key);
    if (t) this.stats.reuses++;
    else t = this.create(format, w, h);
    this.markLive();
    return t;
  }

  /** For CPU uploads: only textures from PREVIOUS batches (queue.writeTexture runs before this batch). */
  acquireUpload(format: WTextureFormat, w: number, h: number): WTexture {
    const key = TexturePool.key(format, w, h);
    let t = this.take(this.stable, key);
    if (t) this.stats.reuses++;
    else t = this.create(format, w, h);
    this.markLive();
    return t;
  }

  release(t: WTexture): void {
    const key = TexturePool.key(t.format as WTextureFormat, t.width, t.height);
    const list = this.recent.get(key) ?? [];
    list.push(t);
    this.recent.set(key, list);
    this.live--;
  }

  /** Called after a batch is submitted: everything released becomes generally reusable. */
  settle(): void {
    for (const [key, list] of this.recent) {
      const s = this.stable.get(key) ?? [];
      s.push(...list);
      this.stable.set(key, s);
    }
    this.recent.clear();
  }

  destroyAll(): void {
    for (const t of this.all) {
      try {
        t.destroy();
      } catch {
        /* best effort */
      }
    }
    this.all = [];
    this.stable.clear();
    this.recent.clear();
  }
}

// ---------------------------------------------------------------------
// Command batching
// ---------------------------------------------------------------------

interface RecordedPass {
  name: KernelName;
  inputs: WTexture[];
  outputs: WTexture[];
  uniform: Uint8Array;
}

interface RecordedCopy {
  src: WTexture;
  sx: number;
  sy: number;
  dst: WTexture;
  dx: number;
  dy: number;
  width: number;
  height: number;
}

/** Encoder order matters: a halo-assembly copy must land before the pass that reads it. */
type Op = { kind: 'pass'; pass: RecordedPass } | { kind: 'copy'; copy: RecordedCopy };

type ReadbackPurpose = keyof TransferStats['readbacks'];
type UploadPurpose = keyof TransferStats['uploads'];

interface Readback {
  texture: WTexture;
  x: number;
  y: number;
  width: number;
  height: number;
  bytesPerPixel: number;
  purpose: ReadbackPurpose;
  /** Receives the mapped 4-byte words as floats (row stride in 4-byte words) BEFORE the buffer is unmapped. */
  consume: (mapped: Float32Array, strideWords: number) => void;
}

const align = (n: number, a: number) => Math.ceil(n / a) * a;
const bytesPerPixelOf = (f: WTextureFormat): number => (f === 'rgba32float' ? 16 : 4);

class Runner {
  readonly pool: TexturePool;
  readonly stats: WebGPUResourceStats = { passesDispatched: 0, submits: 0, texturesCreated: 0, textureReuses: 0, peakTexturesLive: 0 };
  readonly transfers: TransferStats = {
    mode: 'direct',
    uploads: { input: { count: 0, bytes: 0 }, restaged: { count: 0, bytes: 0 } },
    readbacks: { statistics: { count: 0, bytes: 0 }, output: { count: 0, bytes: 0 } },
    gpuCopies: 0,
    residentBytes: 0,
  };
  private ops: Op[] = [];
  private readbacks: Readback[] = [];
  private views = new WeakMap<WTexture, WTextureView>();

  constructor(readonly ctx: WebGPUContext, private readonly compiled: Map<KernelName, Awaited<ReturnType<WebGPUContext['getKernel']>>>) {
    this.pool = new TexturePool(ctx.device);
  }

  get device(): WDevice {
    return this.ctx.device;
  }

  assertAlive(where: string): void {
    if (this.ctx.lost) throw new Error(`WebGPU device was lost during ${where}: ${this.ctx.lost.reason} ${this.ctx.lost.message}`);
  }

  private view(t: WTexture): WTextureView {
    let v = this.views.get(t);
    if (!v) {
      v = t.createView();
      this.views.set(t, v);
    }
    return v;
  }

  out(format: WTextureFormat, w: number, h: number): WTexture {
    return this.pool.acquireOutput(format, w, h);
  }

  /** Upload a CPU array into a fresh (never in-batch-reused) texture. Counted in `transfers.uploads`. */
  upload(format: WTextureFormat, w: number, h: number, data: Float32Array, purpose: UploadPurpose = 'input'): WTexture {
    const t = this.pool.acquireUpload(format, w, h);
    const bpp = bytesPerPixelOf(format);
    this.device.queue.writeTexture({ texture: t }, data, { bytesPerRow: w * bpp, rowsPerImage: h }, { width: w, height: h });
    this.transfers.uploads[purpose].count++;
    this.transfers.uploads[purpose].bytes += w * h * bpp;
    return t;
  }

  pass(name: KernelName, inputs: WTexture[], outputs: WTexture[], uniform: Uint8Array): void {
    const spec = KERNELS[name];
    if (inputs.length !== spec.inputs || outputs.length !== spec.outputs.length) {
      throw new Error(`kernel ${name}: expected ${spec.inputs} inputs / ${spec.outputs.length} outputs`);
    }
    this.ops.push({ kind: 'pass', pass: { name, inputs, outputs, uniform } });
  }

  /** GPU-to-GPU rectangle copy, ordered with the passes around it. Never touches the CPU. */
  copy(src: WTexture, sx: number, sy: number, dst: WTexture, dx: number, dy: number, width: number, height: number): void {
    this.ops.push({ kind: 'copy', copy: { src, sx, sy, dst, dx, dy, width, height } });
  }

  readback(rb: Readback): void {
    this.readbacks.push(rb);
  }

  /** Encodes every recorded op (in order) + readback, submits once, awaits and consumes readbacks. */
  async flush(where: string): Promise<void> {
    const { device } = this;
    const minAlign = device.limits.minUniformBufferOffsetAlignment || 256;

    // Uniform arena: one buffer, alignment-rounded slots (passes only).
    const passOps = this.ops.filter((o): o is Extract<Op, { kind: 'pass' }> => o.kind === 'pass');
    let arenaSize = 0;
    const offsets = new Map<RecordedPass, number>();
    for (const o of passOps) {
      offsets.set(o.pass, arenaSize);
      arenaSize += align(KERNELS[o.pass.name].uniformSize, minAlign);
    }
    const arena = device.createBuffer({ size: Math.max(arenaSize, minAlign), usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST, label: 'ir2vis/uniform-arena' });
    for (const o of passOps) device.queue.writeBuffer(arena, offsets.get(o.pass) as number, o.pass.uniform);

    const encoder = device.createCommandEncoder({ label: `ir2vis/${where}` });
    for (const op of this.ops) {
      if (op.kind === 'copy') {
        const c = op.copy;
        encoder.copyTextureToTexture({ texture: c.src, origin: { x: c.sx, y: c.sy } }, { texture: c.dst, origin: { x: c.dx, y: c.dy } }, { width: c.width, height: c.height });
        this.transfers.gpuCopies++;
        continue;
      }
      const p = op.pass;
      const k = this.compiled.get(p.name);
      if (!k) throw new Error(`kernel ${p.name} was not compiled`);
      const entries: WBindGroupEntry[] = [{ binding: 0, resource: { buffer: arena, offset: offsets.get(p) as number, size: KERNELS[p.name].uniformSize } }];
      p.inputs.forEach((t, j) => entries.push({ binding: 1 + j, resource: this.view(t) }));
      p.outputs.forEach((t, j) => entries.push({ binding: 1 + p.inputs.length + j, resource: this.view(t) }));
      const group = device.createBindGroup({ layout: k.bindGroupLayout, entries, label: `ir2vis/${p.name}` });
      const dims = p.outputs[0];
      const cp = encoder.beginComputePass();
      cp.setPipeline(k.pipeline);
      cp.setBindGroup(0, group);
      cp.dispatchWorkgroups(Math.ceil(dims.width / WORKGROUP_SIZE), Math.ceil(dims.height / WORKGROUP_SIZE), 1);
      cp.end();
    }

    const buffers: { buf: WBuffer; rb: Readback; bytesPerRow: number }[] = [];
    for (const rb of this.readbacks) {
      const bytesPerRow = align(rb.width * rb.bytesPerPixel, 256); // copyTextureToBuffer requires 256 B row alignment
      const buf = device.createBuffer({ size: bytesPerRow * rb.height, usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ, label: 'ir2vis/readback' });
      encoder.copyTextureToBuffer({ texture: rb.texture, origin: { x: rb.x, y: rb.y } }, { buffer: buf, bytesPerRow, rowsPerImage: rb.height }, { width: rb.width, height: rb.height });
      buffers.push({ buf, rb, bytesPerRow });
      this.transfers.readbacks[rb.purpose].count++;
      this.transfers.readbacks[rb.purpose].bytes += rb.width * rb.height * rb.bytesPerPixel;
    }

    device.queue.submit([encoder.finish()]);
    this.stats.submits++;
    this.stats.passesDispatched += passOps.length;
    this.ops = [];
    this.readbacks = [];
    this.pool.settle();

    try {
      for (const { buf, rb, bytesPerRow } of buffers) {
        await buf.mapAsync(MAP_MODE.READ);
        this.assertAlive(where);
        const mapped = new Float32Array(buf.getMappedRange());
        rb.consume(mapped, bytesPerRow / 4);
        buf.unmap();
      }
      await device.queue.onSubmittedWorkDone();
      this.assertAlive(where);
    } finally {
      for (const { buf } of buffers) buf.destroy();
      arena.destroy();
    }
  }

  /** Validation + OOM error scopes around one unit of work; any error throws. */
  async scoped<T>(where: string, fn: () => Promise<T>): Promise<T> {
    const { device } = this;
    device.pushErrorScope('validation');
    device.pushErrorScope('out-of-memory');
    let result: T | undefined;
    let thrown: unknown;
    let didThrow = false;
    try {
      result = await fn();
    } catch (e) {
      didThrow = true;
      thrown = e;
    }
    const oom = await device.popErrorScope();
    const validation = await device.popErrorScope();
    if (didThrow) throw thrown;
    if (oom) throw new Error(`WebGPU out-of-memory during ${where}: ${oom.message}`);
    if (validation) throw new Error(`WebGPU validation error during ${where}: ${validation.message}`);
    this.assertAlive(where);
    return result as T;
  }
}

// ---------------------------------------------------------------------
// Settings -> kernel parameters (derived once per run; same expressions and
// operation order as the WebGL2 paths so the three backends agree)
// ---------------------------------------------------------------------

interface Derived {
  radii: ReturnType<typeof resolveWebGPUStageRadii>;
  noiseAmt: number;
  bilateral: { twoSpatialSigma2: number; twoRangeSigma2: number };
  lcAmt: number;
  tone: { exposureMul: number; brightnessAdd: number; gamma: number; shadowLift: number; highlightRecovery: number; sCurveK: number };
  detailAmt: number;
  colorMapUniform: (sceneEnabled: boolean) => Uint8Array;
  sharpen: { amount: number; threshold: number };
}

function derive(settings: ProcessingSettings, colorStops: ColorStop[], quality: FullPipelineQualityParams): Derived {
  const radii = resolveWebGPUStageRadii(settings, quality);
  const noiseAmt = settings.noiseReduction / 100;
  const rangeSigma = 0.05 + noiseAmt * 0.2;
  const spatialSigma = radii.bilateralRadius / 2 + 0.5;
  const contrastAmt = Math.min(1, Math.max(0, (settings.contrast + 100) / 200));
  const sCurveS = (contrastAmt - 0.5) * 2;

  const colorStrength = Math.min(1, Math.max(0, settings.colorStrength / 100));
  const satAdjust = settings.saturation / 100;
  const hueBiasFrac = (((settings.hueBias / 360) % 1) + 1) % 1;
  const tempAmt = (settings.temperature / 100) * 0.12;

  const colorMapUniform = (sceneEnabled: boolean): Uint8Array => {
    const size = KERNELS.colorMap.uniformSize;
    const pk = new Packer(size);
    for (let i = 0; i < MAX_STOPS; i++) pk.f32(i < colorStops.length ? colorStops[i].t : 0).skip(12); // vec4: .x used
    for (let i = 0; i < MAX_STOPS; i++) {
      const rgb = i < colorStops.length ? colorStops[i].rgb : ([0, 0, 0] as const);
      pk.f32(rgb[0]).f32(rgb[1]).f32(rgb[2]).skip(4);
    }
    pk.i32(colorStops.length).u32(sceneEnabled ? 1 : 0).f32(colorStrength).f32(satAdjust).f32(hueBiasFrac).f32(tempAmt).skip(8);
    return pk.done(size);
  };

  return {
    radii,
    noiseAmt,
    bilateral: { twoSpatialSigma2: 2 * spatialSigma * spatialSigma, twoRangeSigma2: 2 * rangeSigma * rangeSigma },
    lcAmt: settings.localContrast / 100,
    tone: {
      exposureMul: Math.pow(2, settings.exposure),
      brightnessAdd: settings.brightness / 255,
      gamma: Math.max(0.05, settings.gamma),
      shadowLift: Math.min(1, Math.max(0, settings.shadowLift / 100)),
      highlightRecovery: Math.min(1, Math.max(0, settings.highlightRecovery / 100)),
      sCurveK: sCurveS > 0 ? 1 + sCurveS * 3 : 1 / (1 - sCurveS * 3),
    },
    detailAmt: settings.detailPreservation > 0 ? (settings.detailPreservation / 100) * 0.3 * 0.15 : 0,
    colorMapUniform,
    sharpen: {
      amount: Math.min(1, Math.max(0, settings.sharpenAmount / 200)) * 2,
      threshold: settings.sharpenThreshold / 255,
    },
  };
}

interface Planes {
  r: Float32Array;
  g: Float32Array;
  b: Float32Array;
}

const R = 'r32float' as const;
const RGBA = 'rgba32float' as const;

function neededKernels(settings: ProcessingSettings, d: Derived): KernelName[] {
  const k: KernelName[] = ['toneCurve', 'colorMap', 'quantSrgb', 'pack8'];
  if (settings.noiseReduction > 0) {
    k.push('blendR');
    if (settings.noiseMethod === 'gaussian') k.push('boxBlurR');
    else if (settings.noiseMethod === 'bilateral') k.push('bilateralR');
    else if (settings.noiseMethod === 'median') k.push('medianR');
  }
  if (settings.localContrast > 0) k.push('localContrast', 'boxBlurR');
  if (settings.detailPreservation > 0) k.push('blendR');
  if (settings.sceneHeuristics) k.push('squareR', 'boxBlurR', 'varianceCombine', 'scene');
  if (settings.precisionPipeline) k.push('precision');
  if (settings.sharpenAmount > 0) k.push('boxBlurRgba', 'sharpenCombine');
  void d;
  return Array.from(new Set(k));
}

// ---------------------------------------------------------------------
// Pass helpers
// ---------------------------------------------------------------------

function blur(rn: Runner, src: WTexture, w: number, h: number, radius: number): WTexture {
  const tmp = rn.out(R, w, h);
  const dst = rn.out(R, w, h);
  rn.pass('boxBlurR', [src], [tmp], new Packer(16).i32(radius).i32(1).i32(0).i32(0).done(16));
  rn.pass('boxBlurR', [tmp], [dst], new Packer(16).i32(radius).i32(0).i32(1).i32(0).done(16));
  rn.pool.release(tmp); // ping-pong: reusable by the very next pass
  return dst;
}

function blend(rn: Runner, a: WTexture, b: WTexture, w: number, h: number, t: number): WTexture {
  const out = rn.out(R, w, h);
  rn.pass('blendR', [a, b], [out], new Packer(16).f32(t).done(16));
  return out;
}

// ---------------------------------------------------------------------
// Phase A: one tile, intensity -> linear RGB (core written to the planes)
// ---------------------------------------------------------------------

async function runColorMapTile(
  rn: Runner,
  d: Derived,
  settings: ProcessingSettings,
  tile: Tile,
  intensity: Float32Array,
  fullW: number,
  fullH: number,
  dummy: WTexture,
  sink: TileSink
): Promise<void> {
  const { region, core } = tile;
  const w = region.x1 - region.x0;
  const h = region.y1 - region.y0;

  const sub = new Float32Array(w * h);
  for (let j = 0; j < h; j++) {
    const rowStart = (region.y0 + j) * fullW + region.x0;
    sub.set(intensity.subarray(rowStart, rowStart + w), j * w);
  }

  await rn.scoped(`color-map tile (${core.x0},${core.y0})`, async () => {
    const original = rn.upload(R, w, h, sub);
    let current = original;
    const release = (t: WTexture) => rn.pool.release(t);

    // 1. Noise reduction
    if (settings.noiseReduction > 0) {
      if (settings.noiseMethod === 'gaussian') {
        let blurred = current;
        for (let i = 0; i < d.radii.gaussianPasses; i++) {
          const next = blur(rn, blurred, w, h, d.radii.gaussianRadius);
          if (blurred !== current) release(blurred);
          blurred = next;
        }
        const blended = blend(rn, current, blurred, w, h, d.noiseAmt);
        release(current);
        release(blurred);
        current = blended;
      } else if (settings.noiseMethod === 'median') {
        const med = rn.out(R, w, h);
        rn.pass('medianR', [current], [med], new Packer(16).i32(d.radii.medianRadius).done(16));
        const blended = blend(rn, current, med, w, h, d.noiseAmt);
        release(current);
        release(med);
        current = blended;
      } else if (settings.noiseMethod === 'bilateral') {
        const filtered = rn.out(R, w, h);
        rn.pass('bilateralR', [current], [filtered], new Packer(16).i32(d.radii.bilateralRadius).f32(d.bilateral.twoSpatialSigma2).f32(d.bilateral.twoRangeSigma2).i32(0).done(16));
        const blended = blend(rn, current, filtered, w, h, d.noiseAmt);
        release(current);
        release(filtered);
        current = blended;
      }
    }

    // 2. Local contrast
    if (settings.localContrast > 0) {
      const localMean = blur(rn, current, w, h, d.radii.localContrastRadius);
      const out = rn.out(R, w, h);
      rn.pass('localContrast', [current, localMean], [out], new Packer(16).f32(d.lcAmt).done(16));
      release(current);
      release(localMean);
      current = out;
    }

    const preTone = current;

    // 3. Tone curve
    const toneMapped = rn.out(R, w, h);
    rn.pass(
      'toneCurve',
      [preTone],
      [toneMapped],
      new Packer(32).f32(d.tone.exposureMul).f32(d.tone.brightnessAdd).f32(d.tone.gamma).f32(d.tone.shadowLift).f32(d.tone.highlightRecovery).f32(d.tone.sCurveK).done(32)
    );

    // 4. Detail preservation
    let working = toneMapped;
    if (settings.detailPreservation > 0) {
      working = blend(rn, toneMapped, preTone, w, h, d.detailAmt);
      release(toneMapped);
    }
    release(preTone);

    // 5. Scene heuristics
    let sky = dummy;
    let veg = dummy;
    let skyReal: WTexture | null = null;
    let vegReal: WTexture | null = null;
    if (settings.sceneHeuristics) {
      const sq = rn.out(R, w, h);
      rn.pass('squareR', [working], [sq], new Packer(16).done(16));
      const mean = blur(rn, working, w, h, SCENE_VARIANCE_RADIUS);
      const meanSq = blur(rn, sq, w, h, SCENE_VARIANCE_RADIUS);
      release(sq);
      const variance = rn.out(R, w, h);
      rn.pass('varianceCombine', [mean, meanSq], [variance], new Packer(16).done(16));
      release(mean);
      release(meanSq);
      skyReal = rn.out(R, w, h);
      vegReal = rn.out(R, w, h);
      rn.pass('scene', [working, variance], [skyReal, vegReal], new Packer(16).i32(region.y0).f32(fullH).done(16));
      release(variance);
      sky = skyReal;
      veg = vegReal;
    }

    // 6. IR -> RGB color ramp (linear RGB out)
    const colorMapped = rn.out(RGBA, w, h);
    rn.pass('colorMap', [working, sky, veg], [colorMapped], d.colorMapUniform(settings.sceneHeuristics));

    // 7. Hand the finished tile to the sink: it decides what leaves this batch (a statistics readback, a packed
    //    8-bit readback, a resident core copy...). Halo pixels never leave the tile.
    const sinkTemps = sink.record(rn, colorMapped, tile);
    await rn.flush(`color-map tile (${core.x0},${core.y0})`);

    // Everything is settled (submitted + read back): return textures for cross-tile reuse.
    release(working);
    if (skyReal) release(skyReal);
    if (vegReal) release(vegReal);
    release(colorMapped);
    for (const t of sinkTemps) release(t);
    rn.pool.settle();
  });
}

// ---------------------------------------------------------------------
// Where a finished phase-A tile goes next (this is what decides the transfers)
// ---------------------------------------------------------------------

const R32U = 'r32uint' as const;

/** Linear RGB -> sRGB -> 8-bit grid, on the GPU (kernel quantSrgb). Output stays float (k/255). */
function quantize(rn: Runner, linear: WTexture, w: number, h: number): WTexture {
  const q = rn.out(RGBA, w, h);
  rn.pass('quantSrgb', [linear], [q], new Packer(16).done(16));
  return q;
}

/** RGBA float in [0,1] -> one packed u32 per pixel, on the GPU (kernel pack8). */
function pack(rn: Runner, src: WTexture, w: number, h: number): WTexture {
  const pk = rn.out(R32U, w, h);
  rn.pass('pack8', [src], [pk], new Packer(16).done(16));
  return pk;
}

/** Unpacks 4-byte pixels (R | G<<8 | B<<16 | A<<24) into an RGBA byte buffer byte-by-byte (no endianness assumption). */
function packedToBytes(out: Uint8ClampedArray, fullW: number, x0: number, y0: number, cw: number, ch: number) {
  return (m: Float32Array, stride: number): void => {
    const words = new Uint32Array(m.buffer, m.byteOffset, m.length);
    for (let j = 0; j < ch; j++) {
      let dst = ((y0 + j) * fullW + x0) * 4;
      let s = j * stride;
      for (let i = 0; i < cw; i++, dst += 4, s++) {
        const v = words[s];
        out[dst] = v & 255;
        out[dst + 1] = (v >>> 8) & 255;
        out[dst + 2] = (v >>> 16) & 255;
        out[dst + 3] = (v >>> 24) & 255;
      }
    }
  };
}

interface TileSink {
  /** Records ops/readbacks for one finished tile; returns textures to release after the batch is flushed. */
  record(rn: Runner, colorMapped: WTexture, tile: Tile): WTexture[];
}

interface ResidentCore {
  tex: WTexture;
  rect: TileRect;
}

/** Linear float core -> planar CPU buffers. Needed ONLY because the CPU estimators read every pixel. */
function recordStatisticsReadback(rn: Runner, colorMapped: WTexture, tile: Tile, planes: Planes, fullW: number): void {
  const { region, core } = tile;
  const cw = core.x1 - core.x0;
  const ch = core.y1 - core.y0;
  rn.readback({
    texture: colorMapped,
    x: core.x0 - region.x0,
    y: core.y0 - region.y0,
    width: cw,
    height: ch,
    bytesPerPixel: 16,
    purpose: 'statistics',
    consume: (m, stride) => {
      for (let j = 0; j < ch; j++) {
        let dst = (core.y0 + j) * fullW + core.x0;
        let s = j * stride;
        for (let i = 0; i < cw; i++, dst++, s += 4) {
          planes.r[dst] = m[s];
          planes.g[dst] = m[s + 1];
          planes.b[dst] = m[s + 2];
        }
      }
    },
  });
}

/** streaming + precision: only the statistics readback leaves the GPU; the tile is re-uploaded later for precision. */
function planesSink(planes: Planes, fullW: number): TileSink {
  return {
    record(rn, colorMapped, tile) {
      recordStatisticsReadback(rn, colorMapped, tile, planes, fullW);
      return [];
    },
  };
}

/** No precision stage to wait for: quantise + pack on the GPU and read back final bytes (4 B/px). */
function bytesSink(out: Uint8ClampedArray, fullW: number): TileSink {
  return {
    record(rn, colorMapped, tile) {
      const { region, core } = tile;
      const w = region.x1 - region.x0;
      const h = region.y1 - region.y0;
      const cw = core.x1 - core.x0;
      const ch = core.y1 - core.y0;
      const q = quantize(rn, colorMapped, w, h);
      const pk = pack(rn, q, w, h);
      rn.readback({
        texture: pk,
        x: core.x0 - region.x0,
        y: core.y0 - region.y0,
        width: cw,
        height: ch,
        bytesPerPixel: 4,
        purpose: 'output',
        consume: packedToBytes(out, fullW, core.x0, core.y0, cw, ch),
      });
      return [q, pk];
    },
  };
}

/**
 * Resident mode: keep this tile's CORE on the GPU (GPU-to-GPU copy). With a precision stage the core stays LINEAR
 * (precision runs after the statistics sync) and a statistics readback is issued; without one it is quantised now.
 */
function residentSink(cores: ResidentCore[], planes: Planes | null, quantizeNow: boolean, fullW: number): TileSink {
  return {
    record(rn, colorMapped, tile) {
      const { region, core } = tile;
      const w = region.x1 - region.x0;
      const h = region.y1 - region.y0;
      const cw = core.x1 - core.x0;
      const ch = core.y1 - core.y0;
      const temps: WTexture[] = [];
      if (planes) recordStatisticsReadback(rn, colorMapped, tile, planes, fullW);
      let src = colorMapped;
      if (quantizeNow) {
        src = quantize(rn, colorMapped, w, h);
        temps.push(src);
      }
      const coreTex = rn.out(RGBA, cw, ch);
      rn.copy(src, core.x0 - region.x0, core.y0 - region.y0, coreTex, 0, 0, cw, ch);
      cores.push({ tex: coreTex, rect: core });
      return temps;
    },
  };
}

// ---------------------------------------------------------------------
// Precision (whole-image statistics, then pointwise apply per tile)
// ---------------------------------------------------------------------

interface PrecisionParams {
  result: PrecisionPipelineGpuResult;
  uniform: Uint8Array;
}

function estimatePrecisionParams(strengths: { whiteBalance: number; autoTone: number; colorCorrection: number }, planes: Planes): PrecisionParams {
  const wbOpts = { strength: strengths.whiteBalance };
  const atOpts = { strength: strengths.autoTone };
  const ccOpts = { strength: strengths.colorCorrection };
  const wb = computeWhiteBalanceGains(planes.r, planes.g, planes.b, wbOpts);
  const at = computeAutoToneParams(planes.r, planes.g, planes.b, atOpts);
  const cc = computeChannelBalanceGains(planes.r, planes.g, planes.b, ccOpts);
  const uniform = new Packer(48)
    .f32(wb.gainR).f32(wb.gainG).f32(wb.gainB)
    .f32(at.blackPoint)
    .f32(Math.max(at.whitePoint - at.blackPoint, 1e-4))
    .f32(cc.gainR).f32(cc.gainG).f32(cc.gainB)
    .f32(Math.max(0, Math.min(1, ccOpts.strength / 100)))
    .u32(!at.degenerate && atOpts.strength > 0 ? 1 : 0)
    .u32(ccOpts.strength > 0 ? 1 : 0)
    .u32(1)
    .done(48);
  return {
    result: {
      whiteBalance: wb,
      autoTone: { blackPoint: at.blackPoint, whitePoint: at.whitePoint, blackClipPercent: 0, whiteClipPercent: 0 },
      colorCorrection: cc,
      diagnosticsApproximate: true,
    },
    uniform,
  };
}

async function runPrecisionTile(rn: Runner, pp: PrecisionParams, rect: Tile['core'], fullW: number, planes: Planes): Promise<void> {
  const w = rect.x1 - rect.x0;
  const h = rect.y1 - rect.y0;
  const src = new Float32Array(w * h * 4);
  for (let j = 0; j < h; j++) {
    let s = (rect.y0 + j) * fullW + rect.x0;
    let dd = j * w * 4;
    for (let i = 0; i < w; i++, s++, dd += 4) {
      src[dd] = planes.r[s];
      src[dd + 1] = planes.g[s];
      src[dd + 2] = planes.b[s];
      src[dd + 3] = 1;
    }
  }
  await rn.scoped(`precision tile (${rect.x0},${rect.y0})`, async () => {
    const input = rn.upload(RGBA, w, h, src);
    const output = rn.out(RGBA, w, h);
    rn.pass('precision', [input], [output], pp.uniform);
    rn.readback({
      texture: output,
      x: 0,
      y: 0,
      width: w,
      height: h,
      bytesPerPixel: 16,
      purpose: 'output',
      consume: (m, stride) => {
        for (let j = 0; j < h; j++) {
          let dst = (rect.y0 + j) * fullW + rect.x0;
          let s = j * stride;
          for (let i = 0; i < w; i++, dst++, s += 4) {
            planes.r[dst] = m[s];
            planes.g[dst] = m[s + 1];
            planes.b[dst] = m[s + 2];
          }
        }
      },
    });
    await rn.flush(`precision tile (${rect.x0},${rect.y0})`);
    rn.pool.release(input);
    rn.pool.release(output);
    rn.pool.settle();
  });
}

// ---------------------------------------------------------------------
// Sharpening (halo'd, immutable source, separate output)
// ---------------------------------------------------------------------

/** Records the H+V box blur and the unsharp combine for a region texture. Caller releases `blurred`/`sharpened` after the flush. */
function sharpenPasses(rn: Runner, d: Derived, src: WTexture, w: number, h: number): { sharpened: WTexture; blurred: WTexture } {
  const tmp = rn.out(RGBA, w, h);
  const blurred = rn.out(RGBA, w, h);
  const r = d.radii.sharpenRadius;
  rn.pass('boxBlurRgba', [src], [tmp], new Packer(16).i32(r).i32(1).i32(0).i32(0).done(16));
  rn.pass('boxBlurRgba', [tmp], [blurred], new Packer(16).i32(r).i32(0).i32(1).i32(0).done(16));
  rn.pool.release(tmp);
  const sharpened = rn.out(RGBA, w, h);
  rn.pass('sharpenCombine', [src, blurred], [sharpened], new Packer(16).f32(d.sharpen.amount).f32(d.sharpen.threshold).done(16));
  return { sharpened, blurred };
}

/** Streaming precision: upload the (linear) core, precision -> quantise -> pack on the GPU, read back 4 B/px. */
async function runPrecisionTileToBytes(rn: Runner, pp: PrecisionParams, rect: Tile['core'], fullW: number, planes: Planes, out: Uint8ClampedArray): Promise<void> {
  const w = rect.x1 - rect.x0;
  const h = rect.y1 - rect.y0;
  const src = new Float32Array(w * h * 4);
  for (let j = 0; j < h; j++) {
    let s = (rect.y0 + j) * fullW + rect.x0;
    let dd = j * w * 4;
    for (let i = 0; i < w; i++, s++, dd += 4) {
      src[dd] = planes.r[s];
      src[dd + 1] = planes.g[s];
      src[dd + 2] = planes.b[s];
      src[dd + 3] = 1;
    }
  }
  await rn.scoped(`precision tile (${rect.x0},${rect.y0})`, async () => {
    const input = rn.upload(RGBA, w, h, src, 'restaged');
    const lin = rn.out(RGBA, w, h);
    rn.pass('precision', [input], [lin], pp.uniform);
    const q = quantize(rn, lin, w, h);
    const pk = pack(rn, q, w, h);
    rn.readback({ texture: pk, x: 0, y: 0, width: w, height: h, bytesPerPixel: 4, purpose: 'output', consume: packedToBytes(out, fullW, rect.x0, rect.y0, w, h) });
    await rn.flush(`precision tile (${rect.x0},${rect.y0})`);
    rn.pool.release(input);
    rn.pool.release(lin);
    rn.pool.release(q);
    rn.pool.release(pk);
    rn.pool.settle();
  });
}

/** Streaming sharpen: halo'd region re-uploaded from the (immutable) 8-bit image; packed 4 B/px readback into a separate output. */
async function runSharpenTileStreaming(rn: Runner, d: Derived, tile: Tile, fullW: number, srcRgba: Uint8ClampedArray, outRgba: Uint8ClampedArray): Promise<void> {
  const { region, core } = tile;
  const w = region.x1 - region.x0;
  const h = region.y1 - region.y0;
  const srgb = new Float32Array(w * h * 4);
  for (let j = 0; j < h; j++) {
    let s = ((region.y0 + j) * fullW + region.x0) * 4;
    let dd = j * w * 4;
    for (let i = 0; i < w; i++, s += 4, dd += 4) {
      srgb[dd] = srcRgba[s] / 255;
      srgb[dd + 1] = srcRgba[s + 1] / 255;
      srgb[dd + 2] = srcRgba[s + 2] / 255;
      srgb[dd + 3] = 1;
    }
  }
  const cw = core.x1 - core.x0;
  const ch = core.y1 - core.y0;
  await rn.scoped(`sharpen tile (${core.x0},${core.y0})`, async () => {
    const src = rn.upload(RGBA, w, h, srgb, 'restaged');
    const { sharpened, blurred } = sharpenPasses(rn, d, src, w, h);
    const pk = pack(rn, sharpened, w, h);
    rn.readback({ texture: pk, x: core.x0 - region.x0, y: core.y0 - region.y0, width: cw, height: ch, bytesPerPixel: 4, purpose: 'output', consume: packedToBytes(outRgba, fullW, core.x0, core.y0, cw, ch) });
    await rn.flush(`sharpen tile (${core.x0},${core.y0})`);
    rn.pool.release(src);
    rn.pool.release(blurred);
    rn.pool.release(sharpened);
    rn.pool.release(pk);
    rn.pool.settle();
  });
}

/**
 * Resident tail: everything after the statistics sync happens on GPU-resident cores. No upload, and the only
 * readbacks are the final packed pixels. Sharpen halos are assembled with GPU-to-GPU copies from the neighbouring
 * cores, so a tile sees exactly the same quantised neighbour pixels the streaming path would upload.
 */
async function runResidentTail(rn: Runner, d: Derived, cores: ResidentCore[], pp: PrecisionParams | null, planE: TilePlan | null, fullW: number, out: Uint8ClampedArray): Promise<void> {
  // Precision -> quantise (-> pack + read back, when there is no sharpen phase), one batch per core.
  if (pp) {
    for (const c of cores) {
      const w = c.rect.x1 - c.rect.x0;
      const h = c.rect.y1 - c.rect.y0;
      await rn.scoped(`precision tile (${c.rect.x0},${c.rect.y0})`, async () => {
        const lin = rn.out(RGBA, w, h);
        rn.pass('precision', [c.tex], [lin], pp.uniform);
        rn.pool.release(c.tex); // ping-pong: reusable as the quantise output below
        const q = quantize(rn, lin, w, h);
        rn.pool.release(lin);
        c.tex = q;
        let pk: WTexture | null = null;
        if (!planE) {
          pk = pack(rn, q, w, h);
          rn.readback({ texture: pk, x: 0, y: 0, width: w, height: h, bytesPerPixel: 4, purpose: 'output', consume: packedToBytes(out, fullW, c.rect.x0, c.rect.y0, w, h) });
        }
        await rn.flush(`precision tile (${c.rect.x0},${c.rect.y0})`);
        if (pk) rn.pool.release(pk);
        rn.pool.settle();
      });
    }
  }

  if (planE) {
    for (const tile of planE.tiles) {
      const { region, core } = tile;
      const w = region.x1 - region.x0;
      const h = region.y1 - region.y0;
      const cw = core.x1 - core.x0;
      const ch = core.y1 - core.y0;
      await rn.scoped(`sharpen tile (${core.x0},${core.y0})`, async () => {
        const src = rn.out(RGBA, w, h);
        for (const c of cores) {
          const ix0 = Math.max(region.x0, c.rect.x0);
          const iy0 = Math.max(region.y0, c.rect.y0);
          const ix1 = Math.min(region.x1, c.rect.x1);
          const iy1 = Math.min(region.y1, c.rect.y1);
          if (ix1 > ix0 && iy1 > iy0) rn.copy(c.tex, ix0 - c.rect.x0, iy0 - c.rect.y0, src, ix0 - region.x0, iy0 - region.y0, ix1 - ix0, iy1 - iy0);
        }
        const { sharpened, blurred } = sharpenPasses(rn, d, src, w, h);
        const pk = pack(rn, sharpened, w, h);
        rn.readback({ texture: pk, x: core.x0 - region.x0, y: core.y0 - region.y0, width: cw, height: ch, bytesPerPixel: 4, purpose: 'output', consume: packedToBytes(out, fullW, core.x0, core.y0, cw, ch) });
        await rn.flush(`sharpen tile (${core.x0},${core.y0})`);
        rn.pool.release(src);
        rn.pool.release(blurred);
        rn.pool.release(sharpened);
        rn.pool.release(pk);
        rn.pool.settle();
      });
    }
  }

  for (const c of cores) rn.pool.release(c.tex);
  rn.pool.settle();
}

// ---------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------

export async function runFullPipelineWebGPU(
  intensity: Float32Array,
  width: number,
  height: number,
  settings: ProcessingSettings,
  colorStops: ColorStop[],
  quality: FullPipelineQualityParams,
  options: WebGPUPipelineOptions = {}
): Promise<WebGPUPipelineResult> {
  const n = width * height;
  if (intensity.length !== n) throw new Error('runFullPipelineWebGPU: buffer/size mismatch.');
  if (colorStops.length > MAX_STOPS) {
    throw new Error(`Preset has ${colorStops.length} color stops, exceeding this shader's MAX_STOPS (${MAX_STOPS}).`);
  }

  const ctx = await acquireWebGPU(options.acquire);
  const maxTileDim = Math.max(1, Math.min(options.maxTileDim ?? DEFAULT_MAX_TILE_DIM, ctx.limits.maxTextureDimension2D));
  const d = derive(settings, colorStops, quality);
  const colorMapHalo = options.unsafeHaloOverride?.colorMap ?? d.radii.colorMapReach;
  const sharpenHalo = options.unsafeHaloOverride?.sharpen ?? d.radii.sharpenReach;

  // How data moves between stages. 'direct': nothing crosses a phase boundary. 'resident': cores stay in VRAM
  // (16 B/px) while the CPU estimates statistics. 'streaming': over budget => stage through CPU memory.
  const needsCrossStage = settings.precisionPipeline || settings.sharpenAmount > 0;
  const budget = options.residentBudgetBytes ?? DEFAULT_RESIDENT_BUDGET_BYTES;
  const residentBytes = n * 16;
  const mode: TransferMode = !needsCrossStage ? 'direct' : residentBytes <= budget ? 'resident' : 'streaming';

  // Plan everything up front so an untileable request fails BEFORE any GPU work.
  const planA: TilePlan = planTiles(width, height, maxTileDim, colorMapHalo);
  // Precision is pointwise: resident mode applies it to the phase-A cores; only streaming needs its own grid.
  const planC: TilePlan | null = mode === 'streaming' && settings.precisionPipeline ? planTiles(width, height, maxTileDim, 0) : null;
  const planE: TilePlan | null = settings.sharpenAmount > 0 ? planTiles(width, height, maxTileDim, sharpenHalo) : null;

  const compiled = new Map<KernelName, Awaited<ReturnType<WebGPUContext['getKernel']>>>();
  for (const name of neededKernels(settings, d)) compiled.set(name, await ctx.getKernel(name));

  const rn = new Runner(ctx, compiled);
  rn.transfers.mode = mode;
  rn.transfers.residentBytes = mode === 'resident' ? residentBytes : 0;
  try {
    rn.assertAlive('start');
    const dummy = rn.pool.acquireUpload(R, 1, 1); // stands in for sky/veg when scene heuristics are off

    // `staged` is the final image in 'direct'/'resident', and the pre-sharpen image in 'streaming'.
    const staged = new Uint8ClampedArray(n * 4);
    let planes: Planes | null = settings.precisionPipeline && mode !== 'direct' ? { r: new Float32Array(n), g: new Float32Array(n), b: new Float32Array(n) } : null;
    const cores: ResidentCore[] = [];

    let sink: TileSink;
    if (mode === 'direct') sink = bytesSink(staged, width);
    else if (mode === 'streaming') sink = planes ? planesSink(planes, width) : bytesSink(staged, width);
    else sink = residentSink(cores, planes, !settings.precisionPipeline, width);

    for (const tile of planA.tiles) await runColorMapTile(rn, d, settings, tile, intensity, width, height, dummy, sink);

    // Sync point: whole-image statistics from the complete linear planes, by the same CPU estimators as the
    // CPU and WebGL2 paths (they read every pixel, so a sampled gather would change their behaviour).
    let precision: PrecisionPipelineGpuResult | undefined;
    let pp: PrecisionParams | null = null;
    if (planes) {
      pp = estimatePrecisionParams(
        { whiteBalance: settings.whiteBalanceStrength, autoTone: settings.autoToneStrength, colorCorrection: settings.colorCorrectionStrength },
        planes
      );
      precision = pp.result;
    }

    let result = staged;
    if (mode === 'resident') {
      planes = null; // statistics are done: release the three CPU planes before the tail
      await runResidentTail(rn, d, cores, pp, planE, width, staged);
    } else if (mode === 'streaming') {
      if (pp && planC && planes) for (const tile of planC.tiles) await runPrecisionTileToBytes(rn, pp, tile.core, width, planes, staged);
      planes = null;
      if (planE) {
        const out = new Uint8ClampedArray(n * 4);
        for (const tile of planE.tiles) await runSharpenTileStreaming(rn, d, tile, width, staged, out);
        result = out;
      }
    }

    const precisionTiling = settings.precisionPipeline
      ? planC
        ? { cols: planC.cols, rows: planC.rows, tileCount: planC.tiles.length }
        : { cols: planA.cols, rows: planA.rows, tileCount: planA.tiles.length }
      : null;

    return {
      rgba: result,
      precision,
      adapter: ctx.adapter,
      resources: {
        ...rn.stats,
        texturesCreated: rn.pool.stats.created,
        textureReuses: rn.pool.stats.reuses,
        peakTexturesLive: rn.pool.stats.peakLive,
        transfers: rn.transfers,
      },
      tiling: {
        maxTileDim,
        colorMap: { cols: planA.cols, rows: planA.rows, tileCount: planA.tiles.length, halo: planA.halo },
        precision: precisionTiling,
        sharpen: planE ? { cols: planE.cols, rows: planE.rows, tileCount: planE.tiles.length, halo: planE.halo } : null,
        wasTiled: computeWasTiled(
          { tileCount: planA.tiles.length },
          precisionTiling,
          planE ? { tileCount: planE.tiles.length } : null
        ),
      },
    };
  } finally {
    rn.pool.destroyAll();
  }
}

/**
 * WebGPU counterpart of runPrecisionPipelineWebGL2: applies white balance ->
 * AutoTone -> color correction -> gamut protection to planar LINEAR RGB
 * buffers, IN PLACE, and returns the parameters used. Exists so the
 * precision stage can be verified in isolation against the CPU reference
 * (the full pipeline's precision stage uses exactly this code path).
 */
export async function runPrecisionPipelineWebGPU(
  r: Float32Array,
  g: Float32Array,
  b: Float32Array,
  width: number,
  height: number,
  opts: { whiteBalance: { strength: number }; autoTone: { strength: number }; colorCorrection: { strength: number } },
  options: WebGPUPipelineOptions = {}
): Promise<PrecisionPipelineGpuResult> {
  const n = width * height;
  if (r.length !== n || g.length !== n || b.length !== n) throw new Error('runPrecisionPipelineWebGPU: buffer/size mismatch.');
  const ctx = await acquireWebGPU(options.acquire);
  const maxTileDim = Math.max(1, Math.min(options.maxTileDim ?? DEFAULT_MAX_TILE_DIM, ctx.limits.maxTextureDimension2D));
  const plan = planTiles(width, height, maxTileDim, 0);
  const compiled = new Map<KernelName, Awaited<ReturnType<WebGPUContext['getKernel']>>>([['precision', await ctx.getKernel('precision')]]);
  const rn = new Runner(ctx, compiled);
  try {
    const planes: Planes = { r, g, b };
    const pp = estimatePrecisionParams(
      { whiteBalance: opts.whiteBalance.strength, autoTone: opts.autoTone.strength, colorCorrection: opts.colorCorrection.strength },
      planes
    );
    for (const tile of plan.tiles) await runPrecisionTile(rn, pp, tile.core, width, planes);
    return pp.result;
  } finally {
    rn.pool.destroyAll();
  }
}
