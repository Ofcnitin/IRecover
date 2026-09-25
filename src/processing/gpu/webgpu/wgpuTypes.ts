/**
 * Minimal, dependency-free WebGPU type surface.
 * ---------------------------------------------
 * The project's TypeScript (5.x) `lib.dom` does not ship WebGPU types and
 * `@webgpu/types` is not a dependency. Rather than add one (or declare
 * ambient `GPU*` globals that would collide with `@webgpu/types` if it is
 * ever added), this module declares ONLY the subset of the API the
 * WebGPU backend actually calls, under `W`-prefixed module-scoped names,
 * plus the numeric flag constants from the WebGPU spec (so no runtime
 * `GPUBufferUsage`/`GPUTextureUsage` globals are needed either).
 *
 * The real objects satisfy these interfaces structurally; the code
 * reaches the browser only through `navigator.gpu` (see webgpuDevice.ts).
 */

export type WTextureFormat = 'r32float' | 'rgba32float' | 'r32uint';

export interface WAdapterInfo {
  vendor?: string;
  architecture?: string;
  device?: string;
  description?: string;
  isFallbackAdapter?: boolean;
}

export interface WLimits {
  maxTextureDimension2D: number;
  maxComputeWorkgroupSizeX: number;
  maxComputeInvocationsPerWorkgroup: number;
  maxComputeWorkgroupsPerDimension: number;
  maxUniformBufferBindingSize: number;
  minUniformBufferOffsetAlignment: number;
  maxStorageTexturesPerShaderStage: number;
  maxSampledTexturesPerShaderStage: number;
  maxBufferSize: number;
}

export interface WShaderMessage {
  type: 'error' | 'warning' | 'info';
  message: string;
  lineNum: number;
  linePos: number;
}

export interface WShaderModule {
  getCompilationInfo(): Promise<{ messages: readonly WShaderMessage[] }>;
}

export interface WBindGroupLayout {}
export interface WPipelineLayout {}
export interface WBindGroup {}
export interface WComputePipeline {}
export interface WTextureView {}

export interface WBuffer {
  destroy(): void;
  mapAsync(mode: number): Promise<void>;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
}

export interface WTexture {
  readonly width: number;
  readonly height: number;
  readonly format: string;
  createView(): WTextureView;
  destroy(): void;
}

export interface WComputePass {
  setPipeline(p: WComputePipeline): void;
  setBindGroup(index: number, g: WBindGroup): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}

export interface WCommandBuffer {}

export interface WCommandEncoder {
  beginComputePass(): WComputePass;
  /** GPU-to-GPU copy (no CPU involvement); both textures need COPY_SRC / COPY_DST and the same format. */
  copyTextureToTexture(
    source: { texture: WTexture; origin?: { x: number; y: number } },
    destination: { texture: WTexture; origin?: { x: number; y: number } },
    copySize: { width: number; height: number }
  ): void;
  copyTextureToBuffer(
    source: { texture: WTexture; origin?: { x: number; y: number } },
    destination: { buffer: WBuffer; bytesPerRow: number; rowsPerImage?: number },
    copySize: { width: number; height: number }
  ): void;
  finish(): WCommandBuffer;
}

export interface WQueue {
  submit(buffers: WCommandBuffer[]): void;
  writeBuffer(buffer: WBuffer, offset: number, data: ArrayBuffer | ArrayBufferView): void;
  writeTexture(
    destination: { texture: WTexture },
    data: ArrayBufferView,
    dataLayout: { bytesPerRow: number; rowsPerImage?: number },
    size: { width: number; height: number }
  ): void;
  onSubmittedWorkDone(): Promise<void>;
}

export type WErrorFilter = 'validation' | 'out-of-memory' | 'internal';

export interface WBindGroupLayoutEntry {
  binding: number;
  visibility: number;
  buffer?: { type: 'uniform' };
  texture?: { sampleType: 'unfilterable-float'; viewDimension: '2d' };
  storageTexture?: { access: 'write-only'; format: WTextureFormat; viewDimension: '2d' };
}

export interface WBindGroupEntry {
  binding: number;
  resource: WTextureView | { buffer: WBuffer; offset?: number; size?: number };
}

export interface WDevice {
  readonly limits: WLimits;
  readonly queue: WQueue;
  readonly lost: Promise<{ reason: string; message: string }>;
  createShaderModule(desc: { code: string; label?: string }): WShaderModule;
  createBindGroupLayout(desc: { entries: WBindGroupLayoutEntry[]; label?: string }): WBindGroupLayout;
  createPipelineLayout(desc: { bindGroupLayouts: WBindGroupLayout[]; label?: string }): WPipelineLayout;
  createComputePipelineAsync(desc: {
    layout: WPipelineLayout;
    compute: { module: WShaderModule; entryPoint: string };
    label?: string;
  }): Promise<WComputePipeline>;
  createBindGroup(desc: { layout: WBindGroupLayout; entries: WBindGroupEntry[]; label?: string }): WBindGroup;
  createBuffer(desc: { size: number; usage: number; label?: string }): WBuffer;
  createTexture(desc: {
    size: { width: number; height: number };
    format: WTextureFormat;
    usage: number;
    label?: string;
  }): WTexture;
  createCommandEncoder(desc?: { label?: string }): WCommandEncoder;
  pushErrorScope(filter: WErrorFilter): void;
  popErrorScope(): Promise<{ message: string } | null>;
  destroy(): void;
}

export interface WAdapter {
  readonly limits: WLimits;
  readonly info?: WAdapterInfo;
  readonly isFallbackAdapter?: boolean;
  requestDevice(desc?: { label?: string }): Promise<WDevice>;
}

export interface WGpu {
  requestAdapter(options?: { powerPreference?: 'low-power' | 'high-performance' }): Promise<WAdapter | null>;
}

/** Numeric flag values from the WebGPU spec (GPUBufferUsage / GPUTextureUsage / GPUMapMode / GPUShaderStage). */
export const BUFFER_USAGE = {
  MAP_READ: 0x1,
  COPY_SRC: 0x4,
  COPY_DST: 0x8,
  UNIFORM: 0x40,
} as const;

export const TEXTURE_USAGE = {
  COPY_SRC: 0x1,
  COPY_DST: 0x2,
  TEXTURE_BINDING: 0x4,
  STORAGE_BINDING: 0x8,
} as const;

export const MAP_MODE = { READ: 0x1 } as const;
export const SHADER_STAGE = { COMPUTE: 0x4 } as const;
