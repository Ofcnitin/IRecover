import { describe, it, expect } from 'vitest';
import { runFullPipelineWebGPU, DEFAULT_RESIDENT_BUDGET_BYTES } from '../src/processing/gpu/webgpu/webgpuFullPipeline';
import { KERNELS } from '../src/processing/gpu/webgpu/wgslShaders';
import { DEFAULT_SETTINGS, type ProcessingSettings } from '../src/types/processing';
import { PRESETS } from '../src/processing/presets';

// ---------------------------------------------------------------------------
// A recording fake WebGPU device. It computes nothing (every readback is zeros);
// it exists to observe WHAT THE REAL ORCHESTRATOR ASKS THE API TO DO: which
// kernels are dispatched in which order, and every CPU<->GPU transfer.
// Numerical behaviour is verified on a real WebGPU adapter by
// tools/gpu-consistency/webgpu/run-webgpu.ts.
// ---------------------------------------------------------------------------
function makeFakeGpu() {
  const log = {
    events: [] as string[], // 'up:<fmt>' | 'rb:<fmt>'
    dispatches: [] as string[], // kernel names, in dispatch order
    uploadBytes: 0,
    readbackBytes: { float: 0, packed: 0 },
    gpuCopies: 0,
    submits: 0,
  };
  const limits = {
    maxTextureDimension2D: 8192,
    maxComputeWorkgroupSizeX: 256,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupsPerDimension: 65535,
    maxUniformBufferBindingSize: 65536,
    minUniformBufferOffsetAlignment: 256,
    maxStorageTexturesPerShaderStage: 4,
    maxSampledTexturesPerShaderStage: 16,
    maxBufferSize: 1 << 30,
  };
  const device = {
    limits,
    lost: new Promise<never>(() => {}),
    queue: {
      submit: () => void log.submits++,
      writeBuffer: () => {},
      writeTexture: (dest: any, _data: unknown, layout: any, size: any) => {
        log.events.push(`up:${dest.texture.format}`);
        log.uploadBytes += layout.bytesPerRow * size.height;
      },
      onSubmittedWorkDone: async () => {},
    },
    createShaderModule: () => ({ getCompilationInfo: async () => ({ messages: [] }) }),
    createBindGroupLayout: () => ({}),
    createPipelineLayout: () => ({}),
    createComputePipelineAsync: async (d: any) => ({ label: String(d.label).replace('ir2vis/', '') }),
    createBindGroup: () => ({}),
    createBuffer: ({ size }: { size: number }) => ({ destroy() {}, mapAsync: async () => {}, getMappedRange: () => new ArrayBuffer(size), unmap() {} }),
    createTexture: ({ size, format }: any) => ({ width: size.width, height: size.height, format, createView: () => ({}), destroy() {} }),
    createCommandEncoder: () => ({
      beginComputePass: () => {
        let cur = '';
        return { setPipeline: (p: any) => (cur = p.label), setBindGroup() {}, dispatchWorkgroups: () => void log.dispatches.push(cur), end() {} };
      },
      copyTextureToBuffer: (src: any, _dst: any, size: any) => {
        const fmt = src.texture.format as string;
        log.events.push(`rb:${fmt}`);
        const bytes = size.width * size.height * (fmt === 'rgba32float' ? 16 : 4);
        if (fmt === 'rgba32float') log.readbackBytes.float += bytes;
        else log.readbackBytes.packed += bytes;
      },
      copyTextureToTexture: () => void log.gpuCopies++,
      finish: () => ({}),
    }),
    pushErrorScope() {},
    popErrorScope: async () => null,
    destroy() {},
  };
  const gpu = { requestAdapter: async () => ({ limits, info: { vendor: 'fake', architecture: 'fake' }, requestDevice: async () => device }) };
  return { gpu, log };
}

const W = 64;
const H = 48;
const N = W * H;
const Q = { gaussianPasses: 3, localContrastRadius: 24 };
const stops = PRESETS.natural.colorStops;
const base: ProcessingSettings = { ...DEFAULT_SETTINGS, preset: 'natural', noiseReduction: 30, noiseMethod: 'gaussian', localContrast: 25, sceneHeuristics: true };
const allOn: ProcessingSettings = { ...base, precisionPipeline: true, sharpenAmount: 60, sharpenRadius: 2 };
const precisionOnly: ProcessingSettings = { ...base, precisionPipeline: true, sharpenAmount: 0 };
const sharpenOnly: ProcessingSettings = { ...base, precisionPipeline: false, sharpenAmount: 60, sharpenRadius: 2 };
const neither: ProcessingSettings = { ...base, precisionPipeline: false, sharpenAmount: 0 };

async function run(settings: ProcessingSettings, opts: { maxTileDim?: number; residentBudgetBytes?: number } = {}, w = W, h = H) {
  const { gpu, log } = makeFakeGpu();
  const r = await runFullPipelineWebGPU(new Float32Array(w * h).fill(0.5), w, h, settings, stops, Q, { ...opts, acquire: { gpu: gpu as any } });
  return { r, log, t: r.resources.transfers! };
}

describe('WebGPU pipeline: GPU-resident data movement (real orchestrator, recording fake device)', () => {
  it('all stages, one tile: 1 upload, ONE float readback (CPU statistics), ONE packed readback -- nothing else', async () => {
    const { log, t, r } = await run(allOn);
    expect(log.events).toEqual(['up:r32float', 'rb:rgba32float', 'rb:r32uint']);
    expect(t.mode).toBe('resident');
    expect(t.uploads.restaged.count).toBe(0);
    expect(log.uploadBytes + log.readbackBytes.float + log.readbackBytes.packed).toBe((4 + 16 + 4) * N); // 24 B/px
    expect(r.rgba.length).toBe(N * 4);
  });

  it('runs precision -> quantise -> sharpen -> pack on the GPU, in that order, each stage once', async () => {
    const { log } = await run(allOn);
    const order = ['colorMap', 'precision', 'quantSrgb', 'boxBlurRgba', 'sharpenCombine', 'pack8'];
    const idx = order.map((k) => log.dispatches.lastIndexOf(k));
    expect(idx.every((i) => i >= 0)).toBe(true);
    expect([...idx].sort((a, b) => a - b)).toEqual(idx); // strictly the pipeline order
    expect(log.dispatches.filter((k) => k === 'precision')).toHaveLength(1);
    expect(log.dispatches.filter((k) => k === 'pack8')).toHaveLength(1);
    expect(log.dispatches.filter((k) => k === 'quantSrgb')).toHaveLength(1);
  });

  it('never reads a float RGBA image back except for the statistics, and none at all without a precision stage', async () => {
    for (const s of [allOn, precisionOnly]) expect((await run(s)).log.readbackBytes.float).toBe(16 * N);
    for (const s of [sharpenOnly, neither]) expect((await run(s)).log.readbackBytes.float).toBe(0);
    for (const s of [allOn, precisionOnly, sharpenOnly, neither]) expect((await run(s)).log.readbackBytes.packed).toBe(4 * N);
  });

  it('sharpen only: no statistics readback, no precision kernel, no re-upload (1 upload + 1 packed readback)', async () => {
    const { log } = await run(sharpenOnly);
    expect(log.events).toEqual(['up:r32float', 'rb:r32uint']);
    expect(log.dispatches).not.toContain('precision');
  });

  it("neither precision nor sharpen ('direct'): tiles are independent, 1 upload + 1 packed readback, no residency", async () => {
    const { log, t } = await run(neither);
    expect(log.events).toEqual(['up:r32float', 'rb:r32uint']);
    expect(t.mode).toBe('direct');
    expect(t.residentBytes).toBe(0);
    expect(log.gpuCopies).toBe(0);
  });

  it('precision only: statistics readback then a packed readback of the result (no sharpen tail, no re-upload)', async () => {
    const { log } = await run(precisionOnly);
    expect(log.events).toEqual(['up:r32float', 'rb:rgba32float', 'rb:r32uint']);
    expect(log.dispatches).not.toContain('sharpenCombine');
  });

  it('the pipeline\'s self-reported transfers equal what the device actually saw', async () => {
    for (const s of [allOn, precisionOnly, sharpenOnly, neither]) {
      const { log, t } = await run(s);
      const up = log.events.filter((e) => e.startsWith('up:')).length;
      const rbFloat = log.events.filter((e) => e === 'rb:rgba32float').length;
      const rbPacked = log.events.filter((e) => e === 'rb:r32uint').length;
      expect(t.uploads.input.count + t.uploads.restaged.count).toBe(up);
      expect(t.uploads.input.bytes + t.uploads.restaged.bytes).toBe(log.uploadBytes);
      expect(t.readbacks.statistics.count).toBe(rbFloat);
      expect(t.readbacks.statistics.bytes).toBe(log.readbackBytes.float);
      expect(t.readbacks.output.count).toBe(rbPacked);
      expect(t.readbacks.output.bytes).toBe(log.readbackBytes.packed);
      expect(t.gpuCopies).toBe(log.gpuCopies);
    }
  });

  it('multi-tile resident: cores and sharpen halos move GPU-to-GPU; no re-upload; nothing but packed pixels after the last statistics readback', async () => {
    const { log, t, r } = await run(allOn, { maxTileDim: 128 }, 330, 240);
    const tilesA = r.tiling.colorMap.tileCount;
    const tilesE = r.tiling.sharpen!.tileCount;
    expect(tilesA).toBeGreaterThan(4);
    expect(t.mode).toBe('resident');
    expect(t.uploads.input.count).toBe(tilesA);
    expect(t.uploads.restaged.count).toBe(0);
    expect(t.readbacks.statistics.count).toBe(tilesA);
    expect(t.readbacks.output.count).toBe(tilesE);
    expect(log.gpuCopies).toBeGreaterThanOrEqual(tilesA + tilesE);
    const after = log.events.slice(log.events.lastIndexOf('rb:rgba32float') + 1);
    expect(after.length).toBeGreaterThan(0);
    expect(after.every((e) => e === 'rb:r32uint')).toBe(true);
  });

  it('streaming (budget exceeded): re-stages through the CPU, but still reads packed 4 B/px and never float output', async () => {
    const { log, t, r } = await run(allOn, { maxTileDim: 128, residentBudgetBytes: 0 }, 330, 240);
    expect(t.mode).toBe('streaming');
    expect(t.uploads.restaged.count).toBeGreaterThan(0);
    expect(log.gpuCopies).toBe(0);
    expect(t.readbacks.statistics.count).toBe(r.tiling.colorMap.tileCount);
    expect(log.readbackBytes.float).toBe(16 * 330 * 240); // statistics only
    expect(log.readbackBytes.packed).toBe(2 * 4 * 330 * 240); // precision phase + sharpen phase, 4 B/px each
  });

  it('mode selection: resident iff cores fit the budget (16 B/px); direct whenever nothing crosses a phase boundary', async () => {
    expect((await run(allOn, { residentBudgetBytes: 16 * N })).t.mode).toBe('resident'); // exactly at budget
    expect((await run(allOn, { residentBudgetBytes: 16 * N - 1 })).t.mode).toBe('streaming'); // one byte over
    expect((await run(neither, { residentBudgetBytes: 0 })).t.mode).toBe('direct');
    expect((await run(allOn)).t.residentBytes).toBe(16 * N);
    expect(DEFAULT_RESIDENT_BUDGET_BYTES).toBe(256 * 1024 * 1024);
  });

  it('resident and streaming request the same kernels for the same stages (only the staging differs)', async () => {
    const a = new Set((await run(allOn)).log.dispatches);
    const b = new Set((await run(allOn, { residentBudgetBytes: 0 })).log.dispatches);
    expect([...a].sort()).toEqual([...b].sort());
  });
});

describe('WGSL output-stage kernels', () => {
  const table = (): number[] => {
    const m = KERNELS.quantSrgb.wgsl.match(/array<f32, 256>\(([^)]*)\)/);
    expect(m).toBeTruthy();
    return (m as RegExpMatchArray)[1].split(',').map((x) => Number(x.trim()));
  };

  it('quantSrgb carries a 256-entry table whose entry k is EXACTLY Math.fround(k / 255) (no GPU division involved)', () => {
    const t = table();
    expect(t).toHaveLength(256);
    for (let k = 0; k < 256; k++) expect(t[k]).toBe(Math.fround(k / 255));
    expect(t[0]).toBe(0);
    expect(t[255]).toBe(1);
  });

  it('table lookup with round-half-even reproduces Uint8ClampedArray quantisation for every value, including exact ties', () => {
    const t = table();
    const roundHalfEven = (x: number) => {
      const f = Math.floor(x);
      const d = x - f;
      return d < 0.5 ? f : d > 0.5 ? f + 1 : f % 2 === 0 ? f : f + 1;
    };
    const u8 = new Uint8ClampedArray(1);
    const check = (v: number) => {
      const viaTable = t[Math.min(255, Math.max(0, roundHalfEven(v * 255)))];
      u8[0] = v * 255; // what the CPU path did (round-half-even + clamp)
      expect(viaTable).toBe(Math.fround(u8[0] / 255));
    };
    for (let k = 0; k <= 255; k++) check(k / 255); // on-grid
    for (let k = 0; k < 255; k++) check((k + 0.5) / 255); // exact ties: 0.5 -> 0, 1.5 -> 2, 2.5 -> 2 ...
    for (let i = 0; i <= 5000; i++) check(i / 5000);
    check(-0.3);
    check(1.3);
  });

  it('pack8 writes r32uint (4 B/px) and quantSrgb keeps float RGBA; both take one input', () => {
    expect(KERNELS.pack8.outputs).toEqual(['r32uint']);
    expect(KERNELS.quantSrgb.outputs).toEqual(['rgba32float']);
    expect(KERNELS.pack8.inputs).toBe(1);
    expect(KERNELS.quantSrgb.inputs).toBe(1);
    expect(KERNELS.pack8.wgsl).toContain('255u << 24u'); // alpha byte
  });

  it('quantSrgb uses the same sRGB transfer function constants as the colour-map kernel', () => {
    for (const c of ['12.92', '1.055', '0.055', '0.0031308']) {
      expect(KERNELS.quantSrgb.wgsl).toContain(c);
      expect(KERNELS.colorMap.wgsl).toContain(c);
    }
  });
});
