import { describe, it, expect } from 'vitest';
import { KERNELS, KERNEL_NAMES, WORKGROUP_SIZE, type KernelName } from '../src/processing/gpu/webgpu/wgslShaders';
import { MAX_STOPS } from '../src/processing/gpu/fullPipelineShaders';
import { AUTOTONE_KNEE_START } from '../src/processing/autoTone';
import { DEFAULT_MAX_CHROMA_BOOST } from '../src/processing/colorCorrectionPipeline';
import { DEFAULT_STEPS, DEFAULT_EPS } from '../src/processing/gamutProtection';

/**
 * Static checks on the WGSL text. Actually COMPILING the shaders needs a
 * WebGPU device, so that is verified in a real browser by
 * tools/gpu-consistency/webgpu/run-webgpu.ts (all 13 kernels: zero
 * compiler messages). What CAN be verified without a GPU -- and what
 * would otherwise only fail as an opaque validation error or, worse, as
 * a silently misaligned uniform -- is that the WGSL declarations agree
 * with the KernelSpec the orchestrator builds bind-group layouts and
 * packs uniforms against.
 */

const EXPECTED: KernelName[] = [
  'boxBlurR', 'boxBlurRgba', 'blendR', 'bilateralR', 'medianR', 'squareR', 'varianceCombine',
  'localContrast', 'toneCurve', 'scene', 'colorMap', 'sharpenCombine',
  // GPU-resident output stage (production-hardening pass): linear -> 8-bit-quantised sRGB, then packed to one
  // u32/pixel, so sharpening and the final readback never need a CPU float->byte conversion or an intermediate
  // CPU round trip.
  'quantSrgb', 'pack8',
  'precision',
];

interface Binding { index: number; decl: string }

function bindings(wgsl: string): Binding[] {
  const out: Binding[] = [];
  const re = /@group\(0\)\s*@binding\((\d+)\)\s*var(?:<(\w+)>)?\s+(\w+)\s*:\s*([^;]+);/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(wgsl))) out.push({ index: Number(m[1]), decl: `${m[2] ?? ''}|${m[3]}|${m[4].trim()}` });
  return out.sort((a, b) => a.index - b.index);
}

/** Size in bytes of struct P under WGSL uniform layout for the scalar/vec4-array members these kernels use. */
function structPSize(wgsl: string): number {
  const body = /struct P\s*\{([\s\S]*?)\};/.exec(wgsl)?.[1];
  if (!body) throw new Error('no struct P');
  const stripped = body.replace(/\/\/[^\n]*/g, '');
  let size = 0;
  // Members look like `name: type,` where a type may itself contain a comma (array<vec4<f32>, 8>).
  const memberRe = /(\w+)\s*:\s*(array<vec4<f32>,\s*\d+>|[fiu]32)\s*,?/g;
  let m: RegExpExecArray | null;
  let members = 0;
  while ((m = memberRe.exec(stripped))) {
    members++;
    const arr = /^array<vec4<f32>,\s*(\d+)>$/.exec(m[2]);
    if (arr) {
      expect(size % 16).toBe(0); // vec4 arrays must start 16-byte aligned
      size += Number(arr[1]) * 16;
    } else {
      size += 4; // only 4-byte scalars otherwise (no vec3 alignment traps)
    }
  }
  // Guard the parser itself: it must have consumed every member (a silently skipped one would hide a misalignment).
  expect(members).toBe(stripped.split(':').length - 1);
  return Math.ceil(size / 16) * 16;
}

describe('WGSL kernel table', () => {
  it('has exactly the expected kernels (kept in EXPECTED above, not a magic count)', () => {
    expect([...KERNEL_NAMES].sort()).toEqual([...EXPECTED].sort());
  });

  it.each(EXPECTED)('%s: uniformSize is a positive multiple of 16', (name: KernelName) => {
    const { uniformSize } = KERNELS[name];
    expect(uniformSize).toBeGreaterThanOrEqual(16);
    expect(uniformSize % 16).toBe(0);
  });

  it.each(EXPECTED)('%s: declared struct P size equals KernelSpec.uniformSize (packers cannot misalign)', (name: KernelName) => {
    expect(structPSize(KERNELS[name].wgsl)).toBe(KERNELS[name].uniformSize);
  });

  it.each(EXPECTED)('%s: bindings match the spec (uniform, N sampled inputs, storage outputs in order)', (name: KernelName) => {
    const spec = KERNELS[name];
    const b = bindings(spec.wgsl);
    expect(b.map((x) => x.index)).toEqual(Array.from({ length: 1 + spec.inputs + spec.outputs.length }, (_, i) => i));
    expect(b[0].decl.startsWith('uniform|p|P')).toBe(true);
    for (let i = 0; i < spec.inputs; i++) expect(b[1 + i].decl.endsWith('|texture_2d<f32>')).toBe(true);
    spec.outputs.forEach((fmt: string, k: number) => {
      expect(b[1 + spec.inputs + k].decl.endsWith(`|texture_storage_2d<${fmt}, write>`)).toBe(true);
    });
  });

  it.each(EXPECTED)('%s: declares an 8x8 compute entry point named main', (name: KernelName) => {
    const wgsl = KERNELS[name].wgsl;
    expect(WORKGROUP_SIZE).toBe(8);
    expect(wgsl).toMatch(/@compute @workgroup_size\(8, 8, 1\)\s*fn main\(/);
    expect(wgsl).toContain('@builtin(global_invocation_id)');
  });

  it.each(EXPECTED)('%s: bounds-checks against the texture dimensions before any access', (name: KernelName) => {
    const wgsl = KERNELS[name].wgsl;
    const guard = wgsl.indexOf('if (id.x >= dim.x || id.y >= dim.y) { return; }');
    const firstLoad = wgsl.indexOf('textureLoad(', wgsl.indexOf('fn main'));
    expect(guard).toBeGreaterThan(-1);
    expect(guard).toBeLessThan(firstLoad);
  });

  it('every read is an integer-clamped textureLoad (no sampler, no UV math)', () => {
    for (const name of EXPECTED) {
      const wgsl = KERNELS[name].wgsl;
      expect(wgsl).not.toContain('textureSample');
      expect(wgsl).not.toContain('sampler');
    }
  });
});

describe('constants are taken from the CPU reference modules, not re-typed', () => {
  const precision = KERNELS.precision.wgsl;
  it('precision shader constants equal the CPU module exports', () => {
    expect(precision).toContain(`const KNEE_START: f32 = ${AUTOTONE_KNEE_START};`);
    expect(precision).toContain(`const MAX_CHROMA_BOOST: f32 = ${DEFAULT_MAX_CHROMA_BOOST};`);
    expect(precision).toContain(`const GAMUT_STEPS: i32 = ${DEFAULT_STEPS};`);
    expect(precision).toContain(`const GAMUT_EPS: f32 = ${DEFAULT_EPS};`);
  });

  it('color-map uniform arrays are sized by the shared MAX_STOPS', () => {
    expect(MAX_STOPS).toBe(8);
    expect(KERNELS.colorMap.wgsl).toContain(`stopT: array<vec4<f32>, ${MAX_STOPS}>`);
    expect(KERNELS.colorMap.wgsl).toContain(`stopColor: array<vec4<f32>, ${MAX_STOPS}>`);
    expect(KERNELS.colorMap.wgsl).toContain(`i < ${MAX_STOPS - 1}`);
    expect(KERNELS.colorMap.uniformSize).toBe(MAX_STOPS * 32 + 32);
  });
});

describe('math that must match the CPU/WebGL2 pipeline', () => {
  it('scene shader derives the row fraction from the GLOBAL row (tile-invariant), texel-centre like WebGL2', () => {
    const w = KERNELS.scene.wgsl;
    // rowOrigin makes it tile-invariant; +0.5 is the WebGL2 shaders' convention (v_uv.y at a texel centre), which keeps the
    // two GPU APIs bit-identical (the CPU reference's y/height differs by half a row, inside the validated tolerances).
    expect(w).toContain('(f32(p.rowOrigin + c.y) + 0.5) / p.fullHeight');
    expect(w).not.toContain('gl_FragCoord');
  });

  it('median filter selects element floor(k/2) of the sorted window, no arithmetic', () => {
    const w = KERNELS.medianR.wgsl;
    expect(w).toContain('w[k / 2]');
    expect(w).toContain('array<f32, 25>'); // radius 2 => 25 elements (radius is only 1 or 2 in the CPU code)
  });

  it('tone curve keeps the CPU/WebGL2 order: exposure/brightness, shadow lift, highlight recovery, gamma, S-curve', () => {
    const w = KERNELS.toneCurve.wgsl;
    const order = ['p.exposureMul + p.brightnessAdd', 'p.shadowLift * 0.35', 'p.highlightRecovery * 0.5', '1.0 / p.gamma', 'sCurveF(i, p.sCurveK)'];
    const idx = order.map((s) => w.indexOf(s));
    expect(idx.every((i) => i > -1)).toBe(true);
    expect([...idx].sort((a, b) => a - b)).toEqual(idx);
  });

  it('precision shader applies WB -> AutoTone -> color correction -> gamut, in that order', () => {
    const w = KERNELS.precision.wgsl;
    const m = w.indexOf('fn main');
    const order = ['p.wbR, p.wbG, p.wbB', 'autoToneMap(rgb.r)', 'chromaShape(rgb)', 'gamutMap(rgb)'];
    const idx = order.map((s) => w.indexOf(s, m));
    expect(idx.every((i) => i > -1)).toBe(true);
    expect([...idx].sort((a, b) => a - b)).toEqual(idx);
  });
});
