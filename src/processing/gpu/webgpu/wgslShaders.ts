/**
 * WGSL compute kernels for the WebGPU full pipeline.
 * ---------------------------------------------------
 * Every kernel is a line-for-line port of the matching GLSL pass in
 * ../fullPipelineShaders.ts / ../shaders.ts (which are themselves ports
 * of the CPU reference in processing/*.ts): same constants, same order
 * of operations. Those GLSL files are NOT modified or imported for their
 * shader text; only the numeric constants that the precision shader
 * already pulls from the CPU modules are pulled from the same CPU
 * modules here, so the three implementations cannot drift apart.
 *
 * Differences from the WebGL2 shaders, all deliberate:
 *
 *  1. Addressing. WebGL2 samples with normalized UVs clamped to [0,1] on
 *     a NEAREST/CLAMP_TO_EDGE texture. Here every read is
 *     `textureLoad(tex, clamp(coord, 0, dims-1), 0)`: integer texels,
 *     identical edge-replication semantics, no UV rounding at all.
 *  2. Scene row fraction. WebGL2 reads a hardware-interpolated
 *     `v_uv.y`, i.e. the texel centre (y + 0.5) / height, which for
 *     tiling had to be re-derived per tile. Compute kernels have exact
 *     integer coordinates, so the row fraction is
 *     `(rowOrigin + y + 0.5) / fullHeight`: the SAME convention as the
 *     WebGL2 shaders, computed identically whether the image is one tile
 *     or many. (The CPU reference uses the top edge, y / height; the
 *     half-row difference is inside the validated CPU/GPU tolerances and
 *     is deliberately NOT copied: with WebGL2's convention the WebGPU
 *     output is bit-identical to WebGL2's, so switching GPU API never
 *     changes the image. An early revision used the CPU's y/height and
 *     differed from WebGL2 by up to 6 levels on 0.17% of elements -- a
 *     half-row shift flipping the `sky > 0.15` step, amplified by later
 *     stages; a one-line patch experiment showed that WAS the entire
 *     difference.)
 *  3. Median noise reduction is implemented here (radius 1 or 2 => a 9
 *     or 25 element window, exact selection, so it matches the CPU
 *     reference exactly). WebGL2 never ported it.
 *  4. Multiple outputs are separate storage textures rather than MRT.
 *
 * Conventions: group(0) binding(0) is always the uniform block `P`;
 * bindings 1..inputs are `texture_2d<f32>` (float32 textures are
 * unfilterable, read only via textureLoad); the following bindings are
 * write-only storage textures. Workgroups are 8x8.
 */

import { DEFAULT_MAX_CHROMA_BOOST } from '../../colorCorrectionPipeline';
import { DEFAULT_STEPS, DEFAULT_EPS } from '../../gamutProtection';
import { AUTOTONE_KNEE_START } from '../../autoTone';
import { MAX_STOPS } from '../fullPipelineShaders';
import type { WTextureFormat } from './wgpuTypes';

export const WORKGROUP_SIZE = 8;

export type KernelName =
  | 'boxBlurR'
  | 'boxBlurRgba'
  | 'blendR'
  | 'bilateralR'
  | 'medianR'
  | 'squareR'
  | 'varianceCombine'
  | 'localContrast'
  | 'toneCurve'
  | 'scene'
  | 'colorMap'
  | 'sharpenCombine'
  | 'quantSrgb'
  | 'pack8'
  | 'precision';

export interface KernelSpec {
  wgsl: string;
  /** Number of `texture_2d<f32>` inputs (bindings 1..inputs). */
  inputs: number;
  /** Storage-texture output formats, in binding order after the inputs. */
  outputs: readonly WTextureFormat[];
  /** Size in bytes of the uniform struct `P` (multiple of 16). */
  uniformSize: number;
}

/** Formats a JS number as a WGSL f32 literal (always has a '.' or exponent). */
function f(n: number): string {
  if (!Number.isFinite(n)) throw new Error(`WGSL constant is not finite: ${n}`);
  const s = String(n);
  return /[.eE]/.test(s) ? s : `${s}.0`;
}

const ENTRY = `@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE}, 1)`;

// ---- Shared OKLab / sRGB helpers (constants identical to colorSpace.ts and
//      to the two GLSL copies in shaders.ts / fullPipelineShaders.ts). ----
const OKLAB_WGSL = `
fn cbrtSigned(x: f32) -> f32 {
  return sign(x) * pow(abs(x), 1.0 / 3.0);
}

fn linearRgbToOklab(c: vec3<f32>) -> vec3<f32> {
  let l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
  let m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
  let s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
  let l_ = cbrtSigned(l);
  let m_ = cbrtSigned(m);
  let s_ = cbrtSigned(s);
  return vec3<f32>(
    0.2104542553 * l_ + 0.793617785 * m_ - 0.0040720468 * s_,
    1.9779984951 * l_ - 2.428592205 * m_ + 0.4505937099 * s_,
    0.0259040371 * l_ + 0.7827717662 * m_ - 0.808675766 * s_
  );
}

fn oklabToLinearRgb(lab: vec3<f32>) -> vec3<f32> {
  let l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
  let m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
  let s_ = lab.x - 0.0894841775 * lab.y - 1.291485548 * lab.z;
  let l = l_ * l_ * l_;
  let m = m_ * m_ * m_;
  let s = s_ * s_ * s_;
  return vec3<f32>(
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s
  );
}
`;

// Every kernel starts with the same coordinate prologue: global invocation
// -> integer texel, bounds-checked against the (shared) tile dimensions.
const PROLOGUE = (dimsFrom: string): string => `
  let dim = textureDimensions(${dimsFrom});
  if (id.x >= dim.x || id.y >= dim.y) { return; }
  let c = vec2<i32>(id.xy);
  let mx = vec2<i32>(dim) - vec2<i32>(1, 1);
`;

const BOX_BLUR_R = `
struct P { radius: i32, dx: i32, dy: i32, pad0: i32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<r32float, write>;
${ENTRY}
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
${PROLOGUE('src')}
  var acc = 0.0;
  for (var i: i32 = -p.radius; i <= p.radius; i++) {
    let q = clamp(c + vec2<i32>(p.dx, p.dy) * i, vec2<i32>(0, 0), mx);
    acc += textureLoad(src, q, 0).r;
  }
  textureStore(dst, c, vec4<f32>(acc / f32(2 * p.radius + 1), 0.0, 0.0, 1.0));
}
`;

const BOX_BLUR_RGBA = `
struct P { radius: i32, dx: i32, dy: i32, pad0: i32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba32float, write>;
${ENTRY}
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
${PROLOGUE('src')}
  var acc = vec3<f32>(0.0, 0.0, 0.0);
  for (var i: i32 = -p.radius; i <= p.radius; i++) {
    let q = clamp(c + vec2<i32>(p.dx, p.dy) * i, vec2<i32>(0, 0), mx);
    acc += textureLoad(src, q, 0).rgb;
  }
  textureStore(dst, c, vec4<f32>(acc / f32(2 * p.radius + 1), 1.0));
}
`;

// mix(a, b, t) == a*(1-t) + b*t, the CPU blend() in noiseReduction.ts (and
// preserveDetail() in pipeline.ts, which is the same lerp).
const BLEND_R = `
struct P { t: f32, pad0: f32, pad1: f32, pad2: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var a: texture_2d<f32>;
@group(0) @binding(2) var b: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<r32float, write>;
${ENTRY}
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
${PROLOGUE('a')}
  let va = textureLoad(a, c, 0).r;
  let vb = textureLoad(b, c, 0).r;
  textureStore(dst, c, vec4<f32>(mix(va, vb, p.t), 0.0, 0.0, 1.0));
}
`;

const BILATERAL_R = `
struct P { radius: i32, twoSpatialSigma2: f32, twoRangeSigma2: f32, pad0: i32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<r32float, write>;
${ENTRY}
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
${PROLOGUE('src')}
  let centerVal = textureLoad(src, c, 0).r;
  var sum = 0.0;
  var weightSum = 0.0;
  for (var dy: i32 = -p.radius; dy <= p.radius; dy++) {
    for (var dx: i32 = -p.radius; dx <= p.radius; dx++) {
      let q = clamp(c + vec2<i32>(dx, dy), vec2<i32>(0, 0), mx);
      let val = textureLoad(src, q, 0).r;
      let spatialDist2 = f32(dx * dx + dy * dy);
      let rangeDist2 = (val - centerVal) * (val - centerVal);
      let weight = exp(-spatialDist2 / p.twoSpatialSigma2 - rangeDist2 / p.twoRangeSigma2);
      sum += val * weight;
      weightSum += weight;
    }
  }
  let outv = select(centerVal, clamp(sum / weightSum, 0.0, 1.0), weightSum > 0.0);
  textureStore(dst, c, vec4<f32>(outv, 0.0, 0.0, 1.0));
}
`;

// Median filter (radius 1 or 2). Mirrors noiseReduction.ts medianFilter():
// gather the clamped window, sort ascending, take element floor(k/2). Pure
// selection -- no arithmetic -- so it matches the CPU bit for bit.
const MEDIAN_R = `
struct P { radius: i32, pad0: i32, pad1: i32, pad2: i32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<r32float, write>;
${ENTRY}
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
${PROLOGUE('src')}
  var w: array<f32, 25>;
  var k: i32 = 0;
  for (var dy: i32 = -p.radius; dy <= p.radius; dy++) {
    for (var dx: i32 = -p.radius; dx <= p.radius; dx++) {
      let q = clamp(c + vec2<i32>(dx, dy), vec2<i32>(0, 0), mx);
      w[k] = textureLoad(src, q, 0).r;
      k++;
    }
  }
  for (var a: i32 = 1; a < k; a++) {
    let v = w[a];
    var b: i32 = a - 1;
    loop {
      if (b < 0 || w[b] <= v) { break; }
      w[b + 1] = w[b];
      b--;
    }
    w[b + 1] = v;
  }
  textureStore(dst, c, vec4<f32>(w[k / 2], 0.0, 0.0, 1.0));
}
`;

const SQUARE_R = `
struct P { pad0: u32, pad1: u32, pad2: u32, pad3: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<r32float, write>;
${ENTRY}
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
${PROLOGUE('src')}
  let v = textureLoad(src, c, 0).r;
  textureStore(dst, c, vec4<f32>(v * v, 0.0, 0.0, 1.0));
}
`;

// variance = max(0, meanSq - mean^2). Mirrors contrast.ts localVariance().
const VARIANCE_COMBINE = `
struct P { pad0: u32, pad1: u32, pad2: u32, pad3: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var meanTex: texture_2d<f32>;
@group(0) @binding(2) var meanSqTex: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<r32float, write>;
${ENTRY}
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
${PROLOGUE('meanTex')}
  let mean = textureLoad(meanTex, c, 0).r;
  let meanSq = textureLoad(meanSqTex, c, 0).r;
  textureStore(dst, c, vec4<f32>(max(0.0, meanSq - mean * mean), 0.0, 0.0, 1.0));
}
`;

const LOCAL_CONTRAST = `
struct P { amt: f32, pad0: f32, pad1: f32, pad2: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var localMeanTex: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<r32float, write>;
${ENTRY}
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
${PROLOGUE('src')}
  let I = textureLoad(src, c, 0).r;
  let localMean = textureLoad(localMeanTex, c, 0).r;
  let detail = I - localMean;
  textureStore(dst, c, vec4<f32>(clamp(I + detail * p.amt * 1.5, 0.0, 1.0), 0.0, 0.0, 1.0));
}
`;

const TONE_CURVE = `
struct P {
  exposureMul: f32, brightnessAdd: f32, gamma: f32, shadowLift: f32,
  highlightRecovery: f32, sCurveK: f32, pad0: f32, pad1: f32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<r32float, write>;

fn sCurveF(x: f32, k: f32) -> f32 {
  if (x <= 0.5) { return 0.5 * pow(x / 0.5, k); }
  return 1.0 - 0.5 * pow((1.0 - x) / 0.5, k);
}

${ENTRY}
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
${PROLOGUE('src')}
  var i = textureLoad(src, c, 0).r * p.exposureMul + p.brightnessAdd;
  i = clamp(i, 0.0, 1.0);

  if (p.shadowLift > 0.0) {
    i = i + p.shadowLift * 0.35 * (1.0 - i) * exp(-i * 6.0);
  }

  if (p.highlightRecovery > 0.0) {
    let knee = 1.0 - p.highlightRecovery * 0.5;
    if (i > knee) {
      let over = (i - knee) / max(0.0001, 1.0 - knee);
      i = knee + (1.0 - knee) * (1.0 - exp(-over * 2.0));
    }
  }

  i = pow(clamp(i, 0.0, 1.0), 1.0 / p.gamma);
  i = sCurveF(i, p.sCurveK);

  textureStore(dst, c, vec4<f32>(clamp(i, 0.0, 1.0), 0.0, 0.0, 1.0));
}
`;

// Scene heuristics (sky / vegetation confidence). Mirrors pipeline.ts
// computeSceneMaps(). rowFrac is derived from the GLOBAL row index so a
// tile of a larger image computes exactly what the whole image would.
const SCENE = `
struct P { rowOrigin: i32, fullHeight: f32, pad0: f32, pad1: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var intensityTex: texture_2d<f32>;
@group(0) @binding(2) var varianceTex: texture_2d<f32>;
@group(0) @binding(3) var skyOut: texture_storage_2d<r32float, write>;
@group(0) @binding(4) var vegOut: texture_storage_2d<r32float, write>;
${ENTRY}
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
${PROLOGUE('intensityTex')}
  let I = textureLoad(intensityTex, c, 0).r;
  let v = textureLoad(varianceTex, c, 0).r;
  let rowFrac = (f32(p.rowOrigin + c.y) + 0.5) / p.fullHeight; // texel centre; 0 = top of the WHOLE image

  let topBias = clamp(1.0 - rowFrac * 1.6, 0.0, 1.0);
  let brightness = clamp((I - 0.55) / 0.45, 0.0, 1.0);
  let flatness = clamp(1.0 - v * 40.0, 0.0, 1.0);
  let sky = topBias * brightness * flatness;

  let bottomBias = clamp(rowFrac * 1.2 + 0.2, 0.0, 1.0);
  let midtone = clamp(1.0 - abs(I - 0.42) / 0.35, 0.0, 1.0);
  let grain = clamp(v * 30.0, 0.0, 1.0);
  let veg = bottomBias * midtone * grain;

  textureStore(skyOut, c, vec4<f32>(sky, 0.0, 0.0, 1.0));
  textureStore(vegOut, c, vec4<f32>(veg, 0.0, 0.0, 1.0));
}
`;

// IR/NIR -> RGB color-ramp reconstruction. Mirrors colorMapping.ts
// mapIntensityToRgb(); outputs LINEAR RGB (final srgbToLinear) so the
// precision pass can consume it directly.
const COLOR_MAP = `
struct P {
  stopT: array<vec4<f32>, ${MAX_STOPS}>,       // .x used
  stopColor: array<vec4<f32>, ${MAX_STOPS}>,   // .xyz used
  stopCount: i32,
  sceneEnabled: u32,
  colorStrength: f32,
  satAdjust: f32,
  hueBiasFrac: f32,
  tempAmt: f32,
  pad0: f32,
  pad1: f32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var intensityTex: texture_2d<f32>;
@group(0) @binding(2) var skyTex: texture_2d<f32>;
@group(0) @binding(3) var vegTex: texture_2d<f32>;
@group(0) @binding(4) var dst: texture_storage_2d<rgba32float, write>;
${OKLAB_WGSL}
fn srgbToLinearF(c: f32) -> f32 {
  return select(pow((c + 0.055) / 1.055, 2.4), c / 12.92, c <= 0.04045);
}
fn linearToSrgbF(c: f32) -> f32 {
  let v = select(1.055 * pow(max(c, 0.0), 1.0 / 2.4) - 0.055, c * 12.92, c <= 0.0031308);
  return clamp(v, 0.0, 1.0);
}
fn srgbToLinear3(c: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(srgbToLinearF(c.r), srgbToLinearF(c.g), srgbToLinearF(c.b));
}
fn srgbToOklab(c: vec3<f32>) -> vec3<f32> { return linearRgbToOklab(srgbToLinear3(c)); }
fn oklabToSrgb(lab: vec3<f32>) -> vec3<f32> {
  let lin = oklabToLinearRgb(lab);
  return vec3<f32>(linearToSrgbF(lin.r), linearToSrgbF(lin.g), linearToSrgbF(lin.b));
}
fn lerpOklabF(c1: vec3<f32>, c2: vec3<f32>, t: f32) -> vec3<f32> {
  let lab1 = srgbToOklab(c1);
  let lab2 = srgbToOklab(c2);
  return oklabToSrgb(mix(lab1, lab2, t));
}

fn sampleRampF(t: f32) -> vec3<f32> {
  let tc = clamp(t, 0.0, 1.0);
  if (p.stopCount <= 0) { return vec3<f32>(tc, tc, tc); }
  if (tc <= p.stopT[0].x) { return p.stopColor[0].xyz; }
  if (tc >= p.stopT[p.stopCount - 1].x) { return p.stopColor[p.stopCount - 1].xyz; }
  for (var i: i32 = 0; i < ${MAX_STOPS - 1}; i++) {
    if (i >= p.stopCount - 1) { break; }
    let a = p.stopT[i].x;
    let b = p.stopT[i + 1].x;
    if (tc >= a && tc <= b) {
      let span = max(1e-6, b - a);
      let localT = (tc - a) / span;
      return lerpOklabF(p.stopColor[i].xyz, p.stopColor[i + 1].xyz, localT);
    }
  }
  return p.stopColor[p.stopCount - 1].xyz;
}

// HSL -- mirrors colorSpace.ts rgbToHsl/hslToRgb exactly.
fn rgbToHslF(c: vec3<f32>) -> vec3<f32> {
  let maxc = max(c.r, max(c.g, c.b));
  let minc = min(c.r, min(c.g, c.b));
  let l = (maxc + minc) * 0.5;
  if (maxc == minc) { return vec3<f32>(0.0, 0.0, l); }
  let d = maxc - minc;
  var s: f32;
  if (l > 0.5) { s = d / (2.0 - maxc - minc); } else { s = d / (maxc + minc); }
  var h: f32;
  if (maxc == c.r) {
    h = (c.g - c.b) / d + select(0.0, 6.0, c.g < c.b);
  } else if (maxc == c.g) {
    h = (c.b - c.r) / d + 2.0;
  } else {
    h = (c.r - c.g) / d + 4.0;
  }
  h = h / 6.0;
  return vec3<f32>(h, s, l);
}
fn hue2rgbF(pp: f32, q: f32, t: f32) -> f32 {
  var tt = t;
  if (tt < 0.0) { tt += 1.0; }
  if (tt > 1.0) { tt -= 1.0; }
  if (tt < 1.0 / 6.0) { return pp + (q - pp) * 6.0 * tt; }
  if (tt < 0.5) { return q; }
  if (tt < 2.0 / 3.0) { return pp + (q - pp) * (2.0 / 3.0 - tt) * 6.0; }
  return pp;
}
fn hslToRgbF(h: f32, s: f32, l: f32) -> vec3<f32> {
  if (s == 0.0) { return vec3<f32>(l, l, l); }
  var q: f32;
  if (l < 0.5) { q = l * (1.0 + s); } else { q = l + s - l * s; }
  let pp = 2.0 * l - q;
  return vec3<f32>(hue2rgbF(pp, q, h + 1.0 / 3.0), hue2rgbF(pp, q, h), hue2rgbF(pp, q, h - 1.0 / 3.0));
}

${ENTRY}
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
${PROLOGUE('intensityTex')}
  let I = textureLoad(intensityTex, c, 0).r;
  var rgb = sampleRampF(I);

  if (p.colorStrength < 1.0) {
    rgb = rgb * p.colorStrength + vec3<f32>(I, I, I) * (1.0 - p.colorStrength);
  }

  if (p.sceneEnabled != 0u) {
    let sky = textureLoad(skyTex, c, 0).r;
    let veg = textureLoad(vegTex, c, 0).r;
    if (sky > 0.15) {
      let w = sky * 0.35;
      rgb.b = clamp(rgb.b + w * 0.10, 0.0, 1.0);
      rgb.r = clamp(rgb.r - w * 0.04, 0.0, 1.0);
    }
    if (veg > 0.15) {
      let w = veg * 0.35;
      rgb.g = clamp(rgb.g + w * 0.08, 0.0, 1.0);
      rgb.r = clamp(rgb.r - w * 0.03, 0.0, 1.0);
    }
  }

  rgb.r = clamp(rgb.r + p.tempAmt, 0.0, 1.0);
  rgb.b = clamp(rgb.b - p.tempAmt, 0.0, 1.0);

  let hsl = rgbToHslF(rgb);
  var newH = hsl.x + p.hueBiasFrac;
  newH -= floor(newH);
  var newS: f32;
  if (p.satAdjust >= 0.0) {
    newS = clamp(hsl.y + (1.0 - hsl.y) * p.satAdjust, 0.0, 1.0);
  } else {
    newS = clamp(hsl.y * (1.0 + p.satAdjust), 0.0, 1.0);
  }
  rgb = hslToRgbF(newH, newS, hsl.z);

  textureStore(dst, c, vec4<f32>(srgbToLinear3(clamp(rgb, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0))), 1.0));
}
`;

// Unsharp-mask combine. Mirrors sharpening.ts unsharpMask() on all 3 channels.
const SHARPEN_COMBINE = `
struct P { amount: f32, threshold: f32, pad0: f32, pad1: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var blurredTex: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba32float, write>;
${ENTRY}
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
${PROLOGUE('src')}
  let orig = textureLoad(src, c, 0).rgb;
  let blurred = textureLoad(blurredTex, c, 0).rgb;
  let detail = orig - blurred;
  let applied = vec3<f32>(
    select(0.0, detail.r * p.amount, abs(detail.r) >= p.threshold),
    select(0.0, detail.g * p.amount, abs(detail.g) >= p.threshold),
    select(0.0, detail.b * p.amount, abs(detail.b) >= p.threshold)
  );
  textureStore(dst, c, vec4<f32>(clamp(orig + applied, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0)), 1.0));
}
`;

// Precision pipeline: Azusa white balance -> AutoTone -> color correction
// (channel balance + chroma shaping) -> gamut protection. Pointwise. The
// constants are read from the same CPU modules the GLSL version reads.
function buildPrecisionWgsl(): string {
  return `
struct P {
  wbR: f32, wbG: f32, wbB: f32, atBlackPoint: f32,
  atRange: f32, ccR: f32, ccG: f32, ccB: f32,
  ccStrengthFrac: f32, atEnabled: u32, ccEnabled: u32, gamutEnabled: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba32float, write>;

const KNEE_START: f32 = ${f(AUTOTONE_KNEE_START)};
const MAX_CHROMA_BOOST: f32 = ${f(DEFAULT_MAX_CHROMA_BOOST)};
const GAMUT_STEPS: i32 = ${Math.round(DEFAULT_STEPS)};
const GAMUT_EPS: f32 = ${f(DEFAULT_EPS)};
${OKLAB_WGSL}
fn autoToneMap(v: f32) -> f32 {
  var x = (v - p.atBlackPoint) / p.atRange;
  if (x > KNEE_START) {
    let over = (x - KNEE_START) / (1.0 - KNEE_START);
    x = KNEE_START + (1.0 - KNEE_START) * (1.0 - exp(-over));
  }
  return clamp(x, 0.0, 1.0);
}

fn chromaShape(rgb: vec3<f32>) -> vec3<f32> {
  let lab = linearRgbToOklab(rgb);
  let C = length(lab.yz);
  if (C < 1e-5) { return rgb; }
  let midtoneWeight = 1.0 - pow(min(1.0, abs(lab.x - 0.5) / 0.5), 1.5);
  let saturationRolloff = 1.0 / (1.0 + C * 3.0);
  let boost = 1.0 + (MAX_CHROMA_BOOST - 1.0) * p.ccStrengthFrac * midtoneWeight * saturationRolloff;
  let newC = C * boost;
  let scale = newC / C;
  return oklabToLinearRgb(vec3<f32>(lab.x, lab.y * scale, lab.z * scale));
}

fn inGamut(rgb: vec3<f32>) -> bool {
  return all(rgb >= vec3<f32>(-GAMUT_EPS, -GAMUT_EPS, -GAMUT_EPS)) &&
         all(rgb <= vec3<f32>(1.0 + GAMUT_EPS, 1.0 + GAMUT_EPS, 1.0 + GAMUT_EPS));
}

fn gamutMap(rgb: vec3<f32>) -> vec3<f32> {
  let zero = vec3<f32>(0.0, 0.0, 0.0);
  let one = vec3<f32>(1.0, 1.0, 1.0);
  if (inGamut(rgb)) { return clamp(rgb, zero, one); }
  let lab = linearRgbToOklab(rgb);
  let C = length(lab.yz);
  if (C < 1e-6) { return clamp(rgb, zero, one); }
  let hue = lab.yz / C;
  var lo = 0.0;
  var hi = C;
  var best = clamp(rgb, zero, one);
  for (var i: i32 = 0; i < GAMUT_STEPS; i++) {
    let mid = (lo + hi) * 0.5;
    let candidate = oklabToLinearRgb(vec3<f32>(lab.x, hue * mid));
    if (inGamut(candidate)) {
      best = candidate;
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return clamp(best, zero, one);
}

${ENTRY}
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
${PROLOGUE('src')}
  var rgb = textureLoad(src, c, 0).rgb;

  // 1. Azusa white balance.
  rgb = rgb * vec3<f32>(p.wbR, p.wbG, p.wbB);

  // 2. AutoTone.
  if (p.atEnabled != 0u) {
    rgb = vec3<f32>(autoToneMap(rgb.r), autoToneMap(rgb.g), autoToneMap(rgb.b));
  }

  // 3. ColorCorrectionPipeline (channel balance + chroma shaping).
  if (p.ccEnabled != 0u) {
    rgb = rgb * vec3<f32>(p.ccR, p.ccG, p.ccB);
    rgb = chromaShape(rgb);
  }

  // 4. Gamut protection.
  if (p.gamutEnabled != 0u) {
    rgb = gamutMap(rgb);
  } else {
    rgb = clamp(rgb, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0));
  }

  textureStore(dst, c, vec4<f32>(rgb, 1.0));
}
`;
}

// ---- GPU-resident output stage -------------------------------------------
// Replaces the CPU `linearToSrgb(c) * 255 -> Uint8ClampedArray` loop that used
// to sit between the precision stage and sharpening (and after it).
//
// quantSrgb: linear RGB -> sRGB, quantised to the 8-bit grid but kept as a
// float k/255, i.e. exactly the values sharpening consumed when they were
// round-tripped through a Uint8ClampedArray on the CPU:
//   host:  byte = clampRound(linearToSrgbHost(c) * 255);   value = byte / 255
//   here:  K255[ round(linearToSrgbF(c) * 255) ]   (K255[k] == Math.fround(k / 255); see K255_TABLE)
// WGSL round() is round-half-to-even, the same tie rule as Uint8ClampedArray.
// The transfer function is the same expression as the colour-map kernel's.
// The 8-bit grid as f32 values, computed on the CPU: entry k is exactly Math.fround(k / 255), i.e. the very float
// the CPU path produced (`byte / 255` stored into a Float32Array). Looking the value up instead of dividing on the
// GPU makes the sharpen input bit-identical to the CPU-staged path on EVERY implementation (WGSL division is only
// required to be accurate to a couple of ULP, and a 1-ULP difference can flip the sharpen threshold's exact ties,
// because blurred 8-bit values differ by exact multiples of 1/255).
const K255_TABLE = Array.from({ length: 256 }, (_, k) => f(Math.fround(k / 255))).join(', ');

const QUANT_SRGB = `
struct P { pad0: u32, pad1: u32, pad2: u32, pad3: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba32float, write>;

fn linearToSrgbQ(c: f32) -> f32 {
  let v = select(1.055 * pow(max(c, 0.0), 1.0 / 2.4) - 0.055, c * 12.92, c <= 0.0031308);
  return clamp(v, 0.0, 1.0);
}
var<private> K255: array<f32, 256> = array<f32, 256>(${K255_TABLE});
fn q8(v: f32) -> f32 { return K255[u32(clamp(round(v * 255.0), 0.0, 255.0))]; }

${ENTRY}
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
${PROLOGUE('src')}
  let l = textureLoad(src, c, 0).rgb;
  textureStore(dst, c, vec4<f32>(q8(linearToSrgbQ(l.r)), q8(linearToSrgbQ(l.g)), q8(linearToSrgbQ(l.b)), 1.0));
}
`;

// pack8: RGBA float in [0,1] -> one u32 per pixel (R | G<<8 | B<<16 | 255<<24),
// with round-half-even and clamping like Uint8ClampedArray. Reading this back
// moves 4 bytes per pixel instead of the 16 a float RGBA readback costs, and
// no CPU float->byte conversion is needed. Values reaching it are already
// clamped by the kernels upstream.
const PACK8 = `
struct P { pad0: u32, pad1: u32, pad2: u32, pad3: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<r32uint, write>;

fn b8(v: f32) -> u32 { return u32(clamp(round(v * 255.0), 0.0, 255.0)); }

${ENTRY}
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
${PROLOGUE('src')}
  let v = textureLoad(src, c, 0).rgb;
  let word = b8(v.r) | (b8(v.g) << 8u) | (b8(v.b) << 16u) | (255u << 24u);
  textureStore(dst, c, vec4<u32>(word, 0u, 0u, 0u));
}
`;

export const KERNELS: Record<KernelName, KernelSpec> = {
  boxBlurR: { wgsl: BOX_BLUR_R, inputs: 1, outputs: ['r32float'], uniformSize: 16 },
  boxBlurRgba: { wgsl: BOX_BLUR_RGBA, inputs: 1, outputs: ['rgba32float'], uniformSize: 16 },
  blendR: { wgsl: BLEND_R, inputs: 2, outputs: ['r32float'], uniformSize: 16 },
  bilateralR: { wgsl: BILATERAL_R, inputs: 1, outputs: ['r32float'], uniformSize: 16 },
  medianR: { wgsl: MEDIAN_R, inputs: 1, outputs: ['r32float'], uniformSize: 16 },
  squareR: { wgsl: SQUARE_R, inputs: 1, outputs: ['r32float'], uniformSize: 16 },
  varianceCombine: { wgsl: VARIANCE_COMBINE, inputs: 2, outputs: ['r32float'], uniformSize: 16 },
  localContrast: { wgsl: LOCAL_CONTRAST, inputs: 2, outputs: ['r32float'], uniformSize: 16 },
  toneCurve: { wgsl: TONE_CURVE, inputs: 1, outputs: ['r32float'], uniformSize: 32 },
  scene: { wgsl: SCENE, inputs: 2, outputs: ['r32float', 'r32float'], uniformSize: 16 },
  colorMap: { wgsl: COLOR_MAP, inputs: 3, outputs: ['rgba32float'], uniformSize: MAX_STOPS * 32 + 32 },
  sharpenCombine: { wgsl: SHARPEN_COMBINE, inputs: 2, outputs: ['rgba32float'], uniformSize: 16 },
  quantSrgb: { wgsl: QUANT_SRGB, inputs: 1, outputs: ['rgba32float'], uniformSize: 16 },
  pack8: { wgsl: PACK8, inputs: 1, outputs: ['r32uint'], uniformSize: 16 },
  precision: { wgsl: buildPrecisionWgsl(), inputs: 1, outputs: ['rgba32float'], uniformSize: 48 },
};

export const KERNEL_NAMES = Object.keys(KERNELS) as KernelName[];
