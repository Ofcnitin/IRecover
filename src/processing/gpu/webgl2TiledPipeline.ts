/**
 * WebGL2 TILED full pipeline.
 * ---------------------------
 * Runs the same pipeline as webgl2FullPipeline.ts's runFullPipelineWebGL2
 * for images whose width or height exceeds gl.MAX_TEXTURE_SIZE, by
 * processing bounded tiles with overlap ("halo") regions. It reuses every
 * existing shader UNMODIFIED (fullPipelineShaders.ts, shaders.ts); the one
 * position-dependent pass (scene heuristics) uses a variant DERIVED from
 * the original by an asserted substitution (tiledShaders.ts). The
 * untiled path is not touched by this module.
 *
 * Correctness model (see tilePlanner.ts for the halo argument)
 * ------------------------------------------------------------
 * Two spatial phases, separated by ONE full-image synchronisation point
 * that exists in the untiled pipeline too:
 *
 *   Phase A (per tile, halo = noise + local-contrast + scene reach):
 *     intensity -> noise reduction -> local contrast -> tone curve ->
 *     detail preservation -> scene heuristics -> IR->RGB color ramp.
 *     Only each tile's CORE is read back, into full-image planar linear
 *     RGB (the same r/g/b Float32 planes the untiled path builds).
 *
 *   Sync point: the precision stage's white-balance / AutoTone /
 *     color-correction parameters are WHOLE-IMAGE statistics. They are
 *     computed once, from the complete planar buffers, by the very same
 *     CPU estimators the untiled path uses -- so every tile is corrected
 *     with identical parameters and no seams can arise from per-tile
 *     statistics.
 *
 *   Phase C (per tile, no halo -- the shader is pointwise): apply the
 *     precision shader with those global parameters.
 *
 *   Linear -> 8-bit sRGB on the full image (identical to the untiled code).
 *
 *   Phase E (per tile, halo = sharpen radius): unsharp mask. Neighbours
 *     are read from the immutable pre-sharpen image and results are
 *     written to a SEPARATE output, so a tile can never see an
 *     already-sharpened neighbour.
 *
 * With halo >= chained reach, every core pixel sees exactly the same
 * inputs and runs exactly the same arithmetic as in the untiled run, so
 * results are expected to be bit-identical, not merely close (verified
 * by the tiled-vs-untiled checks in tools/gpu-consistency/run.ts).
 *
 * Failure model: any unsupported environment, GL error, context loss, or
 * genuinely untileable request (TilingUnsupportedError) throws, and
 * processing/pipeline.ts treats every throw as "fall back to CPU". This
 * module never returns a partially processed or wrong image: GL errors
 * (notably OUT_OF_MEMORY) are checked after every tile.
 *
 * Memory: the tile working set is bounded by maxTileDim (VRAM), and
 * intermediates are released as soon as they are consumed. CPU memory is
 * O(pixels) -- three full-image Float32 planes plus 8-bit output --
 * the same order as the untiled path's own buffers; the planes are
 * released before the sharpening phase.
 */

import type { ProcessingSettings, ColorStop } from '../../types/processing';
import { linkProgram } from './webgl2Backend';
import type { PrecisionPipelineGpuResult } from './webgl2Backend';
import { PRECISION_PIPELINE_VERTEX_SRC, PRECISION_PIPELINE_FRAGMENT_SRC } from './shaders';
import {
  FULL_PIPELINE_VERTEX_SRC,
  BOX_BLUR_R_FRAG_SRC,
  BOX_BLUR_RGB_FRAG_SRC,
  BLEND_R_FRAG_SRC,
  BILATERAL_R_FRAG_SRC,
  SQUARE_R_FRAG_SRC,
  VARIANCE_COMBINE_FRAG_SRC,
  LOCAL_CONTRAST_FRAG_SRC,
  TONE_CURVE_FRAG_SRC,
  COLOR_MAP_FRAG_SRC,
  SHARPEN_COMBINE_FRAG_SRC,
  MAX_STOPS,
} from './fullPipelineShaders';
import { SCENE_TILED_FRAG_SRC } from './tiledShaders';
import { computeWhiteBalanceGains } from '../whiteBalance';
import { computeAutoToneParams } from '../autoTone';
import { computeChannelBalanceGains } from '../colorCorrectionPipeline';
import {
  computeWasTiled,
  DEFAULT_MAX_TILE_DIM,
  planTiles,
  resolveStageRadii,
  SCENE_VARIANCE_RADIUS,
  type Tile,
  type TilePlan,
  type TilingDiagnostics,
} from './tilePlanner';
import type { FullPipelineGpuResult, FullPipelineQualityParams } from './webgl2FullPipeline';

export interface TiledPipelineOptions {
  /**
   * Maximum tile edge in pixels, halo included. Defaults to
   * min(DEFAULT_MAX_TILE_DIM, gl.MAX_TEXTURE_SIZE); never exceeds the
   * GPU's real limit. Tests pass small values to force multi-tile grids
   * on small images.
   */
  maxTileDim?: number;
  /**
   * DIAGNOSTICS/TESTS ONLY. Replaces the computed halo. Any value below
   * the true reach produces seams by design; the consistency harness
   * uses it as a negative control to prove its seam detection has teeth
   * and that the computed halo is not larger than necessary. Never set
   * this in production code.
   */
  unsafeHaloOverride?: { colorMap?: number; sharpen?: number };
}

// ---------------------------------------------------------------------
// GL plumbing (module-private copies of the tiny helpers the untiled
// orchestrator keeps private; kept separate so the validated untiled
// module needs no edits beyond delegating oversize images here).
// ---------------------------------------------------------------------

function createTileGL(): WebGL2RenderingContext {
  const g = globalThis as unknown as { OffscreenCanvas?: unknown; document?: Document };
  // 1x1 default framebuffer on purpose: every pass renders into FBO
  // textures, and a canvas as large as an oversize image is exactly what
  // browsers refuse to allocate.
  let gl: WebGL2RenderingContext | null = null;
  if (typeof g.OffscreenCanvas !== 'undefined') {
    const canvas = new (g.OffscreenCanvas as new (w: number, h: number) => OffscreenCanvas)(1, 1);
    gl = canvas.getContext('webgl2', { antialias: false, alpha: true }) as WebGL2RenderingContext | null;
  } else if (g.document) {
    const canvas = g.document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;
    gl = canvas.getContext('webgl2', { antialias: false, alpha: true }) as WebGL2RenderingContext | null;
  } else {
    throw new Error('No OffscreenCanvas or document available to create a WebGL2 context.');
  }
  if (!gl) throw new Error('WebGL2 context creation failed.');
  if (!gl.getExtension('EXT_color_buffer_float')) {
    throw new Error('EXT_color_buffer_float is not supported -- cannot render to a float texture.');
  }
  return gl;
}

function releaseContext(gl: WebGL2RenderingContext): void {
  try {
    gl.getExtension('WEBGL_lose_context')?.loseContext();
  } catch {
    /* best effort: the context is unreachable after this anyway */
  }
}

/**
 * Reads gl.MAX_TEXTURE_SIZE from a throwaway 1x1 context (released
 * immediately). Used by the untiled entry point to decide whether to
 * delegate here BEFORE it creates a canvas sized to the image -- an
 * oversize canvas can fail context creation outright, which would turn a
 * tileable image into a needless CPU fallback.
 */
export function probeMaxTextureSize(): number {
  const gl = createTileGL();
  try {
    return gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
  } finally {
    releaseContext(gl);
  }
}

interface Surface {
  tex: WebGLTexture;
  fbo: WebGLFramebuffer;
}

function createSurface(gl: WebGL2RenderingContext, width: number, height: number, channels: 1 | 4): Surface {
  const tex = gl.createTexture();
  if (!tex) throw new Error('gl.createTexture failed.');
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  if (channels === 1) {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, null);
  } else {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
  }
  const fbo = gl.createFramebuffer();
  if (!fbo) throw new Error('gl.createFramebuffer failed.');
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
    throw new Error('WebGL2 framebuffer incomplete (tiled).');
  }
  gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
  // Leave nothing bound: a texture that is both bound to a sampler unit
  // and attached to the framebuffer being drawn is a feedback loop.
  gl.bindTexture(gl.TEXTURE_2D, null);
  return { tex, fbo };
}

interface DualSurface {
  fbo: WebGLFramebuffer;
  texA: WebGLTexture;
  texB: WebGLTexture;
}

/** Two-output (MRT) R32F surface -- used only by the scene-heuristics pass. */
function createDualSurface(gl: WebGL2RenderingContext, width: number, height: number): DualSurface {
  const texA = gl.createTexture();
  const texB = gl.createTexture();
  if (!texA || !texB) throw new Error('gl.createTexture failed (dual).');
  gl.activeTexture(gl.TEXTURE0);
  for (const t of [texA, texB]) {
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, null);
  }
  const fbo = gl.createFramebuffer();
  if (!fbo) throw new Error('gl.createFramebuffer failed (dual).');
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texA, 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, texB, 0);
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
    throw new Error('WebGL2 framebuffer incomplete (tiled dual/MRT).');
  }
  gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return { fbo, texA, texB };
}

function disposeSurface(gl: WebGL2RenderingContext, s: Surface): void {
  gl.deleteTexture(s.tex);
  gl.deleteFramebuffer(s.fbo);
}

function disposeDual(gl: WebGL2RenderingContext, d: DualSurface): void {
  gl.deleteTexture(d.texA);
  gl.deleteTexture(d.texB);
  gl.deleteFramebuffer(d.fbo);
}

/** Uploads a Float32Array into a fresh surface's texture (R32F or RGBA32F). */
function uploadInto(gl: WebGL2RenderingContext, s: Surface, width: number, height: number, channels: 1 | 4, data: Float32Array): void {
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, s.tex);
  if (channels === 1) {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, data);
  } else {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, data);
  }
  gl.bindTexture(gl.TEXTURE_2D, null);
}

function assertHealthy(gl: WebGL2RenderingContext, where: string): void {
  if (gl.isContextLost()) throw new Error(`WebGL2 context was lost during ${where}.`);
  const err = gl.getError();
  if (err !== gl.NO_ERROR) {
    throw new Error(`WebGL2 error 0x${err.toString(16)} during ${where} (likely out of GPU memory).`);
  }
}

type ProgName =
  | 'boxBlurR'
  | 'boxBlurRgb'
  | 'blendR'
  | 'bilateralR'
  | 'squareR'
  | 'varianceCombine'
  | 'localContrast'
  | 'toneCurve'
  | 'scene'
  | 'colorMap'
  | 'sharpenCombine'
  | 'precision';

const PROGRAM_SOURCES: Record<ProgName, readonly [string, string]> = {
  boxBlurR: [FULL_PIPELINE_VERTEX_SRC, BOX_BLUR_R_FRAG_SRC],
  boxBlurRgb: [FULL_PIPELINE_VERTEX_SRC, BOX_BLUR_RGB_FRAG_SRC],
  blendR: [FULL_PIPELINE_VERTEX_SRC, BLEND_R_FRAG_SRC],
  bilateralR: [FULL_PIPELINE_VERTEX_SRC, BILATERAL_R_FRAG_SRC],
  squareR: [FULL_PIPELINE_VERTEX_SRC, SQUARE_R_FRAG_SRC],
  varianceCombine: [FULL_PIPELINE_VERTEX_SRC, VARIANCE_COMBINE_FRAG_SRC],
  localContrast: [FULL_PIPELINE_VERTEX_SRC, LOCAL_CONTRAST_FRAG_SRC],
  toneCurve: [FULL_PIPELINE_VERTEX_SRC, TONE_CURVE_FRAG_SRC],
  scene: [FULL_PIPELINE_VERTEX_SRC, SCENE_TILED_FRAG_SRC],
  colorMap: [FULL_PIPELINE_VERTEX_SRC, COLOR_MAP_FRAG_SRC],
  sharpenCombine: [FULL_PIPELINE_VERTEX_SRC, SHARPEN_COMBINE_FRAG_SRC],
  precision: [PRECISION_PIPELINE_VERTEX_SRC, PRECISION_PIPELINE_FRAGMENT_SRC],
};

interface Env {
  gl: WebGL2RenderingContext;
  vao: WebGLVertexArrayObject;
  prog: (name: ProgName) => WebGLProgram;
}

function loc(env: Env, program: WebGLProgram, name: string): WebGLUniformLocation | null {
  return env.gl.getUniformLocation(program, name);
}

function draw(env: Env, program: WebGLProgram, fbo: WebGLFramebuffer, width: number, height: number, setUniforms: () => void): void {
  const { gl, vao } = env;
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.viewport(0, 0, width, height);
  gl.useProgram(program);
  gl.bindVertexArray(vao);
  setUniforms();
  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.BLEND);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

function bind(env: Env, unit: number, tex: WebGLTexture, program: WebGLProgram, uniformName: string): void {
  const { gl } = env;
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.uniform1i(loc(env, program, uniformName), unit);
}

/** Separable box blur (H then V) on an R32F surface; returns a NEW surface, leaves `src` untouched. */
function boxBlurR(env: Env, src: Surface, w: number, h: number, radius: number): Surface {
  const { gl } = env;
  const p = env.prog('boxBlurR');
  const tmp = createSurface(gl, w, h, 1);
  const dst = createSurface(gl, w, h, 1);
  const pass = (input: WebGLTexture, target: Surface, dx: number, dy: number) =>
    draw(env, p, target.fbo, w, h, () => {
      bind(env, 0, input, p, 'u_src');
      gl.uniform2f(loc(env, p, 'u_texel'), 1 / w, 1 / h);
      gl.uniform2f(loc(env, p, 'u_dir'), dx, dy);
      gl.uniform1i(loc(env, p, 'u_radius'), radius);
    });
  pass(src.tex, tmp, 1, 0);
  pass(tmp.tex, dst, 0, 1);
  disposeSurface(gl, tmp);
  return dst;
}

function blend(env: Env, a: Surface, b: Surface, w: number, h: number, t: number): Surface {
  const { gl } = env;
  const p = env.prog('blendR');
  const out = createSurface(gl, w, h, 1);
  draw(env, p, out.fbo, w, h, () => {
    bind(env, 0, a.tex, p, 'u_a');
    bind(env, 1, b.tex, p, 'u_b');
    gl.uniform1f(loc(env, p, 'u_t'), t);
  });
  return out;
}

// ---------------------------------------------------------------------
// Settings -> shader parameters, derived ONCE per run (identical
// expressions, in identical operation order, to the untiled path).
// ---------------------------------------------------------------------

interface Derived {
  radii: ReturnType<typeof resolveStageRadii>;
  noiseAmt: number;
  bilateral: { twoSpatialSigma2: number; twoRangeSigma2: number };
  lcAmt: number;
  tone: {
    exposureMul: number;
    brightnessAdd: number;
    gamma: number;
    shadowLift: number;
    highlightRecovery: number;
    sCurveK: number;
  };
  detailAmt: number; // 0 => detail preservation off
  stopT: Float32Array;
  stopColor: Float32Array;
  stopCount: number;
  colorStrength: number;
  satAdjust: number;
  hueBiasFrac: number;
  tempAmt: number;
  sharpen: { amount: number; threshold: number };
}

function derive(settings: ProcessingSettings, colorStops: ColorStop[], quality: FullPipelineQualityParams): Derived {
  const radii = resolveStageRadii(settings, quality);

  const noiseAmt = settings.noiseReduction / 100;
  const rangeSigma = 0.05 + noiseAmt * 0.2;
  const spatialSigma = radii.bilateralRadius / 2 + 0.5;

  const contrastAmt = Math.min(1, Math.max(0, (settings.contrast + 100) / 200));
  const sCurveS = (contrastAmt - 0.5) * 2;

  const stopT = new Float32Array(MAX_STOPS);
  const stopColor = new Float32Array(MAX_STOPS * 3);
  for (let i = 0; i < colorStops.length; i++) {
    stopT[i] = colorStops[i].t;
    stopColor[i * 3] = colorStops[i].rgb[0];
    stopColor[i * 3 + 1] = colorStops[i].rgb[1];
    stopColor[i * 3 + 2] = colorStops[i].rgb[2];
  }

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
    stopT,
    stopColor,
    stopCount: colorStops.length,
    colorStrength: Math.min(1, Math.max(0, settings.colorStrength / 100)),
    satAdjust: settings.saturation / 100,
    hueBiasFrac: (((settings.hueBias / 360) % 1) + 1) % 1,
    tempAmt: (settings.temperature / 100) * 0.12,
    sharpen: {
      amount: Math.min(1, Math.max(0, settings.sharpenAmount / 200)) * 2,
      threshold: settings.sharpenThreshold / 255,
    },
  };
}

/** Mirrors colorSpace.ts linearToSrgb() -- same local copy the untiled path keeps. */
function linearToSrgbHost(c: number): number {
  const v = c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(Math.max(c, 0), 1 / 2.4) - 0.055;
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

// ---------------------------------------------------------------------
// Phase A: one tile, intensity -> linear RGB (core written to the planes)
// ---------------------------------------------------------------------

interface Planes {
  r: Float32Array;
  g: Float32Array;
  b: Float32Array;
}

function runColorMapTile(
  env: Env,
  d: Derived,
  settings: ProcessingSettings,
  tile: Tile,
  intensity: Float32Array,
  fullW: number,
  fullH: number,
  dummy: Surface,
  planes: Planes
): void {
  const { gl } = env;
  const { region, core } = tile;
  const w = region.x1 - region.x0;
  const h = region.y1 - region.y0;

  // Extract this tile's region (core + halo) of the input.
  const sub = new Float32Array(w * h);
  for (let j = 0; j < h; j++) {
    const rowStart = (region.y0 + j) * fullW + region.x0;
    sub.set(intensity.subarray(rowStart, rowStart + w), j * w);
  }

  let current = createSurface(gl, w, h, 1);
  uploadInto(gl, current, w, h, 1, sub);

  // ---- 1. Noise reduction. ----
  if (settings.noiseReduction > 0) {
    if (settings.noiseMethod === 'gaussian') {
      let blurred = current;
      for (let i = 0; i < d.radii.gaussianPasses; i++) {
        const next = boxBlurR(env, blurred, w, h, d.radii.gaussianRadius);
        if (blurred !== current) disposeSurface(gl, blurred);
        blurred = next;
      }
      const blended = blend(env, current, blurred, w, h, d.noiseAmt);
      disposeSurface(gl, current);
      disposeSurface(gl, blurred);
      current = blended;
    } else if (settings.noiseMethod === 'bilateral') {
      const p = env.prog('bilateralR');
      const filtered = createSurface(gl, w, h, 1);
      draw(env, p, filtered.fbo, w, h, () => {
        bind(env, 0, current.tex, p, 'u_src');
        gl.uniform2f(loc(env, p, 'u_texel'), 1 / w, 1 / h);
        gl.uniform1i(loc(env, p, 'u_radius'), d.radii.bilateralRadius);
        gl.uniform1f(loc(env, p, 'u_twoSpatialSigma2'), d.bilateral.twoSpatialSigma2);
        gl.uniform1f(loc(env, p, 'u_twoRangeSigma2'), d.bilateral.twoRangeSigma2);
      });
      const blended = blend(env, current, filtered, w, h, d.noiseAmt);
      disposeSurface(gl, current);
      disposeSurface(gl, filtered);
      current = blended;
    }
  }

  // ---- 2. Local contrast. ----
  if (settings.localContrast > 0) {
    const p = env.prog('localContrast');
    const localMean = boxBlurR(env, current, w, h, d.radii.localContrastRadius);
    const out = createSurface(gl, w, h, 1);
    draw(env, p, out.fbo, w, h, () => {
      bind(env, 0, current.tex, p, 'u_src');
      bind(env, 1, localMean.tex, p, 'u_localMean');
      gl.uniform1f(loc(env, p, 'u_amt'), d.lcAmt);
    });
    disposeSurface(gl, current);
    disposeSurface(gl, localMean);
    current = out;
  }

  const preTone = current;

  // ---- 3. Tone curve (pointwise). ----
  const toneP = env.prog('toneCurve');
  const toneMapped = createSurface(gl, w, h, 1);
  draw(env, toneP, toneMapped.fbo, w, h, () => {
    bind(env, 0, preTone.tex, toneP, 'u_src');
    gl.uniform1f(loc(env, toneP, 'u_exposureMul'), d.tone.exposureMul);
    gl.uniform1f(loc(env, toneP, 'u_brightnessAdd'), d.tone.brightnessAdd);
    gl.uniform1f(loc(env, toneP, 'u_gamma'), d.tone.gamma);
    gl.uniform1f(loc(env, toneP, 'u_shadowLift'), d.tone.shadowLift);
    gl.uniform1f(loc(env, toneP, 'u_highlightRecovery'), d.tone.highlightRecovery);
    gl.uniform1f(loc(env, toneP, 'u_sCurveK'), d.tone.sCurveK);
  });

  // ---- 4. Detail preservation (pointwise). ----
  let working = toneMapped;
  if (settings.detailPreservation > 0) {
    working = blend(env, toneMapped, preTone, w, h, d.detailAmt);
    disposeSurface(gl, toneMapped);
  }
  disposeSurface(gl, preTone);

  // ---- 5. Scene heuristics (optional). ----
  let skyTex: WebGLTexture = dummy.tex;
  let vegTex: WebGLTexture = dummy.tex;
  let dual: DualSurface | null = null;
  if (settings.sceneHeuristics) {
    const sqP = env.prog('squareR');
    const sq = createSurface(gl, w, h, 1);
    draw(env, sqP, sq.fbo, w, h, () => {
      bind(env, 0, working.tex, sqP, 'u_src');
    });
    const mean = boxBlurR(env, working, w, h, SCENE_VARIANCE_RADIUS);
    const meanSq = boxBlurR(env, sq, w, h, SCENE_VARIANCE_RADIUS);
    const varP = env.prog('varianceCombine');
    const variance = createSurface(gl, w, h, 1);
    draw(env, varP, variance.fbo, w, h, () => {
      bind(env, 0, mean.tex, varP, 'u_mean');
      bind(env, 1, meanSq.tex, varP, 'u_meanSq');
    });
    dual = createDualSurface(gl, w, h);
    const sceneP = env.prog('scene');
    draw(env, sceneP, dual.fbo, w, h, () => {
      bind(env, 0, working.tex, sceneP, 'u_intensity');
      bind(env, 1, variance.tex, sceneP, 'u_variance');
      // The ONLY position-dependent input in the pipeline: which rows of
      // the WHOLE image this tile's region covers.
      gl.uniform1f(loc(env, sceneP, 'u_rowOrigin'), region.y0);
      gl.uniform1f(loc(env, sceneP, 'u_tileHeight'), h);
      gl.uniform1f(loc(env, sceneP, 'u_fullHeight'), fullH);
    });
    disposeSurface(gl, sq);
    disposeSurface(gl, mean);
    disposeSurface(gl, meanSq);
    disposeSurface(gl, variance);
    skyTex = dual.texA;
    vegTex = dual.texB;
  }

  // ---- 6. IR->RGB color mapping (outputs LINEAR RGB). ----
  const cmP = env.prog('colorMap');
  const colorMapped = createSurface(gl, w, h, 4);
  draw(env, cmP, colorMapped.fbo, w, h, () => {
    bind(env, 0, working.tex, cmP, 'u_intensity');
    bind(env, 1, skyTex, cmP, 'u_sky');
    bind(env, 2, vegTex, cmP, 'u_veg');
    gl.uniform1i(loc(env, cmP, 'u_sceneEnabled'), settings.sceneHeuristics ? 1 : 0);
    gl.uniform1fv(loc(env, cmP, 'u_stopT'), d.stopT);
    gl.uniform3fv(loc(env, cmP, 'u_stopColor'), d.stopColor);
    gl.uniform1i(loc(env, cmP, 'u_stopCount'), d.stopCount);
    gl.uniform1f(loc(env, cmP, 'u_colorStrength'), d.colorStrength);
    gl.uniform1f(loc(env, cmP, 'u_satAdjust'), d.satAdjust);
    gl.uniform1f(loc(env, cmP, 'u_hueBiasFrac'), d.hueBiasFrac);
    gl.uniform1f(loc(env, cmP, 'u_tempAmt'), d.tempAmt);
  });
  disposeSurface(gl, working);
  if (dual) disposeDual(gl, dual);

  // ---- 7. Read back ONLY this tile's core (halo pixels are discarded). ----
  const cw = core.x1 - core.x0;
  const ch = core.y1 - core.y0;
  const buf = new Float32Array(cw * ch * 4);
  gl.bindFramebuffer(gl.FRAMEBUFFER, colorMapped.fbo);
  gl.readPixels(core.x0 - region.x0, core.y0 - region.y0, cw, ch, gl.RGBA, gl.FLOAT, buf);
  assertHealthy(gl, `color-map tile (${core.x0},${core.y0})`);
  disposeSurface(gl, colorMapped);

  for (let j = 0; j < ch; j++) {
    let dst = (core.y0 + j) * fullW + core.x0;
    let src = j * cw * 4;
    for (let i = 0; i < cw; i++, dst++, src += 4) {
      planes.r[dst] = buf[src];
      planes.g[dst] = buf[src + 1];
      planes.b[dst] = buf[src + 2];
    }
  }
}

// ---------------------------------------------------------------------
// Phase C: precision shader, per tile, GLOBAL parameters
// ---------------------------------------------------------------------

interface PrecisionParams {
  result: PrecisionPipelineGpuResult;
  wbGain: [number, number, number];
  atBlackPoint: number;
  atRange: number;
  atEnabled: boolean;
  ccGain: [number, number, number];
  ccStrengthFrac: number;
  ccEnabled: boolean;
  gamutEnabled: boolean;
}

/**
 * Estimates the precision stage's scalar parameters from the COMPLETE
 * image, with the same estimators and the same option objects
 * runPrecisionPipelineWebGL2 uses -- so tiled and untiled runs agree on
 * every gain to the last bit.
 */
function estimatePrecisionParams(settings: ProcessingSettings, planes: Planes): PrecisionParams {
  const wbOpts = { strength: settings.whiteBalanceStrength };
  const atOpts = { strength: settings.autoToneStrength };
  const ccOpts = { strength: settings.colorCorrectionStrength };

  const wb = computeWhiteBalanceGains(planes.r, planes.g, planes.b, wbOpts);
  const at = computeAutoToneParams(planes.r, planes.g, planes.b, atOpts);
  const cc = computeChannelBalanceGains(planes.r, planes.g, planes.b, ccOpts);

  return {
    result: {
      whiteBalance: wb,
      autoTone: { blackPoint: at.blackPoint, whitePoint: at.whitePoint, blackClipPercent: 0, whiteClipPercent: 0 },
      colorCorrection: cc,
      diagnosticsApproximate: true,
    },
    wbGain: [wb.gainR, wb.gainG, wb.gainB],
    atBlackPoint: at.blackPoint,
    atRange: Math.max(at.whitePoint - at.blackPoint, 1e-4),
    atEnabled: !at.degenerate && atOpts.strength > 0,
    ccGain: [cc.gainR, cc.gainG, cc.gainB],
    ccStrengthFrac: Math.max(0, Math.min(1, ccOpts.strength / 100)),
    ccEnabled: ccOpts.strength > 0,
    gamutEnabled: true,
  };
}

function runPrecisionTile(env: Env, pp: PrecisionParams, rect: Tile['core'], fullW: number, planes: Planes): void {
  const { gl } = env;
  const w = rect.x1 - rect.x0;
  const h = rect.y1 - rect.y0;

  const src = new Float32Array(w * h * 4);
  for (let j = 0; j < h; j++) {
    let s = (rect.y0 + j) * fullW + rect.x0;
    let d = j * w * 4;
    for (let i = 0; i < w; i++, s++, d += 4) {
      src[d] = planes.r[s];
      src[d + 1] = planes.g[s];
      src[d + 2] = planes.b[s];
      src[d + 3] = 1;
    }
  }

  const input = createSurface(gl, w, h, 4);
  uploadInto(gl, input, w, h, 4, src);
  const output = createSurface(gl, w, h, 4);

  const p = env.prog('precision');
  draw(env, p, output.fbo, w, h, () => {
    bind(env, 0, input.tex, p, 'u_src');
    gl.uniform3f(loc(env, p, 'u_wbGain'), pp.wbGain[0], pp.wbGain[1], pp.wbGain[2]);
    gl.uniform1f(loc(env, p, 'u_atBlackPoint'), pp.atBlackPoint);
    gl.uniform1f(loc(env, p, 'u_atRange'), pp.atRange);
    gl.uniform1i(loc(env, p, 'u_atEnabled'), pp.atEnabled ? 1 : 0);
    gl.uniform3f(loc(env, p, 'u_ccGain'), pp.ccGain[0], pp.ccGain[1], pp.ccGain[2]);
    gl.uniform1f(loc(env, p, 'u_ccStrengthFrac'), pp.ccStrengthFrac);
    gl.uniform1i(loc(env, p, 'u_ccEnabled'), pp.ccEnabled ? 1 : 0);
    gl.uniform1i(loc(env, p, 'u_gamutEnabled'), pp.gamutEnabled ? 1 : 0);
  });

  const out = new Float32Array(w * h * 4);
  gl.bindFramebuffer(gl.FRAMEBUFFER, output.fbo);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, out);
  assertHealthy(gl, `precision tile (${rect.x0},${rect.y0})`);
  disposeSurface(gl, input);
  disposeSurface(gl, output);

  for (let j = 0; j < h; j++) {
    let dst = (rect.y0 + j) * fullW + rect.x0;
    let s = j * w * 4;
    for (let i = 0; i < w; i++, dst++, s += 4) {
      planes.r[dst] = out[s];
      planes.g[dst] = out[s + 1];
      planes.b[dst] = out[s + 2];
    }
  }
}

// ---------------------------------------------------------------------
// Phase E: sharpening, per tile with halo, reading an IMMUTABLE source
// ---------------------------------------------------------------------

function runSharpenTile(env: Env, d: Derived, tile: Tile, fullW: number, srcRgba: Uint8ClampedArray, outRgba: Uint8ClampedArray): void {
  const { gl } = env;
  const { region, core } = tile;
  const w = region.x1 - region.x0;
  const h = region.y1 - region.y0;

  // Same sRGB float values the untiled path re-uploads (8-bit / 255, alpha 1).
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

  const src = createSurface(gl, w, h, 4);
  uploadInto(gl, src, w, h, 4, srgb);

  const blurP = env.prog('boxBlurRgb');
  const blurTmp = createSurface(gl, w, h, 4);
  const blurred = createSurface(gl, w, h, 4);
  const pass = (input: WebGLTexture, target: Surface, dx: number, dy: number) =>
    draw(env, blurP, target.fbo, w, h, () => {
      bind(env, 0, input, blurP, 'u_src');
      gl.uniform2f(loc(env, blurP, 'u_texel'), 1 / w, 1 / h);
      gl.uniform2f(loc(env, blurP, 'u_dir'), dx, dy);
      gl.uniform1i(loc(env, blurP, 'u_radius'), d.radii.sharpenRadius);
    });
  pass(src.tex, blurTmp, 1, 0);
  pass(blurTmp.tex, blurred, 0, 1);

  const combP = env.prog('sharpenCombine');
  const sharpened = createSurface(gl, w, h, 4);
  draw(env, combP, sharpened.fbo, w, h, () => {
    bind(env, 0, src.tex, combP, 'u_src');
    bind(env, 1, blurred.tex, combP, 'u_blurred');
    gl.uniform1f(loc(env, combP, 'u_amount'), d.sharpen.amount);
    gl.uniform1f(loc(env, combP, 'u_threshold'), d.sharpen.threshold);
  });

  const cw = core.x1 - core.x0;
  const ch = core.y1 - core.y0;
  const buf = new Float32Array(cw * ch * 4);
  gl.bindFramebuffer(gl.FRAMEBUFFER, sharpened.fbo);
  gl.readPixels(core.x0 - region.x0, core.y0 - region.y0, cw, ch, gl.RGBA, gl.FLOAT, buf);
  assertHealthy(gl, `sharpen tile (${core.x0},${core.y0})`);
  disposeSurface(gl, src);
  disposeSurface(gl, blurTmp);
  disposeSurface(gl, blurred);
  disposeSurface(gl, sharpened);

  for (let j = 0; j < ch; j++) {
    let dst = ((core.y0 + j) * fullW + core.x0) * 4;
    let s = j * cw * 4;
    for (let i = 0; i < cw; i++, dst += 4, s += 4) {
      outRgba[dst] = buf[s] * 255;
      outRgba[dst + 1] = buf[s + 1] * 255;
      outRgba[dst + 2] = buf[s + 2] * 255;
      outRgba[dst + 3] = 255;
    }
  }
}

// ---------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------

/**
 * Tiled equivalent of runFullPipelineWebGL2 -- same inputs, same result
 * shape (plus `tiling` diagnostics). Throws on anything it cannot do
 * correctly; callers treat every throw as "fall back to CPU".
 */
export function runTiledFullPipelineWebGL2(
  intensity: Float32Array,
  width: number,
  height: number,
  settings: ProcessingSettings,
  colorStops: ColorStop[],
  quality: FullPipelineQualityParams,
  options: TiledPipelineOptions = {}
): FullPipelineGpuResult & { tiling: TilingDiagnostics } {
  const n = width * height;
  if (intensity.length !== n) throw new Error('runTiledFullPipelineWebGL2: buffer/size mismatch.');
  if (settings.noiseReduction > 0 && settings.noiseMethod === 'median') {
    throw new Error("GPU 'median' noise reduction is not implemented yet -- falling back to CPU.");
  }
  if (colorStops.length > MAX_STOPS) {
    throw new Error(`Preset has ${colorStops.length} color stops, exceeding this shader's MAX_STOPS (${MAX_STOPS}).`);
  }

  const gl = createTileGL();
  let vao: WebGLVertexArrayObject | null = null;
  const programs = new Map<ProgName, WebGLProgram>();
  let dummy: Surface | null = null;

  try {
    const maxTexSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    const maxTileDim = Math.max(1, Math.min(options.maxTileDim ?? DEFAULT_MAX_TILE_DIM, maxTexSize));

    const d = derive(settings, colorStops, quality);
    const colorMapHalo = options.unsafeHaloOverride?.colorMap ?? d.radii.colorMapReach;
    const sharpenHalo = options.unsafeHaloOverride?.sharpen ?? d.radii.sharpenReach;

    // Plan everything up front so an untileable request fails BEFORE any GPU work.
    const planA: TilePlan = planTiles(width, height, maxTileDim, colorMapHalo);
    const planC: TilePlan | null = settings.precisionPipeline ? planTiles(width, height, maxTileDim, 0) : null;
    const planE: TilePlan | null = settings.sharpenAmount > 0 ? planTiles(width, height, maxTileDim, sharpenHalo) : null;

    // Drain any error raised before our first call, so assertHealthy() only reports ours.
    while (gl.getError() !== gl.NO_ERROR) {
      /* drain */
    }

    vao = gl.createVertexArray();
    if (!vao) throw new Error('gl.createVertexArray failed.');
    gl.bindVertexArray(vao);

    const env: Env = {
      gl,
      vao,
      prog: (name) => {
        let p = programs.get(name);
        if (!p) {
          p = linkProgram(gl, PROGRAM_SOURCES[name][0], PROGRAM_SOURCES[name][1]);
          programs.set(name, p);
        }
        return p;
      },
    };

    dummy = createSurface(gl, 1, 1, 1); // stands in for the sky/veg textures when scene heuristics are off

    // ---- Phase A: every tile -> full-image linear RGB planes. ----
    let planes: Planes | null = { r: new Float32Array(n), g: new Float32Array(n), b: new Float32Array(n) };
    for (const tile of planA.tiles) {
      runColorMapTile(env, d, settings, tile, intensity, width, height, dummy, planes);
    }

    // ---- Sync point + Phase C: whole-image statistics, then per-tile apply. ----
    let precision: PrecisionPipelineGpuResult | undefined;
    if (planC) {
      const pp = estimatePrecisionParams(settings, planes);
      precision = pp.result;
      for (const tile of planC.tiles) runPrecisionTile(env, pp, tile.core, width, planes);
    }

    // ---- Linear -> 8-bit sRGB (identical expression to the untiled path). ----
    const rgba = new Uint8ClampedArray(n * 4);
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      rgba[p] = linearToSrgbHost(planes.r[i]) * 255;
      rgba[p + 1] = linearToSrgbHost(planes.g[i]) * 255;
      rgba[p + 2] = linearToSrgbHost(planes.b[i]) * 255;
      rgba[p + 3] = 255;
    }
    planes = null; // release three full-image Float32 planes before the sharpening phase

    // ---- Phase E: sharpening reads `rgba` (immutable) and writes `out`. ----
    let result = rgba;
    if (planE) {
      const out = new Uint8ClampedArray(n * 4);
      for (const tile of planE.tiles) runSharpenTile(env, d, tile, width, rgba, out);
      result = out;
    }

    return {
      rgba: result,
      precision,
      tiling: {
        maxTileDim,
        colorMap: { cols: planA.cols, rows: planA.rows, tileCount: planA.tiles.length, halo: planA.halo },
        precision: planC ? { cols: planC.cols, rows: planC.rows, tileCount: planC.tiles.length } : null,
        sharpen: planE ? { cols: planE.cols, rows: planE.rows, tileCount: planE.tiles.length, halo: planE.halo } : null,
        wasTiled: computeWasTiled(
          { tileCount: planA.tiles.length },
          planC ? { tileCount: planC.tiles.length } : null,
          planE ? { tileCount: planE.tiles.length } : null
        ),
      },
    };
  } finally {
    try {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.bindTexture(gl.TEXTURE_2D, null);
      gl.bindVertexArray(null);
      if (vao) gl.deleteVertexArray(vao);
      if (dummy) disposeSurface(gl, dummy);
      for (const p of programs.values()) gl.deleteProgram(p);
    } finally {
      releaseContext(gl);
    }
  }
}
