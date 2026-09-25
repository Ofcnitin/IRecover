/**
 * WebGL2 "full pipeline" orchestrator.
 * -------------------------------------
 * Chains the passes in fullPipelineShaders.ts into the same stage order
 * as processing/pipeline.ts's CPU steps 4-10 (noise reduction -> local
 * contrast -> tone curve -> detail preservation -> scene heuristics ->
 * IR->RGB color mapping -> precision pipeline -> sharpening), operating
 * on GPU textures throughout each spatial stage rather than reading back
 * to CPU between every pass. Reuses the existing, already-verified
 * runPrecisionPipelineWebGL2() (webgl2Backend.ts) unmodified for the
 * precision sub-stage, via one readback/re-upload handoff -- see "Known
 * transfer points" below for the honest, non-hidden accounting of every
 * CPU<->GPU crossing this orchestrator makes.
 *
 * Images whose width or height exceeds gl.MAX_TEXTURE_SIZE are handed
 * to the tiled implementation (webgl2TiledPipeline.ts), which runs this
 * same pipeline over overlapping tiles and produces the same result;
 * this module's own code path below is unchanged and only ever sees
 * images that fit in one texture.
 *
 * Throws on any unsupported-environment/GPU/compile/link error, or when
 * a requested option has no GPU implementation yet (currently: the
 * 'median' noise-reduction method), or when tiling itself genuinely
 * cannot be done (TilingUnsupportedError -- e.g. a filter overlap so
 * large no useful tile fits). Callers (processing/pipeline.ts) must
 * treat any throw as "fall back to the CPU pipeline", never as fatal.
 *
 * Known transfer points (by design, not oversight):
 *   1. intensity (Float32Array) uploaded once at the start.
 *   2. color-mapped linear RGB read back once, to feed the existing
 *      runPrecisionPipelineWebGL2() (which does its own independent
 *      upload/readback -- see its module header) and to let the
 *      precision-pipeline's scalar stats (white balance / AutoTone /
 *      color-correction gains) be estimated cheaply on CPU, exactly as
 *      that module already does for the precision-pipeline-only path.
 *   3. sRGB rgba re-uploaded once for the sharpening pass (sharpening
 *      must run in gamma-encoded space to match the CPU reference
 *      pipeline's own behavior -- see sharpening's comment below).
 *   4. final sharpened result read back once.
 * Eliminating these remaining handoffs (a single fully-fused upload ->
 * GPU-only -> single readback chain across every stage, including
 * precision) is real further optimization work, honestly left for a
 * follow-up rather than rushed here.
 */

import type { ProcessingSettings, ColorStop } from '../../types/processing';
import { compileShader, linkProgram, runPrecisionPipelineWebGL2 } from './webgl2Backend';
import type { PrecisionPipelineGpuResult } from './webgl2Backend';
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
  SCENE_FRAG_SRC,
  COLOR_MAP_FRAG_SRC,
  SHARPEN_COMBINE_FRAG_SRC,
  MAX_STOPS,
} from './fullPipelineShaders';
import { MAX_SAFE_RADIUS } from './tilePlanner';
import type { TilingDiagnostics } from './tilePlanner';
import { runTiledFullPipelineWebGL2, probeMaxTextureSize } from './webgl2TiledPipeline';

/** WebGL2 (ES 3.0) guarantees MAX_TEXTURE_SIZE >= 2048, so images up to this size never need the probe below. */
const GUARANTEED_MIN_MAX_TEXTURE_SIZE = 2048;

export interface FullPipelineQualityParams {
  gaussianPasses: number;
  localContrastRadius: number;
}

export interface FullPipelineGpuResult {
  rgba: Uint8ClampedArray; // width*height*4, 8-bit sRGB, straight alpha=255
  precision?: PrecisionPipelineGpuResult;
  /** Present only when the image was processed by the tiled path (see webgl2TiledPipeline.ts). */
  tiling?: TilingDiagnostics;
}

/** Every uniform this module's programs need, resolved once from settings. */
interface Programs {
  boxBlurR: WebGLProgram;
  boxBlurRgb: WebGLProgram;
  blendR: WebGLProgram;
  bilateralR: WebGLProgram;
  squareR: WebGLProgram;
  varianceCombine: WebGLProgram;
  localContrast: WebGLProgram;
  toneCurve: WebGLProgram;
  scene: WebGLProgram;
  colorMap: WebGLProgram;
  sharpenCombine: WebGLProgram;
}

function createGL(width: number, height: number): WebGL2RenderingContext {
  const g = globalThis as unknown as { OffscreenCanvas?: unknown; document?: Document };
  let gl: WebGL2RenderingContext | null = null;
  if (typeof g.OffscreenCanvas !== 'undefined') {
    const canvas = new (g.OffscreenCanvas as new (w: number, h: number) => OffscreenCanvas)(width, height);
    gl = canvas.getContext('webgl2', { antialias: false, alpha: true }) as WebGL2RenderingContext | null;
  } else if (g.document) {
    const canvas = g.document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
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

interface Surface {
  tex: WebGLTexture;
  fbo: WebGLFramebuffer;
}

function createSurface(gl: WebGL2RenderingContext, width: number, height: number, channels: 1 | 4): Surface {
  const tex = gl.createTexture();
  if (!tex) throw new Error('gl.createTexture failed.');
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
    throw new Error('WebGL2 framebuffer incomplete.');
  }
  gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
  return { tex, fbo };
}

/** Two-output (MRT) surface, used only by the scene-heuristics pass. */
function createDualSurface(gl: WebGL2RenderingContext, width: number, height: number): { fbo: WebGLFramebuffer; texA: WebGLTexture; texB: WebGLTexture } {
  const texA = gl.createTexture()!;
  const texB = gl.createTexture()!;
  for (const t of [texA, texB]) {
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, null);
  }
  const fbo = gl.createFramebuffer()!;
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texA, 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, texB, 0);
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
    throw new Error('WebGL2 framebuffer incomplete (dual/MRT).');
  }
  gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
  return { fbo, texA, texB };
}

function uploadR32F(gl: WebGL2RenderingContext, tex: WebGLTexture, width: number, height: number, data: Float32Array): void {
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, data);
}

function bindAndDraw(gl: WebGL2RenderingContext, vao: WebGLVertexArrayObject, program: WebGLProgram, fbo: WebGLFramebuffer, width: number, height: number, setUniforms: () => void): void {
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.viewport(0, 0, width, height);
  gl.useProgram(program);
  gl.bindVertexArray(vao);
  setUniforms();
  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.BLEND);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

function bindTex(gl: WebGL2RenderingContext, unit: number, tex: WebGLTexture, program: WebGLProgram, uniformName: string): void {
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.uniform1i(gl.getUniformLocation(program, uniformName), unit);
}

/** Runs one separable box-blur (H then V) on a single-channel surface, returning a NEW surface (input is left untouched). */
function boxBlurR(gl: WebGL2RenderingContext, vao: WebGLVertexArrayObject, programs: Programs, src: Surface, width: number, height: number, radius: number): Surface {
  const tmp = createSurface(gl, width, height, 1);
  const dst = createSurface(gl, width, height, 1);
  const texel: [number, number] = [1 / width, 1 / height];
  bindAndDraw(gl, vao, programs.boxBlurR, tmp.fbo, width, height, () => {
    bindTex(gl, 0, src.tex, programs.boxBlurR, 'u_src');
    gl.uniform2f(gl.getUniformLocation(programs.boxBlurR, 'u_texel'), texel[0], texel[1]);
    gl.uniform2f(gl.getUniformLocation(programs.boxBlurR, 'u_dir'), 1, 0);
    gl.uniform1i(gl.getUniformLocation(programs.boxBlurR, 'u_radius'), radius);
  });
  bindAndDraw(gl, vao, programs.boxBlurR, dst.fbo, width, height, () => {
    bindTex(gl, 0, tmp.tex, programs.boxBlurR, 'u_src');
    gl.uniform2f(gl.getUniformLocation(programs.boxBlurR, 'u_texel'), texel[0], texel[1]);
    gl.uniform2f(gl.getUniformLocation(programs.boxBlurR, 'u_dir'), 0, 1);
    gl.uniform1i(gl.getUniformLocation(programs.boxBlurR, 'u_radius'), radius);
  });
  gl.deleteTexture(tmp.tex);
  gl.deleteFramebuffer(tmp.fbo);
  return dst;
}

// MAX_SAFE_RADIUS (sanity cap; every quality preset stays well under it) is shared with the tiled path via ./tilePlanner.

export function assertFullPipelineWebGL2Supported(): void {
  const gl = createGL(2, 2);
  void gl;
}

/**
 * Runs the full CPU-equivalent pipeline (from post-levels intensity
 * through sharpening) on WebGL2. `intensity` is the same Float32Array
 * processing/pipeline.ts would otherwise feed into noise reduction (i.e.
 * after extractIntensity + applyLevels, both still CPU-side -- they're
 * cheap, histogram/I-O-bound steps, not the hot per-pixel path).
 */
export function runFullPipelineWebGL2(
  intensity: Float32Array,
  width: number,
  height: number,
  settings: ProcessingSettings,
  colorStops: ColorStop[],
  quality: FullPipelineQualityParams
): FullPipelineGpuResult {
  const n = width * height;
  if (intensity.length !== n) throw new Error('runFullPipelineWebGL2: buffer/size mismatch.');
  if (settings.noiseReduction > 0 && settings.noiseMethod === 'median') {
    throw new Error("GPU 'median' noise reduction is not implemented yet -- falling back to CPU.");
  }
  if (colorStops.length > MAX_STOPS) {
    throw new Error(`Preset has ${colorStops.length} color stops, exceeding this shader's MAX_STOPS (${MAX_STOPS}).`);
  }

  // Oversize images go to the tiled path. This must be decided BEFORE
  // createGL(width, height): a canvas as large as an oversize image can
  // fail context creation outright, which would turn a tileable image
  // into a needless CPU fallback. Small images skip the probe entirely.
  if (Math.max(width, height) > GUARANTEED_MIN_MAX_TEXTURE_SIZE) {
    const probedMax = probeMaxTextureSize();
    if (width > probedMax || height > probedMax) {
      return runTiledFullPipelineWebGL2(intensity, width, height, settings, colorStops, quality);
    }
  }

  const gl = createGL(width, height);
  const maxTexSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
  if (width > maxTexSize || height > maxTexSize) {
    // Defensive only: the probe above already routes every oversize image to the tiled path.
    throw new Error(`Image (${width}x${height}) exceeds this GPU's MAX_TEXTURE_SIZE (${maxTexSize}) but was not routed to the tiled path.`);
  }

  const programs: Programs = {
    boxBlurR: linkProgram(gl, FULL_PIPELINE_VERTEX_SRC, BOX_BLUR_R_FRAG_SRC),
    boxBlurRgb: linkProgram(gl, FULL_PIPELINE_VERTEX_SRC, BOX_BLUR_RGB_FRAG_SRC),
    blendR: linkProgram(gl, FULL_PIPELINE_VERTEX_SRC, BLEND_R_FRAG_SRC),
    bilateralR: linkProgram(gl, FULL_PIPELINE_VERTEX_SRC, BILATERAL_R_FRAG_SRC),
    squareR: linkProgram(gl, FULL_PIPELINE_VERTEX_SRC, SQUARE_R_FRAG_SRC),
    varianceCombine: linkProgram(gl, FULL_PIPELINE_VERTEX_SRC, VARIANCE_COMBINE_FRAG_SRC),
    localContrast: linkProgram(gl, FULL_PIPELINE_VERTEX_SRC, LOCAL_CONTRAST_FRAG_SRC),
    toneCurve: linkProgram(gl, FULL_PIPELINE_VERTEX_SRC, TONE_CURVE_FRAG_SRC),
    scene: linkProgram(gl, FULL_PIPELINE_VERTEX_SRC, SCENE_FRAG_SRC),
    colorMap: linkProgram(gl, FULL_PIPELINE_VERTEX_SRC, COLOR_MAP_FRAG_SRC),
    sharpenCombine: linkProgram(gl, FULL_PIPELINE_VERTEX_SRC, SHARPEN_COMBINE_FRAG_SRC),
  };

  const vao = gl.createVertexArray();
  if (!vao) throw new Error('gl.createVertexArray failed.');
  gl.bindVertexArray(vao);

  const allSurfaces: Surface[] = [];
  const track = (s: Surface): Surface => {
    allSurfaces.push(s);
    return s;
  };

  try {
    // ---- 0. Upload starting intensity. ----
    let current = track(createSurface(gl, width, height, 1));
    uploadR32F(gl, current.tex, width, height, intensity);

    // ---- 1. Noise reduction. ----
    if (settings.noiseReduction > 0) {
      const amt = settings.noiseReduction / 100;
      if (settings.noiseMethod === 'gaussian') {
        const radius = Math.min(MAX_SAFE_RADIUS, Math.max(1, Math.round(amt * 4)));
        const passes = Math.max(1, quality.gaussianPasses);
        let blurred = current;
        for (let i = 0; i < passes; i++) {
          const next = track(boxBlurR(gl, vao, programs, blurred, width, height, radius));
          blurred = next;
        }
        const blended = track(createSurface(gl, width, height, 1));
        bindAndDraw(gl, vao, programs.blendR, blended.fbo, width, height, () => {
          bindTex(gl, 0, current.tex, programs.blendR, 'u_a');
          bindTex(gl, 1, blurred.tex, programs.blendR, 'u_b');
          gl.uniform1f(gl.getUniformLocation(programs.blendR, 'u_t'), amt);
        });
        current = blended;
      } else if (settings.noiseMethod === 'bilateral') {
        const radius = Math.min(MAX_SAFE_RADIUS, Math.max(1, Math.round(amt * 3)));
        const rangeSigma = 0.05 + amt * 0.2;
        const spatialSigma = radius / 2 + 0.5;
        const twoSpatialSigma2 = 2 * spatialSigma * spatialSigma;
        const twoRangeSigma2 = 2 * rangeSigma * rangeSigma;
        const filtered = track(createSurface(gl, width, height, 1));
        bindAndDraw(gl, vao, programs.bilateralR, filtered.fbo, width, height, () => {
          bindTex(gl, 0, current.tex, programs.bilateralR, 'u_src');
          gl.uniform2f(gl.getUniformLocation(programs.bilateralR, 'u_texel'), 1 / width, 1 / height);
          gl.uniform1i(gl.getUniformLocation(programs.bilateralR, 'u_radius'), radius);
          gl.uniform1f(gl.getUniformLocation(programs.bilateralR, 'u_twoSpatialSigma2'), twoSpatialSigma2);
          gl.uniform1f(gl.getUniformLocation(programs.bilateralR, 'u_twoRangeSigma2'), twoRangeSigma2);
        });
        const blended = track(createSurface(gl, width, height, 1));
        bindAndDraw(gl, vao, programs.blendR, blended.fbo, width, height, () => {
          bindTex(gl, 0, current.tex, programs.blendR, 'u_a');
          bindTex(gl, 1, filtered.tex, programs.blendR, 'u_b');
          gl.uniform1f(gl.getUniformLocation(programs.blendR, 'u_t'), amt);
        });
        current = blended;
      }
    }

    // ---- 2. Local contrast. ----
    if (settings.localContrast > 0) {
      const radius = Math.min(MAX_SAFE_RADIUS, Math.max(1, Math.round(quality.localContrastRadius)));
      const localMean = track(boxBlurR(gl, vao, programs, current, width, height, radius));
      const out = track(createSurface(gl, width, height, 1));
      bindAndDraw(gl, vao, programs.localContrast, out.fbo, width, height, () => {
        bindTex(gl, 0, current.tex, programs.localContrast, 'u_src');
        bindTex(gl, 1, localMean.tex, programs.localContrast, 'u_localMean');
        gl.uniform1f(gl.getUniformLocation(programs.localContrast, 'u_amt'), settings.localContrast / 100);
      });
      current = out;
    }

    const preTone = current; // == "intensity" right before the tone curve, per pipeline.ts step 6/7

    // ---- 3. Tone curve. ----
    const exposureMul = Math.pow(2, settings.exposure);
    const brightnessAdd = settings.brightness / 255;
    const contrastAmt = Math.min(1, Math.max(0, (settings.contrast + 100) / 200));
    const gamma = Math.max(0.05, settings.gamma);
    const shadowLift = Math.min(1, Math.max(0, settings.shadowLift / 100));
    const highlightRecovery = Math.min(1, Math.max(0, settings.highlightRecovery / 100));
    const sCurveS = (contrastAmt - 0.5) * 2;
    const sCurveK = sCurveS > 0 ? 1 + sCurveS * 3 : 1 / (1 - sCurveS * 3);

    const toneMapped = track(createSurface(gl, width, height, 1));
    bindAndDraw(gl, vao, programs.toneCurve, toneMapped.fbo, width, height, () => {
      bindTex(gl, 0, preTone.tex, programs.toneCurve, 'u_src');
      gl.uniform1f(gl.getUniformLocation(programs.toneCurve, 'u_exposureMul'), exposureMul);
      gl.uniform1f(gl.getUniformLocation(programs.toneCurve, 'u_brightnessAdd'), brightnessAdd);
      gl.uniform1f(gl.getUniformLocation(programs.toneCurve, 'u_gamma'), gamma);
      gl.uniform1f(gl.getUniformLocation(programs.toneCurve, 'u_shadowLift'), shadowLift);
      gl.uniform1f(gl.getUniformLocation(programs.toneCurve, 'u_highlightRecovery'), highlightRecovery);
      gl.uniform1f(gl.getUniformLocation(programs.toneCurve, 'u_sCurveK'), sCurveK);
    });

    // ---- 4. Detail preservation. ----
    let working = toneMapped;
    if (settings.detailPreservation > 0) {
      const amt = (settings.detailPreservation / 100) * 0.3 * 0.15;
      const blended = track(createSurface(gl, width, height, 1));
      bindAndDraw(gl, vao, programs.blendR, blended.fbo, width, height, () => {
        bindTex(gl, 0, toneMapped.tex, programs.blendR, 'u_a');
        bindTex(gl, 1, preTone.tex, programs.blendR, 'u_b');
        gl.uniform1f(gl.getUniformLocation(programs.blendR, 'u_t'), amt);
      });
      working = blended;
    }

    // ---- 5. Scene heuristics (optional). ----
    let skyTex: WebGLTexture;
    let vegTex: WebGLTexture;
    let sceneEnabled = false;
    if (settings.sceneHeuristics) {
      const sq = track(createSurface(gl, width, height, 1));
      bindAndDraw(gl, vao, programs.squareR, sq.fbo, width, height, () => {
        bindTex(gl, 0, working.tex, programs.squareR, 'u_src');
      });
      const mean = track(boxBlurR(gl, vao, programs, working, width, height, 5));
      const meanSq = track(boxBlurR(gl, vao, programs, sq, width, height, 5));
      const variance = track(createSurface(gl, width, height, 1));
      bindAndDraw(gl, vao, programs.varianceCombine, variance.fbo, width, height, () => {
        bindTex(gl, 0, mean.tex, programs.varianceCombine, 'u_mean');
        bindTex(gl, 1, meanSq.tex, programs.varianceCombine, 'u_meanSq');
      });
      const dual = createDualSurface(gl, width, height);
      bindAndDraw(gl, vao, programs.scene, dual.fbo, width, height, () => {
        bindTex(gl, 0, working.tex, programs.scene, 'u_intensity');
        bindTex(gl, 1, variance.tex, programs.scene, 'u_variance');
      });
      skyTex = dual.texA;
      vegTex = dual.texB;
      sceneEnabled = true;
      allSurfaces.push({ tex: dual.texA, fbo: dual.fbo });
      allSurfaces.push({ tex: dual.texB, fbo: dual.fbo }); // fbo appears twice in the list but is only deleted once (see cleanup below)
    } else {
      const dummy = track(createSurface(gl, 1, 1, 1));
      skyTex = dummy.tex;
      vegTex = dummy.tex;
    }

    // ---- 6. IR->RGB color mapping (outputs LINEAR RGB). ----
    const stopT = new Float32Array(MAX_STOPS);
    const stopColor = new Float32Array(MAX_STOPS * 3);
    for (let i = 0; i < colorStops.length; i++) {
      stopT[i] = colorStops[i].t;
      stopColor[i * 3] = colorStops[i].rgb[0];
      stopColor[i * 3 + 1] = colorStops[i].rgb[1];
      stopColor[i * 3 + 2] = colorStops[i].rgb[2];
    }
    const colorMapped = track(createSurface(gl, width, height, 4));
    const colorStrength = Math.min(1, Math.max(0, settings.colorStrength / 100));
    const satAdjust = settings.saturation / 100;
    const hueBiasFrac = (((settings.hueBias / 360) % 1) + 1) % 1;
    const tempAmt = (settings.temperature / 100) * 0.12;
    bindAndDraw(gl, vao, programs.colorMap, colorMapped.fbo, width, height, () => {
      bindTex(gl, 0, working.tex, programs.colorMap, 'u_intensity');
      bindTex(gl, 1, skyTex, programs.colorMap, 'u_sky');
      bindTex(gl, 2, vegTex, programs.colorMap, 'u_veg');
      gl.uniform1i(gl.getUniformLocation(programs.colorMap, 'u_sceneEnabled'), sceneEnabled ? 1 : 0);
      gl.uniform1fv(gl.getUniformLocation(programs.colorMap, 'u_stopT'), stopT);
      gl.uniform3fv(gl.getUniformLocation(programs.colorMap, 'u_stopColor'), stopColor);
      gl.uniform1i(gl.getUniformLocation(programs.colorMap, 'u_stopCount'), colorStops.length);
      gl.uniform1f(gl.getUniformLocation(programs.colorMap, 'u_colorStrength'), colorStrength);
      gl.uniform1f(gl.getUniformLocation(programs.colorMap, 'u_satAdjust'), satAdjust);
      gl.uniform1f(gl.getUniformLocation(programs.colorMap, 'u_hueBiasFrac'), hueBiasFrac);
      gl.uniform1f(gl.getUniformLocation(programs.colorMap, 'u_tempAmt'), tempAmt);
    });

    // ---- 7. Read back linear RGB once, to feed the existing precision-pipeline module. ----
    const packed = new Float32Array(n * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, colorMapped.fbo);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.FLOAT, packed);

    const r = new Float32Array(n);
    const g = new Float32Array(n);
    const b = new Float32Array(n);
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      r[i] = packed[p];
      g[i] = packed[p + 1];
      b[i] = packed[p + 2];
    }

    let precisionResult: PrecisionPipelineGpuResult | undefined;
    if (settings.precisionPipeline) {
      precisionResult = runPrecisionPipelineWebGL2(r, g, b, width, height, {
        whiteBalance: { strength: settings.whiteBalanceStrength },
        autoTone: { strength: settings.autoToneStrength },
        colorCorrection: { strength: settings.colorCorrectionStrength },
      });
    }

    // ---- 8. Linear -> sRGB, pack into 8-bit output. ----
    const rgba = new Uint8ClampedArray(n * 4);
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      rgba[p] = linearToSrgbHost(r[i]) * 255;
      rgba[p + 1] = linearToSrgbHost(g[i]) * 255;
      rgba[p + 2] = linearToSrgbHost(b[i]) * 255;
      rgba[p + 3] = 255;
    }

    // ---- 9. Sharpening (re-uploads the sRGB result; see module header). ----
    if (settings.sharpenAmount > 0) {
      const srcTex = gl.createTexture();
      if (!srcTex) throw new Error('gl.createTexture failed (sharpen source).');
      gl.bindTexture(gl.TEXTURE_2D, srcTex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      const srgbPixels = new Float32Array(n * 4);
      for (let i = 0, p = 0; i < n; i++, p += 4) {
        srgbPixels[p] = rgba[p] / 255;
        srgbPixels[p + 1] = rgba[p + 1] / 255;
        srgbPixels[p + 2] = rgba[p + 2] / 255;
        srgbPixels[p + 3] = 1;
      }
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, srgbPixels);
      allSurfaces.push({ tex: srcTex, fbo: colorMapped.fbo }); // borrow an already-tracked fbo id purely so cleanup finds this texture; not bound as its framebuffer

      const radius = Math.max(0.5, settings.sharpenRadius);
      const rRounded = Math.round(radius);
      const blurTmp = track(createSurface(gl, width, height, 4));
      const blurred = track(createSurface(gl, width, height, 4));
      const texel: [number, number] = [1 / width, 1 / height];
      bindAndDraw(gl, vao, programs.boxBlurRgb, blurTmp.fbo, width, height, () => {
        bindTex(gl, 0, srcTex, programs.boxBlurRgb, 'u_src');
        gl.uniform2f(gl.getUniformLocation(programs.boxBlurRgb, 'u_texel'), texel[0], texel[1]);
        gl.uniform2f(gl.getUniformLocation(programs.boxBlurRgb, 'u_dir'), 1, 0);
        gl.uniform1i(gl.getUniformLocation(programs.boxBlurRgb, 'u_radius'), rRounded);
      });
      bindAndDraw(gl, vao, programs.boxBlurRgb, blurred.fbo, width, height, () => {
        bindTex(gl, 0, blurTmp.tex, programs.boxBlurRgb, 'u_src');
        gl.uniform2f(gl.getUniformLocation(programs.boxBlurRgb, 'u_texel'), texel[0], texel[1]);
        gl.uniform2f(gl.getUniformLocation(programs.boxBlurRgb, 'u_dir'), 0, 1);
        gl.uniform1i(gl.getUniformLocation(programs.boxBlurRgb, 'u_radius'), rRounded);
      });

      const amount = Math.min(1, Math.max(0, settings.sharpenAmount / 200)) * 2;
      const threshold = settings.sharpenThreshold / 255;
      const sharpened = track(createSurface(gl, width, height, 4));
      bindAndDraw(gl, vao, programs.sharpenCombine, sharpened.fbo, width, height, () => {
        bindTex(gl, 0, srcTex, programs.sharpenCombine, 'u_src');
        bindTex(gl, 1, blurred.tex, programs.sharpenCombine, 'u_blurred');
        gl.uniform1f(gl.getUniformLocation(programs.sharpenCombine, 'u_amount'), amount);
        gl.uniform1f(gl.getUniformLocation(programs.sharpenCombine, 'u_threshold'), threshold);
      });

      const finalPixels = new Float32Array(n * 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, sharpened.fbo);
      gl.readPixels(0, 0, width, height, gl.RGBA, gl.FLOAT, finalPixels);
      for (let i = 0, p = 0; i < n; i++, p += 4) {
        rgba[p] = finalPixels[p] * 255;
        rgba[p + 1] = finalPixels[p + 1] * 255;
        rgba[p + 2] = finalPixels[p + 2] * 255;
      }
    }

    return { rgba, precision: precisionResult };
  } finally {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindVertexArray(null);
    gl.deleteVertexArray(vao);
    const deletedFbos = new Set<WebGLFramebuffer>();
    const deletedTexs = new Set<WebGLTexture>();
    for (const s of allSurfaces) {
      if (!deletedTexs.has(s.tex)) {
        deletedTexs.add(s.tex);
        gl.deleteTexture(s.tex);
      }
      if (!deletedFbos.has(s.fbo)) {
        deletedFbos.add(s.fbo);
        gl.deleteFramebuffer(s.fbo);
      }
    }
    for (const key of Object.keys(programs) as (keyof Programs)[]) {
      gl.deleteProgram(programs[key]);
    }
  }
}

/** Mirrors colorSpace.ts linearToSrgb() exactly -- kept local to avoid an extra import cycle for one small pure function. */
function linearToSrgbHost(c: number): number {
  const v = c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(Math.max(c, 0), 1 / 2.4) - 0.055;
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

// Re-exported so callers can reference the shared compile helper alongside this module without a second import from webgl2Backend.ts.
export { compileShader };
