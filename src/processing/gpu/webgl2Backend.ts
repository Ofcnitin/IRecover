/**
 * WebGL2 precision-pipeline backend.
 * ----------------------------------
 * Runs the fused Azusa/AutoTone/ColorCorrectionPipeline/gamut-protection
 * shader (see shaders.ts) on an RGBA32F texture. This is real GPU
 * execution -- not a CPU computation relabeled as GPU: pixel values are
 * uploaded once, processed entirely by the fragment shader on the GPU,
 * and read back once.
 *
 * Scalar parameters (white-balance gains, AutoTone black/white points,
 * color-correction channel gains) are estimated on the CPU from the same
 * robust-statistics helpers the CPU pipeline uses (trimmed means,
 * percentile histograms -- see whiteBalance.ts / autoTone.ts /
 * colorCorrectionPipeline.ts). This mirrors what the CPU implementation
 * already does internally (those estimates are computed from bounded
 * subsamples, not full-resolution passes) and keeps GPU and CPU
 * estimating identical parameters; only the expensive per-pixel apply
 * work (which scales with full image resolution) moves to the GPU.
 *
 * Every failure mode here (missing WebGL2, missing EXT_color_buffer_float,
 * texture-size limits, context-creation failure, shader compile/link
 * failure) throws, so the caller (processing/pipeline.ts) can catch it
 * and fall back to the CPU implementation -- this module never silently
 * produces a wrong or partially-processed image.
 */

import { PRECISION_PIPELINE_VERTEX_SRC, PRECISION_PIPELINE_FRAGMENT_SRC } from './shaders';
import { computeWhiteBalanceGains, type WhiteBalanceOptions, type WhiteBalanceResult } from '../whiteBalance';
import { computeAutoToneParams, type AutoToneOptions } from '../autoTone';
import { computeChannelBalanceGains, type ColorCorrectionOptions, type ColorCorrectionResult } from '../colorCorrectionPipeline';

export interface PrecisionPipelineGpuOptions {
  whiteBalance: WhiteBalanceOptions;
  autoTone: AutoToneOptions;
  colorCorrection: ColorCorrectionOptions;
  gamutEnabled?: boolean; // default true
}

export interface PrecisionPipelineGpuResult {
  whiteBalance: WhiteBalanceResult;
  autoTone: { blackPoint: number; whitePoint: number; blackClipPercent: number; whiteClipPercent: number };
  colorCorrection: ColorCorrectionResult;
  /**
   * Gamut/AutoTone clip diagnostics that the GPU path cannot cheaply
   * compute without an extra readback+scan (the shader doesn't count
   * clipped/out-of-gamut pixels). Always 0 here; callers that need exact
   * counts should use the CPU backend. This is stated explicitly rather
   * than silently guessed.
   */
  diagnosticsApproximate: true;
}

/**
 * Throws if this environment cannot run the shader precision pipeline:
 * no WebGL2, no float-texture rendering support, or no canvas surface.
 * Callers should treat any throw as "fall back to CPU" -- never as a
 * fatal error.
 */
export function assertWebGL2PrecisionPipelineSupported(): void {
  const g = globalThis as unknown as { OffscreenCanvas?: unknown; document?: Document };
  let gl: WebGL2RenderingContext | null = null;
  if (typeof g.OffscreenCanvas !== 'undefined') {
    const canvas = new (g.OffscreenCanvas as new (w: number, h: number) => OffscreenCanvas)(2, 2);
    gl = canvas.getContext('webgl2') as WebGL2RenderingContext | null;
  } else if (g.document) {
    const canvas = g.document.createElement('canvas');
    gl = canvas.getContext('webgl2') as WebGL2RenderingContext | null;
  } else {
    throw new Error('No OffscreenCanvas or document available to create a WebGL2 context.');
  }
  if (!gl) throw new Error('WebGL2 context creation failed.');
  if (!gl.getExtension('EXT_color_buffer_float')) {
    throw new Error('EXT_color_buffer_float is not supported -- cannot render to a float texture.');
  }
}

export function compileShader(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error('gl.createShader failed.');
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`WebGL2 shader compile failed: ${log}`);
  }
  return shader;
}

export function linkProgram(gl: WebGL2RenderingContext, vertSrc: string, fragSrc: string): WebGLProgram {
  const vert = compileShader(gl, gl.VERTEX_SHADER, vertSrc);
  const frag = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
  const program = gl.createProgram();
  if (!program) throw new Error('gl.createProgram failed.');
  gl.attachShader(program, vert);
  gl.attachShader(program, frag);
  gl.linkProgram(program);
  // Shaders can be detached/deleted once linked; the program retains the compiled code.
  gl.deleteShader(vert);
  gl.deleteShader(frag);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`WebGL2 program link failed: ${log}`);
  }
  return program;
}

/**
 * Runs the fused precision-pipeline shader over a planar linear-RGB
 * image, in place (r/g/b are overwritten with the result). Throws on any
 * unsupported-environment or GPU error; see
 * assertWebGL2PrecisionPipelineSupported() for a cheap pre-check.
 */
export function runPrecisionPipelineWebGL2(
  r: Float32Array,
  g: Float32Array,
  b: Float32Array,
  width: number,
  height: number,
  opts: PrecisionPipelineGpuOptions
): PrecisionPipelineGpuResult {
  const n = width * height;
  if (r.length !== n || g.length !== n || b.length !== n) {
    throw new Error('runPrecisionPipelineWebGL2: buffer/size mismatch.');
  }

  // ---- 1. Estimate scalar parameters on CPU (cheap: bounded subsamples
  //         / a single histogram pass), exactly as the CPU backend does. ----
  const wb = computeWhiteBalanceGains(r, g, b, opts.whiteBalance);
  const atParams = computeAutoToneParams(r, g, b, opts.autoTone);
  const atEnabled = !atParams.degenerate && opts.autoTone.strength > 0;
  const cc = computeChannelBalanceGains(r, g, b, opts.colorCorrection);
  const ccEnabled = opts.colorCorrection.strength > 0;
  const gamutEnabled = opts.gamutEnabled ?? true;

  const g_ = globalThis as unknown as { OffscreenCanvas?: unknown; document?: Document };
  let gl: WebGL2RenderingContext | null = null;
  if (typeof g_.OffscreenCanvas !== 'undefined') {
    const canvas = new (g_.OffscreenCanvas as new (w: number, h: number) => OffscreenCanvas)(width, height);
    gl = canvas.getContext('webgl2', { antialias: false, alpha: true }) as WebGL2RenderingContext | null;
  } else if (g_.document) {
    const canvas = g_.document.createElement('canvas');
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

  const maxTexSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
  if (width > maxTexSize || height > maxTexSize) {
    throw new Error(`Image (${width}x${height}) exceeds this GPU's MAX_TEXTURE_SIZE (${maxTexSize}); tiling is not yet implemented, falling back to CPU.`);
  }

  // ---- 2. Pack planar r/g/b into an interleaved RGBA32F source buffer. ----
  const srcPixels = new Float32Array(n * 4);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    srcPixels[p] = r[i];
    srcPixels[p + 1] = g[i];
    srcPixels[p + 2] = b[i];
    srcPixels[p + 3] = 1;
  }

  let srcTex: WebGLTexture | null = null;
  let dstTex: WebGLTexture | null = null;
  let fbo: WebGLFramebuffer | null = null;
  let program: WebGLProgram | null = null;
  let vao: WebGLVertexArrayObject | null = null;

  try {
    program = linkProgram(gl, PRECISION_PIPELINE_VERTEX_SRC, PRECISION_PIPELINE_FRAGMENT_SRC);

    srcTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, srcTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, srcPixels);

    dstTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, dstTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);

    fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, dstTex, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error('WebGL2 framebuffer incomplete for RGBA32F render target.');
    }

    vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    gl.viewport(0, 0, width, height);
    gl.useProgram(program);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, srcTex);
    gl.uniform1i(gl.getUniformLocation(program, 'u_src'), 0);

    gl.uniform3f(gl.getUniformLocation(program, 'u_wbGain'), wb.gainR, wb.gainG, wb.gainB);
    gl.uniform1f(gl.getUniformLocation(program, 'u_atBlackPoint'), atParams.blackPoint);
    gl.uniform1f(gl.getUniformLocation(program, 'u_atRange'), Math.max(atParams.whitePoint - atParams.blackPoint, 1e-4));
    gl.uniform1i(gl.getUniformLocation(program, 'u_atEnabled'), atEnabled ? 1 : 0);
    gl.uniform3f(gl.getUniformLocation(program, 'u_ccGain'), cc.gainR, cc.gainG, cc.gainB);
    gl.uniform1f(gl.getUniformLocation(program, 'u_ccStrengthFrac'), Math.max(0, Math.min(1, opts.colorCorrection.strength / 100)));
    gl.uniform1i(gl.getUniformLocation(program, 'u_ccEnabled'), ccEnabled ? 1 : 0);
    gl.uniform1i(gl.getUniformLocation(program, 'u_gamutEnabled'), gamutEnabled ? 1 : 0);

    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.drawArrays(gl.TRIANGLES, 0, 3); // fullscreen triangle, positions from gl_VertexID

    const out = new Float32Array(n * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.FLOAT, out);

    for (let i = 0, p = 0; i < n; i++, p += 4) {
      r[i] = out[p];
      g[i] = out[p + 1];
      b[i] = out[p + 2];
    }

    return {
      whiteBalance: wb,
      autoTone: {
        blackPoint: atParams.blackPoint,
        whitePoint: atParams.whitePoint,
        blackClipPercent: 0,
        whiteClipPercent: 0,
      },
      colorCorrection: cc,
      diagnosticsApproximate: true,
    };
  } finally {
    // Release GPU resources deterministically rather than waiting on GC --
    // important for repeated processing calls in a long-lived tab/worker.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindVertexArray(null);
    if (vao) gl.deleteVertexArray(vao);
    if (fbo) gl.deleteFramebuffer(fbo);
    if (srcTex) gl.deleteTexture(srcTex);
    if (dstTex) gl.deleteTexture(dstTex);
    if (program) gl.deleteProgram(program);
  }
}
