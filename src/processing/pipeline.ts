import type { ProcessingSettings } from '../types/processing';
import { PRESETS } from './presets';
import { extractIntensity, applyLevels, resolveLevelPoints } from './normalize';
import { buildToneCurve } from './toneMapping';
import { applyLocalContrast, localVariance } from './contrast';
import { reduceNoise } from './noiseReduction';
import { unsharpMask } from './sharpening';
import { mapIntensityToRgb } from './colorMapping';
import type { SceneMaps } from './colorMapping';
import type { HistogramData } from '../types/image';
import { computeHistogram } from './histogram';
import { srgbToLinear, linearToSrgb } from './colorSpace';
import { applyAzusaWhiteBalance } from './whiteBalance';
import { applyAutoTone } from './autoTone';
import { applyColorCorrectionPipeline } from './colorCorrectionPipeline';
import { protectGamut } from './gamutProtection';
import { detectCapabilities } from './backend';
import type { BackendResolution } from './backend';
import { resolveSyncBackend, resolveSyncBackendEx, buildExecutionReport } from './executionReport';
import type { ExecutionAttempt, ExecutionReport } from './executionReport';
import type { TilingDiagnostics } from './gpu/tilePlanner';
import { runPrecisionPipelineWebGL2 } from './gpu/webgl2Backend';
import { runFullPipelineWebGL2 } from './gpu/webgl2FullPipeline';

export interface PrecisionDiagnostics {
  backend: BackendResolution;
  whiteBalance: { gainR: number; gainG: number; gainB: number; confidence: number };
  autoTone: { blackPoint: number; whitePoint: number; blackClipPercent: number; whiteClipPercent: number };
  colorCorrection: { gainR: number; gainG: number; gainB: number };
  outOfGamutPercent: number;
  /**
   * True only when the precision pipeline (white balance / AutoTone /
   * color correction / gamut protection) actually executed as a GPU
   * shader (WebGL2, or WebGPU via runPipelineAsync). False for CPU
   * execution, including when a WebGL2
   * run was requested but failed and transparently fell back to CPU --
   * see backend.reason for why. Never true without real GPU work having
   * happened (no "fake GPU" status).
   */
  gpuAccelerated: boolean;
}

export interface PipelineResult {
  output: ImageData;
  inputHistogram: HistogramData;
  outputHistogram: HistogramData;
  levelsUsed: { black: number; white: number };
  precision?: PrecisionDiagnostics;
  /**
   * What ACTUALLY produced the pixels (cpu | webgl2 | webgpu) and the
   * fallback audit trail. Prefer this over `precision.backend` (which
   * exists only when the precision pipeline is enabled).
   */
  execution?: ExecutionReport;
}

export interface QualityParams {
  /** Box-blur passes used to approximate the Gaussian noise-reduction filter -- more passes, smoother/more accurate approximation. */
  gaussianPasses: number;
  /** Local-mean radius used for local contrast enhancement -- larger radius, broader (more expensive) local-contrast estimate. */
  localContrastRadius: number;
}

/**
 * Processing Quality controls the internal fidelity/cost of the filters
 * that approximate more expensive operations (Gaussian blur via repeated
 * box blur, local-mean radius for local contrast) -- it does not skip any
 * user-enabled effect, only how faithfully/expensively it's computed.
 * "Balanced" matches the pipeline's original defaults exactly.
 */
export function resolveQualityParams(quality: ProcessingSettings['quality']): QualityParams {
  switch (quality) {
    case 'fast':
      return { gaussianPasses: 1, localContrastRadius: 14 };
    case 'high':
      return { gaussianPasses: 4, localContrastRadius: 28 };
    case 'maximum':
      return { gaussianPasses: 6, localContrastRadius: 36 };
    case 'balanced':
    default:
      return { gaussianPasses: 3, localContrastRadius: 24 };
  }
}

/**
 * Runs the full deterministic (non-ML) IR -> visible conversion pipeline on
 * an ImageData buffer. Pure function: same input + settings always produce
 * the same output.
 */
export function runPipeline(input: ImageData, settings: ProcessingSettings): PipelineResult {
  const { width, height } = input;
  const preset = PRESETS[settings.preset];
  const qualityParams = resolveQualityParams(settings.quality);

  // 1. Extract a single intensity channel according to how the source should
  //    be interpreted.
  const { intensity: rawIntensity, histogram: inputHistogram } = extractIntensity(
    input.data,
    width,
    height,
    settings.interpretation
  );

  // 2. Resolve black/white points (auto percentile-based, or manual).
  const levelsUsed = resolveLevelPoints(
    inputHistogram,
    settings.autoLevels,
    settings.blackPoint,
    settings.whitePoint,
    settings.blackPercentile,
    settings.whitePercentile
  );

  // 3. Stretch into normalized [0,1] dynamic range.
  let intensity: Float32Array = applyLevels(rawIntensity, levelsUsed.black, levelsUsed.white);

  // 4-10 (GPU attempt). Tries to run the ENTIRE remaining pipeline --
  // noise reduction through sharpening, including the precision
  // sub-stage -- as WebGL2 shaders (processing/gpu/webgl2FullPipeline.ts)
  // in one GPU session. Any failure (unsupported environment, an
  // unported option like 'median' noise reduction, oversized texture,
  // shader error) throws there and is caught here: `fullGpuResult`
  // stays null and the untouched CPU implementation below runs instead
  // -- including, if that CPU run still has settings.precisionPipeline
  // on, its OWN independent attempt at the (separately-shipped,
  // narrower) GPU precision-pipeline-only path. That gives a graceful
  // three-tier fallback -- full GPU, then CPU-spatial-stages with a
  // GPU-accelerated precision sub-stage, then full CPU -- rather than an
  // all-or-nothing jump straight to CPU.
  let fullGpuResult: { rgba: Uint8ClampedArray; precisionDiagnostics?: PrecisionDiagnostics } | null = null;
  const executionAttempts: ExecutionAttempt[] = [];
  let fullGpuTiling: TilingDiagnostics | undefined;

  {
    // resolveSyncBackendEx: this synchronous pipeline cannot run WebGPU (see
    // executionReport.ts); a 'webgpu' resolution is re-mapped to what it can
    // really execute instead of being reported as active while CPU runs.
    const { backend, webgpuSkipped } = resolveSyncBackendEx(settings.processingEngine, detectCapabilities(), width * height);
    if (webgpuSkipped) {
      executionAttempts.push({ backend: 'webgpu', ok: false, reason: 'not attempted: the synchronous pipeline has no WebGPU path (use runPipelineAsync)' });
    }
    if (settings.processingEngine === 'webgl2' && backend.resolved === 'cpu') {
      executionAttempts.push({ backend: 'webgl2', ok: false, reason: backend.reason });
    }
    if (backend.resolved === 'webgl2') {
      try {
        const gpuResult = runFullPipelineWebGL2(intensity, width, height, settings, preset.colorStops, qualityParams);
        fullGpuTiling = gpuResult.tiling;
        executionAttempts.push({ backend: 'webgl2', ok: true });
        fullGpuResult = {
          rgba: gpuResult.rgba,
          precisionDiagnostics:
            settings.precisionPipeline && gpuResult.precision
              ? {
                  backend,
                  whiteBalance: gpuResult.precision.whiteBalance,
                  autoTone: gpuResult.precision.autoTone,
                  colorCorrection: gpuResult.precision.colorCorrection,
                  // Same honest accounting as the precision-only GPU path:
                  // the shader doesn't scan for an out-of-gamut pixel
                  // count (that's an extra readback+reduction pass) --
                  // see PrecisionPipelineGpuResult.diagnosticsApproximate.
                  outOfGamutPercent: 0,
                  gpuAccelerated: true,
                }
              : undefined,
        };
      } catch (err) {
        executionAttempts.push({ backend: 'webgl2', ok: false, reason: err instanceof Error ? err.message : String(err) });
        // Real, honest fallback: never claim GPU acceleration that
        // didn't happen. fullGpuResult stays null; the CPU branch below
        // still gets its own independent shot at the narrower GPU
        // precision-only path via its own resolveBackend() call.
      }
    }
  }

  let rgba: Uint8ClampedArray;
  let precisionDiagnostics: PrecisionDiagnostics | undefined;

  if (fullGpuResult) {
    rgba = fullGpuResult.rgba;
    precisionDiagnostics = fullGpuResult.precisionDiagnostics;
  } else {
  // 4. Noise reduction (edge-aware where selected), before contrast work so
  //    we don't amplify sensor noise later in the pipeline.
  if (settings.noiseReduction > 0) {
    intensity = reduceNoise(
      intensity,
      width,
      height,
      settings.noiseMethod,
      settings.noiseReduction,
      qualityParams.gaussianPasses
    );
  }

  // 5. Local contrast enhancement (CLAHE-like).
  if (settings.localContrast > 0) {
    intensity = applyLocalContrast(intensity, width, height, settings.localContrast, qualityParams.localContrastRadius);
  }

  // 6. Global tone curve: exposure, brightness, contrast, gamma, shadow
  //    lift, highlight recovery.
  const toneCurve = buildToneCurve({
    exposure: settings.exposure,
    brightness: settings.brightness,
    contrast: settings.contrast,
    gamma: settings.gamma,
    shadowLift: settings.shadowLift,
    highlightRecovery: settings.highlightRecovery,
  });
  const toneMapped = new Float32Array(intensity.length);
  for (let i = 0; i < intensity.length; i++) toneMapped[i] = toneCurve(intensity[i]);

  // 7. Optional detail-preservation pass: blends a touch of the
  //    pre-tone-curve detail back in, so very compressed tone curves don't
  //    flatten fine structure.
  let workingIntensity: Float32Array = toneMapped;
  if (settings.detailPreservation > 0) {
    workingIntensity = preserveDetail(intensity, toneMapped, settings.detailPreservation);
  }

  // 8. Optional deterministic scene heuristics (sky / vegetation likelihood)
  //    used only to nudge color, never to alter structure.
  let scene: SceneMaps | undefined;
  if (settings.sceneHeuristics) {
    scene = computeSceneMaps(workingIntensity, width, height);
  }

  // 9. Natural RGB reconstruction via the preset's OKLab-interpolated ramp,
  //    plus saturation / temperature / hue-bias.
  rgba = mapIntensityToRgb(workingIntensity, width, height, preset.colorStops, {
    colorStrength: settings.colorStrength,
    saturation: settings.saturation,
    temperature: settings.temperature,
    hueBias: settings.hueBias,
    scene,
  });

  // 9b. Precision pipeline: Azusa (white balance) -> AutoTone (tonal
  //     correction) -> ColorCorrectionPipeline (fine color correction) ->
  //     gamut protection. Operates on a high-precision, linear-light
  //     Float32 working buffer so these RGB-domain refinements don't
  //     accumulate 8-bit quantization error, then re-encodes to sRGB once.
  //     This never runs before the deterministic IR->RGB color mapping
  //     above -- it only refines its output (see spec section 11).
  if (settings.precisionPipeline) {
    const n = width * height;
    const rC = new Float32Array(n);
    const gC = new Float32Array(n);
    const bC = new Float32Array(n);
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      rC[i] = srgbToLinear(rgba[p] / 255);
      gC[i] = srgbToLinear(rgba[p + 1] / 255);
      bC[i] = srgbToLinear(rgba[p + 2] / 255);
    }

    let backend = resolveSyncBackend(settings.processingEngine, detectCapabilities(), n);
    const wbOpts = { strength: settings.whiteBalanceStrength };
    const atOpts = { strength: settings.autoToneStrength };
    const ccOpts = { strength: settings.colorCorrectionStrength };

    let gpuAccelerated = false;
    let wb: ReturnType<typeof applyAzusaWhiteBalance> | undefined;
    let atResult: { blackPoint: number; whitePoint: number; blackClipPercent: number; whiteClipPercent: number } | undefined;
    let cc: ReturnType<typeof applyColorCorrectionPipeline> | undefined;
    let outOfGamutPercent = 0;

    if (backend.resolved === 'webgl2') {
      try {
        const gpuResult = runPrecisionPipelineWebGL2(rC, gC, bC, width, height, {
          whiteBalance: wbOpts,
          autoTone: atOpts,
          colorCorrection: ccOpts,
        });
        wb = gpuResult.whiteBalance;
        atResult = gpuResult.autoTone;
        cc = gpuResult.colorCorrection;
        // The GPU shader doesn't scan for clip/out-of-gamut counts (that
        // would need an extra readback+reduction pass); it's not lying
        // about them, it genuinely doesn't compute them. Diagnostics that
        // depend on that count are left at 0 rather than guessed -- see
        // PrecisionPipelineGpuResult.diagnosticsApproximate.
        outOfGamutPercent = 0;
        gpuAccelerated = true;
      } catch (err) {
        // Real, honest fallback: never claim GPU acceleration that didn't
        // happen. Re-resolve as 'cpu' so diagnostics.backend reflects
        // reality, and run the CPU implementation below instead.
        backend = {
          requested: backend.requested,
          resolved: 'cpu',
          fellBack: true,
          reason: `WebGL2 precision pipeline failed at runtime, fell back to CPU: ${err instanceof Error ? err.message : String(err)}`,
        };
      }
    }

    if (!gpuAccelerated) {
      wb = applyAzusaWhiteBalance(rC, gC, bC, wbOpts);
      const at = applyAutoTone(rC, gC, bC, atOpts);
      atResult = {
        blackPoint: at.blackPoint,
        whitePoint: at.whitePoint,
        blackClipPercent: at.blackClipPercent,
        whiteClipPercent: at.whiteClipPercent,
      };
      cc = applyColorCorrectionPipeline(rC, gC, bC, ccOpts);
      const gamut = protectGamut(rC, gC, bC);
      outOfGamutPercent = (gamut.outOfGamutCount / n) * 100;
    }

    for (let i = 0, p = 0; i < n; i++, p += 4) {
      rgba[p] = linearToSrgb(rC[i]) * 255;
      rgba[p + 1] = linearToSrgb(gC[i]) * 255;
      rgba[p + 2] = linearToSrgb(bC[i]) * 255;
    }

    precisionDiagnostics = {
      backend,
      whiteBalance: wb!,
      autoTone: atResult!,
      colorCorrection: cc!,
      outOfGamutPercent,
      gpuAccelerated,
    };
  }

  // 10. Sharpening (unsharp mask) applied per-channel in the final RGB, to
  //     avoid re-introducing color fringing that luminance-only sharpening
  //     on a colored image can cause.
  if (settings.sharpenAmount > 0) {
    rgba = sharpenRgba(rgba, width, height, {
      amount: settings.sharpenAmount,
      radius: settings.sharpenRadius,
      threshold: settings.sharpenThreshold,
    });
  }
  } // end of `else` (CPU steps 4-10) -- see the GPU attempt above step 4.

  const output = new ImageData(new Uint8ClampedArray(rgba), width, height);

  const outIntensity = new Uint8ClampedArray(width * height);
  for (let i = 0, p = 0; i < outIntensity.length; i++, p += 4) {
    outIntensity[i] = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
  }
  const outputHistogram = computeHistogram(outIntensity);

  const executed = fullGpuResult ? 'webgl2' : 'cpu';
  const lastAttempt = executionAttempts[executionAttempts.length - 1];
  if (!(lastAttempt && lastAttempt.backend === executed && lastAttempt.ok)) executionAttempts.push({ backend: executed, ok: true });
  const execution = buildExecutionReport({
    requested: settings.processingEngine,
    executed,
    attempts: executionAttempts,
    precisionStageOnGpu: !fullGpuResult && precisionDiagnostics?.gpuAccelerated === true,
    tiling: fullGpuTiling,
  });

  return { output, inputHistogram, outputHistogram, levelsUsed, precision: precisionDiagnostics, execution };
}

function preserveDetail(
  original: Float32Array,
  toneMapped: Float32Array,
  strength: number
): Float32Array {
  const amt = (strength / 100) * 0.3;
  const out = new Float32Array(original.length);
  for (let i = 0; i < original.length; i++) {
    // High-frequency detail estimated as the difference between the raw and
    // a heavily tone-compressed version; re-inject a fraction of it.
    const detail = original[i] - toneMapped[i];
    out[i] = clamp01(toneMapped[i] + detail * amt * 0.15);
  }
  return out;
}

function clamp01(x: number) {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

export function computeSceneMaps(intensity: Float32Array, width: number, height: number): SceneMaps {
  const variance = localVariance(intensity, width, height, 5);
  const sky = new Float32Array(intensity.length);
  const vegetation = new Float32Array(intensity.length);

  for (let y = 0; y < height; y++) {
    const rowFrac = y / height; // 0 = top, 1 = bottom
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const I = intensity[idx];
      const v = variance[idx];

      // Sky-like: bright, low local texture, biased to the top of the frame.
      const topBias = clamp01(1 - rowFrac * 1.6);
      const brightness = clamp01((I - 0.55) / 0.45);
      const flatness = clamp01(1 - v * 40);
      sky[idx] = topBias * brightness * flatness;

      // Vegetation-like: mid intensity with noticeable local texture,
      // biased away from the very top of the frame.
      const bottomBias = clamp01(rowFrac * 1.2 + 0.2);
      const midtone = clamp01(1 - Math.abs(I - 0.42) / 0.35);
      const texture = clamp01(v * 30);
      vegetation[idx] = bottomBias * midtone * texture;
    }
  }

  return { sky, vegetation };
}

interface SharpenOpts {
  amount: number;
  radius: number;
  threshold: number;
}

function sharpenRgba(rgba: Uint8ClampedArray, width: number, height: number, opts: SharpenOpts): Uint8ClampedArray {
  const n = width * height;
  const r = new Float32Array(n);
  const g = new Float32Array(n);
  const b = new Float32Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    r[i] = rgba[p] / 255;
    g[i] = rgba[p + 1] / 255;
    b[i] = rgba[p + 2] / 255;
  }
  const rs = unsharpMask(r, width, height, opts);
  const gs = unsharpMask(g, width, height, opts);
  const bs = unsharpMask(b, width, height, opts);

  const out = new Uint8ClampedArray(rgba.length);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    out[p] = rs[i] * 255;
    out[p + 1] = gs[i] * 255;
    out[p + 2] = bs[i] * 255;
    out[p + 3] = rgba[p + 3];
  }
  return out;
}
