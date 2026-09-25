/**
 * Async pipeline entry point: WebGPU first, then the EXISTING synchronous
 * pipeline (WebGL2 -> CPU) as the fallback chain.
 * ----------------------------------------------------------------------
 * WebGPU readback is inherently asynchronous (`mapAsync`), so it cannot
 * live inside the synchronous `runPipeline`. This wrapper:
 *
 *  1. Decides whether WebGPU should be attempted: an explicit 'webgpu'
 *     request, or 'auto' when resolveBackend() picks WebGPU (large image,
 *     navigator.gpu present).
 *  2. Runs steps 1-3 (intensity extraction, level resolution, stretch --
 *     the same functions runPipeline uses), then the WebGPU full pipeline
 *     (noise reduction .. sharpening, tiled if needed).
 *  3. On ANY failure (no adapter, shader error, validation/OOM error,
 *     device loss, an untileable request, ...) delegates to the untouched
 *     synchronous runPipeline with the engine forced to 'webgl2', so the
 *     established WebGL2 -> CPU chain runs exactly as before.
 *
 * HONESTY CONTRACT: `result.execution.executed === 'webgpu'` (and
 * `precision.backend.resolved === 'webgpu'`, `gpuAccelerated: true`) only
 * when the WebGPU pipeline itself produced the returned pixels. In every
 * other case those fields name the backend that really ran, `fellBack` is
 * true, and `execution.attempts` records why WebGPU did not.
 */

import type { ProcessingSettings } from '../types/processing';
import { PRESETS } from './presets';
import { extractIntensity, applyLevels, resolveLevelPoints } from './normalize';
import { computeHistogram } from './histogram';
import { detectCapabilities, resolveBackend } from './backend';
import { runPipeline, resolveQualityParams, type PipelineResult, type PrecisionDiagnostics } from './pipeline';
import { buildExecutionReport } from './executionReport';
import { runFullPipelineWebGPU, type WebGPUPipelineOptions } from './gpu/webgpu/webgpuFullPipeline';

export interface AsyncPipelineOptions {
  /** Forwarded to the WebGPU pipeline (maxTileDim, acquire options, ...). */
  webgpu?: WebGPUPipelineOptions;
  /** Test seam: replace the WebGPU implementation (e.g. to inject a failure). */
  runWebGPU?: typeof runFullPipelineWebGPU;
}

function describe(e: unknown): string {
  if (e instanceof Error) return e.name && e.name !== 'Error' ? `${e.name}: ${e.message}` : e.message;
  return String(e);
}

export async function runPipelineAsync(input: ImageData, settings: ProcessingSettings, options: AsyncPipelineOptions = {}): Promise<PipelineResult> {
  const { width, height } = input;
  const requested = settings.processingEngine;
  const resolution = resolveBackend(requested, detectCapabilities(), width * height);
  const attemptWebGPU = requested === 'webgpu' || resolution.resolved === 'webgpu';

  if (!attemptWebGPU) return runPipeline(input, settings);

  let failure: string;
  try {
    return await runOnWebGPU(input, settings, options);
  } catch (e) {
    failure = describe(e);
  }

  // Fallback: the existing synchronous chain, forced to start at WebGL2
  // (its own honest WebGL2 -> CPU fallback applies from there).
  const fallback = runPipeline(input, { ...settings, processingEngine: 'webgl2' });
  const inner = fallback.execution;
  const attempts = [{ backend: 'webgpu' as const, ok: false, reason: failure }, ...(inner?.attempts ?? [])];
  const executed = inner?.executed ?? 'cpu';
  const execution = buildExecutionReport({
    requested,
    executed,
    attempts,
    precisionStageOnGpu: inner?.precisionStageOnGpu,
    tiling: inner?.tiling,
  });
  execution.fellBack = true; // WebGPU was attempted (or explicitly requested) and did not produce the result

  let precision = fallback.precision;
  if (precision) {
    precision = {
      ...precision,
      backend: {
        ...precision.backend,
        requested,
        fellBack: true,
        reason: `WebGPU did not run (${failure}). ${precision.backend.reason ?? ''}`.trim(),
      },
    };
  }
  return { ...fallback, precision, execution };
}

async function runOnWebGPU(input: ImageData, settings: ProcessingSettings, options: AsyncPipelineOptions): Promise<PipelineResult> {
  const { width, height } = input;
  const preset = PRESETS[settings.preset];
  const qualityParams = resolveQualityParams(settings.quality);

  // Steps 1-3: identical to runPipeline (same functions).
  const { intensity: rawIntensity, histogram: inputHistogram } = extractIntensity(input.data, width, height, settings.interpretation);
  const levelsUsed = resolveLevelPoints(
    inputHistogram,
    settings.autoLevels,
    settings.blackPoint,
    settings.whitePoint,
    settings.blackPercentile,
    settings.whitePercentile
  );
  const intensity = applyLevels(rawIntensity, levelsUsed.black, levelsUsed.white);

  const run = options.runWebGPU ?? runFullPipelineWebGPU;
  const gpu = await run(intensity, width, height, settings, preset.colorStops, qualityParams, options.webgpu);

  const output = new ImageData(new Uint8ClampedArray(gpu.rgba), width, height);
  const outIntensity = new Uint8ClampedArray(width * height);
  for (let i = 0, p = 0; i < outIntensity.length; i++, p += 4) {
    outIntensity[i] = 0.299 * gpu.rgba[p] + 0.587 * gpu.rgba[p + 1] + 0.114 * gpu.rgba[p + 2];
  }
  const outputHistogram = computeHistogram(outIntensity);

  const precision: PrecisionDiagnostics | undefined =
    settings.precisionPipeline && gpu.precision
      ? {
          backend: { requested: settings.processingEngine, resolved: 'webgpu', fellBack: false },
          whiteBalance: gpu.precision.whiteBalance,
          autoTone: gpu.precision.autoTone,
          colorCorrection: gpu.precision.colorCorrection,
          // Same honest accounting as the WebGL2 path: no out-of-gamut scan.
          outOfGamutPercent: 0,
          gpuAccelerated: true,
        }
      : undefined;

  const execution = buildExecutionReport({
    requested: settings.processingEngine,
    executed: 'webgpu',
    attempts: [{ backend: 'webgpu', ok: true }],
    tiling: gpu.tiling,
    adapter: gpu.adapter,
    resources: gpu.resources,
  });

  return { output, inputHistogram, outputHistogram, levelsUsed, precision, execution };
}
