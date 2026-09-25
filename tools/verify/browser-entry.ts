import { runFullPipelineWebGL2, assertFullPipelineWebGL2Supported } from '../../src/processing/gpu/webgl2FullPipeline';
import { runPrecisionPipelineWebGL2, assertWebGL2PrecisionPipelineSupported } from '../../src/processing/gpu/webgl2Backend';
import { runTiledFullPipelineWebGL2, probeMaxTextureSize } from '../../src/processing/gpu/webgl2TiledPipeline';
import { runPipeline } from '../../src/processing/pipeline';
import { runPipelineAsync } from '../../src/processing/pipelineAsync';
import { detectCapabilities, isGpuCapableBackend, executesOnCpu } from '../../src/processing/backend';
import { executedOnGpu } from '../../src/processing/executionReport';
import { PRESETS } from '../../src/processing/presets';
import { DEFAULT_SETTINGS } from '../../src/types/processing';
import { runFullPipelineWebGPU, runPrecisionPipelineWebGPU } from '../../src/processing/gpu/webgpu/webgpuFullPipeline';
import { probeWebGPU } from '../../src/processing/gpu/webgpu/webgpuDevice';
import { makeRamp, makeNoisyChecker, makeVerticalGradient, intensityToRgbaBytes } from './fixtures';

(window as unknown as { __v: unknown }).__v = {
  runFullPipelineWebGL2,
  assertFullPipelineWebGL2Supported,
  runPrecisionPipelineWebGL2,
  assertWebGL2PrecisionPipelineSupported,
  runTiledFullPipelineWebGL2,
  probeMaxTextureSize,
  runPipeline,
  runPipelineAsync,
  detectCapabilities,
  isGpuCapableBackend,
  executesOnCpu,
  executedOnGpu,
  PRESETS,
  DEFAULT_SETTINGS,
  probeWebGPU,
  runFullPipelineWebGPU,
  runPrecisionPipelineWebGPU,
  makeRamp,
  makeNoisyChecker,
  makeVerticalGradient,
  intensityToRgbaBytes,
};
