/**
 * Browser-side entry for the WebGPU consistency harness.
 *
 * Bundled (esbuild) into an IIFE and loaded into a real Chromium page
 * served from http://localhost (WebGPU requires a secure context).
 * Exposes the shipped WebGPU, WebGL2 and pipeline entry points -- no
 * reimplementation, no test doubles -- so the harness can compare all
 * three backends in the SAME browser on the SAME inputs.
 */
import { runFullPipelineWebGPU, runPrecisionPipelineWebGPU } from '../../../src/processing/gpu/webgpu/webgpuFullPipeline';
import { acquireWebGPU, compileAllKernels, probeWebGPU, resetWebGPUCache } from '../../../src/processing/gpu/webgpu/webgpuDevice';
import { KERNELS } from '../../../src/processing/gpu/webgpu/wgslShaders';
import { runFullPipelineWebGL2 } from '../../../src/processing/gpu/webgl2FullPipeline';
import { runTiledFullPipelineWebGL2, probeMaxTextureSize } from '../../../src/processing/gpu/webgl2TiledPipeline';
import { runPipeline } from '../../../src/processing/pipeline';
import { runPipelineAsync } from '../../../src/processing/pipelineAsync';

(window as unknown as { __gpuHarness: unknown }).__gpuHarness = {
  runFullPipelineWebGPU,
  runPrecisionPipelineWebGPU,
  runPipeline,
  runPipelineAsync,
  acquireWebGPU,
  compileAllKernels,
  probeWebGPU,
  resetWebGPUCache,
  KERNELS,
  runFullPipelineWebGL2,
  runTiledFullPipelineWebGL2,
  probeMaxTextureSize,
};
