/**
 * Bundled (via esbuild, see run.mjs) into a single IIFE and loaded into a
 * real browser page by the Playwright-driven harness. Exposes exactly
 * the production GPU entry points -- no reimplementation, no test
 * doubles -- so what gets measured is the real shipped code path. The
 * tiled entry point and the texture-size probe are the same functions
 * runFullPipelineWebGL2 itself delegates to for oversize images.
 */
import { runFullPipelineWebGL2, assertFullPipelineWebGL2Supported } from '../../src/processing/gpu/webgl2FullPipeline';
import { runPrecisionPipelineWebGL2, assertWebGL2PrecisionPipelineSupported } from '../../src/processing/gpu/webgl2Backend';
import { runTiledFullPipelineWebGL2, probeMaxTextureSize } from '../../src/processing/gpu/webgl2TiledPipeline';
import { runPipeline } from '../../src/processing/pipeline';

(window as unknown as { __gpuHarness: unknown }).__gpuHarness = {
  runFullPipelineWebGL2,
  runPrecisionPipelineWebGL2,
  assertFullPipelineWebGL2Supported,
  assertWebGL2PrecisionPipelineSupported,
  runTiledFullPipelineWebGL2,
  probeMaxTextureSize,
  runPipeline,
};
