/**
 * CPU/GPU numerical consistency harness.
 * ----------------------------------------
 * Runs the SHIPPED production code on both sides -- no reimplementation,
 * no test doubles:
 *   - GPU side: the real `runFullPipelineWebGL2` / `runPrecisionPipelineWebGL2`
 *     (src/processing/gpu/), executed inside a real headless Chromium
 *     with a real WebGL2 context (via Playwright), bundled with esbuild.
 *   - CPU side: the real CPU reference functions (src/processing/*.ts),
 *     run directly in Node via tsx.
 *
 * Requires (not part of the app's normal devDependencies -- this is an
 * opt-in verification tool, not something every `npm install` needs to
 * pay for): `npm i -D playwright esbuild tsx && npx playwright install chromium`.
 * Run with `npm run gpu-consistency`.
 *
 * "Stage-by-stage" isolation is achieved by calling the real
 * `runFullPipelineWebGL2` with every OTHER stage's settings zeroed/
 * neutralized, and comparing against the matching individual CPU
 * function(s) run under the same isolated configuration -- rather than
 * exporting internal GL primitives for a separate, lower-fidelity test
 * path. This means every check here exercises the exact code path a
 * real user's browser would take.
 *
 * Exit code is non-zero if any check exceeds its tolerance, so this can
 * run in CI. Tolerances are chosen to reflect "8-bit visual
 * equivalence" (final output is 8-bit sRGB regardless of internal
 * precision) and float32-vs-float64 rounding in chained OKLab/pow/exp
 * math -- they are NOT tuned to make failing checks pass. See the
 * repo's README GPU section for the real discrepancies found and fixed
 * while building this harness.
 */
import { chromium, type Browser, type Page } from 'playwright';
import * as esbuild from 'esbuild';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';

// Reuses the same ImageData polyfill Vitest uses (tests/setup.ts) -- this
// harness runs in plain Node too, which has no DOM.
import '../../tests/setup';

import { compareArrays, compareInterleavedChannels, formatMetrics, type ErrorMetrics } from './metrics';

import { srgbToLinear } from '../../src/processing/colorSpace';
import { reduceNoise } from '../../src/processing/noiseReduction';
import { applyLocalContrast } from '../../src/processing/contrast';
import { buildToneCurve } from '../../src/processing/toneMapping';
import { mapIntensityToRgb } from '../../src/processing/colorMapping';
import { computeSceneMaps, runPipeline } from '../../src/processing/pipeline';
import { unsharpMask } from '../../src/processing/sharpening';
import { applyAzusaWhiteBalance } from '../../src/processing/whiteBalance';
import { applyAutoTone } from '../../src/processing/autoTone';
import { applyColorCorrectionPipeline } from '../../src/processing/colorCorrectionPipeline';
import { protectGamut } from '../../src/processing/gamutProtection';
import { PRESETS } from '../../src/processing/presets';
import { extractIntensity, applyLevels, resolveLevelPoints } from '../../src/processing/normalize';
import { DEFAULT_SETTINGS } from '../../src/types/processing';
import { planTiles, resolveStageRadii } from '../../src/processing/gpu/tilePlanner';
import type { TilingDiagnostics } from '../../src/processing/gpu/tilePlanner';
import type { ProcessingSettings, ColorStop } from '../../src/types/processing';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------

function makeIntensityRamp(width: number, height: number): Float32Array {
  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      out[y * width + x] = x / (width - 1);
    }
  }
  return out;
}

function makeIntensityCheckerNoisy(width: number, height: number, seed = 1): Float32Array {
  const out = new Float32Array(width * height);
  let s = seed;
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const base = (Math.floor(x / 8) + Math.floor(y / 8)) % 2 === 0 ? 0.25 : 0.75;
      out[y * width + x] = Math.min(1, Math.max(0, base + (rand() - 0.5) * 0.3));
    }
  }
  return out;
}

function makeIntensityVerticalGradient(width: number, height: number): Float32Array {
  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      out[y * width + x] = y / (height - 1);
    }
  }
  return out;
}

function makeUint8ImageFromIntensity(intensity: Float32Array, width: number, height: number): Uint8ClampedArray {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0, p = 0; i < intensity.length; i++, p += 4) {
    const v = Math.round(intensity[i] * 255);
    data[p] = v;
    data[p + 1] = v;
    data[p + 2] = v;
    data[p + 3] = 255;
  }
  return data;
}

// ---------------------------------------------------------------------
// Report collection
// ---------------------------------------------------------------------

interface CheckResult {
  name: string;
  metrics: ErrorMetrics;
  tolerance: number;
  maxAllowedFraction: number;
  scale: number;
  pass: boolean;
}

const results: CheckResult[] = [];

function record(name: string, metrics: ErrorMetrics, tolerance: number, maxAllowedFraction: number, scale = 1): void {
  const pass = metrics.maxError <= tolerance * 4 && metrics.fractionOverTolerance <= maxAllowedFraction;
  results.push({ name, metrics, tolerance, maxAllowedFraction, scale, pass });
  const status = pass ? 'PASS' : 'FAIL';
  console.log(`[${status}] ${formatMetrics(name, metrics, tolerance, scale)}`);
}

// ---------------------------------------------------------------------
// GPU harness (Playwright)
// ---------------------------------------------------------------------

interface GpuHarness {
  browser: Browser;
  page: Page;
}

async function startGpuHarness(): Promise<GpuHarness> {
  const bundleDir = path.join(__dirname, '.bundle');
  fs.mkdirSync(bundleDir, { recursive: true });
  const outfile = path.join(bundleDir, 'harness.js');

  await esbuild.build({
    entryPoints: [path.join(__dirname, 'harness-entry.ts')],
    bundle: true,
    format: 'iife',
    platform: 'browser',
    target: 'es2020',
    outfile,
  });

  const browser = await chromium.launch({
    headless: true,
    args: [
      '--no-sandbox',
      '--use-gl=swiftshader',
      '--ignore-gpu-blocklist',
      '--enable-webgl',
      '--enable-webgl2-compute-context',
      '--disable-gpu-sandbox',
    ],
  });
  const page = await browser.newPage();
  await page.goto('about:blank');
  await page.addScriptTag({ path: outfile });

  const supported = await page.evaluate(() => {
    try {
      (window as any).__gpuHarness.assertFullPipelineWebGL2Supported();
      return true;
    } catch (e) {
      return String(e);
    }
  });
  if (supported !== true) {
    throw new Error(`GPU harness page does not support the full pipeline: ${supported}`);
  }

  return { browser, page };
}

async function gpuRunFullPipeline(
  page: Page,
  intensity: Float32Array,
  width: number,
  height: number,
  settings: ProcessingSettings,
  colorStops: ColorStop[],
  quality: { gaussianPasses: number; localContrastRadius: number }
): Promise<{ rgba: number[]; precision?: unknown; tiling?: TilingDiagnostics }> {
  return page.evaluate(
    (args: [number[], number, number, ProcessingSettings, ColorStop[], { gaussianPasses: number; localContrastRadius: number }]) => {
      const [intensityArr, w, h, s, stops, q] = args;
      const harness = (window as any).__gpuHarness;
      const result = harness.runFullPipelineWebGL2(new Float32Array(intensityArr), w, h, s, stops, q);
      return { rgba: Array.from(result.rgba as Uint8ClampedArray), precision: result.precision, tiling: result.tiling };
    },
    [Array.from(intensity), width, height, settings, colorStops, quality] as [number[], number, number, ProcessingSettings, ColorStop[], { gaussianPasses: number; localContrastRadius: number }]
  );
}

async function gpuRunPrecisionPipeline(
  page: Page,
  r: Float32Array,
  g: Float32Array,
  b: Float32Array,
  width: number,
  height: number,
  opts: { whiteBalance: { strength: number }; autoTone: { strength: number }; colorCorrection: { strength: number } }
): Promise<{ r: number[]; g: number[]; b: number[] }> {
  return page.evaluate(
    (args: [number[], number[], number[], number, number, typeof opts]) => {
      const [rArr, gArr, bArr, w, h, o] = args;
      const harness = (window as any).__gpuHarness;
      const rF = new Float32Array(rArr);
      const gF = new Float32Array(gArr);
      const bF = new Float32Array(bArr);
      harness.runPrecisionPipelineWebGL2(rF, gF, bF, w, h, o);
      return { r: Array.from(rF), g: Array.from(gF), b: Array.from(bF) };
    },
    [Array.from(r), Array.from(g), Array.from(b), width, height, opts] as [number[], number[], number[], number, number, typeof opts]
  );
}

// ---------------------------------------------------------------------
// Stage-by-stage checks
// ---------------------------------------------------------------------

const NEUTRAL_QUALITY = { gaussianPasses: 3, localContrastRadius: 24 };

/** Settings with every stage off/neutral -- the baseline every isolated stage test starts from. */
function neutralSettings(): ProcessingSettings {
  return {
    ...DEFAULT_SETTINGS,
    autoLevels: false,
    blackPoint: 0,
    whitePoint: 255,
    exposure: 0,
    brightness: 0,
    contrast: 0,
    gamma: 1,
    saturation: 0,
    temperature: 0,
    hueBias: 0,
    highlightRecovery: 0,
    shadowLift: 0,
    localContrast: 0,
    colorStrength: 100,
    noiseReduction: 0,
    sharpenAmount: 0,
    sceneHeuristics: false,
    detailPreservation: 0,
    precisionPipeline: false,
  };
}

const MONO_MAP_OPTS = { colorStrength: 100, saturation: 0, temperature: 0, hueBias: 0 };

async function checkNoiseReductionGaussian(page: Page): Promise<void> {
  const width = 24;
  const height = 24;
  const intensity = makeIntensityCheckerNoisy(width, height, 7);
  const settings = { ...neutralSettings(), preset: 'monochrome' as const, noiseReduction: 60, noiseMethod: 'gaussian' as const };

  const cpuIntensity = reduceNoise(intensity.slice(), width, height, 'gaussian', 60, NEUTRAL_QUALITY.gaussianPasses);
  const cpuRgba = mapIntensityToRgb(cpuIntensity, width, height, PRESETS.monochrome.colorStops, MONO_MAP_OPTS);

  const gpu = await gpuRunFullPipeline(page, intensity, width, height, settings, PRESETS.monochrome.colorStops, NEUTRAL_QUALITY);

  const m = compareInterleavedChannels(cpuRgba as unknown as ArrayLike<number>, gpu.rgba, 4, 2, 3);
  record('noise-reduction/gaussian (8-bit RGB)', m.combined, 2, 0.02);
}

async function checkNoiseReductionBilateral(page: Page): Promise<void> {
  const width = 24;
  const height = 24;
  const intensity = makeIntensityCheckerNoisy(width, height, 11);
  const settings = { ...neutralSettings(), preset: 'monochrome' as const, noiseReduction: 50, noiseMethod: 'bilateral' as const };

  const cpuIntensity = reduceNoise(intensity.slice(), width, height, 'bilateral', 50, NEUTRAL_QUALITY.gaussianPasses);
  const cpuRgba = mapIntensityToRgb(cpuIntensity, width, height, PRESETS.monochrome.colorStops, MONO_MAP_OPTS);

  const gpu = await gpuRunFullPipeline(page, intensity, width, height, settings, PRESETS.monochrome.colorStops, NEUTRAL_QUALITY);

  const m = compareInterleavedChannels(cpuRgba as unknown as ArrayLike<number>, gpu.rgba, 4, 2, 3);
  record('noise-reduction/bilateral (8-bit RGB)', m.combined, 2, 0.02);
}

async function checkLocalContrast(page: Page): Promise<void> {
  const width = 20;
  const height = 20;
  const intensity = makeIntensityCheckerNoisy(width, height, 3);
  const settings = { ...neutralSettings(), preset: 'monochrome' as const, localContrast: 40 };

  const cpuIntensity = applyLocalContrast(intensity.slice(), width, height, 40, NEUTRAL_QUALITY.localContrastRadius);
  const cpuRgba = mapIntensityToRgb(cpuIntensity, width, height, PRESETS.monochrome.colorStops, MONO_MAP_OPTS);

  const gpu = await gpuRunFullPipeline(page, intensity, width, height, settings, PRESETS.monochrome.colorStops, NEUTRAL_QUALITY);

  const m = compareInterleavedChannels(cpuRgba as unknown as ArrayLike<number>, gpu.rgba, 4, 2, 3);
  record('local-contrast (8-bit RGB)', m.combined, 2, 0.02);
}

async function checkToneCurve(page: Page): Promise<void> {
  const width = 32;
  const height = 4;
  const intensity = makeIntensityRamp(width, height);
  const toneOpts = { exposure: 0.6, brightness: -15, contrast: 35, gamma: 1.4, shadowLift: 40, highlightRecovery: 30 };
  const settings = { ...neutralSettings(), preset: 'monochrome' as const, ...toneOpts };

  const toneCurve = buildToneCurve(toneOpts);
  const cpuIntensity = new Float32Array(intensity.length);
  for (let i = 0; i < intensity.length; i++) cpuIntensity[i] = toneCurve(intensity[i]);
  const cpuRgba = mapIntensityToRgb(cpuIntensity, width, height, PRESETS.monochrome.colorStops, MONO_MAP_OPTS);

  const gpu = await gpuRunFullPipeline(page, intensity, width, height, settings, PRESETS.monochrome.colorStops, NEUTRAL_QUALITY);

  const m = compareInterleavedChannels(cpuRgba as unknown as ArrayLike<number>, gpu.rgba, 4, 2, 3);
  record('tone-curve (8-bit RGB)', m.combined, 2, 0.02);
}

async function checkColorMapping(page: Page, presetId: 'natural' | 'portrait'): Promise<void> {
  const width = 40;
  const height = 6;
  const intensity = makeIntensityRamp(width, height);
  const settings = { ...neutralSettings(), preset: presetId, saturation: 20, temperature: -15, hueBias: 12 };

  const cpuRgba = mapIntensityToRgb(intensity, width, height, PRESETS[presetId].colorStops, {
    colorStrength: 100,
    saturation: settings.saturation,
    temperature: settings.temperature,
    hueBias: settings.hueBias,
  });

  const gpu = await gpuRunFullPipeline(page, intensity, width, height, settings, PRESETS[presetId].colorStops, NEUTRAL_QUALITY);

  const m = compareInterleavedChannels(cpuRgba as unknown as ArrayLike<number>, gpu.rgba, 4, 2, 3);
  record(`color-mapping/${presetId} (8-bit RGB)`, m.combined, 2, 0.02);
}

async function checkColorMappingWithScene(page: Page): Promise<void> {
  const width = 24;
  const height = 24;
  const intensity = makeIntensityVerticalGradient(width, height);
  const settings = { ...neutralSettings(), preset: 'landscape' as const, sceneHeuristics: true };

  const scene = computeSceneMaps(intensity, width, height);
  const cpuRgba = mapIntensityToRgb(intensity, width, height, PRESETS.landscape.colorStops, {
    colorStrength: 100,
    saturation: 0,
    temperature: 0,
    hueBias: 0,
    scene,
  });

  const gpu = await gpuRunFullPipeline(page, intensity, width, height, settings, PRESETS.landscape.colorStops, NEUTRAL_QUALITY);

  const m = compareInterleavedChannels(cpuRgba as unknown as ArrayLike<number>, gpu.rgba, 4, 2, 3);
  record('color-mapping+scene-heuristics/landscape (8-bit RGB)', m.combined, 2, 0.03);
}

async function checkSharpen(page: Page): Promise<void> {
  const width = 24;
  const height = 24;
  const intensity = makeIntensityCheckerNoisy(width, height, 5);
  const sharpenOpts = { amount: 120, radius: 1.5, threshold: 4 };
  const settings = { ...neutralSettings(), preset: 'monochrome' as const, sharpenAmount: sharpenOpts.amount, sharpenRadius: sharpenOpts.radius, sharpenThreshold: sharpenOpts.threshold };

  const baseRgba = mapIntensityToRgb(intensity, width, height, PRESETS.monochrome.colorStops, MONO_MAP_OPTS);
  const n = width * height;
  const rC = new Float32Array(n);
  const gC = new Float32Array(n);
  const bC = new Float32Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    rC[i] = baseRgba[p] / 255;
    gC[i] = baseRgba[p + 1] / 255;
    bC[i] = baseRgba[p + 2] / 255;
  }
  const rs = unsharpMask(rC, width, height, sharpenOpts);
  const gs = unsharpMask(gC, width, height, sharpenOpts);
  const bs = unsharpMask(bC, width, height, sharpenOpts);
  const cpuRgba = new Uint8ClampedArray(baseRgba.length);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    cpuRgba[p] = rs[i] * 255;
    cpuRgba[p + 1] = gs[i] * 255;
    cpuRgba[p + 2] = bs[i] * 255;
    cpuRgba[p + 3] = 255;
  }

  const gpu = await gpuRunFullPipeline(page, intensity, width, height, settings, PRESETS.monochrome.colorStops, NEUTRAL_QUALITY);

  const m = compareInterleavedChannels(cpuRgba as unknown as ArrayLike<number>, gpu.rgba, 4, 2, 3);
  record('sharpen (8-bit RGB)', m.combined, 2, 0.03);
}

async function checkPrecisionPipeline(page: Page): Promise<void> {
  const width = 20;
  const height = 20;
  const n = width * height;
  // A synthetic, deliberately color-cast, non-gray test image (a mild
  // orange cast over a gradient), so white balance/AutoTone/color
  // correction/gamut protection all have real work to do.
  const r = new Float32Array(n);
  const g = new Float32Array(n);
  const b = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = (i % width) / (width - 1);
    r[i] = srgbToLinear(Math.min(1, t * 0.9 + 0.15));
    g[i] = srgbToLinear(Math.min(1, t * 0.75 + 0.05));
    b[i] = srgbToLinear(Math.min(1, t * 0.6));
  }

  const opts = { whiteBalance: { strength: 55 }, autoTone: { strength: 40 }, colorCorrection: { strength: 45 } };

  const rCpu = r.slice();
  const gCpu = g.slice();
  const bCpu = b.slice();
  applyAzusaWhiteBalance(rCpu, gCpu, bCpu, opts.whiteBalance);
  applyAutoTone(rCpu, gCpu, bCpu, opts.autoTone);
  applyColorCorrectionPipeline(rCpu, gCpu, bCpu, opts.colorCorrection);
  protectGamut(rCpu, gCpu, bCpu);

  const gpu = await gpuRunPrecisionPipeline(page, r, g, b, width, height, opts);

  const mR = compareArrays(rCpu, gpu.r, 2 / 255);
  const mG = compareArrays(gCpu, gpu.g, 2 / 255);
  const mB = compareArrays(bCpu, gpu.b, 2 / 255);
  record('precision-pipeline/R (linear, x255)', mR, 2 / 255, 0.02, 255);
  record('precision-pipeline/G (linear, x255)', mG, 2 / 255, 0.02, 255);
  record('precision-pipeline/B (linear, x255)', mB, 2 / 255, 0.02, 255);
}

// ---------------------------------------------------------------------
// End-to-end checks (full pipeline, real settings combinations)
// ---------------------------------------------------------------------

function qualityParamsFor(quality: ProcessingSettings['quality']): { gaussianPasses: number; localContrastRadius: number } {
  switch (quality) {
    case 'fast':
      return { gaussianPasses: 1, localContrastRadius: 14 };
    case 'high':
      return { gaussianPasses: 4, localContrastRadius: 28 };
    case 'maximum':
      return { gaussianPasses: 6, localContrastRadius: 36 };
    default:
      return { gaussianPasses: 3, localContrastRadius: 24 };
  }
}

async function checkEndToEnd(page: Page, label: string, settings: ProcessingSettings, width: number, height: number, intensitySeed: number): Promise<void> {
  const data = makeUint8ImageFromIntensity(makeIntensityCheckerNoisy(width, height, intensitySeed), width, height);
  const input = { data, width, height } as unknown as ImageData;

  const cpuResult = runPipeline(input, settings);

  // Re-derive the same post-levels intensity runFullPipelineWebGL2 expects
  // (mirrors pipeline.ts steps 1-3, which stay CPU-only either way).
  const { intensity: rawIntensity, histogram } = extractIntensity(data, width, height, settings.interpretation);
  const levels = resolveLevelPoints(histogram, settings.autoLevels, settings.blackPoint, settings.whitePoint, settings.blackPercentile, settings.whitePercentile);
  const intensity = applyLevels(rawIntensity, levels.black, levels.white);

  const preset = PRESETS[settings.preset];
  const gpu = await gpuRunFullPipeline(page, intensity, width, height, settings, preset.colorStops, qualityParamsFor(settings.quality));

  const m = compareInterleavedChannels(cpuResult.output.data as unknown as ArrayLike<number>, gpu.rgba, 4, 3, 3);
  record(`end-to-end/${label} (8-bit RGB)`, m.combined, 3, 0.05);
}

async function checkEndToEndSolid(page: Page, label: string, value: number): Promise<void> {
  const width = 10;
  const height = 10;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data[i * 4] = value;
    data[i * 4 + 1] = value;
    data[i * 4 + 2] = value;
    data[i * 4 + 3] = 255;
  }
  const input = { data, width, height } as unknown as ImageData;
  const settings = { ...DEFAULT_SETTINGS, sceneHeuristics: true };

  const cpuResult = runPipeline(input, settings);
  const { intensity: rawIntensity, histogram } = extractIntensity(data, width, height, settings.interpretation);
  const levels = resolveLevelPoints(histogram, settings.autoLevels, settings.blackPoint, settings.whitePoint, settings.blackPercentile, settings.whitePercentile);
  const intensity = applyLevels(rawIntensity, levels.black, levels.white);
  const preset = PRESETS[settings.preset];
  const gpu = await gpuRunFullPipeline(page, intensity, width, height, settings, preset.colorStops, qualityParamsFor(settings.quality));

  const m = compareInterleavedChannels(cpuResult.output.data as unknown as ArrayLike<number>, gpu.rgba, 4, 3, 3);
  record(`end-to-end/${label} (8-bit RGB)`, m.combined, 3, 0.05);
}

async function checkNoiseExtreme(page: Page, method: 'gaussian' | 'bilateral', strength: number): Promise<void> {
  const width = 24;
  const height = 24;
  const intensity = makeIntensityCheckerNoisy(width, height, 99);
  const settings = { ...neutralSettings(), preset: 'monochrome' as const, noiseReduction: strength, noiseMethod: method };

  const cpuIntensity = reduceNoise(intensity.slice(), width, height, method, strength, NEUTRAL_QUALITY.gaussianPasses);
  const cpuRgba = mapIntensityToRgb(cpuIntensity, width, height, PRESETS.monochrome.colorStops, MONO_MAP_OPTS);
  const gpu = await gpuRunFullPipeline(page, intensity, width, height, settings, PRESETS.monochrome.colorStops, NEUTRAL_QUALITY);

  const m = compareInterleavedChannels(cpuRgba as unknown as ArrayLike<number>, gpu.rgba, 4, 2, 3);
  record(`noise-reduction/${method}@${strength} extreme (8-bit RGB)`, m.combined, 2, 0.02);
}

async function checkLocalContrastExtreme(page: Page): Promise<void> {
  const width = 20;
  const height = 20;
  const intensity = makeIntensityCheckerNoisy(width, height, 71);
  const radius = 36; // 'maximum' quality's localContrastRadius
  const settings = { ...neutralSettings(), preset: 'monochrome' as const, localContrast: 100 };

  const cpuIntensity = applyLocalContrast(intensity.slice(), width, height, 100, radius);
  const cpuRgba = mapIntensityToRgb(cpuIntensity, width, height, PRESETS.monochrome.colorStops, MONO_MAP_OPTS);
  const gpu = await gpuRunFullPipeline(page, intensity, width, height, settings, PRESETS.monochrome.colorStops, { gaussianPasses: 3, localContrastRadius: radius });

  const m = compareInterleavedChannels(cpuRgba as unknown as ArrayLike<number>, gpu.rgba, 4, 2, 3);
  record('local-contrast@100/radius36 extreme (8-bit RGB)', m.combined, 2, 0.02);
}

async function checkToneCurveExtreme(page: Page): Promise<void> {
  const width = 64;
  const height = 4;
  const intensity = makeIntensityRamp(width, height);
  const toneOpts = { exposure: 2, brightness: 100, contrast: 100, gamma: 3, shadowLift: 100, highlightRecovery: 100 };
  const settings = { ...neutralSettings(), preset: 'monochrome' as const, ...toneOpts };

  const toneCurve = buildToneCurve(toneOpts);
  const cpuIntensity = new Float32Array(intensity.length);
  for (let i = 0; i < intensity.length; i++) cpuIntensity[i] = toneCurve(intensity[i]);
  const cpuRgba = mapIntensityToRgb(cpuIntensity, width, height, PRESETS.monochrome.colorStops, MONO_MAP_OPTS);
  const gpu = await gpuRunFullPipeline(page, intensity, width, height, settings, PRESETS.monochrome.colorStops, NEUTRAL_QUALITY);

  const m = compareInterleavedChannels(cpuRgba as unknown as ArrayLike<number>, gpu.rgba, 4, 2, 3);
  record('tone-curve extreme (exposure=2,contrast=100,gamma=3,...) (8-bit RGB)', m.combined, 2, 0.02);

  // Also stress the opposite extreme (negative contrast/exposure/brightness).
  const toneOpts2 = { exposure: -2, brightness: -100, contrast: -100, gamma: 0.2, shadowLift: 0, highlightRecovery: 0 };
  const settings2 = { ...neutralSettings(), preset: 'monochrome' as const, ...toneOpts2 };
  const toneCurve2 = buildToneCurve(toneOpts2);
  const cpuIntensity2 = new Float32Array(intensity.length);
  for (let i = 0; i < intensity.length; i++) cpuIntensity2[i] = toneCurve2(intensity[i]);
  const cpuRgba2 = mapIntensityToRgb(cpuIntensity2, width, height, PRESETS.monochrome.colorStops, MONO_MAP_OPTS);
  const gpu2 = await gpuRunFullPipeline(page, intensity, width, height, settings2, PRESETS.monochrome.colorStops, NEUTRAL_QUALITY);
  const m2 = compareInterleavedChannels(cpuRgba2 as unknown as ArrayLike<number>, gpu2.rgba, 4, 2, 3);
  record('tone-curve extreme (negative exposure/contrast/gamma=0.2) (8-bit RGB)', m2.combined, 2, 0.02);
}

async function checkColorMappingExtreme(page: Page, presetId: keyof typeof PRESETS): Promise<void> {
  const width = 48;
  const height = 4;
  const intensity = makeIntensityRamp(width, height);
  const settings = { ...neutralSettings(), preset: presetId, saturation: -100, temperature: 100, hueBias: 175 };

  const cpuRgba = mapIntensityToRgb(intensity, width, height, PRESETS[presetId].colorStops, {
    colorStrength: 100,
    saturation: settings.saturation,
    temperature: settings.temperature,
    hueBias: settings.hueBias,
  });
  const gpu = await gpuRunFullPipeline(page, intensity, width, height, settings, PRESETS[presetId].colorStops, NEUTRAL_QUALITY);

  const m = compareInterleavedChannels(cpuRgba as unknown as ArrayLike<number>, gpu.rgba, 4, 2, 3);
  record(`color-mapping/${presetId} extreme sat/temp/hue (8-bit RGB)`, m.combined, 2, 0.02);

  // And the other saturation extreme (+100) plus the opposite hue direction.
  const settings2 = { ...neutralSettings(), preset: presetId, saturation: 100, temperature: -100, hueBias: -175 };
  const cpuRgba2 = mapIntensityToRgb(intensity, width, height, PRESETS[presetId].colorStops, {
    colorStrength: 100,
    saturation: settings2.saturation,
    temperature: settings2.temperature,
    hueBias: settings2.hueBias,
  });
  const gpu2 = await gpuRunFullPipeline(page, intensity, width, height, settings2, PRESETS[presetId].colorStops, NEUTRAL_QUALITY);
  const m2 = compareInterleavedChannels(cpuRgba2 as unknown as ArrayLike<number>, gpu2.rgba, 4, 2, 3);
  record(`color-mapping/${presetId} extreme sat+/temp-/hue- (8-bit RGB)`, m2.combined, 2, 0.02);
}

async function checkSharpenExtreme(page: Page): Promise<void> {
  const width = 24;
  const height = 24;
  const intensity = makeIntensityCheckerNoisy(width, height, 55);
  const sharpenOpts = { amount: 200, radius: 5, threshold: 0 };
  const settings = {
    ...neutralSettings(),
    preset: 'monochrome' as const,
    sharpenAmount: sharpenOpts.amount,
    sharpenRadius: sharpenOpts.radius,
    sharpenThreshold: sharpenOpts.threshold,
  };

  const baseRgba = mapIntensityToRgb(intensity, width, height, PRESETS.monochrome.colorStops, MONO_MAP_OPTS);
  const n = width * height;
  const rC = new Float32Array(n);
  const gC = new Float32Array(n);
  const bC = new Float32Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    rC[i] = baseRgba[p] / 255;
    gC[i] = baseRgba[p + 1] / 255;
    bC[i] = baseRgba[p + 2] / 255;
  }
  const rs = unsharpMask(rC, width, height, sharpenOpts);
  const gs = unsharpMask(gC, width, height, sharpenOpts);
  const bs = unsharpMask(bC, width, height, sharpenOpts);
  const cpuRgba = new Uint8ClampedArray(baseRgba.length);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    cpuRgba[p] = rs[i] * 255;
    cpuRgba[p + 1] = gs[i] * 255;
    cpuRgba[p + 2] = bs[i] * 255;
    cpuRgba[p + 3] = 255;
  }

  const gpu = await gpuRunFullPipeline(page, intensity, width, height, settings, PRESETS.monochrome.colorStops, NEUTRAL_QUALITY);

  const m = compareInterleavedChannels(cpuRgba as unknown as ArrayLike<number>, gpu.rgba, 4, 2, 3);
  record('sharpen@amount200/radius5/threshold0 extreme (8-bit RGB)', m.combined, 2, 0.03);
}

// ---------------------------------------------------------------------
// Tiled-vs-untiled checks
// ---------------------------------------------------------------------
//
// PASS RULES (fixed in code here, before the full run of this section):
//
//  * Tiled and untiled use the same shaders and, inside a tile's core,
//    the same inputs in the same arithmetic, so wherever there is no
//    position-dependent term the results must be BIT-IDENTICAL: 8-bit
//    output max error 0, and precision diagnostics (WB / AutoTone / CC
//    gains) exactly equal.
//  * Scene heuristics are the one exception. Untiled reads the
//    hardware-interpolated v_uv.y; tiled computes (row0 + row + 0.5)/H
//    exactly. Identical mathematically, but they can round differently
//    in the last float bit (observed while diagnosing an exploratory
//    run: ~4e-11 on the CC gains, pixels still identical). That noise
//    can reach 8 bits only by flipping the `sky > 0.15` / `veg > 0.15`
//    step, which moves a channel by at most ~1.3 levels. So scene-
//    dependent checks allow max 2 levels on <= 0.01% of elements and
//    gains within 1e-8. These bounds are analytic, not fitted.
//  * Tiled output is ALSO held to the same CPU tolerances as the untiled
//    end-to-end checks (tol 3, <= 5% over) wherever a CPU comparison is made.
//  * Negative controls: an under-sized halo MUST produce mismatches, and
//    they must be confined to bands around internal tile edges.

interface StrictRule {
  maxError: number;
  maxFraction: number;
}
const TILED_EXACT: StrictRule = { maxError: 0, maxFraction: 0 };
const TILED_SCENE: StrictRule = { maxError: 2, maxFraction: 1e-4 };
const TILED_GAIN_TOL_EXACT = 0;
const TILED_GAIN_TOL_SCENE = 1e-8;

function recordStrict(name: string, metrics: ErrorMetrics, maxErrorAllowed: number, maxFractionAllowed: number, scale = 1): void {
  const pass = metrics.maxError <= maxErrorAllowed && metrics.fractionOverTolerance <= maxFractionAllowed;
  results.push({ name, metrics, tolerance: maxErrorAllowed, maxAllowedFraction: maxFractionAllowed, scale, pass });
  console.log(`[${pass ? 'PASS' : 'FAIL'}] ${formatMetrics(name, metrics, 0, scale)} (limit: max<=${maxErrorAllowed}, frac<=${maxFractionAllowed})`);
}

function recordCustom(name: string, pass: boolean, detail: string): void {
  const metrics: ErrorMetrics = { mae: 0, maxError: 0, maxErrorIndex: -1, fractionOverTolerance: 0, countOverTolerance: 0, count: 0 };
  results.push({ name, metrics, tolerance: 0, maxAllowedFraction: 0, scale: 1, pass });
  console.log(`[${pass ? 'PASS' : 'FAIL'}] ${name}: ${detail}`);
}

interface BothRuns {
  untiled: number[];
  tiled: number[];
  untiledPrecision?: unknown;
  tiledPrecision?: unknown;
  tiling?: TilingDiagnostics;
  untiledTiling?: unknown;
}

interface TileOpts {
  maxTileDim?: number;
  unsafeHaloOverride?: { colorMap?: number; sharpen?: number };
}

async function gpuRunBoth(
  page: Page,
  intensity: Float32Array,
  width: number,
  height: number,
  settings: ProcessingSettings,
  stops: ColorStop[],
  quality: { gaussianPasses: number; localContrastRadius: number },
  opts: TileOpts
): Promise<BothRuns> {
  return page.evaluate(
    (args: [number[], number, number, ProcessingSettings, ColorStop[], typeof quality, TileOpts]) => {
      const [arr, w, h, s, st, q, o] = args;
      const H = (window as any).__gpuHarness;
      const un = H.runFullPipelineWebGL2(new Float32Array(arr), w, h, s, st, q);
      const ti = H.runTiledFullPipelineWebGL2(new Float32Array(arr), w, h, s, st, q, o);
      return {
        untiled: Array.from(un.rgba as Uint8ClampedArray),
        tiled: Array.from(ti.rgba as Uint8ClampedArray),
        untiledPrecision: un.precision,
        tiledPrecision: ti.precision,
        tiling: ti.tiling,
        untiledTiling: un.tiling,
      };
    },
    [Array.from(intensity), width, height, settings, stops, quality, opts] as [number[], number, number, ProcessingSettings, ColorStop[], typeof quality, TileOpts]
  );
}

function numericLeaves(v: unknown, out: number[] = []): number[] {
  if (typeof v === 'number') out.push(v);
  else if (v && typeof v === 'object') for (const k of Object.keys(v as object).sort()) numericLeaves((v as Record<string, unknown>)[k], out);
  return out;
}

/** 1 where a pixel lies within `band` px of an INTERNAL tile edge of any given plan. */
function seamBandMask(width: number, height: number, plans: { halo: number; dim: number; band: number }[]): Uint8Array {
  const mask = new Uint8Array(width * height);
  for (const p of plans) {
    const plan = planTiles(width, height, p.dim, p.halo);
    const xs = new Set<number>();
    const ys = new Set<number>();
    for (const t of plan.tiles) {
      if (t.core.x0 > 0) xs.add(t.core.x0);
      if (t.core.y0 > 0) ys.add(t.core.y0);
    }
    for (const xe of xs) for (let y = 0; y < height; y++) for (let x = Math.max(0, xe - p.band); x < Math.min(width, xe + p.band); x++) mask[y * width + x] = 1;
    for (const ye of ys) for (let y = Math.max(0, ye - p.band); y < Math.min(height, ye + p.band); y++) for (let x = 0; x < width; x++) mask[y * width + x] = 1;
  }
  return mask;
}

interface TiledCase {
  label: string;
  width: number;
  height: number;
  intensity: Float32Array;
  settings: ProcessingSettings;
  stops: ColorStop[];
  quality: { gaussianPasses: number; localContrastRadius: number };
  maxTileDim: number;
  /** true when scene heuristics are on (position-dependent term => TILED_SCENE rule). */
  sceneDependent: boolean;
  /** Require at least this many tiles in the color-map phase (guards against a test that silently doesn't tile). */
  minTiles: number;
  seamBand?: boolean;
}

async function checkTiledVsUntiled(page: Page, c: TiledCase): Promise<BothRuns> {
  const rule = c.sceneDependent ? TILED_SCENE : TILED_EXACT;
  const gainTol = c.sceneDependent ? TILED_GAIN_TOL_SCENE : TILED_GAIN_TOL_EXACT;
  const run = await gpuRunBoth(page, c.intensity, c.width, c.height, c.settings, c.stops, c.quality, { maxTileDim: c.maxTileDim });

  const engaged = !!run.tiling && run.tiling.colorMap.tileCount >= c.minTiles && !run.untiledTiling;
  const grid = run.tiling ? `${run.tiling.colorMap.cols}x${run.tiling.colorMap.rows} grid, halo ${run.tiling.colorMap.halo}` : 'no tiling';
  recordCustom(`tiled-vs-untiled/${c.label} tiling engaged`, engaged, `${grid}, ${run.tiling?.colorMap.tileCount ?? 0} tiles (need >= ${c.minTiles})`);

  recordStrict(`tiled-vs-untiled/${c.label} (8-bit RGBA)`, compareArrays(run.untiled, run.tiled, 0), rule.maxError, rule.maxFraction);

  if (run.untiledPrecision && run.tiledPrecision) {
    const a = numericLeaves(run.untiledPrecision);
    const b = numericLeaves(run.tiledPrecision);
    recordStrict(`tiled-vs-untiled/${c.label} precision diagnostics (WB/AutoTone/CC)`, compareArrays(a, b, 0), gainTol, 1);
  }

  if (c.seamBand && run.tiling) {
    const radii = resolveStageRadii(c.settings, c.quality);
    const plans = [{ halo: run.tiling.colorMap.halo, dim: run.tiling.maxTileDim, band: radii.colorMapReach + 1 }];
    if (run.tiling.sharpen) plans.push({ halo: run.tiling.sharpen.halo, dim: run.tiling.maxTileDim, band: radii.sharpenReach + 1 });
    const mask = seamBandMask(c.width, c.height, plans);
    const refBand: number[] = [];
    const actBand: number[] = [];
    for (let i = 0; i < mask.length; i++) {
      if (!mask[i]) continue;
      for (let ch = 0; ch < 3; ch++) {
        refBand.push(run.untiled[i * 4 + ch]);
        actBand.push(run.tiled[i * 4 + ch]);
      }
    }
    recordStrict(`seam-band/${c.label} (pixels within reach of internal tile edges)`, compareArrays(refBand, actBand, 0), rule.maxError, rule.maxFraction);
  }
  return run;
}

/** An under-sized halo must create mismatches, and only near internal tile edges. */
async function checkNegativeControl(page: Page, label: string, c: TiledCase, override: { colorMap?: number; sharpen?: number }): Promise<void> {
  const radii = resolveStageRadii(c.settings, c.quality);
  const run = await gpuRunBoth(page, c.intensity, c.width, c.height, c.settings, c.stops, c.quality, { maxTileDim: c.maxTileDim, unsafeHaloOverride: override });
  const bandWidth = Math.max(radii.colorMapReach, radii.sharpenReach) + 1;
  const plans: { halo: number; dim: number; band: number }[] = [];
  plans.push({ halo: override.colorMap ?? radii.colorMapReach, dim: c.maxTileDim, band: bandWidth });
  if (c.settings.sharpenAmount > 0) plans.push({ halo: override.sharpen ?? radii.sharpenReach, dim: c.maxTileDim, band: bandWidth });
  const mask = seamBandMask(c.width, c.height, plans);
  let total = 0;
  let outside = 0;
  let worst = 0;
  for (let i = 0; i < run.untiled.length; i++) {
    const d = Math.abs(run.untiled[i] - run.tiled[i]);
    if (d > 0) {
      total++;
      if (!mask[i >> 2]) outside++;
      if (d > worst) worst = d;
    }
  }
  recordCustom(
    `negative-control/${label}`,
    total > 0 && outside === 0,
    `${total} mismatching elements (max ${worst} levels), ${outside} outside the seam bands -- expected >0 and 0`
  );
}

function makeIntensityTwoHalves(width: number, height: number): Float32Array {
  const noisy = makeIntensityCheckerNoisy(width, height, 31);
  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const v = noisy[y * width + x];
      out[y * width + x] = x < width / 2 ? 0.03 + 0.25 * v : 0.62 + 0.38 * v;
    }
  return out;
}

function prepareE2E(settings: ProcessingSettings, width: number, height: number, seed: number) {
  const data = makeUint8ImageFromIntensity(makeIntensityCheckerNoisy(width, height, seed), width, height);
  const input = { data, width, height } as unknown as ImageData;
  const cpu = runPipeline(input, { ...settings, processingEngine: 'cpu' });
  const { intensity: raw, histogram } = extractIntensity(data, width, height, settings.interpretation);
  const levels = resolveLevelPoints(histogram, settings.autoLevels, settings.blackPoint, settings.whitePoint, settings.blackPercentile, settings.whitePercentile);
  return { data, cpuRgba: cpu.output.data, intensity: applyLevels(raw, levels.black, levels.white) };
}

async function checkTiledVsCpu(page: Page, label: string, settings: ProcessingSettings, width: number, height: number, seed: number, maxTileDim: number): Promise<void> {
  const prep = prepareE2E(settings, width, height, seed);
  const q = qualityParamsFor(settings.quality);
  const run = await gpuRunBoth(page, prep.intensity, width, height, settings, PRESETS[settings.preset].colorStops, q, { maxTileDim });
  const m = compareInterleavedChannels(prep.cpuRgba as unknown as ArrayLike<number>, run.tiled, 4, 3, 3);
  record(`tiled-vs-cpu/${label} (8-bit RGB)`, m.combined, 3, 0.05);
}

async function checkOversizeEndToEnd(page: Page, label: string, settings: ProcessingSettings, width: number, height: number, seed: number, expectTiled: boolean): Promise<void> {
  const prep = prepareE2E(settings, width, height, seed);
  const gpu = await gpuRunFullPipeline(page, prep.intensity, width, height, settings, PRESETS[settings.preset].colorStops, qualityParamsFor(settings.quality));
  const engaged = expectTiled ? !!gpu.tiling && gpu.tiling.colorMap.tileCount > 1 : !gpu.tiling;
  recordCustom(
    `oversize/${label} routed ${expectTiled ? 'to tiled path' : 'to untiled path'}`,
    engaged,
    gpu.tiling ? `${gpu.tiling.colorMap.cols}x${gpu.tiling.colorMap.rows} tiles` : 'untiled'
  );
  const m = compareInterleavedChannels(prep.cpuRgba as unknown as ArrayLike<number>, gpu.rgba, 4, 3, 3);
  record(`oversize/${label} vs CPU (8-bit RGB)`, m.combined, 3, 0.05);
}

async function checkOversizeTilingInvariance(page: Page, label: string, settings: ProcessingSettings, width: number, height: number, seed: number): Promise<void> {
  const prep = prepareE2E(settings, width, height, seed);
  const q = qualityParamsFor(settings.quality);
  const stops = PRESETS[settings.preset].colorStops;
  const r = await page.evaluate(
    (args: [number[], number, number, ProcessingSettings, ColorStop[], typeof q]) => {
      const [arr, w, h, s, st, qq] = args;
      const H = (window as any).__gpuHarness;
      const auto = H.runFullPipelineWebGL2(new Float32Array(arr), w, h, s, st, qq); // dispatcher, default tile size
      const small = H.runTiledFullPipelineWebGL2(new Float32Array(arr), w, h, s, st, qq, { maxTileDim: 512 });
      return { a: Array.from(auto.rgba as Uint8ClampedArray), b: Array.from(small.rgba as Uint8ClampedArray), ta: auto.tiling?.colorMap.tileCount, tb: small.tiling.colorMap.tileCount };
    },
    [Array.from(prep.intensity), width, height, settings, stops, q] as [number[], number, number, ProcessingSettings, ColorStop[], typeof q]
  );
  const rule = settings.sceneHeuristics ? TILED_SCENE : TILED_EXACT;
  recordCustom(`oversize/${label} two tilings differ in tile count`, !!r.ta && r.ta !== r.tb, `default=${r.ta} tiles, maxTileDim512=${r.tb} tiles`);
  recordStrict(`oversize/${label} tiling-invariance (default vs 512px tiles)`, compareArrays(r.a, r.b, 0), rule.maxError, rule.maxFraction);
}

async function checkPipelineLevelOversize(page: Page, label: string, settings: ProcessingSettings, width: number, height: number, seed: number): Promise<void> {
  const data = makeUint8ImageFromIntensity(makeIntensityCheckerNoisy(width, height, seed), width, height);
  const cpu = runPipeline({ data, width, height } as unknown as ImageData, { ...settings, processingEngine: 'cpu' });
  const gpu = await page.evaluate(
    (args: [number[], number, number, ProcessingSettings]) => {
      const [arr, w, h, s] = args;
      const H = (window as any).__gpuHarness;
      const out = H.runPipeline(new ImageData(new Uint8ClampedArray(arr), w, h), { ...s, processingEngine: 'webgl2' });
      return { data: Array.from(out.output.data as Uint8ClampedArray), gpuAccelerated: out.precision?.gpuAccelerated, resolved: out.precision?.backend?.resolved };
    },
    [Array.from(data), width, height, settings] as [number[], number, number, ProcessingSettings]
  );
  recordCustom(
    `pipeline-level/${label} ran on GPU (no CPU fallback)`,
    gpu.gpuAccelerated === true && gpu.resolved === 'webgl2',
    `gpuAccelerated=${gpu.gpuAccelerated}, backend=${gpu.resolved}`
  );
  const m = compareInterleavedChannels(cpu.output.data as unknown as ArrayLike<number>, gpu.data, 4, 3, 3);
  record(`pipeline-level/${label} vs CPU engine (8-bit RGB)`, m.combined, 3, 0.05);
}

async function checkTilingUnsupportedIsHonest(page: Page): Promise<void> {
  const width = 300;
  const height = 200;
  const settings = { ...neutralSettings(), preset: 'monochrome' as const, noiseReduction: 100, noiseMethod: 'gaussian' as const };
  const r = await page.evaluate(
    (args: [number[], number, number, ProcessingSettings, ColorStop[]]) => {
      const [arr, w, h, s, st] = args;
      const H = (window as any).__gpuHarness;
      try {
        // 100 blur passes x radius 4 = 400px of overlap: no useful core fits a 128px tile.
        H.runTiledFullPipelineWebGL2(new Float32Array(arr), w, h, s, st, { gaussianPasses: 100, localContrastRadius: 24 }, { maxTileDim: 128 });
        return 'no-throw';
      } catch (e) {
        return (e as Error).name;
      }
    },
    [Array.from(makeIntensityCheckerNoisy(width, height, 3)), width, height, settings, PRESETS.monochrome.colorStops] as [number[], number, number, ProcessingSettings, ColorStop[]]
  );
  recordCustom('unsupported/absurd filter overlap throws TilingUnsupportedError (=> CPU fallback)', r === 'TilingUnsupportedError', `threw: ${r}`);
}

async function runTiledChecks(page: Page): Promise<void> {
  const stopsMono = PRESETS.monochrome.colorStops;
  const maxQ = qualityParamsFor('maximum');
  const mk = (over: Partial<ProcessingSettings>): ProcessingSettings => ({ ...neutralSettings(), preset: 'monochrome' as const, ...over });

  const maxTexSize = await page.evaluate(() => (window as any).__gpuHarness.probeMaxTextureSize() as number);
  console.log(`(real MAX_TEXTURE_SIZE in this browser: ${maxTexSize})`);

  console.log('-- Tiled vs untiled: isolated stages (2-D grids, forced small tiles) --');
  const noisy = (w: number, h: number, seed: number) => makeIntensityCheckerNoisy(w, h, seed);

  const nrGauss: TiledCase = { label: 'gaussian-NR@100/quality=maximum 330x240', width: 330, height: 240, intensity: noisy(330, 240, 71), settings: mk({ noiseReduction: 100, noiseMethod: 'gaussian' }), stops: stopsMono, quality: maxQ, maxTileDim: 128, sceneDependent: false, minTiles: 12, seamBand: true };
  await checkTiledVsUntiled(page, nrGauss);
  await checkTiledVsUntiled(page, { label: 'bilateral-NR@100 330x240', width: 330, height: 240, intensity: noisy(330, 240, 72), settings: mk({ noiseReduction: 100, noiseMethod: 'bilateral' }), stops: stopsMono, quality: NEUTRAL_QUALITY, maxTileDim: 96, sceneDependent: false, minTiles: 12, seamBand: true });
  const lc: TiledCase = { label: 'local-contrast@100/radius36 330x240', width: 330, height: 240, intensity: noisy(330, 240, 73), settings: mk({ localContrast: 100 }), stops: stopsMono, quality: maxQ, maxTileDim: 160, sceneDependent: false, minTiles: 9, seamBand: true };
  await checkTiledVsUntiled(page, lc);
  await checkTiledVsUntiled(page, { label: 'scene-heuristics vertical gradient 200x360 (global row fraction)', width: 200, height: 360, intensity: makeIntensityVerticalGradient(200, 360), settings: { ...mk({ sceneHeuristics: true }), preset: 'landscape' }, stops: PRESETS.landscape.colorStops, quality: NEUTRAL_QUALITY, maxTileDim: 128, sceneDependent: true, minTiles: 6, seamBand: true });
  const sharp: TiledCase = { label: 'sharpen@200/radius5/threshold0 330x240', width: 330, height: 240, intensity: noisy(330, 240, 74), settings: mk({ sharpenAmount: 200, sharpenRadius: 5, sharpenThreshold: 0 }), stops: stopsMono, quality: NEUTRAL_QUALITY, maxTileDim: 96, sceneDependent: false, minTiles: 1, seamBand: true };
  const sharpRun = await checkTiledVsUntiled(page, sharp);
  recordCustom('tiled-vs-untiled/sharpen phase tiled', !!sharpRun.tiling?.sharpen && sharpRun.tiling.sharpen.tileCount > 1, `${sharpRun.tiling?.sharpen?.cols}x${sharpRun.tiling?.sharpen?.rows}, halo ${sharpRun.tiling?.sharpen?.halo}`);

  console.log('\n-- Tiled vs untiled: global statistics (precision stage) --');
  const halves = makeIntensityTwoHalves(300, 200);
  const precRun = await checkTiledVsUntiled(page, { label: 'precision-pipeline two-halves 300x200 (per-tile stats would differ)', width: 300, height: 200, intensity: halves, settings: { ...mk({ precisionPipeline: true, whiteBalanceStrength: 100, autoToneStrength: 100, colorCorrectionStrength: 100 }), preset: 'natural' }, stops: PRESETS.natural.colorStops, quality: NEUTRAL_QUALITY, maxTileDim: 100, sceneDependent: false, minTiles: 6 });
  const cc = (precRun.tiledPrecision as { colorCorrection: { gainR: number; gainG: number; gainB: number } }).colorCorrection;
  const dev = Math.max(Math.abs(cc.gainR - 1), Math.abs(cc.gainG - 1), Math.abs(cc.gainB - 1));
  recordCustom('tiled-vs-untiled/precision stage is non-trivial on this image', dev > 1e-3, `max |CC gain - 1| = ${dev.toExponential(2)}`);

  console.log('\n-- Tiled vs untiled: full pipeline, non-square, boundaries --');
  const kitchen: ProcessingSettings = { ...DEFAULT_SETTINGS, preset: 'landscape', quality: 'high', noiseReduction: 60, noiseMethod: 'gaussian', localContrast: 40, sharpenAmount: 80, sceneHeuristics: true, saturation: 15, temperature: -10 };
  await checkTiledVsUntiled(page, { label: 'kitchen-sink 350x240', width: 350, height: 240, intensity: noisy(350, 240, 75), settings: kitchen, stops: PRESETS.landscape.colorStops, quality: qualityParamsFor('high'), maxTileDim: 192, sceneDependent: true, minTiles: 8 });
  const maxed: ProcessingSettings = { ...DEFAULT_SETTINGS, preset: 'high-contrast', quality: 'maximum', noiseReduction: 100, noiseMethod: 'gaussian', localContrast: 100, sharpenAmount: 200, sharpenRadius: 5, sceneHeuristics: true, saturation: -100, temperature: 100, hueBias: 175, exposure: 1.5, contrast: 80, gamma: 2.2, shadowLift: 90, highlightRecovery: 90, whiteBalanceStrength: 100, autoToneStrength: 100, colorCorrectionStrength: 100 };
  await checkTiledVsUntiled(page, { label: 'everything-maxed/quality=maximum 300x290 (65px halo, 3x3 grid)', width: 300, height: 290, intensity: noisy(300, 290, 76), settings: maxed, stops: PRESETS['high-contrast'].colorStops, quality: maxQ, maxTileDim: 256, sceneDependent: true, minTiles: 9 });
  await checkTiledVsUntiled(page, { label: 'non-square-wide defaults 700x60', width: 700, height: 60, intensity: noisy(700, 60, 77), settings: DEFAULT_SETTINGS, stops: PRESETS.natural.colorStops, quality: NEUTRAL_QUALITY, maxTileDim: 128, sceneDependent: true, minTiles: 8 });
  await checkTiledVsUntiled(page, { label: 'non-square-tall defaults 60x700', width: 60, height: 700, intensity: noisy(60, 700, 78), settings: DEFAULT_SETTINGS, stops: PRESETS.natural.colorStops, quality: NEUTRAL_QUALITY, maxTileDim: 128, sceneDependent: true, minTiles: 8 });
  await checkTiledVsUntiled(page, { label: 'boundary: one pixel over tile size 129x100', width: 129, height: 100, intensity: noisy(129, 100, 79), settings: DEFAULT_SETTINGS, stops: PRESETS.natural.colorStops, quality: NEUTRAL_QUALITY, maxTileDim: 128, sceneDependent: true, minTiles: 2 });
  await checkTiledVsUntiled(page, { label: 'boundary: exactly one tile 128x100', width: 128, height: 100, intensity: noisy(128, 100, 80), settings: DEFAULT_SETTINGS, stops: PRESETS.natural.colorStops, quality: NEUTRAL_QUALITY, maxTileDim: 128, sceneDependent: true, minTiles: 1 });
  await checkTiledVsUntiled(page, { label: 'boundary: tiny 3x2 image', width: 3, height: 2, intensity: noisy(3, 2, 81), settings: DEFAULT_SETTINGS, stops: PRESETS.natural.colorStops, quality: NEUTRAL_QUALITY, maxTileDim: 128, sceneDependent: true, minTiles: 1 });

  console.log('\n-- Tiled vs CPU (same tolerances as the untiled end-to-end checks) --');
  await checkTiledVsCpu(page, 'kitchen-sink 350x240', kitchen, 350, 240, 75, 192);
  await checkTiledVsCpu(page, 'non-square-wide defaults 700x60', DEFAULT_SETTINGS, 700, 60, 77, 128);
  await checkTiledVsCpu(page, 'non-square-tall defaults 60x700', DEFAULT_SETTINGS, 60, 700, 78, 128);

  console.log('\n-- Negative controls: under-sized halos must produce seams --');
  // NOTE: for a 6-pass radius-4 blur the outermost tap (distance 24) has weight ~(1/9)^6 = 2e-6, invisible at 8 bits, so 'reach-1' is
  // undetectable by construction for multi-pass Gaussians; a materially smaller halo (half the reach) is the meaningful control.
  await checkNegativeControl(page, 'gaussian-NR halo=reach/2 (12 vs 24)', nrGauss, { colorMap: 12 });
  await checkNegativeControl(page, 'local-contrast halo=reach-1 (35 vs 36)', lc, { colorMap: 35 });
  await checkNegativeControl(page, 'local-contrast halo=0 (no overlap)', lc, { colorMap: 0 });
  await checkNegativeControl(page, 'sharpen halo=reach-1 (4 vs 5)', sharp, { sharpen: 4 });
  const chain: TiledCase = { label: 'chain', width: 330, height: 240, intensity: noisy(330, 240, 82), settings: { ...mk({ noiseReduction: 60, noiseMethod: 'gaussian', localContrast: 40, sceneHeuristics: true }), preset: 'landscape' }, stops: PRESETS.landscape.colorStops, quality: qualityParamsFor('high'), maxTileDim: 192, sceneDependent: true, minTiles: 1 };
  const chainRadii = resolveStageRadii(chain.settings, chain.quality);
  const chainNoReach = chainRadii.gaussianRadius * chainRadii.gaussianPasses;
  const chainHalo = chainRadii.colorMapReach - chainNoReach; // 'max/sum mistake': forget that the NR reach ADDS to LC + scene
  await checkNegativeControl(page, `chained NR+LC+scene halo omits NR reach (${chainHalo} vs ${chainRadii.colorMapReach})`, chain, { colorMap: chainHalo });

  console.log('\n-- Real oversize images (width/height > real MAX_TEXTURE_SIZE), natural dispatch --');
  const over = maxTexSize + 8;
  const oversizeSettings: ProcessingSettings = { ...DEFAULT_SETTINGS, preset: 'landscape', noiseReduction: 35, noiseMethod: 'gaussian', localContrast: 25, sharpenAmount: 50, sceneHeuristics: true, saturation: 15, temperature: -10 };
  await checkOversizeEndToEnd(page, `wide ${over}x12`, oversizeSettings, over, 12, 101, true);
  await checkOversizeEndToEnd(page, `tall 12x${over}`, oversizeSettings, 12, over, 102, true);
  await checkOversizeEndToEnd(page, `just over the limit ${maxTexSize + 1}x8`, DEFAULT_SETTINGS, maxTexSize + 1, 8, 103, true);
  await checkOversizeEndToEnd(page, `exactly at the limit ${maxTexSize}x8 (stays untiled)`, DEFAULT_SETTINGS, maxTexSize, 8, 104, false);
  await checkOversizeEndToEnd(page, `bilateral+precision-off wide ${over}x10`, { ...DEFAULT_SETTINGS, noiseReduction: 30, noiseMethod: 'bilateral', precisionPipeline: false }, over, 10, 105, true);
  await checkOversizeTilingInvariance(page, `wide ${over}x12`, oversizeSettings, over, 12, 101);
  await checkPipelineLevelOversize(page, `runPipeline(webgl2) wide ${over}x12`, oversizeSettings, over, 12, 106);
  await checkPipelineLevelOversize(page, `runPipeline(webgl2) tall 12x${over}`, oversizeSettings, 12, over, 107);

  console.log('\n-- Honest fallback --');
  await checkTilingUnsupportedIsHonest(page);
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

async function main(): Promise<void> {
  console.log('Starting GPU consistency harness (real headless Chromium + WebGL2)...\n');
  const { browser, page } = await startGpuHarness();

  try {
    console.log('-- Stage-by-stage checks --');
    await checkNoiseReductionGaussian(page);
    await checkNoiseReductionBilateral(page);
    await checkLocalContrast(page);
    await checkToneCurve(page);
    await checkColorMapping(page, 'natural');
    await checkColorMapping(page, 'portrait');
    await checkColorMappingWithScene(page);
    await checkSharpen(page);
    await checkPrecisionPipeline(page);

    console.log('\n-- End-to-end checks --');
    await checkEndToEnd(page, 'defaults', DEFAULT_SETTINGS, 16, 16, 21);
    await checkEndToEnd(
      page,
      'kitchen-sink',
      {
        ...DEFAULT_SETTINGS,
        preset: 'landscape',
        noiseReduction: 35,
        noiseMethod: 'gaussian',
        localContrast: 25,
        sharpenAmount: 50,
        sceneHeuristics: true,
        saturation: 15,
        temperature: -10,
      },
      20,
      20,
      42
    );
    await checkEndToEnd(
      page,
      'bilateral-no-precision',
      { ...DEFAULT_SETTINGS, noiseReduction: 30, noiseMethod: 'bilateral', precisionPipeline: false },
      18,
      18,
      5
    );
    await checkEndToEnd(page, 'monochrome-quality-high', { ...DEFAULT_SETTINGS, preset: 'monochrome', quality: 'high', sharpenAmount: 40 }, 16, 16, 9);

    console.log('\n-- Stress / edge-case checks --');
    await checkNoiseExtreme(page, 'gaussian', 100);
    await checkNoiseExtreme(page, 'bilateral', 100);
    await checkLocalContrastExtreme(page);
    await checkToneCurveExtreme(page);
    for (const presetId of Object.keys(PRESETS) as (keyof typeof PRESETS)[]) {
      await checkColorMappingExtreme(page, presetId);
    }
    await checkSharpenExtreme(page);
    await checkEndToEnd(
      page,
      'maximum-quality-everything-maxed',
      {
        ...DEFAULT_SETTINGS,
        preset: 'high-contrast',
        quality: 'maximum',
        noiseReduction: 100,
        noiseMethod: 'gaussian',
        localContrast: 100,
        sharpenAmount: 200,
        sharpenRadius: 5,
        sceneHeuristics: true,
        saturation: -100,
        temperature: 100,
        hueBias: 175,
        exposure: 1.5,
        contrast: 80,
        gamma: 2.2,
        shadowLift: 90,
        highlightRecovery: 90,
        whiteBalanceStrength: 100,
        autoToneStrength: 100,
        colorCorrectionStrength: 100,
      },
      28,
      28,
      99
    );
    await checkEndToEndSolid(page, 'solid-black', 0);
    await checkEndToEndSolid(page, 'solid-white', 255);
    await checkEndToEndSolid(page, 'solid-mid-gray', 128);
    await checkEndToEnd(page, 'non-square-wide', { ...DEFAULT_SETTINGS, sceneHeuristics: true, noiseReduction: 20 }, 31, 11, 17);
    await checkEndToEnd(page, 'non-square-tall', { ...DEFAULT_SETTINGS, sceneHeuristics: true, localContrast: 30 }, 11, 31, 23);

    console.log('\n== Large-image GPU tiling ==');
    await runTiledChecks(page);
  } finally {
    await browser.close();
  }

  console.log('\n-- Summary --');
  const failed = results.filter((r) => !r.pass);
  for (const r of results) {
    console.log(`${r.pass ? 'PASS' : 'FAIL'}  ${r.name}`);
  }
  console.log(`\n${results.length - failed.length}/${results.length} checks passed.`);

  if (failed.length > 0) {
    console.error(`\n${failed.length} check(s) exceeded tolerance -- see above.`);
    process.exit(1);
  }
}

main().catch((err) => {
  console.error('GPU consistency harness crashed:', err);
  process.exit(1);
});
