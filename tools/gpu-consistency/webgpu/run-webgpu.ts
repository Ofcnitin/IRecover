/**
 * WebGPU consistency harness (real headless Chromium + real WebGPU).
 * ------------------------------------------------------------------
 * Verifies the WGSL/WebGPU backend against the CPU reference and the
 * WebGL2 backend, in a real browser, on the shipped code.
 *
 * The ported checks in the first half of this file (fixtures, stage
 * checks, end-to-end/stress checks) were extracted MECHANICALLY from
 * tools/gpu-consistency/run.ts (function bodies unchanged; only the GPU
 * runner they call and the check-name prefix differ), so WebGPU is held
 * to IDENTICAL fixtures and tolerances as WebGL2. Diff any ported
 * function against run.ts to confirm. The second half is new and
 * WebGPU-specific.
 *
 * Run:   npm run gpu-consistency:webgpu
 * Needs: Playwright's Chromium with WebGPU. Flags used (SwiftShader
 *        Vulkan software adapter): see CHROMIUM_WEBGPU_ARGS below.
 *
 * If no WebGPU adapter can be acquired the harness exits with code 2 and
 * states that NOTHING was verified -- it never reports a pass without
 * having run on a real WebGPU device.
 */
import { chromium, type Browser, type Page } from 'playwright';
import * as esbuild from 'esbuild';
import http from 'node:http';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';

// Same Node-side test setup the WebGL2 harness uses (ImageData polyfill etc.).
import '../../../tests/setup';

import { compareArrays, compareInterleavedChannels, formatMetrics, type ErrorMetrics } from '../metrics';
import { srgbToLinear } from '../../../src/processing/colorSpace';
import { reduceNoise } from '../../../src/processing/noiseReduction';
import { applyLocalContrast } from '../../../src/processing/contrast';
import { buildToneCurve } from '../../../src/processing/toneMapping';
import { mapIntensityToRgb } from '../../../src/processing/colorMapping';
import { computeSceneMaps, runPipeline } from '../../../src/processing/pipeline';
import { unsharpMask } from '../../../src/processing/sharpening';
import { applyAzusaWhiteBalance } from '../../../src/processing/whiteBalance';
import { applyAutoTone } from '../../../src/processing/autoTone';
import { applyColorCorrectionPipeline } from '../../../src/processing/colorCorrectionPipeline';
import { protectGamut } from '../../../src/processing/gamutProtection';
import { PRESETS } from '../../../src/processing/presets';
import { extractIntensity, applyLevels, resolveLevelPoints } from '../../../src/processing/normalize';
import { DEFAULT_SETTINGS } from '../../../src/types/processing';
import { planTiles } from '../../../src/processing/gpu/tilePlanner';
import type { TilingDiagnostics } from '../../../src/processing/gpu/tilePlanner';
import { resolveWebGPUStageRadii } from '../../../src/processing/gpu/webgpu/webgpuFullPipeline';
import { KERNEL_NAMES } from '../../../src/processing/gpu/webgpu/wgslShaders';
import type { ProcessingSettings, ColorStop } from '../../../src/types/processing';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// =====================================================================
// PORTED FROM tools/gpu-consistency/run.ts (mechanical extraction)
// =====================================================================


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

const NEUTRAL_QUALITY = { gaussianPasses: 3, localContrastRadius: 24 };
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
  record('webgpu-vs-cpu/noise-reduction/gaussian (8-bit RGB)', m.combined, 2, 0.02);
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
  record('webgpu-vs-cpu/noise-reduction/bilateral (8-bit RGB)', m.combined, 2, 0.02);
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
  record('webgpu-vs-cpu/local-contrast (8-bit RGB)', m.combined, 2, 0.02);
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
  record('webgpu-vs-cpu/tone-curve (8-bit RGB)', m.combined, 2, 0.02);
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
  record(`webgpu-vs-cpu/color-mapping/${presetId} (8-bit RGB)`, m.combined, 2, 0.02);
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
  record('webgpu-vs-cpu/color-mapping+scene-heuristics/landscape (8-bit RGB)', m.combined, 2, 0.03);
}

/**
 * Sky/vegetation colour effect, on a fixture that actually CONTAINS sky and
 * vegetation. The gradient fixture above yields almost no pixels with
 * sky/veg confidence above the 0.15 colour-shift threshold, so a bug in the
 * sky/veg colour offsets slips through it (demonstrated by mutation: raising
 * the sky blue boost 0.10 -> 0.30 leaves that check green). Here the top
 * third is bright and flat (sky) and the rest is textured midtone
 * (vegetation). Same CPU reference and SAME tolerance as the check above.
 */
async function checkColorMappingSceneSkyVeg(page: Page): Promise<void> {
  const width = 32;
  const height = 36;
  const intensity = new Float32Array(width * height);
  let seed = 12345;
  const rnd = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 0xffffffff;
  };
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      intensity[y * width + x] = y < height / 3 ? 0.9 : 0.42 + (rnd() - 0.5) * 0.24;
    }
  }
  const settings = { ...neutralSettings(), preset: 'landscape' as const, sceneHeuristics: true };

  const scene = computeSceneMaps(intensity, width, height);
  let skyPixels = 0;
  let vegPixels = 0;
  for (let i = 0; i < width * height; i++) {
    if ((scene.sky?.[i] ?? 0) > 0.15) skyPixels++;
    if ((scene.vegetation?.[i] ?? 0) > 0.15) vegPixels++;
  }
  // Guard the fixture itself: if it stops exercising both offsets, this check is worthless.
  recordCustom('webgpu-vs-cpu/scene sky+veg fixture exercises both colour offsets', skyPixels > 100 && vegPixels > 100, `${skyPixels} sky px, ${vegPixels} vegetation px above the 0.15 threshold`);

  const cpuRgba = mapIntensityToRgb(intensity, width, height, PRESETS.landscape.colorStops, {
    colorStrength: 100,
    saturation: 0,
    temperature: 0,
    hueBias: 0,
    scene,
  });

  const gpu = await gpuRunFullPipeline(page, intensity, width, height, settings, PRESETS.landscape.colorStops, NEUTRAL_QUALITY);

  const m = compareInterleavedChannels(cpuRgba as unknown as ArrayLike<number>, gpu.rgba, 4, 2, 3);
  record('webgpu-vs-cpu/color-mapping+scene-heuristics/sky+vegetation fixture (8-bit RGB)', m.combined, 2, 0.03);
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
  record('webgpu-vs-cpu/sharpen (8-bit RGB)', m.combined, 2, 0.03);
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
  record('webgpu-vs-cpu/precision-pipeline/R (linear, x255)', mR, 2 / 255, 0.02, 255);
  record('webgpu-vs-cpu/precision-pipeline/G (linear, x255)', mG, 2 / 255, 0.02, 255);
  record('webgpu-vs-cpu/precision-pipeline/B (linear, x255)', mB, 2 / 255, 0.02, 255);
}

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
  record(`webgpu-vs-cpu/end-to-end/${label} (8-bit RGB)`, m.combined, 3, 0.05);
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
  record(`webgpu-vs-cpu/end-to-end/${label} (8-bit RGB)`, m.combined, 3, 0.05);
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
  record(`webgpu-vs-cpu/noise-reduction/${method}@${strength} extreme (8-bit RGB)`, m.combined, 2, 0.02);
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
  record('webgpu-vs-cpu/local-contrast@100/radius36 extreme (8-bit RGB)', m.combined, 2, 0.02);
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
  record('webgpu-vs-cpu/tone-curve extreme (exposure=2,contrast=100,gamma=3,...) (8-bit RGB)', m.combined, 2, 0.02);

  // Also stress the opposite extreme (negative contrast/exposure/brightness).
  const toneOpts2 = { exposure: -2, brightness: -100, contrast: -100, gamma: 0.2, shadowLift: 0, highlightRecovery: 0 };
  const settings2 = { ...neutralSettings(), preset: 'monochrome' as const, ...toneOpts2 };
  const toneCurve2 = buildToneCurve(toneOpts2);
  const cpuIntensity2 = new Float32Array(intensity.length);
  for (let i = 0; i < intensity.length; i++) cpuIntensity2[i] = toneCurve2(intensity[i]);
  const cpuRgba2 = mapIntensityToRgb(cpuIntensity2, width, height, PRESETS.monochrome.colorStops, MONO_MAP_OPTS);
  const gpu2 = await gpuRunFullPipeline(page, intensity, width, height, settings2, PRESETS.monochrome.colorStops, NEUTRAL_QUALITY);
  const m2 = compareInterleavedChannels(cpuRgba2 as unknown as ArrayLike<number>, gpu2.rgba, 4, 2, 3);
  record('webgpu-vs-cpu/tone-curve extreme (negative exposure/contrast/gamma=0.2) (8-bit RGB)', m2.combined, 2, 0.02);
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
  record(`webgpu-vs-cpu/color-mapping/${presetId} extreme sat/temp/hue (8-bit RGB)`, m.combined, 2, 0.02);

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
  record(`webgpu-vs-cpu/color-mapping/${presetId} extreme sat+/temp-/hue- (8-bit RGB)`, m2.combined, 2, 0.02);
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
  record('webgpu-vs-cpu/sharpen@amount200/radius5/threshold0 extreme (8-bit RGB)', m.combined, 2, 0.03);
}

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

function numericLeaves(v: unknown, out: number[] = []): number[] {
  if (typeof v === 'number') out.push(v);
  else if (v && typeof v === 'object') for (const k of Object.keys(v as object).sort()) numericLeaves((v as Record<string, unknown>)[k], out);
  return out;
}

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


/** New (WebGL2 has no median): CPU median reference vs WebGPU, same fixture style/tolerance as gaussian/bilateral. */
async function checkNoiseReductionMedian(page: Page): Promise<void> {
  for (const strength of [40, 80]) {
    const width = 24;
    const height = 24;
    const intensity = makeIntensityCheckerNoisy(width, height, 13);
    const settings = { ...neutralSettings(), preset: 'monochrome' as const, noiseReduction: strength, noiseMethod: 'median' as const };
    const cpuIntensity = reduceNoise(intensity.slice(), width, height, 'median', strength, NEUTRAL_QUALITY.gaussianPasses);
    const cpuRgba = mapIntensityToRgb(cpuIntensity, width, height, PRESETS.monochrome.colorStops, MONO_MAP_OPTS);
    const gpu = await gpuRunFullPipeline(page, intensity, width, height, settings, PRESETS.monochrome.colorStops, NEUTRAL_QUALITY);
    const m = compareInterleavedChannels(cpuRgba as unknown as ArrayLike<number>, gpu.rgba, 4, 2, 3);
    record(`webgpu-vs-cpu/noise-reduction/median@${strength} radius ${strength / 100 > 0.66 ? 2 : 1} (8-bit RGB)`, m.combined, 2, 0.02);
  }
}

/** The "everything maxed" settings the WebGL2 harness uses for its worst-case end-to-end check. */
const MAXED_SETTINGS: ProcessingSettings = {
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
};


// =====================================================================
// WebGPU-specific harness (everything below is new; everything above was
// extracted mechanically from tools/gpu-consistency/run.ts so the
// WebGPU-vs-CPU checks use IDENTICAL fixtures and tolerances)
// =====================================================================
//
// PASS RULES FOR THE NEW COMPARISONS (fixed here, before the first run):
//
//  * WebGPU vs CPU: the same tolerances as the WebGL2-vs-CPU checks
//    (stage checks: tol 2 / <=2-3% ; end-to-end: tol 3 / <=5%).
//  * WebGPU vs WebGL2 (both float32 GPU implementations of the same
//    math, run in the same browser on the same input): max 8-bit error
//    <= 4 levels and <= 1% of elements differing by more than 1 level.
//    Precision-diagnostic gains agree within 1e-4.
//    (The WGSL kernels are line-for-line ports, so identical results are
//    EXPECTED; the bound only leaves room for GPU math-library rounding.)
//  * WebGPU should be no FARTHER from the CPU reference than WebGL2 is:
//    MAE(cpu, webgpu) <= MAE(cpu, webgl2) + 0.1 levels.
//  * WebGPU tiled vs untiled: BIT-IDENTICAL for every config (including
//    scene heuristics: integer global rows make the row fraction exact),
//    diagnostics exactly equal.
//  * Negative controls: an under-sized halo MUST create mismatches, all
//    confined to bands around internal tile edges.
//  REVISION HISTORY (kept for transparency): the first WGSL revision used
//  the CPU's scene row convention (y/H). The first run then FAILED the
//  WebGL2-vs-WebGPU rule above on one config (max 6 levels vs the <=4
//  bound; 0.166% of elements) -- my a-priori estimate that a half-row
//  shift is worth ~0.6 level was wrong, because it flips the sky>0.15 /
//  veg>0.15 step and later stages amplify it. A one-line WGSL patch
//  experiment (WebGL2's (y+0.5)/H) made WebGPU bit-identical to WebGL2
//  (max 0), proving that convention was the ENTIRE difference; both
//  conventions are equally far from the CPU (MAE 0.452 vs 0.454). The
//  rule was NOT loosened: the kernel was changed to WebGL2's convention
//  so switching GPU API never changes the image.
//
//  * Honesty: `execution.executed === 'webgpu'` iff the WebGPU pipeline
//    produced the pixels; in every injected-failure scenario it must
//    name the backend that really ran and record why WebGPU did not.

const CHROMIUM_WEBGPU_ARGS = [
  '--no-sandbox',
  '--enable-unsafe-webgpu',
  '--enable-features=Vulkan,WebGPU',
  '--use-angle=swiftshader',
  '--use-vulkan=swiftshader',
  '--disable-vulkan-surface',
  '--ignore-gpu-blocklist',
  '--enable-unsafe-swiftshader',
  '--disable-gpu-sandbox',
];

interface WebGPUHarness {
  browser: Browser;
  page: Page;
  server: http.Server;
  adapter: { vendor: string; architecture: string; software: boolean; isFallbackAdapter: boolean };
  maxTextureDimension2D: number;
}

async function startWebGPUHarness(): Promise<WebGPUHarness> {
  const bundleDir = path.join(__dirname, '.bundle');
  fs.mkdirSync(bundleDir, { recursive: true });
  const outfile = path.join(bundleDir, 'harness-webgpu.js');
  await esbuild.build({
    entryPoints: [path.join(__dirname, 'harness-entry-webgpu.ts')],
    bundle: true,
    format: 'iife',
    platform: 'browser',
    target: 'es2022',
    outfile,
  });

  // The SHIPPED worker (src/workers/imageProcessor.worker.ts), bundled exactly as the app would load it.
  const workerOut = path.join(bundleDir, 'imageProcessor.worker.js');
  await esbuild.build({
    entryPoints: [path.join(__dirname, '../../../src/workers/imageProcessor.worker.ts')],
    bundle: true,
    format: 'iife',
    platform: 'browser',
    target: 'es2022',
    outfile: workerOut,
  });

  // WebGPU is only exposed in secure contexts: serve a blank page (and the worker script) from http://localhost.
  const server = http.createServer((req, res) => {
    if (req.url === '/worker.js') {
      res.setHeader('content-type', 'text/javascript');
      res.end(fs.readFileSync(workerOut));
      return;
    }
    res.setHeader('content-type', 'text/html');
    res.end('<!doctype html><title>webgpu harness</title>');
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const port = (server.address() as { port: number }).port;

  const browser = await chromium.launch({ headless: true, args: CHROMIUM_WEBGPU_ARGS });
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.log(`[page error] ${e.message}`));
  await page.goto(`http://localhost:${port}/`);
  await page.addScriptTag({ path: outfile });
  // tsx/esbuild wraps named inner functions in __name(); page.evaluate bodies run in the browser, where it must exist.
  await page.evaluate('window.__name = (f) => f;');

  const probe = await page.evaluate(async () => {
    const H = (window as any).__gpuHarness;
    const r = await H.probeWebGPU({ fresh: true });
    if (!r.available) return { ok: false as const, reason: r.reason as string };
    const ctx = await H.acquireWebGPU();
    return { ok: true as const, adapter: r.adapter, max: ctx.limits.maxTextureDimension2D as number };
  });
  if (!probe.ok) {
    await browser.close();
    server.close();
    throw new WebGPUUnavailableInHarness(probe.reason);
  }
  return { browser, page, server, adapter: probe.adapter, maxTextureDimension2D: probe.max };
}

class WebGPUUnavailableInHarness extends Error {}

/** WebGPU version of the runner the extracted checks call (same signature as the WebGL2 harness's). */
async function gpuRunFullPipeline(
  page: Page,
  intensity: Float32Array,
  width: number,
  height: number,
  settings: ProcessingSettings,
  colorStops: ColorStop[],
  quality: { gaussianPasses: number; localContrastRadius: number }
): Promise<{ rgba: number[]; precision?: unknown; tiling?: TilingDiagnostics; resources?: any; adapter?: any }> {
  return page.evaluate(
    async (args: [number[], number, number, ProcessingSettings, ColorStop[], { gaussianPasses: number; localContrastRadius: number }]) => {
      const [arr, w, h, s, stops, q] = args;
      const H = (window as any).__gpuHarness;
      const r = await H.runFullPipelineWebGPU(new Float32Array(arr), w, h, s, stops, q);
      return { rgba: Array.from(r.rgba as Uint8ClampedArray), precision: r.precision, tiling: r.tiling, resources: r.resources, adapter: r.adapter };
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
    async (args: [number[], number[], number[], number, number, typeof opts]) => {
      const [rArr, gArr, bArr, w, h, o] = args;
      const H = (window as any).__gpuHarness;
      const rF = new Float32Array(rArr);
      const gF = new Float32Array(gArr);
      const bF = new Float32Array(bArr);
      await H.runPrecisionPipelineWebGPU(rF, gF, bF, w, h, o);
      return { r: Array.from(rF), g: Array.from(gF), b: Array.from(bF) };
    },
    [Array.from(r), Array.from(g), Array.from(b), width, height, opts] as [number[], number[], number[], number, number, typeof opts]
  );
}

// ---------------------------------------------------------------------
// Section A: shaders / device
// ---------------------------------------------------------------------

async function checkShadersCompile(page: Page): Promise<void> {
  const r = await page.evaluate(async () => {
    const H = (window as any).__gpuHarness;
    const ctx = await H.acquireWebGPU();
    const messages: string[] = [];
    let count = 0;
    for (const [name, spec] of Object.entries(H.KERNELS) as [string, { wgsl: string }][]) {
      const m = (ctx.device as any).createShaderModule({ code: spec.wgsl });
      const info = await m.getCompilationInfo();
      for (const x of info.messages) messages.push(`${name}: ${x.type} L${x.lineNum} ${x.message}`);
      count++;
    }
    // Also build every compute pipeline (validates bind group layouts against the WGSL bindings).
    const fresh = await H.acquireWebGPU({ fresh: true });
    await H.compileAllKernels(fresh);
    return { count, messages };
  });
  const N = KERNEL_NAMES.length;
  recordCustom(`webgpu/${N} WGSL kernels: zero compiler messages (no errors, no warnings)`, r.count === N && r.messages.length === 0, `${r.count} kernels (expected ${N}), ${r.messages.length} messages ${r.messages.join(' | ')}`);
  recordCustom(`webgpu/all ${N} compute pipelines + bind-group layouts create successfully`, true, 'compileAllKernels resolved');
}

// ---------------------------------------------------------------------
// Section B: three-way CPU / WebGL2 / WebGPU
// ---------------------------------------------------------------------

const WGPU_VS_GL = { maxError: 4, maxFractionOver1: 0.01, gainTol: 1e-4 };

async function runBothGpus(page: Page, intensity: Float32Array, width: number, height: number, settings: ProcessingSettings, stops: ColorStop[], quality: { gaussianPasses: number; localContrastRadius: number }) {
  return page.evaluate(
    async (args: [number[], number, number, ProcessingSettings, ColorStop[], typeof quality]) => {
      const [arr, w, h, s, st, q] = args;
      const H = (window as any).__gpuHarness;
      const wg = await H.runFullPipelineWebGPU(new Float32Array(arr), w, h, s, st, q);
      const gl = H.runFullPipelineWebGL2(new Float32Array(arr), w, h, s, st, q);
      return { wg: Array.from(wg.rgba as Uint8ClampedArray), gl: Array.from(gl.rgba as Uint8ClampedArray), wgP: wg.precision, glP: gl.precision };
    },
    [Array.from(intensity), width, height, settings, stops, quality] as [number[], number, number, ProcessingSettings, ColorStop[], typeof quality]
  );
}

async function checkThreeWay(page: Page, label: string, settings: ProcessingSettings, width: number, height: number, seed: number): Promise<void> {
  const prep = prepareE2E(settings, width, height, seed);
  const q = qualityParamsFor(settings.quality);
  const stops = PRESETS[settings.preset].colorStops;
  const r = await runBothGpus(page, prep.intensity, width, height, settings, stops, q);
  const cpu = prep.cpuRgba as unknown as ArrayLike<number>;

  const cpuGl = compareInterleavedChannels(cpu, r.gl, 4, 3, 3).combined;
  const cpuWg = compareInterleavedChannels(cpu, r.wg, 4, 3, 3).combined;
  record(`three-way/${label}: CPU vs WebGL2`, cpuGl, 3, 0.05);
  record(`three-way/${label}: CPU vs WebGPU`, cpuWg, 3, 0.05);

  const glWg = compareInterleavedChannels(r.gl, r.wg, 4, 1, 3).combined;
  recordStrict(`three-way/${label}: WebGL2 vs WebGPU`, glWg, WGPU_VS_GL.maxError, WGPU_VS_GL.maxFractionOver1);

  recordCustom(
    `three-way/${label}: WebGPU no farther from CPU than WebGL2`,
    cpuWg.mae <= cpuGl.mae + 0.1,
    `MAE cpu-webgpu ${cpuWg.mae.toFixed(4)} vs cpu-webgl2 ${cpuGl.mae.toFixed(4)} (allowed +0.1)`
  );

  if (r.glP && r.wgP) {
    const m = compareArrays(numericLeaves(r.glP), numericLeaves(r.wgP), 0);
    recordStrict(`three-way/${label}: precision diagnostics WebGL2 vs WebGPU`, m, WGPU_VS_GL.gainTol, 1);
  }
}

// ---------------------------------------------------------------------
// Section C: WebGPU tiling (tiled vs untiled, seams, negative controls, real oversize)
// ---------------------------------------------------------------------

interface WgCase {
  label: string;
  width: number;
  height: number;
  intensity: Float32Array;
  settings: ProcessingSettings;
  stops: ColorStop[];
  quality: { gaussianPasses: number; localContrastRadius: number };
  maxTileDim: number;
  minTiles: number;
  seamBand?: boolean;
}

interface WgPair {
  single: number[];
  tiled: number[];
  singleP?: unknown;
  tiledP?: unknown;
  singleTiling: TilingDiagnostics;
  tiledTiling: TilingDiagnostics;
  tiledRes: { passesDispatched: number; submits: number; texturesCreated: number; textureReuses: number; peakTexturesLive: number };
}

async function runWgPair(page: Page, c: WgCase, override?: { colorMap?: number; sharpen?: number }, singleDim = 8192): Promise<WgPair> {
  return page.evaluate(
    async (args: [number[], number, number, ProcessingSettings, ColorStop[], typeof c.quality, number, number, typeof override]) => {
      const [arr, w, h, s, st, q, tileDim, sDim, ov] = args;
      const H = (window as any).__gpuHarness;
      const single = await H.runFullPipelineWebGPU(new Float32Array(arr), w, h, s, st, q, { maxTileDim: sDim });
      const tiled = await H.runFullPipelineWebGPU(new Float32Array(arr), w, h, s, st, q, { maxTileDim: tileDim, unsafeHaloOverride: ov });
      return {
        single: Array.from(single.rgba as Uint8ClampedArray),
        tiled: Array.from(tiled.rgba as Uint8ClampedArray),
        singleP: single.precision,
        tiledP: tiled.precision,
        singleTiling: single.tiling,
        tiledTiling: tiled.tiling,
        tiledRes: tiled.resources,
      };
    },
    [Array.from(c.intensity), c.width, c.height, c.settings, c.stops, c.quality, c.maxTileDim, singleDim, override] as [number[], number, number, ProcessingSettings, ColorStop[], typeof c.quality, number, number, typeof override]
  );
}

async function checkWgTiledVsUntiled(page: Page, c: WgCase): Promise<WgPair> {
  const r = await runWgPair(page, c);
  const engaged = r.tiledTiling.colorMap.tileCount >= c.minTiles && r.singleTiling.colorMap.tileCount === 1;
  recordCustom(
    `webgpu-tiled-vs-untiled/${c.label} tiling engaged`,
    engaged,
    `${r.tiledTiling.colorMap.cols}x${r.tiledTiling.colorMap.rows} grid, halo ${r.tiledTiling.colorMap.halo}, ${r.tiledTiling.colorMap.tileCount} tiles (need >= ${c.minTiles}); single-tile run: ${r.singleTiling.colorMap.tileCount}`
  );
  recordStrict(`webgpu-tiled-vs-untiled/${c.label} (8-bit RGBA)`, compareArrays(r.single, r.tiled, 0), 0, 0);
  if (r.singleP && r.tiledP) {
    recordStrict(`webgpu-tiled-vs-untiled/${c.label} precision diagnostics`, compareArrays(numericLeaves(r.singleP), numericLeaves(r.tiledP), 0), 0, 1);
  }
  if (c.seamBand) {
    const radii = resolveWebGPUStageRadii(c.settings, c.quality);
    const plans = [{ halo: r.tiledTiling.colorMap.halo, dim: r.tiledTiling.maxTileDim, band: radii.colorMapReach + 1 }];
    if (r.tiledTiling.sharpen) plans.push({ halo: r.tiledTiling.sharpen.halo, dim: r.tiledTiling.maxTileDim, band: radii.sharpenReach + 1 });
    const mask = seamBandMask(c.width, c.height, plans);
    const a: number[] = [];
    const b: number[] = [];
    for (let i = 0; i < mask.length; i++) {
      if (!mask[i]) continue;
      for (let ch = 0; ch < 3; ch++) {
        a.push(r.single[i * 4 + ch]);
        b.push(r.tiled[i * 4 + ch]);
      }
    }
    recordStrict(`webgpu-seam-band/${c.label} (pixels within reach of internal tile edges)`, compareArrays(a, b, 0), 0, 0);
  }
  return r;
}

async function checkWgNegativeControl(page: Page, label: string, c: WgCase, override: { colorMap?: number; sharpen?: number }): Promise<void> {
  const radii = resolveWebGPUStageRadii(c.settings, c.quality);
  const r = await runWgPair(page, c, override);
  const band = Math.max(radii.colorMapReach, radii.sharpenReach) + 1;
  const plans = [{ halo: override.colorMap ?? radii.colorMapReach, dim: c.maxTileDim, band }];
  if (c.settings.sharpenAmount > 0) plans.push({ halo: override.sharpen ?? radii.sharpenReach, dim: c.maxTileDim, band });
  const mask = seamBandMask(c.width, c.height, plans);
  let total = 0;
  let outside = 0;
  let worst = 0;
  for (let i = 0; i < r.single.length; i++) {
    const d = Math.abs(r.single[i] - r.tiled[i]);
    if (d > 0) {
      total++;
      if (!mask[i >> 2]) outside++;
      if (d > worst) worst = d;
    }
  }
  recordCustom(`webgpu-negative-control/${label}`, total > 0 && outside === 0, `${total} mismatching elements (max ${worst} levels), ${outside} outside the seam bands -- expected >0 and 0`);
}

async function checkWgOversize(page: Page, deviceMax: number, label: string, settings: ProcessingSettings, width: number, height: number, seed: number, expectTiles: number | 'many'): Promise<void> {
  const prep = prepareE2E(settings, width, height, seed);
  const q = qualityParamsFor(settings.quality);
  const stops = PRESETS[settings.preset].colorStops;
  const r = await page.evaluate(
    async (args: [number[], number, number, ProcessingSettings, ColorStop[], typeof q]) => {
      const [arr, w, h, s, st, qq] = args;
      const H = (window as any).__gpuHarness;
      // Ask for an absurdly large tile: the pipeline must clamp it to the device's real texture limit.
      const big = await H.runFullPipelineWebGPU(new Float32Array(arr), w, h, s, st, qq, { maxTileDim: 1_000_000 });
      const dflt = await H.runFullPipelineWebGPU(new Float32Array(arr), w, h, s, st, qq);
      return { big: Array.from(big.rgba as Uint8ClampedArray), dflt: Array.from(dflt.rgba as Uint8ClampedArray), bigTiling: big.tiling, dfltTiling: dflt.tiling };
    },
    [Array.from(prep.intensity), width, height, settings, stops, q] as [number[], number, number, ProcessingSettings, ColorStop[], typeof q]
  );
  const clamped = r.bigTiling.maxTileDim === deviceMax;
  const tilesOk = expectTiles === 'many' ? r.bigTiling.colorMap.tileCount > 1 : r.bigTiling.colorMap.tileCount === expectTiles;
  recordCustom(
    `webgpu-oversize/${label} tile size clamped to device limit ${deviceMax}`,
    clamped && tilesOk,
    `maxTileDim=${r.bigTiling.maxTileDim}, ${r.bigTiling.colorMap.cols}x${r.bigTiling.colorMap.rows} tiles (expected ${expectTiles}); default run used ${r.dfltTiling.colorMap.tileCount} tiles @${r.dfltTiling.maxTileDim}`
  );
  const m = compareInterleavedChannels(prep.cpuRgba as unknown as ArrayLike<number>, r.big, 4, 3, 3).combined;
  record(`webgpu-oversize/${label} vs CPU (8-bit RGB)`, m, 3, 0.05);
  recordStrict(`webgpu-oversize/${label} tiling-invariance (device-limit tiles vs default tiles)`, compareArrays(r.big, r.dflt, 0), 0, 0);
}

// ---------------------------------------------------------------------
// Section D: fallback honesty in a real browser
// ---------------------------------------------------------------------

type Injection = 'none' | 'queue-submit-throws' | 'queue-submit-throws+no-webgl2' | 'mid-run-device-loss' | 'no-adapter' | 'no-navigator-gpu' | 'tiling-unsupported';

interface AsyncRun {
  data: number[];
  execution: any;
  precision: any;
}

async function runAsyncInPage(page: Page, data: Uint8ClampedArray, width: number, height: number, settings: ProcessingSettings, inject: Injection): Promise<AsyncRun> {
  return page.evaluate(
    async (args: [number[], number, number, ProcessingSettings, string]) => {
      const [arr, w, h, s, inj] = args;
      const H = (window as any).__gpuHarness;
      const restore: Array<() => void> = [];
      const patch = (obj: any, key: string, fn: (orig: any) => any) => {
        const orig = obj[key];
        obj[key] = fn(orig);
        restore.push(() => (obj[key] = orig));
      };
      let opts: any = undefined;
      try {
        if (inj === 'queue-submit-throws' || inj === 'queue-submit-throws+no-webgl2') {
          patch((window as any).GPUQueue.prototype, 'submit', () => function () { throw new Error('injected: queue.submit failure'); });
        }
        if (inj === 'queue-submit-throws+no-webgl2') {
          patch(OffscreenCanvas.prototype as any, 'getContext', (orig) => function (this: any, type: string, ...a: any[]) { return type === 'webgl2' ? null : orig.call(this, type, ...a); });
          patch(HTMLCanvasElement.prototype as any, 'getContext', (orig) => function (this: any, type: string, ...a: any[]) { return type === 'webgl2' ? null : orig.call(this, type, ...a); });
        }
        if (inj === 'mid-run-device-loss') {
          const ctx = await H.acquireWebGPU();
          patch((window as any).GPUBuffer.prototype, 'mapAsync', (orig) => function (this: any, ...a: any[]) { ctx.device.destroy(); return orig.apply(this, a); });
        }
        if (inj === 'no-adapter') {
          H.resetWebGPUCache();
          patch((navigator as any).gpu, 'requestAdapter', () => async () => null);
        }
        if (inj === 'no-navigator-gpu') {
          H.resetWebGPUCache();
          const desc = Object.getOwnPropertyDescriptor(navigator, 'gpu');
          Object.defineProperty(navigator, 'gpu', { value: undefined, configurable: true });
          restore.push(() => { if (desc) Object.defineProperty(navigator, 'gpu', desc); else delete (navigator as any).gpu; });
        }
        // An image that fits in ONE tile is never untileable (the planner ignores overlap then), so the tile size must be
        // smaller than the image (24 > 16) while the ~30px overlap this pipeline needs can't fit in a 16px tile.
        if (inj === 'tiling-unsupported') opts = { webgpu: { maxTileDim: 16 } };

        const out = await H.runPipelineAsync(new ImageData(new Uint8ClampedArray(arr), w, h), s, opts);
        return { data: Array.from(out.output.data as Uint8ClampedArray), execution: out.execution, precision: out.precision };
      } finally {
        for (const r of restore.reverse()) r();
      }
    },
    [Array.from(data), width, height, settings, inject] as [number[], number, number, ProcessingSettings, string]
  );
}

/** The invariants that make "WebGPU active" trustworthy. */
function assertHonest(name: string, run: AsyncRun, expectExecuted: 'webgpu' | 'webgl2' | 'cpu', mustHaveWebGPUFailure: boolean): void {
  const ex = run.execution;
  const p = run.precision;
  const last = ex.attempts[ex.attempts.length - 1];
  const wgFail = ex.attempts.find((a: any) => a.backend === 'webgpu' && !a.ok);
  const reportsWebGPU = ex.executed === 'webgpu';
  const checks: [string, boolean][] = [
    [`executed=${ex.executed} (expected ${expectExecuted})`, ex.executed === expectExecuted],
    [`gpuAccelerated ${ex.gpuAccelerated} matches executed!=cpu`, ex.gpuAccelerated === (ex.executed !== 'cpu')],
    [`last attempt is the executed backend and ok`, last && last.backend === ex.executed && last.ok === true],
    [`adapter info present iff executed=webgpu`, reportsWebGPU === (ex.adapter !== undefined)],
    [`precision.backend.resolved never 'webgpu' unless executed=webgpu`, !p || (p.backend.resolved === 'webgpu') === reportsWebGPU],
    [`precision.gpuAccelerated=false when executed=cpu (no precision-only GL tier)`, !p || ex.executed !== 'cpu' || ex.precisionStageOnGpu || p.gpuAccelerated === false],
    [mustHaveWebGPUFailure ? `records the WebGPU failure with a reason` : `no WebGPU failure recorded`, mustHaveWebGPUFailure ? !!wgFail && typeof wgFail.reason === 'string' && wgFail.reason.length > 0 : !wgFail],
    [mustHaveWebGPUFailure ? `fellBack=true` : `fellBack=${ex.fellBack}`, mustHaveWebGPUFailure ? ex.fellBack === true : true],
  ];
  const failed = checks.filter(([, ok]) => !ok).map(([d]) => d);
  recordCustom(`honesty/${name}`, failed.length === 0, failed.length === 0 ? `executed=${ex.executed}; attempts=${ex.attempts.map((a: any) => `${a.backend}:${a.ok ? 'ok' : 'FAIL'}`).join(' -> ')}${wgFail ? ` [${String(wgFail.reason).slice(0, 90)}]` : ''}` : `violations: ${failed.join('; ')}`);
}

async function runFallbackChecks(page: Page): Promise<void> {
  const w = 24;
  const h = 20;
  const data = makeUint8ImageFromIntensity(makeIntensityCheckerNoisy(w, h, 61), w, h);
  const settings: ProcessingSettings = { ...DEFAULT_SETTINGS, sceneHeuristics: true, noiseReduction: 25, noiseMethod: 'gaussian', sharpenAmount: 40 };
  const cpuRef = runPipeline({ data, width: w, height: h } as unknown as ImageData, { ...settings, processingEngine: 'cpu' }).output.data as unknown as ArrayLike<number>;
  const gpuTol = (name: string, out: AsyncRun) => record(`honesty/${name}: pixels vs CPU (8-bit RGB)`, compareInterleavedChannels(cpuRef, out.data, 4, 3, 3).combined, 3, 0.05);
  const cpuTol = (name: string, out: AsyncRun) => recordStrict(`honesty/${name}: pixels equal the CPU pipeline`, compareInterleavedChannels(cpuRef, out.data, 4, 1, 3).combined, 1, 0.001);

  let out = await runAsyncInPage(page, data, w, h, { ...settings, processingEngine: 'webgpu' }, 'none');
  assertHonest("real success: engine='webgpu'", out, 'webgpu', false);
  gpuTol("real success", out);
  recordCustom('honesty/real success reports the adapter truthfully (software renderer flagged)', out.execution.adapter?.software === true && !!out.execution.adapter?.architecture, JSON.stringify(out.execution.adapter));
  recordCustom('honesty/real success reports resource + tiling stats', !!out.execution.resources && out.execution.resources.passesDispatched > 0 && !!out.execution.tiling, JSON.stringify(out.execution.resources));

  out = await runAsyncInPage(page, data, w, h, { ...settings, processingEngine: 'webgl2' }, 'none');
  assertHonest("engine='webgl2' never touches WebGPU", out, 'webgl2', false);

  out = await runAsyncInPage(page, data, w, h, { ...settings, processingEngine: 'cpu' }, 'none');
  assertHonest("engine='cpu'", out, 'cpu', false);
  cpuTol("engine='cpu'", out);

  out = await runAsyncInPage(page, data, w, h, { ...settings, processingEngine: 'auto' }, 'none');
  assertHonest("engine='auto' + small image stays on CPU (WebGPU not attempted)", out, 'cpu', false);

  out = await runAsyncInPage(page, data, w, h, { ...settings, processingEngine: 'webgpu' }, 'queue-submit-throws');
  assertHonest('injected queue.submit failure -> WebGL2', out, 'webgl2', true);
  gpuTol('injected queue.submit failure -> WebGL2', out);

  out = await runAsyncInPage(page, data, w, h, { ...settings, processingEngine: 'webgpu' }, 'queue-submit-throws+no-webgl2');
  assertHonest('WebGPU failure AND WebGL2 unavailable -> CPU', out, 'cpu', true);
  recordCustom('honesty/WebGPU+WebGL2 both failed: WebGL2 failure is also recorded', out.execution.attempts.some((a: any) => a.backend === 'webgl2' && !a.ok), out.execution.attempts.map((a: any) => `${a.backend}:${a.ok}`).join(' -> '));
  cpuTol('WebGPU+WebGL2 failed -> CPU', out);

  out = await runAsyncInPage(page, data, w, h, { ...settings, processingEngine: 'webgpu' }, 'mid-run-device-loss');
  assertHonest('device destroyed MID-RUN (mapAsync) -> WebGL2', out, 'webgl2', true);
  gpuTol('mid-run device loss -> WebGL2', out);

  out = await runAsyncInPage(page, data, w, h, { ...settings, processingEngine: 'webgpu' }, 'none');
  assertHonest('after device loss the NEXT run re-acquires a device and is WebGPU again', out, 'webgpu', false);

  out = await runAsyncInPage(page, data, w, h, { ...settings, processingEngine: 'webgpu' }, 'no-adapter');
  assertHonest('requestAdapter() -> null -> WebGL2', out, 'webgl2', true);

  out = await runAsyncInPage(page, data, w, h, { ...settings, processingEngine: 'webgpu' }, 'no-navigator-gpu');
  assertHonest('navigator.gpu undefined -> WebGL2', out, 'webgl2', true);

  out = await runAsyncInPage(page, data, w, h, { ...settings, processingEngine: 'webgpu' }, 'tiling-unsupported');
  assertHonest('untileable request (overlap > tile) -> WebGL2', out, 'webgl2', true);
  recordCustom('honesty/untileable request names TilingUnsupportedError', String(out.execution.attempts[0].reason).includes('TilingUnsupportedError'), String(out.execution.attempts[0].reason).slice(0, 120));

  // 'auto' on a large image DOES pick WebGPU (>= 512*512 pixels)
  const bw = 640;
  const bh = 420;
  const bigData = makeUint8ImageFromIntensity(makeIntensityCheckerNoisy(bw, bh, 62), bw, bh);
  const bigSettings: ProcessingSettings = { ...DEFAULT_SETTINGS, processingEngine: 'auto' };
  const bigCpu = runPipeline({ data: bigData, width: bw, height: bh } as unknown as ImageData, { ...bigSettings, processingEngine: 'cpu' }).output.data as unknown as ArrayLike<number>;
  const big = await runAsyncInPage(page, bigData, bw, bh, bigSettings, 'none');
  assertHonest("engine='auto' + large image (640x420) uses WebGPU", big, 'webgpu', false);
  record('honesty/auto+large: pixels vs CPU (8-bit RGB)', compareInterleavedChannels(bigCpu, big.data, 4, 3, 3).combined, 3, 0.05);
}

async function runPipelineLevelWebGPU(page: Page, deviceMax: number): Promise<void> {
  const settings: ProcessingSettings = { ...DEFAULT_SETTINGS, preset: 'landscape', noiseReduction: 35, noiseMethod: 'gaussian', localContrast: 25, sharpenAmount: 50, sceneHeuristics: true, saturation: 15, temperature: -10, processingEngine: 'webgpu' };
  for (const [w, h, seed] of [[deviceMax + 8, 12, 106], [12, deviceMax + 8, 107]] as const) {
    const data = makeUint8ImageFromIntensity(makeIntensityCheckerNoisy(w, h, seed), w, h);
    const cpu = runPipeline({ data, width: w, height: h } as unknown as ImageData, { ...settings, processingEngine: 'cpu' });
    const out = await runAsyncInPage(page, data, w, h, settings, 'none');
    assertHonest(`runPipelineAsync(webgpu) beyond device texture limit ${w}x${h}`, out, 'webgpu', false);
    recordCustom(`pipeline-level/${w}x${h} tiled on WebGPU`, !!out.execution.tiling && out.execution.tiling.colorMap.tileCount > 1, JSON.stringify(out.execution.tiling?.colorMap));
    record(`pipeline-level/${w}x${h} WebGPU vs CPU engine (8-bit RGB)`, compareInterleavedChannels(cpu.output.data as unknown as ArrayLike<number>, out.data, 4, 3, 3).combined, 3, 0.05);
  }
}

// ---------------------------------------------------------------------
// Section E: the SHIPPED worker, in a real Web Worker
// ---------------------------------------------------------------------

interface WorkerReq {
  id: number;
  w: number;
  h: number;
  data: number[];
  settings: ProcessingSettings;
}

/** Posts every request back-to-back (no awaiting between them) to one persistent worker; resolves with responses in ARRIVAL order. */
async function workerCall(page: Page, reqs: WorkerReq[]): Promise<any[]> {
  return page.evaluate(async (rs: WorkerReq[]) => {
    const win = window as any;
    if (!win.__imgWorker) win.__imgWorker = new Worker('/worker.js');
    const w: Worker = win.__imgWorker;
    const out: any[] = [];
    const done = new Promise<any[]>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('worker timeout')), 120000);
      w.onmessage = (ev: MessageEvent) => {
        const r = ev.data;
        out.push({ ...r, buffer: Array.from(new Uint8ClampedArray(r.buffer)) });
        if (out.length === rs.length) {
          clearTimeout(timer);
          resolve(out);
        }
      };
      w.onerror = (e: ErrorEvent) => reject(new Error('worker error: ' + e.message));
    });
    for (const q of rs) {
      const buf = new Uint8ClampedArray(q.data).buffer;
      w.postMessage({ requestId: q.id, width: q.w, height: q.h, buffer: buf, settings: q.settings }, [buf]);
    }
    return done;
  }, reqs);
}

async function runWorkerChecks(page: Page): Promise<void> {
  const w = 24;
  const h = 20;
  const data = makeUint8ImageFromIntensity(makeIntensityCheckerNoisy(w, h, 61), w, h);
  const settings: ProcessingSettings = { ...DEFAULT_SETTINGS, sceneHeuristics: true, noiseReduction: 25, noiseMethod: 'gaussian', sharpenAmount: 40, processingEngine: 'webgpu' };
  const cpuRef = runPipeline({ data, width: w, height: h } as unknown as ImageData, { ...settings, processingEngine: 'cpu' }).output.data as unknown as ArrayLike<number>;
  const asRun = (r: any): AsyncRun => ({ data: r.buffer, execution: r.execution, precision: r.precision });

  // 1. Real WebGPU inside a real Worker.
  let [r] = await workerCall(page, [{ id: 1, w, h, data: Array.from(data), settings }]);
  recordCustom('worker/no error on a valid request', !r.error && r.requestId === 1 && r.width === w && r.height === h, r.error ?? `id=${r.requestId} ${r.width}x${r.height}, ${r.durationMs.toFixed(0)}ms`);
  assertHonest("worker: engine='webgpu' runs on WebGPU inside the Worker", asRun(r), 'webgpu', false);
  record('worker/webgpu pixels vs CPU (8-bit RGB)', compareInterleavedChannels(cpuRef, r.buffer, 4, 3, 3).combined, 3, 0.05);

  // 2. Other engines through the same worker report truthfully.
  [r] = await workerCall(page, [{ id: 2, w, h, data: Array.from(data), settings: { ...settings, processingEngine: 'cpu' } }]);
  assertHonest("worker: engine='cpu'", asRun(r), 'cpu', false);
  recordStrict("worker: engine='cpu' pixels equal the CPU pipeline", compareInterleavedChannels(cpuRef, r.buffer, 4, 1, 3).combined, 1, 0.001);
  [r] = await workerCall(page, [{ id: 3, w, h, data: Array.from(data), settings: { ...settings, processingEngine: 'webgl2' } }]);
  assertHonest("worker: engine='webgl2' (never touches WebGPU)", asRun(r), 'webgl2', false);

  // 3. A burst of requests is processed strictly in order (serialized queue): ids come back 10,11,12,13,14.
  const burst: WorkerReq[] = [10, 11, 12, 13, 14].map((id, i) => ({
    id,
    w: 16 + i,
    h: 12 + i,
    data: Array.from(makeUint8ImageFromIntensity(makeIntensityCheckerNoisy(16 + i, 12 + i, 70 + i), 16 + i, 12 + i)),
    settings: { ...settings, noiseReduction: 10 * i },
  }));
  const resp = await workerCall(page, burst);
  recordCustom('worker/burst of 5 requests answered in submission order', resp.map((x) => x.requestId).join(',') === '10,11,12,13,14', `arrival order: ${resp.map((x) => x.requestId).join(',')}`);
  recordCustom(
    'worker/each burst response has its own request dimensions and ran on WebGPU',
    resp.every((x, i) => !x.error && x.width === 16 + i && x.height === 12 + i && x.buffer.length === (16 + i) * (12 + i) * 4 && x.execution?.executed === 'webgpu'),
    resp.map((x) => `${x.width}x${x.height}:${x.execution?.executed}`).join(' ')
  );

  // 4. A failing request reports an error, and does NOT poison the queue for what follows.
  const mixed = await workerCall(page, [
    { id: 20, w: 10, h: 10, data: [1, 2, 3, 4, 5, 6, 7, 8], settings }, // 8 bytes for a 10x10 RGBA image: ImageData constructor throws
    { id: 21, w, h, data: Array.from(data), settings },
  ]);
  recordCustom('worker/a malformed request answers with an error string (no crash)', mixed[0].requestId === 20 && typeof mixed[0].error === 'string' && mixed[0].error.length > 0, String(mixed[0].error).slice(0, 100));
  recordCustom('worker/the request AFTER a failure still succeeds on WebGPU (queue not poisoned)', mixed[1].requestId === 21 && !mixed[1].error && mixed[1].execution?.executed === 'webgpu', `id=${mixed[1].requestId} executed=${mixed[1].execution?.executed}`);
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// Section C5: data movement -- the GPU-resident pipeline
// ---------------------------------------------------------------------
//
// Transfers are counted at the WebGPU API boundary by wrapping
// GPUQueue.writeTexture / GPUCommandEncoder.copyTextureToBuffer /
// copyTextureToTexture IN THE PAGE, independently of the pipeline's own
// `resources.transfers` self-report, and the two are cross-checked. Float
// RGBA readbacks (rgba32float) are the CPU-statistics readback; packed
// readbacks (r32uint) are final 8-bit pixels at 4 B/px.

interface WireCounts {
  upR: number; // r32float uploads (the intensity input)
  upRgba: number; // rgba32float uploads (data re-staged through the CPU)
  rbFloat: number; // rgba32float readbacks
  rbPacked: number; // r32uint readbacks
  rbFloatBytes: number;
  rbPackedBytes: number;
  upBytes: number;
  gpuCopies: number;
  events: string[];
}

interface CountedRun {
  wire: WireCounts;
  self: any;
  rgba: number[];
  tiling: TilingDiagnostics;
  precision?: unknown;
}

async function runCounted(page: Page, c: { width: number; height: number; intensity: Float32Array; settings: ProcessingSettings; stops: ColorStop[]; quality: { gaussianPasses: number; localContrastRadius: number } }, opts: { maxTileDim?: number; residentBudgetBytes?: number }): Promise<CountedRun> {
  return page.evaluate(
    async (args: [number[], number, number, ProcessingSettings, ColorStop[], typeof c.quality, typeof opts]) => {
      const [arr, w, h, st, stops, q, o] = args;
      const G = window as any;
      const wire = { upR: 0, upRgba: 0, rbFloat: 0, rbPacked: 0, rbFloatBytes: 0, rbPackedBytes: 0, upBytes: 0, gpuCopies: 0, events: [] as string[] };
      const undo: Array<() => void> = [];
      const wrap = (proto: any, key: string, fn: (orig: any) => any) => {
        const orig = proto[key];
        proto[key] = fn(orig);
        undo.push(() => (proto[key] = orig));
      };
      wrap(G.GPUQueue.prototype, 'writeTexture', (orig) => function (this: any, dest: any, data: any, layout: any, size: any) {
        const fmt = dest.texture.format as string;
        if (fmt === 'r32float') { wire.upR++; wire.events.push('up:r32float'); } else { wire.upRgba++; wire.events.push('up:' + fmt); }
        wire.upBytes += layout.bytesPerRow * (size.height ?? 1);
        return orig.call(this, dest, data, layout, size);
      });
      wrap(G.GPUCommandEncoder.prototype, 'copyTextureToBuffer', (orig) => function (this: any, src: any, dst: any, size: any) {
        const fmt = src.texture.format as string;
        const bytes = size.width * size.height * (fmt === 'rgba32float' ? 16 : 4);
        if (fmt === 'rgba32float') { wire.rbFloat++; wire.rbFloatBytes += bytes; } else { wire.rbPacked++; wire.rbPackedBytes += bytes; }
        wire.events.push('rb:' + fmt);
        return orig.call(this, src, dst, size);
      });
      wrap(G.GPUCommandEncoder.prototype, 'copyTextureToTexture', (orig) => function (this: any, ...a: any[]) {
        wire.gpuCopies++;
        return orig.apply(this, a);
      });
      try {
        const H = G.__gpuHarness;
        const r = await H.runFullPipelineWebGPU(new Float32Array(arr), w, h, st, stops, q, o);
        return { wire, self: r.resources.transfers, rgba: Array.from(r.rgba as Uint8ClampedArray), tiling: r.tiling, precision: r.precision };
      } finally {
        for (const u of undo.reverse()) u();
      }
    },
    [Array.from(c.intensity), c.width, c.height, c.settings, c.stops, c.quality, opts] as [number[], number, number, ProcessingSettings, ColorStop[], typeof c.quality, typeof opts]
  );
}

function selfMatchesWire(r: CountedRun): string[] {
  const s = r.self;
  const w = r.wire;
  const bad: string[] = [];
  const eq = (name: string, a: number, b: number) => { if (a !== b) bad.push(`${name}: self-report ${a} != observed ${b}`); };
  eq('input uploads', s.uploads.input.count, w.upR);
  eq('restaged uploads', s.uploads.restaged.count, w.upRgba);
  eq('upload bytes', s.uploads.input.bytes + s.uploads.restaged.bytes, w.upBytes);
  eq('statistics readbacks', s.readbacks.statistics.count, w.rbFloat);
  eq('statistics bytes', s.readbacks.statistics.bytes, w.rbFloatBytes);
  eq('output readbacks', s.readbacks.output.count, w.rbPacked);
  eq('output bytes', s.readbacks.output.bytes, w.rbPackedBytes);
  eq('gpu copies', s.gpuCopies, w.gpuCopies);
  return bad;
}

async function runDataMovementChecks(page: Page): Promise<void> {
  const stops = PRESETS.natural.colorStops;
  const base: ProcessingSettings = { ...neutralSettings(), preset: 'natural' as const, noiseReduction: 30, noiseMethod: 'gaussian' as const, localContrast: 25, sceneHeuristics: true, detailPreservation: 30 };
  const variants: Record<string, ProcessingSettings> = {
    'all-on': { ...base, precisionPipeline: true, sharpenAmount: 60, sharpenRadius: 2 },
    'precision-only': { ...base, precisionPipeline: true, sharpenAmount: 0 },
    'sharpen-only': { ...base, precisionPipeline: false, sharpenAmount: 80, sharpenRadius: 3 },
    neither: { ...base, precisionPipeline: false, sharpenAmount: 0 },
  };
  const Q = NEUTRAL_QUALITY;

  // ---- single tile: exact event sequences and byte totals --------------------------------------------------------
  const W = 200;
  const H = 150;
  const n = W * H;
  const inten = makeIntensityCheckerNoisy(W, H, 91);
  const expectSingle: Record<string, { events: string[]; bytesPerPx: number; mode: string }> = {
    'all-on': { events: ['up:r32float', 'rb:rgba32float', 'rb:r32uint'], bytesPerPx: 4 + 16 + 4, mode: 'resident' },
    'precision-only': { events: ['up:r32float', 'rb:rgba32float', 'rb:r32uint'], bytesPerPx: 4 + 16 + 4, mode: 'resident' },
    'sharpen-only': { events: ['up:r32float', 'rb:r32uint'], bytesPerPx: 4 + 4, mode: 'resident' },
    neither: { events: ['up:r32float', 'rb:r32uint'], bytesPerPx: 4 + 4, mode: 'direct' },
  };
  for (const [name, settings] of Object.entries(variants)) {
    const r = await runCounted(page, { width: W, height: H, intensity: inten, settings, stops, quality: Q }, {});
    const e = expectSingle[name];
    const total = r.wire.upBytes + r.wire.rbFloatBytes + r.wire.rbPackedBytes;
    const seqOk = JSON.stringify(r.wire.events) === JSON.stringify(e.events);
    recordCustom(`data-movement/single tile ${name}: exact transfer sequence`, seqOk && r.self.mode === e.mode, `observed [${r.wire.events.join(', ')}], mode=${r.self.mode} (expected [${e.events.join(', ')}], ${e.mode})`);
    recordCustom(`data-movement/single tile ${name}: ${e.bytesPerPx} B/px total (was 3 uploads + 3 float readbacks before the GPU-resident rewrite)`, total === e.bytesPerPx * n, `${total / n} B/px observed`);
    const mism = selfMatchesWire(r);
    recordCustom(`data-movement/single tile ${name}: self-reported transfers match the API-boundary count`, mism.length === 0, mism.length === 0 ? 'all counters equal' : mism.join('; '));
  }

  // ---- multi tile: per-phase transfer counts follow directly from the tile plans --------------------------------
  const W2 = 330;
  const H2 = 240;
  const n2 = W2 * H2;
  const inten2 = makeIntensityCheckerNoisy(W2, H2, 92);
  for (const streaming of [false, true]) {
    for (const [name, settings] of Object.entries(variants)) {
      const opts = streaming ? { maxTileDim: 128, residentBudgetBytes: 0 } : { maxTileDim: 128 };
      const r = await runCounted(page, { width: W2, height: H2, intensity: inten2, settings, stops, quality: Q }, opts);
      const t = r.tiling;
      const tilesA = t.colorMap.tileCount;
      const tilesE = t.sharpen?.tileCount ?? 0;
      const tilesP = t.precision?.tileCount ?? 0;
      const precision = settings.precisionPipeline;
      const sharpen = settings.sharpenAmount > 0;
      const direct = !precision && !sharpen;
      const mode = direct ? 'direct' : streaming ? 'streaming' : 'resident';
      const exp = direct || !streaming
        ? { upR: tilesA, upRgba: 0, rbFloat: precision ? tilesA : 0, rbPacked: sharpen ? tilesE : tilesA }
        : { upR: tilesA, upRgba: (precision ? tilesP : 0) + (sharpen ? tilesE : 0), rbFloat: precision ? tilesA : 0, rbPacked: (precision ? tilesP : tilesA) + (sharpen ? tilesE : 0) };
      const got = { upR: r.wire.upR, upRgba: r.wire.upRgba, rbFloat: r.wire.rbFloat, rbPacked: r.wire.rbPacked };
      const label = `${streaming ? 'streaming' : 'resident'} ${name} ${W2}x${H2}/128px tiles (${t.colorMap.cols}x${t.colorMap.rows})`;
      recordCustom(`data-movement/${label}: transfer counts follow the tile plans`, JSON.stringify(got) === JSON.stringify(exp) && r.self.mode === mode, `observed ${JSON.stringify(got)} mode=${r.self.mode}; expected ${JSON.stringify(exp)} mode=${mode}`);
      // Whatever the mode: the only float RGBA readback is the CPU-statistics one, and it covers the image exactly once.
      recordCustom(`data-movement/${label}: float readback is ONLY the statistics readback (${precision ? '16 B/px x image once' : 'none'})`, r.wire.rbFloatBytes === (precision ? 16 * n2 : 0), `${r.wire.rbFloatBytes} B (expected ${precision ? 16 * n2 : 0})`);
      // Packed output volume is exact: one 4 B/px pass over the image, plus a second one only when streaming has a
      // sharpen phase (streaming stages the pre-sharpen image through the CPU, so both phases read packed pixels back).
      const expPackedBytes = 4 * n2 * (streaming && sharpen ? 2 : 1);
      recordCustom(`data-movement/${label}: output readbacks are packed 8-bit, exactly ${streaming && sharpen ? '2' : '1'} x 4 B/px`, r.wire.rbPackedBytes === expPackedBytes, `${r.wire.rbPackedBytes} B observed, ${expPackedBytes} B expected`);
      if (!streaming) {
        // 'resident => no re-staged upload, ever' holds regardless of whether there is a statistics sync to
        // speak of (precision off => cores are quantised/packed straight from phase A, still with 0 uploads
        // after the FIRST phase-A upload). Only meaningful WITH precision: nothing but packed readbacks can
        // follow the statistics readback that ends the sync (a later tile's phase-A upload would be a re-fetch
        // of data the GPU already held -- the resident promise this check exists to guard).
        recordCustom(`data-movement/${label}: resident => zero re-staged uploads (nothing leaves and re-enters the GPU)`, r.wire.upRgba === 0, `${r.wire.upRgba} re-staged uploads (observed events: ${[...new Set(r.wire.events)].join(',')})`);
        if (precision) {
          const lastStat = r.wire.events.lastIndexOf('rb:rgba32float');
          const after = r.wire.events.slice(lastStat + 1);
          recordCustom(`data-movement/${label}: nothing but packed output readbacks follow the statistics sync (no re-upload, no re-fetch of source data)`, lastStat >= 0 && after.length > 0 && after.every((e) => e === 'rb:r32uint'), `events after the last statistics readback: ${[...new Set(after)].join(',') || '(none)'}`);
        }
        if (!direct) recordCustom(`data-movement/${label}: resident halos/cores moved GPU-to-GPU`, r.wire.gpuCopies >= tilesA + (sharpen ? tilesE : 0), `${r.wire.gpuCopies} GPU copies for ${tilesA} cores${sharpen ? ` + ${tilesE} sharpen tiles` : ''}`);
      } else if (!direct) {
        recordCustom(`data-movement/${label}: streaming used no GPU-side halo copies (CPU-staged)`, r.wire.gpuCopies === 0, `${r.wire.gpuCopies} GPU copies`);
      }
      const mism = selfMatchesWire(r);
      recordCustom(`data-movement/${label}: self-reported transfers match the API-boundary count`, mism.length === 0, mism.length === 0 ? 'all counters equal' : mism.join('; '));
    }
  }

  // ---- equivalence: resident vs streaming, tiled vs non-tiled (must be bit-identical) -----------------------------
  const eqCases: Array<{ label: string; w: number; h: number; seed: number; settings: ProcessingSettings; dim: number }> = [
    { label: 'all-on 200x150', w: 200, h: 150, seed: 93, settings: variants['all-on'], dim: 160 },
    { label: 'all-on 330x240', w: 330, h: 240, seed: 94, settings: variants['all-on'], dim: 128 },
    { label: 'precision-only 330x240', w: 330, h: 240, seed: 95, settings: variants['precision-only'], dim: 128 },
    { label: 'sharpen-only 330x240', w: 330, h: 240, seed: 96, settings: variants['sharpen-only'], dim: 128 },
    { label: 'median all-on 330x240', w: 330, h: 240, seed: 97, settings: { ...variants['all-on'], noiseMethod: 'median' as const, noiseReduction: 60 }, dim: 128 },
    { label: 'everything-maxed 300x290 (65px halo)', w: 300, h: 290, seed: 98, settings: MAXED_SETTINGS, dim: 256 },
  ];
  for (const ec of eqCases) {
    const st = PRESETS[ec.settings.preset].colorStops;
    const q = qualityParamsFor(ec.settings.quality);
    const res = await page.evaluate(
      async (args: [number[], number, number, ProcessingSettings, ColorStop[], typeof q, number]) => {
        const [arr, w, h, s, stops2, qq, dim] = args;
        const HH = (window as any).__gpuHarness;
        const run = async (o: any) => { const r = await HH.runFullPipelineWebGPU(new Float32Array(arr), w, h, s, stops2, qq, o); return { d: Array.from(r.rgba as Uint8ClampedArray), mode: r.resources.transfers.mode, tiles: r.tiling.colorMap.tileCount, p: r.precision }; };
        return { resSingle: await run({ maxTileDim: 8192 }), strSingle: await run({ maxTileDim: 8192, residentBudgetBytes: 0 }), resTiled: await run({ maxTileDim: dim }), strTiled: await run({ maxTileDim: dim, residentBudgetBytes: 0 }) };
      },
      [Array.from(makeIntensityCheckerNoisy(ec.w, ec.h, ec.seed)), ec.w, ec.h, ec.settings, st, q, ec.dim] as [number[], number, number, ProcessingSettings, ColorStop[], typeof q, number]
    );
    const modesOk = res.resSingle.mode === 'resident' && res.strSingle.mode === 'streaming' && res.resTiled.mode === 'resident' && res.strTiled.mode === 'streaming' && res.resSingle.tiles === 1 && res.resTiled.tiles > 1;
    recordCustom(`equivalence/${ec.label}: the four runs really are resident/streaming x single/tiled`, modesOk, `${res.resSingle.mode}/${res.strSingle.mode}/${res.resTiled.mode}/${res.strTiled.mode}; tiles ${res.resSingle.tiles} vs ${res.resTiled.tiles}`);
    recordStrict(`equivalence/${ec.label}: resident == streaming (single tile, bit-identical)`, compareArrays(res.resSingle.d, res.strSingle.d, 0), 0, 0);
    recordStrict(`equivalence/${ec.label}: resident tiled == resident single (bit-identical)`, compareArrays(res.resSingle.d, res.resTiled.d, 0), 0, 0);
    recordStrict(`equivalence/${ec.label}: streaming tiled == resident single (bit-identical)`, compareArrays(res.resSingle.d, res.strTiled.d, 0), 0, 0);
    if (res.resSingle.p) recordStrict(`equivalence/${ec.label}: precision diagnostics identical in all four`, compareArrays(numericLeaves(res.resSingle.p), numericLeaves(res.strTiled.p), 0), 0, 1);
  }
}

// ---------------------------------------------------------------------
// Section D2: fallback truthfulness when the failure happens BETWEEN stages
// ---------------------------------------------------------------------
//
// With GPU-resident stages, a failure can now occur after real GPU work has
// completed (the precision/sharpen tail). Force a failure at EVERY
// queue.submit index of a multi-tile run; each must report the fallback
// truthfully and return exactly the pixels the WebGL2 pipeline produces.

async function runFailAtSubmit(page: Page, data: Uint8ClampedArray, width: number, height: number, settings: ProcessingSettings, failAt: number, tileDim: number): Promise<AsyncRun & { submits: number }> {
  return page.evaluate(
    async (args: [number[], number, number, ProcessingSettings, number, number]) => {
      const [arr, w, h, s, k, dim] = args;
      const G = window as any;
      const orig = G.GPUQueue.prototype.submit;
      let calls = 0;
      G.GPUQueue.prototype.submit = function (this: any, ...a: any[]) {
        calls++;
        if (calls === k) throw new Error(`injected: queue.submit #${k} failure`);
        return orig.apply(this, a);
      };
      try {
        const out = await G.__gpuHarness.runPipelineAsync(new ImageData(new Uint8ClampedArray(arr), w, h), s, { webgpu: { maxTileDim: dim } });
        return { data: Array.from(out.output.data as Uint8ClampedArray), execution: out.execution, precision: out.precision, submits: calls };
      } finally {
        G.GPUQueue.prototype.submit = orig;
      }
    },
    [Array.from(data), width, height, settings, failAt, tileDim] as [number[], number, number, ProcessingSettings, number, number]
  );
}

async function runSubmitFailureSweep(page: Page): Promise<void> {
  const w = 260;
  const h = 100;
  const data = makeUint8ImageFromIntensity(makeIntensityCheckerNoisy(w, h, 96), w, h);
  const settings: ProcessingSettings = { ...DEFAULT_SETTINGS, sceneHeuristics: true, noiseReduction: 25, noiseMethod: 'gaussian', sharpenAmount: 40, processingEngine: 'webgpu' };
  const dim = 128;
  const ref = await runAsyncInPage(page, data, w, h, { ...settings, processingEngine: 'webgl2' }, 'none');
  const clean = await runFailAtSubmit(page, data, w, h, settings, 0, dim); // failAt 0 => never fails: counts submits
  const K = clean.submits;
  recordCustom('submit-sweep/clean multi-tile WebGPU run executes on WebGPU and has several submits', clean.execution.executed === 'webgpu' && K >= 8, `executed=${clean.execution.executed}, ${K} submits, ${clean.execution.tiling?.colorMap.tileCount} colour-map tiles`);
  let allHonest = true;
  const problems: string[] = [];
  for (let k = 1; k <= K; k++) {
    const r = await runFailAtSubmit(page, data, w, h, settings, k, dim);
    const ex = r.execution;
    const first = ex.attempts[0];
    const ok =
      ex.executed === 'webgl2' &&
      ex.fellBack === true &&
      first?.backend === 'webgpu' && first?.ok === false && String(first?.reason).includes(`submit #${k}`) &&
      !ex.adapter && !ex.resources &&
      (!r.precision || r.precision.backend.resolved !== 'webgpu') &&
      compareArrays(ref.data, r.data, 0).maxError === 0;
    if (!ok) { allHonest = false; problems.push(`k=${k}: executed=${ex.executed} fellBack=${ex.fellBack} reason=${String(first?.reason).slice(0, 60)} pixelsEqualWebGL2=${compareArrays(ref.data, r.data, 0).maxError === 0}`); }
  }
  recordCustom(`submit-sweep/a failure at EACH of the ${K} submits (incl. mid-tail, after GPU work completed) -> truthful WebGL2 fallback with pixels == WebGL2`, allHonest, allHonest ? `${K}/${K} injected failures reported executed=webgl2, never webgpu, adapter/resources absent, pixels identical to the WebGL2 pipeline` : problems.slice(0, 3).join(' | '));
}

async function main(): Promise<void> {
  console.log('Starting WebGPU consistency harness (real headless Chromium + WebGPU)...\n');
  let h: WebGPUHarness;
  try {
    h = await startWebGPUHarness();
  } catch (e) {
    if (e instanceof WebGPUUnavailableInHarness) {
      console.error(`\nWebGPU is NOT available in this environment: ${e.message}\nThis harness cannot run and NOTHING was verified. Exiting with code 2 (not a pass).`);
      process.exit(2);
    }
    throw e;
  }
  const { browser, page, server, adapter, maxTextureDimension2D: deviceMax } = h;
  console.log(`WebGPU adapter: vendor=${adapter.vendor} architecture=${adapter.architecture} software=${adapter.software} isFallbackAdapter=${adapter.isFallbackAdapter}; maxTextureDimension2D=${deviceMax}`);
  if (adapter.software) console.log('NOTE: this is a SOFTWARE WebGPU adapter (Chrome Dawn on SwiftShader Vulkan), not a hardware GPU.\n');

  try {
    console.log('-- A. Shaders / device --');
    await checkShadersCompile(page);

    console.log('\n-- B1. WebGPU vs CPU: stage-by-stage (identical fixtures/tolerances to the WebGL2 harness) --');
    await checkNoiseReductionGaussian(page);
    await checkNoiseReductionBilateral(page);
    await checkNoiseReductionMedian(page);
    await checkLocalContrast(page);
    await checkToneCurve(page);
    await checkColorMapping(page, 'natural');
    await checkColorMapping(page, 'portrait');
    await checkColorMappingWithScene(page);
    await checkColorMappingSceneSkyVeg(page);
    await checkSharpen(page);
    await checkPrecisionPipeline(page);

    console.log('\n-- B2. WebGPU vs CPU: end-to-end + stress / edge cases --');
    await checkEndToEnd(page, 'defaults', DEFAULT_SETTINGS, 16, 16, 21);
    await checkEndToEnd(page, 'kitchen-sink', { ...DEFAULT_SETTINGS, preset: 'landscape', noiseReduction: 35, noiseMethod: 'gaussian', localContrast: 25, sharpenAmount: 50, sceneHeuristics: true, saturation: 15, temperature: -10 }, 20, 20, 42);
    await checkEndToEnd(page, 'bilateral-no-precision', { ...DEFAULT_SETTINGS, noiseReduction: 30, noiseMethod: 'bilateral', precisionPipeline: false }, 18, 18, 5);
    await checkEndToEnd(page, 'median-NR (not available on WebGL2)', { ...DEFAULT_SETTINGS, noiseReduction: 50, noiseMethod: 'median' }, 20, 20, 8);
    await checkEndToEnd(page, 'monochrome-quality-high', { ...DEFAULT_SETTINGS, preset: 'monochrome', quality: 'high', sharpenAmount: 40 }, 16, 16, 9);
    await checkNoiseExtreme(page, 'gaussian', 100);
    await checkNoiseExtreme(page, 'bilateral', 100);
    await checkLocalContrastExtreme(page);
    await checkToneCurveExtreme(page);
    for (const presetId of Object.keys(PRESETS) as (keyof typeof PRESETS)[]) await checkColorMappingExtreme(page, presetId);
    await checkSharpenExtreme(page);
    await checkEndToEnd(page, 'maximum-quality-everything-maxed', MAXED_SETTINGS, 28, 28, 99);
    await checkEndToEndSolid(page, 'solid-black', 0);
    await checkEndToEndSolid(page, 'solid-white', 255);
    await checkEndToEndSolid(page, 'solid-mid-gray', 128);
    await checkEndToEnd(page, 'non-square-wide', { ...DEFAULT_SETTINGS, sceneHeuristics: true, noiseReduction: 20 }, 31, 11, 17);
    await checkEndToEnd(page, 'non-square-tall', { ...DEFAULT_SETTINGS, sceneHeuristics: true, localContrast: 30 }, 11, 31, 23);

    console.log('\n-- B3. Three-way CPU / WebGL2 / WebGPU on the same inputs --');
    const kitchen: ProcessingSettings = { ...DEFAULT_SETTINGS, preset: 'landscape', quality: 'high', noiseReduction: 60, noiseMethod: 'gaussian', localContrast: 40, sharpenAmount: 80, sceneHeuristics: true, saturation: 15, temperature: -10 };
    await checkThreeWay(page, 'defaults 16x16', DEFAULT_SETTINGS, 16, 16, 21);
    await checkThreeWay(page, 'kitchen-sink 350x240', kitchen, 350, 240, 75);
    await checkThreeWay(page, 'bilateral+precision-off 200x150', { ...DEFAULT_SETTINGS, noiseReduction: 30, noiseMethod: 'bilateral', precisionPipeline: false }, 200, 150, 5);
    await checkThreeWay(page, 'everything-maxed 300x290', MAXED_SETTINGS, 300, 290, 76);
    await checkThreeWay(page, 'non-square-wide 700x60', { ...DEFAULT_SETTINGS, sceneHeuristics: true }, 700, 60, 77);
    await checkThreeWay(page, 'non-square-tall 60x700', { ...DEFAULT_SETTINGS, sceneHeuristics: true }, 60, 700, 78);

    console.log('\n-- C. WebGPU tiling: tiled vs untiled (must be bit-identical) --');
    const stopsMono = PRESETS.monochrome.colorStops;
    const maxQ = qualityParamsFor('maximum');
    const mk = (over: Partial<ProcessingSettings>): ProcessingSettings => ({ ...neutralSettings(), preset: 'monochrome' as const, ...over });
    const noisy = makeIntensityCheckerNoisy;
    const nrGauss: WgCase = { label: 'gaussian-NR@100/quality=maximum 330x240', width: 330, height: 240, intensity: noisy(330, 240, 71), settings: mk({ noiseReduction: 100, noiseMethod: 'gaussian' }), stops: stopsMono, quality: maxQ, maxTileDim: 128, minTiles: 12, seamBand: true };
    const nrGaussRes = await checkWgTiledVsUntiled(page, nrGauss);
    await checkWgTiledVsUntiled(page, { label: 'bilateral-NR@100 330x240', width: 330, height: 240, intensity: noisy(330, 240, 72), settings: mk({ noiseReduction: 100, noiseMethod: 'bilateral' }), stops: stopsMono, quality: NEUTRAL_QUALITY, maxTileDim: 96, minTiles: 12, seamBand: true });
    const median2: WgCase = { label: 'median-NR@100 (radius 2) 330x240', width: 330, height: 240, intensity: noisy(330, 240, 73), settings: mk({ noiseReduction: 100, noiseMethod: 'median' }), stops: stopsMono, quality: NEUTRAL_QUALITY, maxTileDim: 96, minTiles: 12, seamBand: true };
    await checkWgTiledVsUntiled(page, median2);
    await checkWgTiledVsUntiled(page, { label: 'median-NR@40 (radius 1) 330x240', width: 330, height: 240, intensity: noisy(330, 240, 74), settings: mk({ noiseReduction: 40, noiseMethod: 'median' }), stops: stopsMono, quality: NEUTRAL_QUALITY, maxTileDim: 96, minTiles: 12, seamBand: true });
    const lc: WgCase = { label: 'local-contrast@100/radius36 330x240', width: 330, height: 240, intensity: noisy(330, 240, 75), settings: mk({ localContrast: 100 }), stops: stopsMono, quality: maxQ, maxTileDim: 160, minTiles: 9, seamBand: true };
    await checkWgTiledVsUntiled(page, lc);
    await checkWgTiledVsUntiled(page, { label: 'scene-heuristics vertical gradient 200x360 (global row fraction)', width: 200, height: 360, intensity: makeIntensityVerticalGradient(200, 360), settings: { ...mk({ sceneHeuristics: true }), preset: 'landscape' }, stops: PRESETS.landscape.colorStops, quality: NEUTRAL_QUALITY, maxTileDim: 128, minTiles: 6, seamBand: true });
    const sharp: WgCase = { label: 'sharpen@200/radius5/threshold0 330x240', width: 330, height: 240, intensity: noisy(330, 240, 76), settings: mk({ sharpenAmount: 200, sharpenRadius: 5, sharpenThreshold: 0 }), stops: stopsMono, quality: NEUTRAL_QUALITY, maxTileDim: 96, minTiles: 1, seamBand: true };
    const sharpRes = await checkWgTiledVsUntiled(page, sharp);
    recordCustom('webgpu-tiled-vs-untiled/sharpen phase tiled', !!sharpRes.tiledTiling.sharpen && sharpRes.tiledTiling.sharpen.tileCount > 1, `${sharpRes.tiledTiling.sharpen?.cols}x${sharpRes.tiledTiling.sharpen?.rows}, halo ${sharpRes.tiledTiling.sharpen?.halo}`);
    const precRun = await checkWgTiledVsUntiled(page, { label: 'precision-pipeline two-halves 300x200 (per-tile stats would differ)', width: 300, height: 200, intensity: makeIntensityTwoHalves(300, 200), settings: { ...mk({ precisionPipeline: true, whiteBalanceStrength: 100, autoToneStrength: 100, colorCorrectionStrength: 100 }), preset: 'natural' }, stops: PRESETS.natural.colorStops, quality: NEUTRAL_QUALITY, maxTileDim: 100, minTiles: 6 });
    const cc = (precRun.tiledP as { colorCorrection: { gainR: number; gainG: number; gainB: number } }).colorCorrection;
    recordCustom('webgpu-tiled-vs-untiled/precision stage is non-trivial on this image', Math.max(Math.abs(cc.gainR - 1), Math.abs(cc.gainG - 1), Math.abs(cc.gainB - 1)) > 1e-3, `CC gains ${cc.gainR.toFixed(4)} ${cc.gainG.toFixed(4)} ${cc.gainB.toFixed(4)}`);
    await checkWgTiledVsUntiled(page, { label: 'kitchen-sink 350x240', width: 350, height: 240, intensity: noisy(350, 240, 77), settings: kitchen, stops: PRESETS.landscape.colorStops, quality: qualityParamsFor('high'), maxTileDim: 192, minTiles: 8 });
    await checkWgTiledVsUntiled(page, { label: 'everything-maxed/quality=maximum 300x290 (65px halo, 3x3 grid)', width: 300, height: 290, intensity: noisy(300, 290, 78), settings: MAXED_SETTINGS, stops: PRESETS['high-contrast'].colorStops, quality: maxQ, maxTileDim: 256, minTiles: 9 });
    await checkWgTiledVsUntiled(page, { label: 'median+everything 300x250', width: 300, height: 250, intensity: noisy(300, 250, 79), settings: { ...MAXED_SETTINGS, noiseMethod: 'median' as const, quality: 'high' as const }, stops: PRESETS['high-contrast'].colorStops, quality: qualityParamsFor('high'), maxTileDim: 192, minTiles: 4 });
    await checkWgTiledVsUntiled(page, { label: 'non-square-wide defaults 700x60', width: 700, height: 60, intensity: noisy(700, 60, 80), settings: DEFAULT_SETTINGS, stops: PRESETS.natural.colorStops, quality: NEUTRAL_QUALITY, maxTileDim: 128, minTiles: 8 });
    await checkWgTiledVsUntiled(page, { label: 'non-square-tall defaults 60x700', width: 60, height: 700, intensity: noisy(60, 700, 81), settings: DEFAULT_SETTINGS, stops: PRESETS.natural.colorStops, quality: NEUTRAL_QUALITY, maxTileDim: 128, minTiles: 8 });
    await checkWgTiledVsUntiled(page, { label: 'boundary: one pixel over tile size 129x100', width: 129, height: 100, intensity: noisy(129, 100, 82), settings: DEFAULT_SETTINGS, stops: PRESETS.natural.colorStops, quality: NEUTRAL_QUALITY, maxTileDim: 128, minTiles: 2 });
    await checkWgTiledVsUntiled(page, { label: 'boundary: exactly one tile 128x100', width: 128, height: 100, intensity: noisy(128, 100, 83), settings: DEFAULT_SETTINGS, stops: PRESETS.natural.colorStops, quality: NEUTRAL_QUALITY, maxTileDim: 128, minTiles: 1 });
    await checkWgTiledVsUntiled(page, { label: 'boundary: tiny 3x2 image', width: 3, height: 2, intensity: noisy(3, 2, 84), settings: DEFAULT_SETTINGS, stops: PRESETS.natural.colorStops, quality: NEUTRAL_QUALITY, maxTileDim: 128, minTiles: 1 });

    console.log('\n-- C2. Resource management (texture pool / ping-pong) --');
    const res = nrGaussRes.tiledRes;
    recordCustom('resources/multi-tile run pools textures (reuses >> allocations)', res.textureReuses > 3 * res.texturesCreated, `created ${res.texturesCreated}, reused ${res.textureReuses}, peak live ${res.peakTexturesLive}, ${res.passesDispatched} passes in ${res.submits} submits`);
    recordCustom('resources/peak live textures is bounded (no per-tile leak)', res.peakTexturesLive <= 12, `peak ${res.peakTexturesLive} across ${nrGaussRes.tiledTiling.colorMap.tileCount} tiles`);
    const single = await runWgPair(page, { ...nrGauss, maxTileDim: 8192 });
    const s1 = single.tiledRes;
    recordCustom('resources/6-pass Gaussian ping-pongs a small texture set (one tile, 14+ blur passes)', s1.passesDispatched >= 14 && s1.texturesCreated <= 8 && s1.textureReuses >= 8, `${s1.passesDispatched} passes, created ${s1.texturesCreated}, reused ${s1.textureReuses}`);

    console.log('\n-- C3. Negative controls: under-sized halos must produce seams --');
    await checkWgNegativeControl(page, 'gaussian-NR halo=reach/2 (12 vs 24)', nrGauss, { colorMap: 12 });
    await checkWgNegativeControl(page, 'local-contrast halo=reach-1 (35 vs 36)', lc, { colorMap: 35 });
    await checkWgNegativeControl(page, 'local-contrast halo=0 (no overlap)', lc, { colorMap: 0 });
    await checkWgNegativeControl(page, 'sharpen halo=reach-1 (4 vs 5)', sharp, { sharpen: 4 });
    await checkWgNegativeControl(page, 'median radius-2 halo=reach-1 (1 vs 2)', median2, { colorMap: 1 });
    const chain: WgCase = { label: 'chain', width: 330, height: 240, intensity: noisy(330, 240, 85), settings: { ...mk({ noiseReduction: 60, noiseMethod: 'gaussian', localContrast: 40, sceneHeuristics: true }), preset: 'landscape' }, stops: PRESETS.landscape.colorStops, quality: qualityParamsFor('high'), maxTileDim: 192, minTiles: 1 };
    const chainRadii = resolveWebGPUStageRadii(chain.settings, chain.quality);
    const chainHalo = chainRadii.colorMapReach - chainRadii.gaussianRadius * chainRadii.gaussianPasses;
    await checkWgNegativeControl(page, `chained NR+LC+scene halo omits NR reach (${chainHalo} vs ${chainRadii.colorMapReach})`, chain, { colorMap: chainHalo });

    console.log('\n-- C4. Real oversize (beyond the DEVICE texture limit), tile size clamped to the device --');
    const overSettings: ProcessingSettings = { ...DEFAULT_SETTINGS, preset: 'landscape', noiseReduction: 35, noiseMethod: 'gaussian', localContrast: 25, sharpenAmount: 50, sceneHeuristics: true, saturation: 15, temperature: -10 };
    await checkWgOversize(page, deviceMax, `wide ${deviceMax + 8}x12`, overSettings, deviceMax + 8, 12, 101, 'many');
    await checkWgOversize(page, deviceMax, `tall 12x${deviceMax + 8}`, overSettings, 12, deviceMax + 8, 102, 'many');
    await checkWgOversize(page, deviceMax, `just over the limit ${deviceMax + 1}x8`, DEFAULT_SETTINGS, deviceMax + 1, 8, 103, 'many');
    await checkWgOversize(page, deviceMax, `exactly at the limit ${deviceMax}x8 (single tile)`, DEFAULT_SETTINGS, deviceMax, 8, 104, 1);
    await checkWgOversize(page, deviceMax, `median NR wide ${deviceMax + 8}x10`, { ...DEFAULT_SETTINGS, noiseReduction: 50, noiseMethod: 'median', precisionPipeline: false }, deviceMax + 8, 10, 105, 'many');

    console.log('\n-- C5. Data movement: GPU-resident pipeline (independently counted at the WebGPU API boundary) --');
    await runDataMovementChecks(page);

    console.log('\n-- D. Fallback honesty in a real browser (WebGPU -> WebGL2 -> CPU) --');
    await runFallbackChecks(page);

    console.log('\n-- D2. Fallback truthfulness for a failure at every queue.submit (incl. mid-tail) --');
    await runSubmitFailureSweep(page);
    await runPipelineLevelWebGPU(page, deviceMax);

    console.log('\n-- E. The shipped Web Worker (real Worker, real WebGPU) --');
    await runWorkerChecks(page);
  } finally {
    await browser.close();
    server.close();
  }

  console.log('\n-- Summary --');
  const failed = results.filter((r) => !r.pass);
  for (const r of failed) console.log(`FAILED: ${r.name}`);
  console.log(`\n${results.length - failed.length}/${results.length} WebGPU checks passed.`);
  process.exit(failed.length === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
