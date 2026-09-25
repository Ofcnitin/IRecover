import '../../tests/setup';
import { chromium, type Browser, type Page } from 'playwright';
import { runPipeline } from '../../src/processing/pipeline';
import { runPipelineAsync } from '../../src/processing/pipelineAsync';
import { DEFAULT_SETTINGS, type ProcessingSettings } from '../../src/types/processing';
import { makeRamp, makeNoisyChecker, makeVerticalGradient, intensityToRgbaBytes } from './fixtures';
import { executesOnCpu, isGpuCapableBackend } from '../../src/processing/backend';

function metrics(a: Uint8ClampedArray | number[], b: Uint8ClampedArray | number[]) {
  let sum = 0, max = 0, over = 0;
  const tol = 3;
  const n = a.length;
  for (let i = 0; i < n; i++) {
    const d = Math.abs((a[i] as number) - (b[i] as number));
    sum += d;
    if (d > max) max = d;
    if (d > tol) over++;
  }
  return { mae: sum / n, max, overPct: (100 * over) / n, n };
}

type Fixture = 'ramp' | 'noisy' | 'gradient';
function buildIntensity(fixture: Fixture, w: number, h: number): Float32Array {
  if (fixture === 'ramp') return makeRamp(w, h);
  if (fixture === 'gradient') return makeVerticalGradient(w, h);
  return makeNoisyChecker(w, h);
}

interface CheckSpec {
  name: string;
  width: number;
  height: number;
  fixture: Fixture;
  overrides: Partial<ProcessingSettings>;
  skipWebgpu?: boolean;
}

const CHECKS: CheckSpec[] = [
  { name: 'defaults (ramp, small)', width: 64, height: 48, fixture: 'ramp', overrides: {} },
  { name: 'noisy checker + gaussian NR@100 + LC@100', width: 96, height: 64, fixture: 'noisy',
    overrides: { noiseReduction: 100, noiseMethod: 'gaussian', localContrast: 100 } },
  { name: 'noisy checker + bilateral NR@80', width: 80, height: 64, fixture: 'noisy',
    overrides: { noiseReduction: 80, noiseMethod: 'bilateral' } },
  { name: 'sharpen extreme (amt=200,r=5,thr=0)', width: 72, height: 56, fixture: 'ramp',
    overrides: { sharpenAmount: 200, sharpenRadius: 5, sharpenThreshold: 0 } },
  { name: 'preset landscape, extreme sat/temp/hue', width: 70, height: 50, fixture: 'gradient',
    overrides: { preset: 'landscape', saturation: 100, temperature: 100, hueBias: 180 } },
  { name: 'preset monochrome, extreme sat/temp/hue', width: 70, height: 50, fixture: 'gradient',
    overrides: { preset: 'monochrome', saturation: -100, temperature: -100, hueBias: -180 } },
  { name: 'precision pipeline @100/100/100, precision=maximum', width: 90, height: 60, fixture: 'noisy',
    overrides: { precisionPipeline: true, whiteBalanceStrength: 100, autoToneStrength: 100, colorCorrectionStrength: 100, precision: 'maximum' } },
  { name: 'quality=maximum kitchen sink', width: 64, height: 64, fixture: 'noisy',
    overrides: { quality: 'maximum', noiseReduction: 100, localContrast: 100, sharpenAmount: 200, sharpenRadius: 5,
      saturation: 100, temperature: 100, hueBias: 180, precisionPipeline: true, whiteBalanceStrength: 100,
      autoToneStrength: 100, colorCorrectionStrength: 100, precision: 'maximum', sceneHeuristics: true } },
  { name: 'solid black', width: 40, height: 30, fixture: 'ramp', overrides: {}, },
  { name: 'non-square tall', width: 30, height: 90, fixture: 'gradient', overrides: { localContrast: 60, noiseReduction: 40 } },
];

async function evalGpu(page: Page, spec: CheckSpec, engine: 'webgl2' | 'webgpu') {
  return page.evaluate(async ({ spec, engine, DEFAULT_SETTINGS }) => {
    const v = (window as any).__v;
    const fixtureFn = spec.fixture === 'ramp' ? v.makeRamp : spec.fixture === 'gradient' ? v.makeVerticalGradient : v.makeNoisyChecker;
    const intensity = fixtureFn(spec.width, spec.height);
    const bytes = v.intensityToRgbaBytes(intensity, spec.width, spec.height);
    const input = new ImageData(bytes, spec.width, spec.height);
    const settings = { ...DEFAULT_SETTINGS, ...spec.overrides, processingEngine: engine };
    const t0 = performance.now();
    let result;
    if (engine === 'webgpu') {
      result = await v.runPipelineAsync(input, settings);
    } else {
      result = v.runPipeline(input, settings);
    }
    const t1 = performance.now();
    return {
      rgba: Array.from(result.output.data as Uint8ClampedArray),
      execution: result.execution,
      precision: result.precision ? { gpuAccelerated: result.precision.gpuAccelerated, backendResolved: result.precision.backend.resolved } : undefined,
      ms: t1 - t0,
    };
  }, { spec, engine, DEFAULT_SETTINGS });
}

async function main() {
  const browser = await chromium.launch({
    executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
    args: [
      '--use-gl=swiftshader',
      '--enable-unsafe-swiftshader',
      '--enable-unsafe-webgpu',
      '--ignore-gpu-blocklist',
      '--no-sandbox',
    ],
  });
  const page = await browser.newPage();
  const pageErrors: string[] = [];
  page.on('pageerror', (e) => pageErrors.push(e.message));
  await page.goto('http://127.0.0.1:8931/.browser-dist/pages/verify.html');
  await page.waitForTimeout(300);

  const results: any[] = [];

  for (const spec of CHECKS) {
    const intensity = buildIntensity(spec.fixture, spec.width, spec.height);
    const bytes = intensityToRgbaBytes(intensity, spec.width, spec.height);
    const input = new (globalThis as any).ImageData(bytes, spec.width, spec.height);
    const cpuSettings: ProcessingSettings = { ...DEFAULT_SETTINGS, ...spec.overrides, processingEngine: 'cpu' };
    const cpuResult = runPipeline(input, cpuSettings);

    const gl = await evalGpu(page, spec, 'webgl2');
    const glMetrics = metrics(cpuResult.output.data, gl.rgba);

    let wgpu: any = null;
    let wgpuMetrics: any = null;
    let glVsGpuMetrics: any = null;
    if (!spec.skipWebgpu) {
      wgpu = await evalGpu(page, spec, 'webgpu');
      wgpuMetrics = metrics(cpuResult.output.data, wgpu.rgba);
      glVsGpuMetrics = metrics(gl.rgba, wgpu.rgba);
    }

    results.push({
      name: spec.name,
      size: `${spec.width}x${spec.height}`,
      cpuExecuted: cpuResult.execution?.executed,
      webgl2: { executed: gl.execution?.executed, gpuAccelerated: gl.execution?.gpuAccelerated, tiling: !!gl.execution?.tiling, ms: gl.ms.toFixed(2), ...glMetrics },
      webgpu: wgpu ? { executed: wgpu.execution?.executed, gpuAccelerated: wgpu.execution?.gpuAccelerated, tiling: !!wgpu.execution?.tiling, ms: wgpu.ms.toFixed(2), ...wgpuMetrics } : null,
      glVsGpu: glVsGpuMetrics,
    });
  }

  console.log('\n=== Pixel-consistency checks (real CPU in Node vs real WebGL2/WebGPU in real headless Chromium) ===\n');
  for (const r of results) {
    console.log(`- ${r.name} [${r.size}]`);
    console.log(`    CPU executed=${r.cpuExecuted}`);
    console.log(`    WebGL2: executed=${r.webgl2.executed} gpuAccel=${r.webgl2.gpuAccelerated} tiling=${r.webgl2.tiling} time=${r.webgl2.ms}ms  MAE=${r.webgl2.mae.toFixed(4)} max=${r.webgl2.max} overTol(>3)=${r.webgl2.overPct.toFixed(3)}%`);
    if (r.webgpu) {
      console.log(`    WebGPU: executed=${r.webgpu.executed} gpuAccel=${r.webgpu.gpuAccelerated} tiling=${r.webgpu.tiling} time=${r.webgpu.ms}ms  MAE=${r.webgpu.mae.toFixed(4)} max=${r.webgpu.max} overTol(>3)=${r.webgpu.overPct.toFixed(3)}%`);
      console.log(`    WebGL2 vs WebGPU (direct): MAE=${r.glVsGpu.mae.toFixed(4)} max=${r.glVsGpu.max} overTol(>3)=${r.glVsGpu.overPct.toFixed(3)}%`);
    }
  }

  console.log('\n=== Static backend-classification sanity (executesOnCpu / isGpuCapableBackend) ===');
  console.log({
    'executesOnCpu(cpu)': executesOnCpu('cpu'),
    'executesOnCpu(webgl2)': executesOnCpu('webgl2'),
    'executesOnCpu(webgpu)': executesOnCpu('webgpu'),
    'isGpuCapableBackend(cpu)': isGpuCapableBackend('cpu'),
    'isGpuCapableBackend(webgl2)': isGpuCapableBackend('webgl2'),
    'isGpuCapableBackend(webgpu)': isGpuCapableBackend('webgpu'),
  });

  if (pageErrors.length) {
    console.log('\n=== Uncaught page errors ===');
    for (const e of pageErrors) console.log(' -', e);
  }

  await browser.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
