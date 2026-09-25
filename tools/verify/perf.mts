import '../../tests/setup';
import { chromium } from 'playwright';
import { runPipeline } from '../../src/processing/pipeline';
import { DEFAULT_SETTINGS, type ProcessingSettings } from '../../src/types/processing';
import { makeNoisyChecker, intensityToRgbaBytes } from './fixtures';

const SIZES: [number, number, string][] = [
  [256, 256, 'small (256x256, 65K px)'],
  [1024, 768, 'medium (1024x768, 786K px)'],
  [8300, 100, 'oversize width, forces tiling (8300x100, 830K px)'],
];

async function main() {
  const browser = await chromium.launch({
    executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
    args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--no-sandbox'],
  });
  const page = await browser.newPage();
  await page.goto('http://127.0.0.1:8931/.browser-dist/pages/verify.html');
  await page.waitForTimeout(300);

  console.log('\n=== Performance (representative sizes; default settings; SOFTWARE GPU only -- SwiftShader/Dawn, no hardware GPU in this sandbox) ===\n');
  console.log('size'.padEnd(45), 'CPU(ms)'.padEnd(12), 'WebGL2(ms)'.padEnd(12), 'WebGL2 2nd(ms)'.padEnd(16), 'WebGPU(ms)'.padEnd(12), 'WebGPU 2nd(ms)');

  for (const [w, h, label] of SIZES) {
    process.stdout.write(`[running] ${label} ...\n`);
    const intensity = makeNoisyChecker(w, h);
    const bytes = intensityToRgbaBytes(intensity, w, h);
    const input = new (globalThis as any).ImageData(bytes, w, h);
    const settings: ProcessingSettings = { ...DEFAULT_SETTINGS, localContrast: 40, noiseReduction: 30, processingEngine: 'cpu' };

    const cpuStart = process.hrtime.bigint();
    const cpuResult = runPipeline(input, settings);
    const cpuMs = Number(process.hrtime.bigint() - cpuStart) / 1e6;

    // Run twice per GPU backend in the SAME page: first call pays context/
    // device acquisition cost, second call shows steady-state per-frame cost.
    const gpuTimes = await page.evaluate(async ({ w, h, DEFAULT_SETTINGS }) => {
      const v = (window as any).__v;
      const intensity = v.makeNoisyChecker(w, h);
      const bytes = v.intensityToRgbaBytes(intensity, w, h);
      const input1 = new ImageData(bytes, w, h);
      const glSettings = { ...DEFAULT_SETTINGS, localContrast: 40, noiseReduction: 30, processingEngine: 'webgl2' };
      const t0 = performance.now();
      const gl1 = v.runPipeline(input1, glSettings);
      const t1 = performance.now();
      const input2 = new ImageData(bytes, w, h);
      const gl2 = v.runPipeline(input2, glSettings);
      const t2 = performance.now();

      const gpuSettings = { ...DEFAULT_SETTINGS, localContrast: 40, noiseReduction: 30, processingEngine: 'webgpu' };
      const input3 = new ImageData(bytes, w, h);
      const t3 = performance.now();
      const gp1 = await v.runPipelineAsync(input3, gpuSettings);
      const t4 = performance.now();
      const input4 = new ImageData(bytes, w, h);
      const gp2 = await v.runPipelineAsync(input4, gpuSettings);
      const t5 = performance.now();

      return {
        webgl2First: t1 - t0, webgl2Second: t2 - t1,
        webgpuFirst: t4 - t3, webgpuSecond: t5 - t4,
        glExecuted: gl1.execution?.executed, gpuExecuted: gp1.execution?.executed,
        glTiling: gl1.execution?.tiling?.colorMap, gpuTiling: gp1.execution?.tiling?.colorMap,
      };
    }, { w, h, DEFAULT_SETTINGS });

    console.log(
      label.padEnd(45),
      cpuMs.toFixed(1).padEnd(12),
      gpuTimes.webgl2First.toFixed(1).padEnd(12),
      gpuTimes.webgl2Second.toFixed(1).padEnd(16),
      gpuTimes.webgpuFirst.toFixed(1).padEnd(12),
      gpuTimes.webgpuSecond.toFixed(1)
    );
    console.log(`   webgl2 executed=${gpuTimes.glExecuted} tiling=${JSON.stringify(gpuTimes.glTiling)}  webgpu executed=${gpuTimes.gpuExecuted} tiling=${JSON.stringify(gpuTimes.gpuTiling)}  cpu executed=${cpuResult.execution?.executed}`);
  }

  await browser.close();
}
main().catch((e) => { console.error(e); process.exit(1); });
