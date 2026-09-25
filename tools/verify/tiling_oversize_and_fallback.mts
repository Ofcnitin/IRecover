import '../../tests/setup';
import { chromium } from 'playwright';
import { runPipeline } from '../../src/processing/pipeline';
import { DEFAULT_SETTINGS, type ProcessingSettings } from '../../src/types/processing';
import { makeVerticalGradient, intensityToRgbaBytes } from './fixtures';

function metrics(a: Uint8ClampedArray | number[], b: Uint8ClampedArray | number[]) {
  let sum = 0, max = 0, over = 0;
  const tol = 3;
  const n = a.length;
  for (let i = 0; i < n; i++) {
    const d = Math.abs((a[i] as number) - (b[i] as number));
    sum += d; if (d > max) max = d; if (d > tol) over++;
  }
  return { mae: sum / n, max, overPct: (100 * over) / n };
}

async function main() {
  const browser = await chromium.launch({
    executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
    args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--no-sandbox'],
  });
  const page = await browser.newPage();
  await page.goto('http://127.0.0.1:8931/.browser-dist/pages/verify.html');
  await page.waitForTimeout(300);

  // --- 1. Real oversize (beyond MAX_TEXTURE_SIZE=8192) forcing multi-tile dispatch, both backends ---
  for (const [w, h] of [[8300, 12], [12, 8300]] as const) {
    const intensity = makeVerticalGradient(w, h);
    const bytes = intensityToRgbaBytes(intensity, w, h);
    const input = new (globalThis as any).ImageData(bytes, w, h);
    const cpuSettings: ProcessingSettings = { ...DEFAULT_SETTINGS, localContrast: 40, noiseReduction: 30, processingEngine: 'cpu' };
    const cpu = runPipeline(input, cpuSettings);

    const out = await page.evaluate(async ({ w, h, DEFAULT_SETTINGS }) => {
      const v = (window as any).__v;
      const intensity = v.makeVerticalGradient(w, h);
      const bytes = v.intensityToRgbaBytes(intensity, w, h);
      const input = new ImageData(bytes, w, h);
      const glSettings = { ...DEFAULT_SETTINGS, localContrast: 40, noiseReduction: 30, processingEngine: 'webgl2' };
      const gl = v.runPipeline(input, glSettings);
      const gpuSettings = { ...DEFAULT_SETTINGS, localContrast: 40, noiseReduction: 30, processingEngine: 'webgpu' };
      const gpu = await v.runPipelineAsync(input, gpuSettings);
      return {
        gl: { rgba: Array.from(gl.output.data), tiling: gl.execution?.tiling, executed: gl.execution?.executed, gpuAccelerated: gl.execution?.gpuAccelerated },
        gpu: { rgba: Array.from(gpu.output.data), tiling: gpu.execution?.tiling, executed: gpu.execution?.executed, gpuAccelerated: gpu.execution?.gpuAccelerated },
      };
    }, { w, h, DEFAULT_SETTINGS });

    console.log(`\n[oversize ${w}x${h}] CPU executed=${cpu.execution?.executed}`);
    console.log(`  WebGL2: executed=${out.gl.executed} gpuAccel=${out.gl.gpuAccelerated} tiling=${JSON.stringify(out.gl.tiling)}`);
    console.log(`  WebGL2 vs CPU:`, metrics(cpu.output.data, out.gl.rgba));
    console.log(`  WebGPU: executed=${out.gpu.executed} gpuAccel=${out.gpu.gpuAccelerated} tiling=${JSON.stringify(out.gpu.tiling)}`);
    console.log(`  WebGPU vs CPU:`, metrics(cpu.output.data, out.gpu.rgba));
  }

  // --- 2. median noise method: WebGL2 should fall back, verify honesty of executed/attempts ---
  {
    const w = 48, h = 40;
    const intensity = makeVerticalGradient(w, h);
    const bytes = intensityToRgbaBytes(intensity, w, h);
    const input = new (globalThis as any).ImageData(bytes, w, h);
    const cpuSettings: ProcessingSettings = { ...DEFAULT_SETTINGS, noiseReduction: 50, noiseMethod: 'median', processingEngine: 'cpu' };
    const cpu = runPipeline(input, cpuSettings);

    const out = await page.evaluate(async ({ w, h, DEFAULT_SETTINGS }) => {
      const v = (window as any).__v;
      const intensity = v.makeVerticalGradient(w, h);
      const bytes = v.intensityToRgbaBytes(intensity, w, h);
      const input = new ImageData(bytes, w, h);
      const glSettings = { ...DEFAULT_SETTINGS, noiseReduction: 50, noiseMethod: 'median', processingEngine: 'webgl2' };
      const gl = v.runPipeline(input, glSettings);
      const gpuSettings = { ...DEFAULT_SETTINGS, noiseReduction: 50, noiseMethod: 'median', processingEngine: 'webgpu' };
      const gpu = await v.runPipelineAsync(input, gpuSettings);
      return {
        gl: { rgba: Array.from(gl.output.data), executed: gl.execution?.executed, attempts: gl.execution?.attempts },
        gpu: { rgba: Array.from(gpu.output.data), executed: gpu.execution?.executed, attempts: gpu.execution?.attempts },
      };
    }, { w, h, DEFAULT_SETTINGS });

    console.log(`\n[median noise method] CPU executed=${cpu.execution?.executed}`);
    console.log(`  WebGL2 requested='webgl2': executed=${out.gl.executed}`, JSON.stringify(out.gl.attempts));
    console.log(`  WebGL2 vs CPU:`, metrics(cpu.output.data, out.gl.rgba));
    console.log(`  WebGPU requested='webgpu': executed=${out.gpu.executed}`, JSON.stringify(out.gpu.attempts));
    console.log(`  WebGPU vs CPU:`, metrics(cpu.output.data, out.gpu.rgba));
  }

  // --- 3. Fallback behavior: deny WebGL2 context creation, deny navigator.gpu, verify graceful CPU fallback ---
  {
    const page2 = await browser.newPage();
    await page2.addInitScript(() => {
      // Force WebGL2 unavailable on BOTH HTMLCanvasElement and
      // OffscreenCanvas (detectCapabilities() prefers OffscreenCanvas
      // when present -- patching only HTMLCanvasElement is a no-op).
      for (const proto of [
        (window as any).HTMLCanvasElement?.prototype,
        (window as any).OffscreenCanvas?.prototype,
      ]) {
        if (!proto) continue;
        const orig = proto.getContext;
        proto.getContext = function (type: string, ...args: any[]) {
          if (type === 'webgl2') return null;
          return orig.call(this, type, ...args);
        };
      }
      // Force WebGPU unavailable. navigator.gpu is a getter-only
      // accessor on the prototype in Chromium, so a plain `delete`
      // silently no-ops -- redefine it as an own property instead.
      Object.defineProperty(Object.getPrototypeOf(navigator), 'gpu', { get: () => undefined, configurable: true });
    });
    await page2.goto('http://127.0.0.1:8931/.browser-dist/pages/verify.html');
    await page2.waitForTimeout(300);
    const out = await page2.evaluate(async ({ DEFAULT_SETTINGS }) => {
      const v = (window as any).__v;
      const w = 48, h = 40;
      const intensity = v.makeRamp(w, h);
      const bytes = v.intensityToRgbaBytes(intensity, w, h);
      const input = new ImageData(bytes, w, h);
      const caps = v.detectCapabilities();
      const glSettings = { ...DEFAULT_SETTINGS, processingEngine: 'webgl2' };
      const gl = v.runPipeline(input, glSettings);
      const gpuSettings = { ...DEFAULT_SETTINGS, processingEngine: 'webgpu' };
      const gpu = await v.runPipelineAsync(input, gpuSettings);
      const autoSettings = { ...DEFAULT_SETTINGS, processingEngine: 'auto' };
      const auto = v.runPipeline(input, autoSettings);
      return {
        caps,
        glExecuted: gl.execution?.executed, glAttempts: gl.execution?.attempts,
        gpuExecuted: gpu.execution?.executed, gpuAttempts: gpu.execution?.attempts,
        autoExecuted: auto.execution?.executed,
      };
    }, { DEFAULT_SETTINGS });
    console.log('\n[fallback behavior: WebGL2 + WebGPU both forced unavailable]');
    console.log(JSON.stringify(out, null, 2));
    await page2.close();
  }

  await browser.close();
}
main().catch((e) => { console.error(e); process.exit(1); });
