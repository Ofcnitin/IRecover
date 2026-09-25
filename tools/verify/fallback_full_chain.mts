import { chromium } from 'playwright';
const browser = await chromium.launch({
  executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
  args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--no-sandbox'],
});
const page = await browser.newPage();
await page.addInitScript(() => {
  if ((window as any).GPUAdapter) {
    (window as any).GPUAdapter.prototype.requestDevice = async () => {
      throw new DOMException('Injected device-creation failure', 'OperationError');
    };
  }
  for (const proto of [(window as any).HTMLCanvasElement?.prototype, (window as any).OffscreenCanvas?.prototype]) {
    if (!proto) continue;
    const orig = proto.getContext;
    proto.getContext = function (type: string, ...args: any[]) {
      if (type === 'webgl2') return null;
      return orig.call(this, type, ...args);
    };
  }
});
await page.goto('http://127.0.0.1:8931/.browser-dist/pages/verify.html');
await page.waitForTimeout(300);

// No nested function/arrow definitions inside the evaluate callback --
// tsx's esbuild transform injects `__name(...)` helper calls for those,
// and Playwright serializes only the callback's own source text (not the
// module scope the helper lives in), which throws ReferenceError in the
// page. All logic below is flat statements; metrics are computed in Node
// afterwards on the returned raw arrays instead.
const out = await page.evaluate(async () => {
  const v = (window as any).__v;
  const DEFAULT_SETTINGS = v.DEFAULT_SETTINGS;
  const w = 48, h = 40;
  const cpuInput = new ImageData(v.intensityToRgbaBytes(v.makeRamp(w, h), w, h), w, h);
  const cpuRef = v.runPipeline(cpuInput, { ...DEFAULT_SETTINGS, processingEngine: 'cpu' });
  const input = new ImageData(v.intensityToRgbaBytes(v.makeRamp(w, h), w, h), w, h);
  const gpuSettings = { ...DEFAULT_SETTINGS, processingEngine: 'webgpu' };
  const gpu = await v.runPipelineAsync(input, gpuSettings);
  const autoSettings = { ...DEFAULT_SETTINGS, processingEngine: 'auto' };
  const auto = await v.runPipelineAsync(input, autoSettings);
  return {
    executed: gpu.execution?.executed,
    attempts: gpu.execution?.attempts,
    gpuAccelerated: gpu.execution?.gpuAccelerated,
    cpuRgba: Array.from(cpuRef.output.data),
    gpuRgba: Array.from(gpu.output.data),
    autoExecuted: auto.execution?.executed,
  };
});

let sum = 0, max = 0;
for (let i = 0; i < out.cpuRgba.length; i++) {
  const d = Math.abs(out.cpuRgba[i] - out.gpuRgba[i]);
  sum += d;
  if (d > max) max = d;
}
console.log(JSON.stringify({
  executed: out.executed,
  attempts: out.attempts,
  gpuAccelerated: out.gpuAccelerated,
  autoExecuted: out.autoExecuted,
  vsCpuExact: { mae: sum / out.cpuRgba.length, max },
}, null, 2));
await browser.close();
