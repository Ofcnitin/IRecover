import { chromium } from 'playwright';
const browser = await chromium.launch({
  executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
  args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--no-sandbox'],
});
const page = await browser.newPage();
page.on('console', m => console.log('[page]', m.text()));
await page.addInitScript(() => {
  console.log('addInitScript: typeof GPUAdapter =', typeof (window as any).GPUAdapter);
  if ((window as any).GPUAdapter) {
    (window as any).GPUAdapter.prototype.requestDevice = async function () {
      console.log('PATCHED (eager) requestDevice called');
      throw new DOMException('Injected device-creation failure', 'OperationError');
    };
  }
});
await page.goto('http://127.0.0.1:8931/.browser-dist/pages/verify.html');
await page.waitForTimeout(300);
const out = await page.evaluate(async () => {
  const v = (window as any).__v;
  const DEFAULT_SETTINGS = v.DEFAULT_SETTINGS;
  const w = 48, h = 40;
  const intensity = v.makeRamp(w, h);
  const bytes = v.intensityToRgbaBytes(intensity, w, h);
  const input = new ImageData(bytes, w, h);
  const gpuSettings = { ...DEFAULT_SETTINGS, processingEngine: 'webgpu' };
  const gpu = await v.runPipelineAsync(input, gpuSettings);
  return { executed: gpu.execution?.executed, attempts: gpu.execution?.attempts, gpuAccelerated: gpu.execution?.gpuAccelerated };
});
console.log(JSON.stringify(out, null, 2));
await browser.close();
