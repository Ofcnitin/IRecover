import { chromium } from 'playwright';
function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(`ASSERTION FAILED: ${msg}`);
}
async function main() {
  const browser = await chromium.launch({
    executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
    args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--no-sandbox'],
  });
  const page = await browser.newPage();
  await page.goto('http://127.0.0.1:8931/.browser-dist/pages/verify.html');
  await page.waitForTimeout(300);

  // Case A: small image, precision pipeline explicitly OFF -> expect 'direct'
  // mode, one packed output readback, zero statistics readback.
  const a = await page.evaluate(async () => {
    const v = (window as any).__v;
    const w = 64, h = 48;
    const input = new ImageData(v.intensityToRgbaBytes(v.makeRamp(w, h), w, h), w, h);
    const res = await v.runPipelineAsync(input, { ...v.DEFAULT_SETTINGS, processingEngine: 'webgpu', precisionPipeline: false, sharpenAmount: 0 });
    return res.execution?.resources ?? null;
  });
  console.log('Case A (small, no precision pipeline):', JSON.stringify(a, null, 2));
  assert(a, 'expected resources to be present for a webgpu-executed run');
  assert(a.transfers, 'expected transfer stats to be present');
  assert(a.transfers.mode === 'direct', `expected 'direct' mode for a small image with no precision pipeline, got ${a.transfers.mode}`);
  assert(a.transfers.uploads.input.count >= 1, 'expected at least one input upload');
  assert(a.transfers.readbacks.output.count >= 1, 'expected at least one packed output readback');
  assert(a.transfers.readbacks.statistics.count === 0, `expected zero statistics readbacks with precision pipeline off, got ${a.transfers.readbacks.statistics.count}`);

  // Case B: small image, precision pipeline ON -> expect exactly one statistics readback (the documented CPU sync point).
  const b = await page.evaluate(async () => {
    const v = (window as any).__v;
    const w = 64, h = 48;
    const input = new ImageData(v.intensityToRgbaBytes(v.makeRamp(w, h), w, h), w, h);
    const res = await v.runPipelineAsync(input, { ...v.DEFAULT_SETTINGS, processingEngine: 'webgpu', precisionPipeline: true, whiteBalanceStrength: 50, autoToneStrength: 50, colorCorrectionStrength: 50 });
    return res.execution?.resources ?? null;
  });
  console.log('Case B (small, precision pipeline ON):', JSON.stringify(b, null, 2));
  assert(b.transfers.readbacks.statistics.count === 1, `expected exactly one statistics readback (the CPU sync point) with precision pipeline on, got ${b.transfers.readbacks.statistics.count}`);
  assert(b.transfers.readbacks.statistics.bytes > 0, 'expected nonzero bytes for the statistics readback');

  // Case C: oversize (forces tiling), precision pipeline off so the mode
  // reflects tiling alone -> expect 'resident' or 'streaming', with one
  // input upload per tile.
  const c = await page.evaluate(async () => {
    const v = (window as any).__v;
    const w = 8300, h = 12;
    const input = new ImageData(v.intensityToRgbaBytes(v.makeVerticalGradient(w, h), w, h), w, h);
    const res = await v.runPipelineAsync(input, { ...v.DEFAULT_SETTINGS, processingEngine: 'webgpu', precisionPipeline: false });
    return { resources: res.execution?.resources ?? null, tileCount: res.execution?.tiling?.colorMap.tileCount };
  });
  console.log('Case C (oversize, forces tiling):', JSON.stringify(c, null, 2));
  assert(c.tileCount! > 1, `expected multi-tile dispatch, got ${c.tileCount}`);
  assert(c.resources.transfers.mode === 'resident' || c.resources.transfers.mode === 'streaming', `expected resident or streaming mode for oversize image, got ${c.resources.transfers.mode}`);
  assert(c.resources.transfers.uploads.input.count === c.tileCount, `expected one input upload per tile (${c.tileCount}), got ${c.resources.transfers.uploads.input.count}`);

  console.log('\nPASS: transfer accounting matches the documented contract (direct/resident/streaming modes, statistics-readback count, per-tile upload count).');
  await browser.close();
}
main().catch((e) => { console.error(e); process.exit(1); });
