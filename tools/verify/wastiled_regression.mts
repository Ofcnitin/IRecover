/**
 * Real-browser regression for the `execution.tiling.wasTiled` fix.
 * Asserts (not just prints) on real Chromium + the real shipped code:
 *  - a small image (single tile on both backends) reports wasTiled=false
 *  - WebGL2 still omits `tiling` entirely when untiled (unchanged contract)
 *  - an oversize image forcing 5 real tiles reports wasTiled=true on both
 *    backends, with colorMap.tileCount matching on both
 * Exits non-zero on any assertion failure so this can gate a release.
 */
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

  const small = await page.evaluate(async () => {
    const v = (window as any).__v;
    const w = 64, h = 48;
    const input = new ImageData(v.intensityToRgbaBytes(v.makeRamp(w, h), w, h), w, h);
    const gl = v.runPipeline(input, { ...v.DEFAULT_SETTINGS, processingEngine: 'webgl2' });
    const gpu = await v.runPipelineAsync(input, { ...v.DEFAULT_SETTINGS, processingEngine: 'webgpu' });
    return { glTiling: gl.execution?.tiling ?? null, gpuTiling: gpu.execution?.tiling ?? null };
  });

  assert(small.glTiling === null, `expected WebGL2 to omit tiling entirely for a small image, got ${JSON.stringify(small.glTiling)}`);
  assert(small.gpuTiling !== null, 'expected WebGPU to always attach a tiling object');
  assert(small.gpuTiling.colorMap.tileCount === 1, `expected single-tile colorMap, got ${small.gpuTiling.colorMap.tileCount}`);
  assert(small.gpuTiling.wasTiled === false, 'expected wasTiled === false for a single-tile WebGPU run (this is the bug being regression-tested)');

  const big = await page.evaluate(async () => {
    const v = (window as any).__v;
    const w = 8300, h = 12;
    const input = new ImageData(v.intensityToRgbaBytes(v.makeVerticalGradient(w, h), w, h), w, h);
    const gl = v.runPipeline(input, { ...v.DEFAULT_SETTINGS, processingEngine: 'webgl2' });
    const gpu = await v.runPipelineAsync(input, { ...v.DEFAULT_SETTINGS, processingEngine: 'webgpu' });
    return {
      glWasTiled: gl.execution?.tiling?.wasTiled, glTileCount: gl.execution?.tiling?.colorMap.tileCount,
      gpuWasTiled: gpu.execution?.tiling?.wasTiled, gpuTileCount: gpu.execution?.tiling?.colorMap.tileCount,
    };
  });

  assert(big.glTileCount! > 1, `expected WebGL2 to actually split into multiple tiles, got ${big.glTileCount}`);
  assert(big.glWasTiled === true, 'expected wasTiled === true for a real multi-tile WebGL2 run');
  assert(big.gpuTileCount! > 1, `expected WebGPU to actually split into multiple tiles, got ${big.gpuTileCount}`);
  assert(big.gpuWasTiled === true, 'expected wasTiled === true for a real multi-tile WebGPU run');
  assert(big.glTileCount === big.gpuTileCount, `expected both backends to agree on tile count, got webgl2=${big.glTileCount} webgpu=${big.gpuTileCount}`);

  console.log('PASS: wasTiled regression (small=untiled on both backends, oversize=tiled on both backends, all assertions held)');
  await browser.close();
}

main().catch((e) => { console.error(e); process.exit(1); });
