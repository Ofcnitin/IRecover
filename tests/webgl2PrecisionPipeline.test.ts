import { describe, it, expect } from 'vitest';
import { applyAzusaWhiteBalance, computeWhiteBalanceGains } from '../src/processing/whiteBalance';
import { applyAutoTone, computeAutoToneParams } from '../src/processing/autoTone';
import { applyColorCorrectionPipeline, computeChannelBalanceGains } from '../src/processing/colorCorrectionPipeline';
import {
  assertWebGL2PrecisionPipelineSupported,
  runPrecisionPipelineWebGL2,
} from '../src/processing/gpu/webgl2Backend';

function makeTestImage(n: number): { r: Float32Array; g: Float32Array; b: Float32Array } {
  const r = new Float32Array(n);
  const g = new Float32Array(n);
  const b = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    r[i] = (i % 7) / 6;
    g[i] = (i % 11) / 10;
    b[i] = (i % 5) / 4;
  }
  return { r, g, b };
}

describe('stats-only helpers stay in exact parity with the mutating CPU apply functions', () => {
  // These helpers are the single source of truth the WebGL2 shader path
  // (webgl2Backend.ts) uses for its uniforms. If a future edit to the
  // apply* functions drifted from the compute* functions, CPU and GPU
  // output would silently diverge -- this test exists to catch that.

  it('white balance: computeWhiteBalanceGains matches the gains applyAzusaWhiteBalance actually applies', () => {
    const img1 = makeTestImage(500);
    const img2 = makeTestImage(500);
    const opts = { strength: 60 };

    const computed = computeWhiteBalanceGains(img1.r, img1.g, img1.b, opts);
    const applied = applyAzusaWhiteBalance(img2.r, img2.g, img2.b, opts);

    expect(computed.gainR).toBeCloseTo(applied.gainR, 12);
    expect(computed.gainG).toBeCloseTo(applied.gainG, 12);
    expect(computed.gainB).toBeCloseTo(applied.gainB, 12);
    expect(computed.confidence).toBeCloseTo(applied.confidence, 12);
  });

  it('AutoTone: computeAutoToneParams matches the black/white points applyAutoTone actually applies', () => {
    const img1 = makeTestImage(500);
    const img2 = makeTestImage(500);
    const opts = { strength: 50 };

    const computed = computeAutoToneParams(img1.r, img1.g, img1.b, opts);
    const applied = applyAutoTone(img2.r, img2.g, img2.b, opts);

    expect(computed.degenerate).toBe(false);
    expect(computed.blackPoint).toBeCloseTo(applied.blackPoint, 12);
    expect(computed.whitePoint).toBeCloseTo(applied.whitePoint, 12);
  });

  it('ColorCorrectionPipeline: computeChannelBalanceGains matches the channel gains applyColorCorrectionPipeline actually applies', () => {
    const img1 = makeTestImage(500);
    const img2 = makeTestImage(500);
    const opts = { strength: 40 };

    const computed = computeChannelBalanceGains(img1.r, img1.g, img1.b, opts);
    const applied = applyColorCorrectionPipeline(img2.r, img2.g, img2.b, opts);

    expect(computed.gainR).toBeCloseTo(applied.gainR, 12);
    expect(computed.gainG).toBeCloseTo(applied.gainG, 12);
    expect(computed.gainB).toBeCloseTo(applied.gainB, 12);
  });
});

describe('WebGL2 precision pipeline: environment safety', () => {
  // Vitest runs under Node with no DOM/GPU (see tests/setup.ts) -- there is
  // no real WebGL2 context available here, so these tests validate the
  // *fallback contract* (throws cleanly, never silently no-ops or crashes
  // the process), which is exactly the path browsers without WebGL2 or
  // without EXT_color_buffer_float take too. Real GPU-execution
  // correctness must be verified in-browser; this suite cannot fabricate
  // that without a native GL binding.

  it('assertWebGL2PrecisionPipelineSupported throws (not crashes) with no GPU/canvas available', () => {
    expect(() => assertWebGL2PrecisionPipelineSupported()).toThrow();
  });

  it('runPrecisionPipelineWebGL2 throws cleanly rather than silently no-op-ing when unsupported', () => {
    const { r, g, b } = makeTestImage(16);
    const before = { r: r.slice(), g: g.slice(), b: b.slice() };
    expect(() =>
      runPrecisionPipelineWebGL2(r, g, b, 4, 4, {
        whiteBalance: { strength: 50 },
        autoTone: { strength: 50 },
        colorCorrection: { strength: 50 },
      })
    ).toThrow();
    // A thrown, caught GPU call must not have partially mutated the buffers
    // -- pipeline.ts relies on this to safely fall back to the CPU path.
    expect(Array.from(r)).toEqual(Array.from(before.r));
    expect(Array.from(g)).toEqual(Array.from(before.g));
    expect(Array.from(b)).toEqual(Array.from(before.b));
  });

  it('runPrecisionPipelineWebGL2 rejects mismatched buffer/size arguments before touching the GPU', () => {
    const r = new Float32Array(10);
    const g = new Float32Array(10);
    const b = new Float32Array(10);
    expect(() =>
      runPrecisionPipelineWebGL2(r, g, b, 4, 4 /* implies n=16, not 10 */, {
        whiteBalance: { strength: 50 },
        autoTone: { strength: 50 },
        colorCorrection: { strength: 50 },
      })
    ).toThrow(/buffer\/size mismatch/);
  });
});
