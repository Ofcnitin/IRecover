import { describe, it, expect } from 'vitest';
import { analyzeGeography, computeNdvi } from '../src/processing/geographicAnalysis';

function solidImage(width: number, height: number, [r, g, b]: [number, number, number]): ImageData {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data[i * 4] = r;
    data[i * 4 + 1] = g;
    data[i * 4 + 2] = b;
    data[i * 4 + 3] = 255;
  }
  return new ImageData(data, width, height);
}

describe('computeNdvi', () => {
  it('is unavailable when no bands are supplied', () => {
    const result = computeNdvi(null);
    expect(result.available).toBe(false);
    if (!result.available) {
      expect(result.reason).toMatch(/unavailable/i);
    }
  });

  it('is unavailable when band arrays are empty or mismatched', () => {
    expect(computeNdvi({ nir: [], red: [] }).available).toBe(false);
    expect(computeNdvi({ nir: [1, 2], red: [1] }).available).toBe(false);
  });

  it('computes the standard NDVI formula when real bands are supplied', () => {
    // NDVI = (NIR - Red) / (NIR + Red)
    const nir = [200, 100];
    const red = [50, 100];
    const result = computeNdvi({ nir, red });
    expect(result.available).toBe(true);
    if (result.available) {
      // (200-50)/(200+50) = 0.6 ; (100-100)/(200) = 0
      expect(result.mean).toBeCloseTo(0.3, 2);
      expect(result.sampleCount).toBe(2);
    }
  });
});

describe('analyzeGeography', () => {
  it('never reports NDVI as available unless real bands were supplied', () => {
    const image = solidImage(20, 20, [40, 160, 40]); // green
    const local = analyzeGeography({ rgb: image, inputType: 'unknown', ndviBands: null });
    expect(local.vegetation.ndvi.available).toBe(false);
    expect(local.limitations.join(' ')).toMatch(/NDVI/);
  });

  it('classifies a strongly green image as vegetation-heavy', () => {
    const image = solidImage(20, 20, [40, 170, 40]);
    const local = analyzeGeography({ rgb: image, inputType: 'unknown' });
    expect(local.vegetation.detected).toBe(true);
    const veg = local.landCover.find((f) => f.type === 'vegetation' || f.type === 'agriculture');
    expect(veg).toBeDefined();
    expect(veg!.estimatedCoveragePercent).toBeGreaterThan(50);
  });

  it('classifies a dark blue image as water-heavy', () => {
    const image = solidImage(20, 20, [10, 30, 90]);
    const local = analyzeGeography({ rgb: image, inputType: 'unknown' });
    expect(local.water.detected).toBe(true);
    expect(local.water.estimatedCoveragePercent).toBeGreaterThan(50);
  });

  it('classifies a very bright, low-saturation image as snow/ice', () => {
    const image = solidImage(20, 20, [245, 245, 245]);
    const local = analyzeGeography({ rgb: image, inputType: 'unknown' });
    const snow = local.landCover.find((f) => f.type === 'snow-ice');
    expect(snow).toBeDefined();
    expect(snow!.estimatedCoveragePercent).toBeGreaterThan(50);
  });

  it('always attaches a confidence level to every land-cover estimate', () => {
    const image = solidImage(16, 16, [120, 80, 60]);
    const local = analyzeGeography({ rgb: image, inputType: 'unknown' });
    for (const f of local.landCover) {
      expect(['high', 'medium', 'low']).toContain(f.confidence);
    }
  });

  it('never claims elevation -- terrain confidence stays low', () => {
    const image = solidImage(20, 20, [90, 90, 90]);
    const local = analyzeGeography({ rgb: image, inputType: 'unknown' });
    expect(local.terrain.confidence).toBe('low');
  });

  it('records how many pixels were actually sampled', () => {
    const image = solidImage(50, 50, [100, 100, 100]);
    const local = analyzeGeography({ rgb: image, inputType: 'unknown' });
    expect(local.sampledPixels).toBeGreaterThan(0);
    expect(local.sampledPixels).toBeLessThanOrEqual(2500);
  });
});
