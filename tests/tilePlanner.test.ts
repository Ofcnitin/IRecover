import { describe, it, expect } from 'vitest';
import {
  planTiles,
  resolveStageRadii,
  TilingUnsupportedError,
  MAX_SAFE_RADIUS,
  MIN_CORE_TILE_DIM,
  DEFAULT_MAX_TILE_DIM,
  computeWasTiled,
  type TilePlan,
} from '../src/processing/gpu/tilePlanner';
import { DEFAULT_SETTINGS } from '../src/types/processing';

/** Every image pixel must be covered by exactly one tile core -- no gaps (missing pixels) and no overlaps (double writes). */
function coverageCounts(plan: TilePlan): Uint8Array {
  const counts = new Uint8Array(plan.width * plan.height);
  for (const t of plan.tiles) {
    for (let y = t.core.y0; y < t.core.y1; y++) {
      for (let x = t.core.x0; x < t.core.x1; x++) counts[y * plan.width + x]++;
    }
  }
  return counts;
}

function assertPlanInvariants(plan: TilePlan): void {
  const counts = coverageCounts(plan);
  expect(counts.every((c) => c === 1)).toBe(true);

  for (const { core, region } of plan.tiles) {
    // region contains core, is clipped to the image, and never exceeds the tile size limit
    expect(region.x0).toBeLessThanOrEqual(core.x0);
    expect(region.y0).toBeLessThanOrEqual(core.y0);
    expect(region.x1).toBeGreaterThanOrEqual(core.x1);
    expect(region.y1).toBeGreaterThanOrEqual(core.y1);
    expect(region.x0).toBeGreaterThanOrEqual(0);
    expect(region.y0).toBeGreaterThanOrEqual(0);
    expect(region.x1).toBeLessThanOrEqual(plan.width);
    expect(region.y1).toBeLessThanOrEqual(plan.height);
    expect(region.x1 - region.x0).toBeLessThanOrEqual(plan.maxTileDim);
    expect(region.y1 - region.y0).toBeLessThanOrEqual(plan.maxTileDim);

    // The halo is EXACTLY `halo` on each side unless the image border clips it; and a side
    // touches the true image border iff its core does (so clamp-to-edge is only ever the real border).
    expect(region.x0).toBe(Math.max(0, core.x0 - plan.halo));
    expect(region.y0).toBe(Math.max(0, core.y0 - plan.halo));
    expect(region.x1).toBe(Math.min(plan.width, core.x1 + plan.halo));
    expect(region.y1).toBe(Math.min(plan.height, core.y1 + plan.halo));
  }
  expect(plan.tiles).toHaveLength(plan.cols * plan.rows);
}

describe('planTiles: geometry', () => {
  it('does not tile an image that fits, and needs no halo for it', () => {
    const plan = planTiles(800, 600, 2048, 500);
    expect(plan.tiles).toHaveLength(1);
    expect(plan.tiles[0].core).toEqual({ x0: 0, y0: 0, x1: 800, y1: 600 });
    expect(plan.tiles[0].region).toEqual({ x0: 0, y0: 0, x1: 800, y1: 600 });
  });

  it('an image exactly equal to the tile size is a single tile; one pixel more splits', () => {
    expect(planTiles(128, 100, 128, 29).tiles).toHaveLength(1);
    const over = planTiles(129, 100, 128, 29);
    expect(over.cols).toBe(2);
    expect(over.rows).toBe(1);
    assertPlanInvariants(over);
  });

  it('partitions oversized wide, tall and 2-D images exactly (no gaps, no overlaps)', () => {
    for (const [w, h] of [
      [700, 60],
      [60, 700],
      [330, 240],
      [1000, 1000],
      [1, 5000],
      [5000, 1],
      [8193, 8],
    ] as const) {
      assertPlanInvariants(planTiles(w, h, 128, 29));
    }
  });

  it('handles non-square images: only the long axis is split when the short one fits', () => {
    const wide = planTiles(700, 60, 128, 29);
    expect(wide.rows).toBe(1);
    expect(wide.cols).toBeGreaterThan(1);
    const tall = planTiles(60, 700, 128, 29);
    expect(tall.cols).toBe(1);
    expect(tall.rows).toBeGreaterThan(1);
  });

  it('halos are clipped at the true image border (so clamp-to-edge there is the real behaviour)', () => {
    const plan = planTiles(400, 300, 128, 20);
    const first = plan.tiles[0];
    const last = plan.tiles[plan.tiles.length - 1];
    expect(first.region.x0).toBe(0);
    expect(first.region.y0).toBe(0);
    expect(last.region.x1).toBe(400);
    expect(last.region.y1).toBe(300);
    // ...but interior sides DO get the full halo
    expect(first.region.x1).toBe(first.core.x1 + 20);
    expect(first.region.y1).toBe(first.core.y1 + 20);
  });

  it('adjacent tiles overlap by at least the halo on each shared edge (spatial-filter overlap)', () => {
    const halo = 36;
    const plan = planTiles(500, 400, 200, halo);
    for (let j = 0; j < plan.rows; j++) {
      for (let i = 0; i < plan.cols - 1; i++) {
        const a = plan.tiles[j * plan.cols + i];
        const b = plan.tiles[j * plan.cols + i + 1];
        expect(a.core.x1).toBe(b.core.x0);
        expect(a.region.x1 - a.core.x1).toBe(halo); // a sees `halo` px into b's core
        expect(b.core.x0 - b.region.x0).toBe(halo); // b sees `halo` px into a's core
      }
    }
  });

  it('produces balanced tiles (sizes differ by at most 1) rather than a sliver remainder', () => {
    const plan = planTiles(1001, 50, 128, 20); // coreMax 88 -> 12 cols
    const widths = plan.tiles.map((t) => t.core.x1 - t.core.x0);
    expect(Math.max(...widths) - Math.min(...widths)).toBeLessThanOrEqual(1);
    expect(Math.max(...widths)).toBeLessThanOrEqual(128 - 2 * 20);
    expect(Math.min(...widths)).toBeGreaterThanOrEqual(MIN_CORE_TILE_DIM / 2);
  });

  it('holds its invariants across many pseudo-random sizes, tile sizes and halos', () => {
    let s = 12345;
    const rnd = (n: number) => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return 1 + (s % n);
    };
    for (let k = 0; k < 150; k++) {
      const maxTileDim = 128 + rnd(400);
      const halo = rnd(Math.max(1, Math.floor((maxTileDim - MIN_CORE_TILE_DIM) / 2))) - 1;
      const w = rnd(1500);
      const h = rnd(1500);
      assertPlanInvariants(planTiles(w, h, maxTileDim, halo));
    }
  });

  it('plans a realistically huge image without ever exceeding the default tile size', () => {
    const plan = planTiles(20000, 15000, DEFAULT_MAX_TILE_DIM, 65);
    assertPlanInvariants({ ...plan }); // structural checks on the real grid
    expect(plan.tiles.every((t) => t.region.x1 - t.region.x0 <= DEFAULT_MAX_TILE_DIM && t.region.y1 - t.region.y0 <= DEFAULT_MAX_TILE_DIM)).toBe(true);
  }, 60000);
});

describe('planTiles: unsupported / invalid requests', () => {
  it('accepts exactly the minimum core (128 - 2*32 = 64) and rejects one pixel less', () => {
    const ok = planTiles(300, 200, 128, 32);
    expect(ok.tiles.length).toBeGreaterThan(1);
    expect(() => planTiles(300, 200, 128, 33)).toThrow(TilingUnsupportedError); // core would be 62
  });

  it('throws TilingUnsupportedError (=> CPU fallback) when the overlap leaves no useful core', () => {
    expect(() => planTiles(300, 200, 128, 400)).toThrow(TilingUnsupportedError);
    expect(() => planTiles(300, 200, 128, 400)).toThrow(/cannot be tiled/);
  });

  it('never rejects an image that fits in one tile, however large the halo', () => {
    expect(() => planTiles(100, 100, 128, 500)).not.toThrow();
  });

  it('validates its arguments', () => {
    expect(() => planTiles(0, 10, 128, 0)).toThrow(RangeError);
    expect(() => planTiles(10, 1.5, 128, 0)).toThrow(RangeError);
    expect(() => planTiles(10, 10, 0, 0)).toThrow(RangeError);
    expect(() => planTiles(10, 10, 128, -1)).toThrow(RangeError);
  });
});

const QUALITY = { gaussianPasses: 3, localContrastRadius: 24 };
const off = { noiseReduction: 0, localContrast: 0, sceneHeuristics: false, sharpenAmount: 0 };

describe('resolveStageRadii: spatial-filter overlap requirements', () => {
  it('needs no overlap when every spatial stage is off', () => {
    const r = resolveStageRadii({ ...DEFAULT_SETTINGS, ...off }, QUALITY);
    expect(r.colorMapReach).toBe(0);
    expect(r.sharpenReach).toBe(0);
  });

  it('gaussian noise reduction reach is radius x passes (chained box blurs compound)', () => {
    // amt 1.0 -> round(4) = 4 px per pass
    const r = resolveStageRadii({ ...DEFAULT_SETTINGS, ...off, noiseReduction: 100, noiseMethod: 'gaussian' }, { ...QUALITY, gaussianPasses: 6 });
    expect(r.gaussianRadius).toBe(4);
    expect(r.gaussianPasses).toBe(6);
    expect(r.colorMapReach).toBe(24);
    // amt 0.1 -> round(0.4) = 0 -> clamped to a minimum of 1 px
    const small = resolveStageRadii({ ...DEFAULT_SETTINGS, ...off, noiseReduction: 10, noiseMethod: 'gaussian' }, QUALITY);
    expect(small.gaussianRadius).toBe(1);
    expect(small.colorMapReach).toBe(3);
  });

  it('bilateral reach is its window radius (single pass)', () => {
    const r = resolveStageRadii({ ...DEFAULT_SETTINGS, ...off, noiseReduction: 100, noiseMethod: 'bilateral' }, QUALITY);
    expect(r.bilateralRadius).toBe(3);
    expect(r.colorMapReach).toBe(3);
  });

  it('local contrast contributes its mean radius; scene heuristics contribute 5', () => {
    const lc = resolveStageRadii({ ...DEFAULT_SETTINGS, ...off, localContrast: 40 }, { ...QUALITY, localContrastRadius: 36 });
    expect(lc.colorMapReach).toBe(36);
    const scene = resolveStageRadii({ ...DEFAULT_SETTINGS, ...off, sceneHeuristics: true }, QUALITY);
    expect(scene.colorMapReach).toBe(5);
  });

  it('SUMS the reaches of chained stages rather than taking the max', () => {
    const r = resolveStageRadii(
      { ...DEFAULT_SETTINGS, ...off, noiseReduction: 60, noiseMethod: 'gaussian', localContrast: 40, sceneHeuristics: true },
      { gaussianPasses: 4, localContrastRadius: 28 }
    );
    expect(r.gaussianRadius).toBe(2); // round(0.6*4)
    expect(r.colorMapReach).toBe(2 * 4 + 28 + 5);
  });

  it('sharpening is its own phase: it does not inflate the color-map overlap', () => {
    const r = resolveStageRadii({ ...DEFAULT_SETTINGS, ...off, sharpenAmount: 80, sharpenRadius: 5 }, QUALITY);
    expect(r.sharpenReach).toBe(5);
    expect(r.colorMapReach).toBe(0);
  });

  it('rounds the sharpen radius like the shader path does (minimum 0.5 -> 1 px)', () => {
    const r = (radius: number) => resolveStageRadii({ ...DEFAULT_SETTINGS, ...off, sharpenAmount: 10, sharpenRadius: radius }, QUALITY).sharpenReach;
    expect(r(0.3)).toBe(1);
    expect(r(1.4)).toBe(1);
    expect(r(1.5)).toBe(2);
    expect(r(5)).toBe(5);
  });

  it('caps runaway radii at MAX_SAFE_RADIUS, matching the untiled path', () => {
    const r = resolveStageRadii({ ...DEFAULT_SETTINGS, ...off, localContrast: 10 }, { ...QUALITY, localContrastRadius: 9999 });
    expect(r.localContrastRadius).toBe(MAX_SAFE_RADIUS);
  });

  it("the 'maximum' quality worst case (everything on) still tiles comfortably at the default tile size", () => {
    const r = resolveStageRadii(
      { ...DEFAULT_SETTINGS, noiseReduction: 100, noiseMethod: 'gaussian', localContrast: 100, sceneHeuristics: true, sharpenAmount: 200, sharpenRadius: 5 },
      { gaussianPasses: 6, localContrastRadius: 36 }
    );
    expect(r.colorMapReach).toBe(24 + 36 + 5);
    expect(() => planTiles(20000, 20000, DEFAULT_MAX_TILE_DIM, r.colorMapReach)).not.toThrow();
  });
});

describe('computeWasTiled: unambiguous "did this run actually split into multiple tiles" signal', () => {
  // Regression for the diagnosed ambiguity: the WebGPU orchestrator always
  // attaches a `tiling` object (even for a single tile), while WebGL2 only
  // attaches one when it actually tiled -- so `!!execution.tiling` alone is
  // not a reliable "was tiled" check on WebGPU. `wasTiled` must be false
  // for a single-tile run on every phase, and true as soon as any phase
  // split into more than one tile.

  it('is false when every phase reports a single tile (the WebGPU "always-planned, never-split" case)', () => {
    expect(
      computeWasTiled({ tileCount: 1 }, { tileCount: 1 }, { tileCount: 1 })
    ).toBe(false);
  });

  it('is false when the optional phases are off (null) and the required color-map phase is a single tile', () => {
    expect(computeWasTiled({ tileCount: 1 }, null, null)).toBe(false);
  });

  it('is true as soon as the color-map phase actually splits', () => {
    expect(computeWasTiled({ tileCount: 5 }, { tileCount: 1 }, { tileCount: 1 })).toBe(true);
  });

  it('is true when only the precision phase splits (color-map and sharpen are both single-tile)', () => {
    expect(computeWasTiled({ tileCount: 1 }, { tileCount: 3 }, { tileCount: 1 })).toBe(true);
  });

  it('is true when only the sharpen phase splits', () => {
    expect(computeWasTiled({ tileCount: 1 }, null, { tileCount: 2 })).toBe(true);
  });
});
