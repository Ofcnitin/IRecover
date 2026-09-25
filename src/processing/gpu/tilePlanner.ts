/**
 * Tile planning for the WebGL2 full pipeline.
 * --------------------------------------------
 * Pure geometry + halo (overlap) arithmetic -- no GL, no DOM -- so every
 * rule here is unit-testable in plain Node (tests/tilePlanner.test.ts)
 * independently of whether a GPU is available.
 *
 * WHY A HALO, AND HOW BIG
 * Every spatial pass in the pipeline (separable box blur, bilateral,
 * local-contrast mean, scene variance, unsharp blur) reads a square
 * neighbourhood and clamps coordinates to the texture edge. Inside a
 * tile, the outer edge of the uploaded region is a *fake* border: pixels
 * within one filter-radius of it were computed from replicated (wrong)
 * data instead of the true neighbours. Chained passes compound that: a
 * radius-A pass followed by a radius-B pass contaminates A+B pixels
 * inward. So the halo must cover the SUM of the reaches of every spatial
 * stage chained before the next point where data leaves the GPU -- not
 * just the largest one. With halo >= that sum, every pixel in the tile's
 * *core* is computed from exactly the same input values, in the same
 * arithmetic, as in the untiled run, so the result is identical -- not
 * merely close.
 *
 * Tile edges that coincide with the TRUE image border get no halo (the
 * region is clipped there), so clamp-to-edge replication at that border
 * is the real, intended behaviour and matches the untiled pipeline.
 *
 * The pipeline has two spatial phases separated by a full-image
 * synchronisation point (the precision stage needs whole-image
 * statistics before it can run), so it is planned as two independent
 * tilings with their own halos -- see resolveStageRadii().
 */

import type { ProcessingSettings } from '../../types/processing';

/**
 * Sanity cap on any single blur/bilateral radius. Shared with
 * webgl2FullPipeline.ts (which imports it from here) so the tiled and
 * untiled paths can never disagree about it.
 */
export const MAX_SAFE_RADIUS = 200;

/** Radius of the box blur the scene-heuristics variance map uses (fixed in the pipeline). */
export const SCENE_VARIANCE_RADIUS = 5;

/**
 * Default ceiling on a tile's edge length (halo included). Deliberately
 * far below typical MAX_TEXTURE_SIZE values (4096-16384): the pipeline
 * keeps several R32F/RGBA32F surfaces alive per tile, and an RGBA32F
 * 2048x2048 surface is 64 MiB, so this bounds per-tile VRAM to a few
 * hundred MiB rather than gambling on the driver's texture limit.
 */
export const DEFAULT_MAX_TILE_DIM = 2048;

/**
 * Smallest useful core tile edge. Below this, halo overhead dominates
 * (a tile that is mostly halo re-computes more than it produces), so
 * tiling is reported unsupported instead -- the caller then falls back to
 * CPU. Only reachable with absurd radii; every quality preset needs a
 * halo well under 100 px.
 */
export const MIN_CORE_TILE_DIM = 64;

/** Thrown when tiling genuinely cannot be done (never for "just big" images). */
export class TilingUnsupportedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'TilingUnsupportedError';
  }
}

export interface TileRect {
  /** Inclusive left / top, exclusive right / bottom (pixel coordinates in the full image). */
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface Tile {
  /** The pixels this tile is responsible for producing. Cores partition the image exactly. */
  core: TileRect;
  /** core grown by the halo and clipped to the image: what gets uploaded and processed. */
  region: TileRect;
}

export interface TilePlan {
  width: number;
  height: number;
  halo: number;
  maxTileDim: number;
  cols: number;
  rows: number;
  tiles: Tile[]; // row-major
}

/** What the tiled run actually did -- surfaced on the result so callers/tests never have to guess. */
export interface TilingDiagnostics {
  /** Effective max tile edge (halo included) used for this run. */
  maxTileDim: number;
  /** Color-map phase: noise reduction -> local contrast -> tone -> scene -> color ramp. */
  colorMap: { cols: number; rows: number; tileCount: number; halo: number };
  /** Precision phase (pointwise, so no halo); null when the precision pipeline is off. */
  precision: { cols: number; rows: number; tileCount: number } | null;
  /** Sharpening phase; null when sharpening is off. */
  sharpen: { cols: number; rows: number; tileCount: number; halo: number } | null;
  /**
   * Unambiguous "did this run actually split the image into more than one
   * tile" signal. The WebGPU orchestrator always routes through the tile
   * planner (even a single-tile image gets a `tiling` object with
   * `colorMap.tileCount === 1`), while the WebGL2 orchestrator only attaches
   * `tiling` at all when it took the tiled path -- so `!!execution.tiling`
   * alone is a reliable "was tiled" check for WebGL2 but NOT for WebGPU.
   * Use `wasTiled` (or `tileCount > 1` on any phase) instead of checking
   * object presence, on either backend.
   */
  wasTiled: boolean;
}

/** Derives `TilingDiagnostics.wasTiled` from the three phase tile counts. */
export function computeWasTiled(
  colorMap: { tileCount: number },
  precision: { tileCount: number } | null,
  sharpen: { tileCount: number } | null
): boolean {
  return colorMap.tileCount > 1 || (precision?.tileCount ?? 0) > 1 || (sharpen?.tileCount ?? 0) > 1;
}

/** Per-stage radii, derived exactly as the untiled orchestrator derives them. */
export interface StageRadii {
  /** 0 when the stage is off. */
  gaussianRadius: number;
  gaussianPasses: number;
  bilateralRadius: number;
  localContrastRadius: number;
  sceneRadius: number;
  sharpenRadius: number;
  /** Chebyshev reach of every spatial stage that feeds the color-map pass (noise + local contrast + scene variance). */
  colorMapReach: number;
  /** Reach of the (later, separate) sharpening phase. */
  sharpenReach: number;
}

interface QualityLike {
  gaussianPasses: number;
  localContrastRadius: number;
}

type SettingsLike = Pick<
  ProcessingSettings,
  'noiseReduction' | 'noiseMethod' | 'localContrast' | 'sceneHeuristics' | 'sharpenAmount' | 'sharpenRadius'
>;

const clampRadius = (r: number): number => Math.min(MAX_SAFE_RADIUS, Math.max(1, r));

/**
 * Mirrors, line for line, how webgl2FullPipeline.ts derives each pass's
 * radius from settings (gaussian: round(amt*4) x passes; bilateral:
 * round(amt*3); local contrast: round(quality radius); scene: 5;
 * sharpen: round(max(0.5, radius))). The tiled orchestrator uses THESE
 * values for both the halo and the actual shader uniforms, so within the
 * tiled path halo and radii cannot disagree; the tiled-vs-untiled harness
 * checks guard against drift from the untiled path's inline copies.
 */
export function resolveStageRadii(settings: SettingsLike, quality: QualityLike): StageRadii {
  let gaussianRadius = 0;
  let gaussianPasses = 0;
  let bilateralRadius = 0;
  if (settings.noiseReduction > 0) {
    const amt = settings.noiseReduction / 100;
    if (settings.noiseMethod === 'gaussian') {
      gaussianRadius = clampRadius(Math.round(amt * 4));
      gaussianPasses = Math.max(1, quality.gaussianPasses);
    } else if (settings.noiseMethod === 'bilateral') {
      bilateralRadius = clampRadius(Math.round(amt * 3));
    }
  }
  const localContrastRadius = settings.localContrast > 0 ? clampRadius(Math.round(quality.localContrastRadius)) : 0;
  const sceneRadius = settings.sceneHeuristics ? SCENE_VARIANCE_RADIUS : 0;
  const sharpenRadius = settings.sharpenAmount > 0 ? Math.round(Math.max(0.5, settings.sharpenRadius)) : 0;

  const noiseReach = gaussianRadius * gaussianPasses + bilateralRadius;
  return {
    gaussianRadius,
    gaussianPasses,
    bilateralRadius,
    localContrastRadius,
    sceneRadius,
    sharpenRadius,
    colorMapReach: noiseReach + localContrastRadius + sceneRadius,
    sharpenReach: sharpenRadius,
  };
}

/**
 * Splits [0, len) into balanced, contiguous segments each no longer than
 * `coreMax`. Balanced (sizes differ by at most 1) rather than
 * "coreMax, coreMax, ..., sliver" so no tile is a wasteful sliver.
 */
function splitAxis(len: number, coreMax: number): number[] {
  const n = Math.ceil(len / coreMax);
  const cuts: number[] = [];
  for (let i = 0; i <= n; i++) cuts.push(Math.floor((i * len) / n));
  return cuts;
}

/**
 * Plans a row-major grid of tiles. A tile's full region (core + halo,
 * clipped) never exceeds `maxTileDim` on either axis. An axis that
 * already fits in one tile is not split (and needs no halo there, since
 * it has no interior edges). Throws TilingUnsupportedError only when the
 * halo is so large that a useful core cannot fit.
 */
export function planTiles(width: number, height: number, maxTileDim: number, halo: number): TilePlan {
  for (const [name, v, min] of [
    ['width', width, 1],
    ['height', height, 1],
    ['maxTileDim', maxTileDim, 1],
    ['halo', halo, 0],
  ] as const) {
    if (!Number.isInteger(v) || v < min) {
      throw new RangeError(`planTiles: ${name} must be an integer >= ${min} (got ${v}).`);
    }
  }

  const coreMax = maxTileDim - 2 * halo;
  const needsSplit = width > maxTileDim || height > maxTileDim;
  if (needsSplit && coreMax < MIN_CORE_TILE_DIM) {
    throw new TilingUnsupportedError(
      `Spatial-filter overlap of ${halo}px per side leaves only a ${Math.max(coreMax, 0)}px core in a ${maxTileDim}px tile ` +
        `(minimum ${MIN_CORE_TILE_DIM}px); this image/settings combination cannot be tiled on this GPU.`
    );
  }

  const xCuts = width > maxTileDim ? splitAxis(width, coreMax) : [0, width];
  const yCuts = height > maxTileDim ? splitAxis(height, coreMax) : [0, height];

  const tiles: Tile[] = [];
  for (let j = 0; j < yCuts.length - 1; j++) {
    for (let i = 0; i < xCuts.length - 1; i++) {
      const core: TileRect = { x0: xCuts[i], y0: yCuts[j], x1: xCuts[i + 1], y1: yCuts[j + 1] };
      const region: TileRect = {
        x0: Math.max(0, core.x0 - halo),
        y0: Math.max(0, core.y0 - halo),
        x1: Math.min(width, core.x1 + halo),
        y1: Math.min(height, core.y1 + halo),
      };
      tiles.push({ core, region });
    }
  }

  return { width, height, halo, maxTileDim, cols: xCuts.length - 1, rows: yCuts.length - 1, tiles };
}
