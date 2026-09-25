/**
 * Shader variants needed only by the tiled pipeline.
 * ---------------------------------------------------
 * Exactly ONE pass in the pipeline depends on a pixel's position in the
 * WHOLE image rather than only on its neighbourhood: scene heuristics
 * (fullPipelineShaders.ts SCENE_FRAG_SRC) biases sky toward the top and
 * vegetation toward the bottom using `v_uv.y` as "fraction of the way
 * down the image". Inside a tile, v_uv.y is the fraction down the TILE,
 * which would paint a fresh sky/vegetation gradient into every tile and
 * produce a visible seam at every horizontal tile boundary.
 *
 * Rather than copy-paste the shader (which could silently drift from the
 * validated original), the tiled variant is DERIVED from SCENE_FRAG_SRC
 * by two exact string substitutions, each asserted to match exactly once.
 * If the original shader is ever edited so that either anchor changes,
 * this module throws at load time instead of quietly shipping a wrong
 * shader. The original SCENE_FRAG_SRC is never modified.
 *
 * The replacement is arithmetically the same quantity the untiled shader
 * computes: v_uv.y at a pixel centre is (row + 0.5) / height, so
 * (u_rowOrigin + floor(v_uv.y * u_tileHeight) + 0.5) / u_fullHeight is
 * the identical (row + 0.5) / fullHeight with `row` made global.
 * floor() is safe because v_uv.y * tileHeight lands at row + 0.5, half a
 * texel from any rounding boundary.
 */

import { SCENE_FRAG_SRC } from './fullPipelineShaders';

function replaceExactlyOnce(src: string, needle: string, replacement: string, what: string): string {
  const first = src.indexOf(needle);
  if (first === -1 || src.indexOf(needle, first + needle.length) !== -1) {
    throw new Error(
      `tiledShaders: expected exactly one occurrence of the ${what} anchor in SCENE_FRAG_SRC; ` +
        `the scene shader changed -- update tiledShaders.ts to match.`
    );
  }
  return src.slice(0, first) + replacement + src.slice(first + needle.length);
}

export const SCENE_ROW_ANCHOR = 'float rowFrac = v_uv.y;';
export const SCENE_UNIFORM_ANCHOR = 'uniform sampler2D u_variance;';

export const SCENE_TILED_FRAG_SRC = replaceExactlyOnce(
  replaceExactlyOnce(
    SCENE_FRAG_SRC,
    SCENE_UNIFORM_ANCHOR,
    `${SCENE_UNIFORM_ANCHOR}\nuniform float u_rowOrigin;  // first row of this tile's region, in full-image rows\nuniform float u_tileHeight;  // height of this tile's region, in rows\nuniform float u_fullHeight;  // height of the whole image, in rows`,
    'uniform'
  ),
  SCENE_ROW_ANCHOR,
  'float rowFrac = (u_rowOrigin + floor(v_uv.y * u_tileHeight) + 0.5) / u_fullHeight;',
  'rowFrac'
);
