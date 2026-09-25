import { describe, it, expect } from 'vitest';
import { SCENE_FRAG_SRC } from '../src/processing/gpu/fullPipelineShaders';
import { SCENE_TILED_FRAG_SRC, SCENE_ROW_ANCHOR, SCENE_UNIFORM_ANCHOR } from '../src/processing/gpu/tiledShaders';

describe('tiled scene shader (derived from the validated original)', () => {
  it('leaves the original scene shader untouched and position-local', () => {
    expect(SCENE_FRAG_SRC).toContain(SCENE_ROW_ANCHOR);
    expect(SCENE_FRAG_SRC).not.toContain('u_rowOrigin');
  });

  it('replaces the tile-local row fraction with a global one', () => {
    expect(SCENE_TILED_FRAG_SRC).not.toContain(SCENE_ROW_ANCHOR);
    expect(SCENE_TILED_FRAG_SRC).toContain('u_rowOrigin');
    expect(SCENE_TILED_FRAG_SRC).toContain('u_tileHeight');
    expect(SCENE_TILED_FRAG_SRC).toContain('u_fullHeight');
    expect(SCENE_TILED_FRAG_SRC).toMatch(/float rowFrac = \(u_rowOrigin \+ floor\(v_uv\.y \* u_tileHeight\) \+ 0\.5\) \/ u_fullHeight;/);
  });

  it('differs from the original ONLY by the two intended substitutions (no silent drift)', () => {
    const reverted = SCENE_TILED_FRAG_SRC.replace(
      /\nuniform float u_rowOrigin;[^\n]*\nuniform float u_tileHeight;[^\n]*\nuniform float u_fullHeight;[^\n]*/,
      ''
    ).replace('float rowFrac = (u_rowOrigin + floor(v_uv.y * u_tileHeight) + 0.5) / u_fullHeight;', SCENE_ROW_ANCHOR);
    expect(reverted).toBe(SCENE_FRAG_SRC);
    expect(SCENE_TILED_FRAG_SRC).toContain(SCENE_UNIFORM_ANCHOR);
  });

  it('keeps every other line of the sky/vegetation math byte-identical', () => {
    for (const line of ['float topBias = clamp(1.0 - rowFrac * 1.6, 0.0, 1.0);', 'float bottomBias = clamp(rowFrac * 1.2 + 0.2, 0.0, 1.0);', 'float flatness = clamp(1.0 - v * 40.0, 0.0, 1.0);']) {
      expect(SCENE_FRAG_SRC).toContain(line);
      expect(SCENE_TILED_FRAG_SRC).toContain(line);
    }
  });
});
