import type { LoadedImage } from '../types/image';
import { analyzeImage } from '../processing/imageAnalysis';

export type SamplePatternId = 'gradient' | 'high-contrast-scene' | 'simulated-nir' | 'noisy';

const SAMPLE_LABELS: Record<SamplePatternId, string> = {
  gradient: 'Gradient',
  'high-contrast-scene': 'High Contrast',
  'simulated-nir': 'Simulated NIR',
  noisy: 'Noisy',
};

export const SAMPLE_PATTERNS: { id: SamplePatternId; label: string }[] = (
  Object.keys(SAMPLE_LABELS) as SamplePatternId[]
).map((id) => ({ id, label: SAMPLE_LABELS[id] }));

/**
 * Generates a locally-synthesized test pattern (no external downloads, no
 * copyrighted imagery) so the app is usable without a real IR photo on hand.
 */
export function generateSamplePattern(id: SamplePatternId, width = 640, height = 480): LoadedImage {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  const imageData = ctx.createImageData(width, height);
  const d = imageData.data;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      let v = 128;

      switch (id) {
        case 'gradient': {
          v = Math.round((x / (width - 1)) * 255);
          break;
        }
        case 'high-contrast-scene': {
          // A few large blocks at very different intensities plus a soft gradient band.
          if (y < height * 0.35) v = 235; // "sky"
          else if (y < height * 0.75) v = 60 + Math.round(40 * Math.sin(x / 18)); // "terrain" texture
          else v = 15; // "shadow band"
          break;
        }
        case 'simulated-nir': {
          // NIR-like: vegetation reflects strongly (bright), sky mid-bright, water dark.
          const horizon = height * 0.4;
          if (y < horizon) {
            v = 150 + Math.round(20 * Math.sin(x / 40)); // sky, brighter in NIR
          } else {
            const rowT = (y - horizon) / (height - horizon);
            const texture = Math.sin(x / 6 + y / 5) * 20;
            v = Math.round(70 + rowT * 140 + texture); // bright "foliage" gradient with texture
          }
          break;
        }
        case 'noisy': {
          const base = 100 + Math.round(80 * Math.sin(x / 30) * Math.cos(y / 30));
          const noise = (pseudoRandom(x, y) - 0.5) * 90;
          v = base + noise;
          break;
        }
      }

      const c = clampByte(v);
      d[i] = c;
      d[i + 1] = c;
      d[i + 2] = c;
      d[i + 3] = 255;
    }
  }

  ctx.putImageData(imageData, 0, 0);

  return {
    fileName: `sample-${id}.png`,
    fileSize: 0,
    mimeType: 'image/png',
    width,
    height,
    data: imageData,
    detected: analyzeImage(imageData),
  };
}

function clampByte(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : Math.round(v);
}

/** Deterministic pseudo-random noise (no Math.random, keeps output reproducible). */
function pseudoRandom(x: number, y: number): number {
  const s = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453;
  return s - Math.floor(s);
}
