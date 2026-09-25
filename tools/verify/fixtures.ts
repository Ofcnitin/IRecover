// Shared, deterministic fixture generators. Imported directly by both the
// Node-side CPU reference run and (transpiled, unmodified) the browser-side
// GPU run, so both sides build byte-identical inputs without serializing
// large arrays over the wire.
export function makeRamp(width: number, height: number): Float32Array {
  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      out[y * width + x] = width > 1 ? x / (width - 1) : 0;
    }
  }
  return out;
}

export function makeNoisyChecker(width: number, height: number, seed = 1): Float32Array {
  const out = new Float32Array(width * height);
  let s = seed;
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const base = (Math.floor(x / 8) + Math.floor(y / 8)) % 2 === 0 ? 0.25 : 0.75;
      out[y * width + x] = Math.min(1, Math.max(0, base + (rand() - 0.5) * 0.3));
    }
  }
  return out;
}

export function makeVerticalGradient(width: number, height: number): Float32Array {
  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      out[y * width + x] = height > 1 ? y / (height - 1) : 0;
    }
  }
  return out;
}

// Turns a single-channel [0,1] intensity map into an 8-bit grayscale
// ImageData-shaped RGBA buffer (what `extractIntensity` expects to read
// back out under 'grayscale-ir' interpretation with autoLevels off,
// black=0/white=255) so we can drive the real `runPipeline`/`runPipelineAsync`
// entry points (which take ImageData in, not raw intensity).
export function intensityToRgbaBytes(intensity: Float32Array, width: number, height: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const v = Math.round(Math.min(1, Math.max(0, intensity[i])) * 255);
    out[i * 4] = v;
    out[i * 4 + 1] = v;
    out[i * 4 + 2] = v;
    out[i * 4 + 3] = 255;
  }
  return out;
}
