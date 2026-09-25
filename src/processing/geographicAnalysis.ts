import type {
  BuiltUpEstimate,
  ConfidenceLevel,
  InputImageKind,
  LandCoverEstimate,
  LandCoverType,
  LocalGeographicAnalysis,
  TerrainEstimate,
  VegetationEstimate,
  WaterEstimate,
} from '../types/geographic';

/**
 * Classical, deterministic image-analysis heuristics for land-cover style
 * classification. This is NOT machine learning, NOT object detection, and
 * NOT a substitute for an authoritative remote-sensing classifier -- it is
 * a set of RGB/HSV threshold rules applied to the already-converted
 * visible-light approximation, meant to give a rough, clearly-labelled
 * "estimated" breakdown.
 *
 * Every number this module produces must be presented to the user as an
 * estimate with an attached confidence level, never as a measurement.
 */

const LABELS: Record<LandCoverType, string> = {
  vegetation: 'Vegetation',
  water: 'Water',
  'bare-soil': 'Bare Soil',
  rock: 'Rock',
  'snow-ice': 'Snow/Ice',
  urban: 'Urban/Built-up',
  agriculture: 'Agriculture',
  'open-terrain': 'Open Terrain',
  unknown: 'Unknown',
};

interface Hsv {
  h: number; // 0-360
  s: number; // 0-1
  v: number; // 0-1 (0-255 scale input handled by caller)
}

function rgbToHsv(r: number, g: number, b: number): Hsv {
  const rf = r / 255;
  const gf = g / 255;
  const bf = b / 255;
  const max = Math.max(rf, gf, bf);
  const min = Math.min(rf, gf, bf);
  const delta = max - min;
  let h = 0;
  if (delta !== 0) {
    if (max === rf) h = 60 * (((gf - bf) / delta) % 6);
    else if (max === gf) h = 60 * ((bf - rf) / delta + 2);
    else h = 60 * ((rf - gf) / delta + 4);
  }
  if (h < 0) h += 360;
  const s = max === 0 ? 0 : delta / max;
  return { h, s, v: max };
}

function classifyPixel(r: number, g: number, b: number): LandCoverType {
  const { h, s, v } = rgbToHsv(r, g, b);

  // Snow/ice: very bright, low saturation.
  if (v > 0.85 && s < 0.18) return 'snow-ice';

  // Water: darker, blue-leaning, low-to-moderate saturation, low variance hue band.
  if (v < 0.55 && s > 0.1 && h >= 170 && h <= 250) return 'water';
  if (v < 0.35 && s < 0.25) return 'water';

  // Vegetation: green-band hue with reasonable saturation.
  if (h >= 70 && h <= 170 && s > 0.15 && v > 0.12) {
    return v > 0.55 ? 'agriculture' : 'vegetation';
  }

  // Bare soil / rock: warm hues (orange/brown/red), low-to-mid saturation.
  if (h >= 10 && h < 45 && s > 0.2 && v > 0.15 && v < 0.75) return 'bare-soil';
  if ((h < 10 || h >= 340) && s > 0.15 && v < 0.6) return 'bare-soil';

  // Rock/urban: low saturation, mid brightness gray tones.
  if (s < 0.14 && v >= 0.25 && v <= 0.85) return 'urban';
  if (s < 0.1) return 'rock';

  return 'open-terrain';
}

export interface ClassificationSample {
  counts: Record<LandCoverType, number>;
  total: number;
  /** Simple local-gradient magnitude samples, used for terrain/built-up heuristics. */
  gradientSamples: number[];
}

function sampleClassification(data: ImageData, maxSamples = 60000): ClassificationSample {
  const { data: rgba, width, height } = data;
  const n = width * height;
  const step = Math.max(1, Math.floor(n / maxSamples));

  const counts: Record<LandCoverType, number> = {
    vegetation: 0,
    water: 0,
    'bare-soil': 0,
    rock: 0,
    'snow-ice': 0,
    urban: 0,
    agriculture: 0,
    'open-terrain': 0,
    unknown: 0,
  };
  const gradientSamples: number[] = [];
  let total = 0;

  for (let idx = 0; idx < n; idx += step) {
    const x = idx % width;
    const y = Math.floor(idx / width);
    const p = idx * 4;
    const r = rgba[p];
    const g = rgba[p + 1];
    const b = rgba[p + 2];
    counts[classifyPixel(r, g, b)]++;
    total++;

    // Cheap horizontal luminance gradient for texture/edge-density heuristics.
    if (x < width - 1) {
      const pNext = (idx + 1) * 4;
      const lumHere = 0.299 * r + 0.587 * g + 0.114 * b;
      const lumNext = 0.299 * rgba[pNext] + 0.587 * rgba[pNext + 1] + 0.114 * rgba[pNext + 2];
      gradientSamples.push(Math.abs(lumHere - lumNext));
    }
    void y;
  }

  return { counts, total, gradientSamples };
}

function pct(count: number, total: number): number {
  if (total === 0) return 0;
  return Math.round((count / total) * 1000) / 10;
}

function buildLandCover(sample: ClassificationSample): LandCoverEstimate[] {
  const { counts, total } = sample;
  const types = Object.keys(counts) as LandCoverType[];
  return types
    .map((type) => {
      const coverage = pct(counts[type], total);
      const confidence: ConfidenceLevel = coverage > 30 ? 'medium' : 'low';
      return {
        type,
        label: LABELS[type],
        estimatedCoveragePercent: coverage,
        confidence: type === 'unknown' ? 'low' : confidence,
      };
    })
    .sort((a, b) => b.estimatedCoveragePercent - a.estimatedCoveragePercent)
    .filter((f) => f.estimatedCoveragePercent > 0);
}

function buildVegetation(sample: ClassificationSample, ndvi: VegetationEstimate['ndvi']): VegetationEstimate {
  const vegCount = sample.counts.vegetation + sample.counts.agriculture;
  const coverage = pct(vegCount, sample.total);
  const detected = coverage > 3;
  const confidence: ConfidenceLevel = ndvi.available ? 'high' : coverage > 25 ? 'medium' : 'low';
  return {
    detected,
    estimatedCoveragePercent: detected ? coverage : 0,
    confidence,
    description: detected
      ? `Green-band response consistent with vegetation across roughly ${coverage.toFixed(0)}% of the sampled image.`
      : 'No significant vegetation-like response detected in this image.',
    ndvi,
  };
}

function buildWater(sample: ClassificationSample): WaterEstimate {
  const coverage = pct(sample.counts.water, sample.total);
  const detected = coverage > 3;
  const confidence: ConfidenceLevel = coverage > 20 ? 'high' : coverage > 5 ? 'medium' : 'low';
  return {
    detected,
    estimatedCoveragePercent: detected ? coverage : 0,
    confidence,
    description: detected
      ? `A region consistent with a contiguous water body was found in roughly ${coverage.toFixed(0)}% of the sampled image.`
      : 'No large region consistent with open water was detected.',
  };
}

function buildTerrain(sample: ClassificationSample): TerrainEstimate {
  if (sample.gradientSamples.length === 0) {
    return { description: 'Unable to estimate terrain texture.', confidence: 'low' };
  }
  const mean = sample.gradientSamples.reduce((a, b) => a + b, 0) / sample.gradientSamples.length;
  // Higher average luminance gradient roughly correlates with more
  // visually "rough"/textured terrain in a 2D image -- this is a texture
  // proxy, not an elevation measurement.
  let description: string;
  if (mean < 4) description = 'Flat or open plain';
  else if (mean < 9) description = 'Rolling terrain';
  else if (mean < 16) description = 'Hilly terrain';
  else description = 'Steep or mountainous-looking terrain';

  return {
    description,
    confidence: 'low', // 2D visual texture is a weak proxy for actual terrain relief.
  };
}

function buildBuiltUp(sample: ClassificationSample): BuiltUpEstimate {
  const coverage = pct(sample.counts.urban, sample.total);
  const detected = coverage > 8;
  return {
    detected,
    confidence: detected ? 'medium' : 'low',
    description: detected
      ? `Regions consistent with built-up/developed surfaces are possible in roughly ${coverage.toFixed(0)}% of the sampled image.`
      : 'No strong indication of built-up or developed surfaces was found.',
  };
}

export interface NdviBands {
  /** Per-sample NIR intensity, 0-255 scale. */
  nir: Float32Array | number[];
  /** Per-sample Red intensity, 0-255 scale, same length/order as nir. */
  red: Float32Array | number[];
}

/**
 * Computes mean NDVI ONLY when real, distinct NIR and Red band samples are
 * supplied. This must never be called with values derived from the
 * deterministic IR->RGB *approximation* -- that would fabricate a
 * spectral index from data that was never actually spectral.
 */
export function computeNdvi(bands: NdviBands | null): VegetationEstimate['ndvi'] {
  if (!bands || bands.nir.length === 0 || bands.nir.length !== bands.red.length) {
    return {
      available: false,
      reason: 'Spectral vegetation index unavailable for this image.',
    };
  }
  let sum = 0;
  let count = 0;
  for (let i = 0; i < bands.nir.length; i++) {
    const nir = bands.nir[i];
    const red = bands.red[i];
    const denom = nir + red;
    if (denom === 0) continue;
    sum += (nir - red) / denom;
    count++;
  }
  if (count === 0) {
    return { available: false, reason: 'Spectral vegetation index unavailable for this image.' };
  }
  return { available: true, mean: Math.round((sum / count) * 1000) / 1000, sampleCount: count };
}

export interface LocalGeographicAnalysisInput {
  /** The locally converted, visible-light RGB approximation -- never the raw IR data. */
  rgb: ImageData;
  inputType: InputImageKind;
  /** Only pass real band samples; omit or pass null otherwise. */
  ndviBands?: NdviBands | null;
}

export function analyzeGeography(input: LocalGeographicAnalysisInput): LocalGeographicAnalysis {
  const sample = sampleClassification(input.rgb);
  const ndvi = computeNdvi(input.ndviBands ?? null);

  const limitations: string[] = [
    'All land-cover, vegetation, water, and terrain values are estimates from classical image heuristics, not an authoritative geographic classification.',
    'Terrain description is inferred from 2D visual texture only; no elevation data was used.',
  ];
  if (!ndvi.available) {
    limitations.push('Spectral vegetation index (NDVI) unavailable for this image.');
  }

  return {
    inputType: input.inputType,
    landCover: buildLandCover(sample),
    vegetation: buildVegetation(sample, ndvi),
    water: buildWater(sample),
    terrain: buildTerrain(sample),
    builtUp: buildBuiltUp(sample),
    sampledPixels: sample.total,
    limitations,
  };
}
