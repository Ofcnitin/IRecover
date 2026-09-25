import type { DetectedCharacteristics } from '../types/image';

/**
 * Deterministic, non-ML heuristics that describe an image's basic
 * characteristics. This is NOT object/scene recognition -- just channel
 * statistics that help the UI suggest a sensible default interpretation.
 */
export function analyzeImage(data: ImageData): DetectedCharacteristics {
  const { data: rgba, width, height } = data;
  const n = width * height;

  let sumR = 0,
    sumG = 0,
    sumB = 0;
  let hasAlpha = false;
  let sampleStep = Math.max(1, Math.floor(n / 200000)); // subsample huge images for speed

  let sampled = 0;
  for (let i = 0; i < n; i += sampleStep) {
    const p = i * 4;
    sumR += rgba[p];
    sumG += rgba[p + 1];
    sumB += rgba[p + 2];
    if (rgba[p + 3] < 255) hasAlpha = true;
    sampled++;
  }
  const meanR = sumR / sampled;
  const meanG = sumG / sampled;
  const meanB = sumB / sampled;

  let varR = 0,
    varG = 0,
    varB = 0;
  let maxChannelDiff = 0;
  for (let i = 0; i < n; i += sampleStep) {
    const p = i * 4;
    varR += (rgba[p] - meanR) ** 2;
    varG += (rgba[p + 1] - meanG) ** 2;
    varB += (rgba[p + 2] - meanB) ** 2;
    const diff = Math.max(
      Math.abs(rgba[p] - rgba[p + 1]),
      Math.abs(rgba[p + 1] - rgba[p + 2]),
      Math.abs(rgba[p] - rgba[p + 2])
    );
    if (diff > maxChannelDiff) maxChannelDiff = diff;
  }
  const stdR = Math.sqrt(varR / sampled);
  const stdG = Math.sqrt(varG / sampled);
  const stdB = Math.sqrt(varB / sampled);

  // If channels never diverge meaningfully, this is effectively grayscale.
  const isGrayscale = maxChannelDiff < 6;

  let likelyType: DetectedCharacteristics['likelyType'] = 'auto';
  let confidenceNote =
    'Automatic detection is based on simple channel statistics only, not scene understanding. Please confirm the image type below.';

  if (isGrayscale) {
    likelyType = 'nir-grayscale';
    confidenceNote =
      'Channels are nearly identical, consistent with a single-band grayscale IR/NIR or thermal capture. Choose "Thermal" below if this came from a thermal camera.';
  } else {
    // Colorful with a dominant channel pattern often indicates a false-color
    // palette (e.g. thermal cameras often export magenta/orange/blue palettes).
    likelyType = 'false-color-thermal';
    confidenceNote =
      'This image has distinct color channels, which often indicates a false-color palette (e.g. from a thermal camera) rather than natural RGB. Confirm below.';
  }

  return {
    isGrayscale,
    hasAlpha,
    channelMeans: [meanR, meanG, meanB],
    channelStdDevs: [stdR, stdG, stdB],
    likelyType,
    confidenceNote,
  };
}
