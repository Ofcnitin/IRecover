import type { ImageGeographicMetadata } from './geographic';

export type InputInterpretation = 'native-rgb' | 'false-color-ir' | 'grayscale-ir';

export type SceneType = 'auto' | 'nir-grayscale' | 'thermal-grayscale' | 'false-color-thermal';

export interface LoadedImage {
  fileName: string;
  fileSize: number;
  mimeType: string;
  width: number;
  height: number;
  /** Original decoded pixel data, RGBA, straight from the canvas. Never mutated. */
  data: ImageData;
  /** Best-guess automatic characteristics, purely descriptive (not ML). */
  detected: DetectedCharacteristics;
  /** File-level metadata (EXIF/GPS/camera) actually found in the file, if any. Optional -- absent for generated sample images. */
  geoMetadata?: ImageGeographicMetadata;
}

export interface DetectedCharacteristics {
  isGrayscale: boolean;
  hasAlpha: boolean;
  channelMeans: [number, number, number];
  channelStdDevs: [number, number, number];
  likelyType: SceneType;
  confidenceNote: string;
}

export interface HistogramData {
  /** 256-bucket luminance histogram of the (already-selected) intensity channel. */
  luminance: number[];
  min: number;
  max: number;
  mean: number;
  median: number;
  /** Percentile lookup, e.g. p1, p2, p50, p98, p99 */
  percentiles: Record<number, number>;
}
