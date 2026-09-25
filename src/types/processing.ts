export type PresetId =
  | 'natural'
  | 'neutral'
  | 'natural-warm'
  | 'natural-cool'
  | 'high-contrast'
  | 'landscape'
  | 'portrait'
  | 'monochrome';

export type QualityMode = 'fast' | 'balanced' | 'high' | 'maximum';

export interface ProcessingSettings {
  preset: PresetId;
  interpretation: import('./image').InputInterpretation;
  sceneType: import('./image').SceneType;
  quality: QualityMode;

  // Levels
  autoLevels: boolean;
  blackPoint: number; // 0-255 (in source intensity space)
  whitePoint: number; // 0-255
  blackPercentile: number; // used when autoLevels is on
  whitePercentile: number;

  // Basic tone
  exposure: number; // stops, -2..2
  brightness: number; // -100..100
  contrast: number; // -100..100
  gamma: number; // 0.2..3
  saturation: number; // -100..100
  temperature: number; // -100 (cool) .. 100 (warm)
  hueBias: number; // -180..180 degrees

  // Advanced tone
  highlightRecovery: number; // 0..100
  shadowLift: number; // 0..100
  localContrast: number; // 0..100
  colorStrength: number; // 0..100, blends between grayscale (0) and full color mapping (100)

  // Filters
  noiseReduction: number; // 0..100
  noiseMethod: 'gaussian' | 'median' | 'bilateral';
  sharpenAmount: number; // 0..200 (%)
  sharpenRadius: number; // 0.3..5 px
  sharpenThreshold: number; // 0..255

  // Heuristics
  sceneHeuristics: boolean;

  detailPreservation: number; // 0..100

  // Precision pipeline (Azusa white balance / AutoTone / ColorCorrectionPipeline / gamut protection)
  precisionPipeline: boolean; // master switch for the RGB-domain refinement stages below
  whiteBalanceStrength: number; // 0..100 -- Azusa
  autoToneStrength: number; // 0..100 -- AutoTone
  colorCorrectionStrength: number; // 0..100 -- ColorCorrectionPipeline
  precision: 'standard' | 'high' | 'maximum';
  processingEngine: ProcessingEngineSetting;
}

export type ProcessingEngineSetting = 'auto' | 'cpu' | 'webgl2' | 'webgpu';

export interface PresetDefinition {
  id: PresetId;
  label: string;
  description: string;
  overrides: Partial<ProcessingSettings>;
  /** Color ramp: intensity stops (0..1) to RGB (0..1) in OKLab-interpolated color mapping. */
  colorStops: ColorStop[];
}

export interface ColorStop {
  t: number; // 0..1 position on the tone curve
  rgb: [number, number, number]; // 0..1 sRGB
}

export const DEFAULT_SETTINGS: ProcessingSettings = {
  preset: 'natural',
  interpretation: 'grayscale-ir',
  sceneType: 'auto',
  quality: 'balanced',

  autoLevels: true,
  blackPoint: 0,
  whitePoint: 255,
  blackPercentile: 0.5,
  whitePercentile: 99.5,

  exposure: 0,
  brightness: 0,
  contrast: 0,
  gamma: 1,
  saturation: 0,
  temperature: 0,
  hueBias: 0,

  highlightRecovery: 20,
  shadowLift: 10,
  localContrast: 25,
  colorStrength: 100,

  noiseReduction: 0,
  noiseMethod: 'gaussian',
  sharpenAmount: 30,
  sharpenRadius: 1,
  sharpenThreshold: 4,

  sceneHeuristics: true,
  detailPreservation: 60,

  precisionPipeline: true,
  whiteBalanceStrength: 35,
  autoToneStrength: 25,
  colorCorrectionStrength: 30,
  precision: 'high',
  processingEngine: 'auto',
};
