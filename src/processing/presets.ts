import type { ColorStop, PresetDefinition, PresetId } from '../types/processing';

const NATURAL_STOPS: ColorStop[] = [
  { t: 0.0, rgb: [0.05, 0.06, 0.08] }, // deep, slightly cool shadow (not pure black)
  { t: 0.22, rgb: [0.14, 0.12, 0.11] }, // dark neutral brown
  { t: 0.45, rgb: [0.38, 0.35, 0.28] }, // olive / earth midtone
  { t: 0.68, rgb: [0.64, 0.58, 0.47] }, // warm tan
  { t: 0.86, rgb: [0.85, 0.81, 0.72] }, // light warm highlight
  { t: 1.0, rgb: [0.97, 0.96, 0.92] }, // near-white, slightly warm
];

const NEUTRAL_STOPS: ColorStop[] = [
  { t: 0.0, rgb: [0.04, 0.04, 0.05] },
  { t: 0.5, rgb: [0.5, 0.5, 0.5] },
  { t: 1.0, rgb: [0.97, 0.97, 0.97] },
];

const WARM_STOPS: ColorStop[] = [
  { t: 0.0, rgb: [0.07, 0.06, 0.06] },
  { t: 0.22, rgb: [0.18, 0.13, 0.09] },
  { t: 0.45, rgb: [0.45, 0.36, 0.24] },
  { t: 0.68, rgb: [0.72, 0.6, 0.43] },
  { t: 0.86, rgb: [0.89, 0.79, 0.63] },
  { t: 1.0, rgb: [0.98, 0.94, 0.86] },
];

const COOL_STOPS: ColorStop[] = [
  { t: 0.0, rgb: [0.04, 0.06, 0.09] },
  { t: 0.22, rgb: [0.11, 0.13, 0.16] },
  { t: 0.45, rgb: [0.32, 0.38, 0.4] },
  { t: 0.68, rgb: [0.56, 0.63, 0.63] },
  { t: 0.86, rgb: [0.8, 0.85, 0.85] },
  { t: 1.0, rgb: [0.95, 0.97, 0.98] },
];

const HIGH_CONTRAST_STOPS: ColorStop[] = [
  { t: 0.0, rgb: [0.02, 0.02, 0.03] },
  { t: 0.18, rgb: [0.1, 0.08, 0.07] },
  { t: 0.42, rgb: [0.34, 0.3, 0.22] },
  { t: 0.62, rgb: [0.66, 0.58, 0.44] },
  { t: 0.82, rgb: [0.88, 0.82, 0.68] },
  { t: 1.0, rgb: [1.0, 0.99, 0.96] },
];

const LANDSCAPE_STOPS: ColorStop[] = [
  { t: 0.0, rgb: [0.04, 0.06, 0.07] },
  { t: 0.2, rgb: [0.13, 0.14, 0.09] }, // dark earth
  { t: 0.42, rgb: [0.27, 0.34, 0.18] }, // foliage green
  { t: 0.62, rgb: [0.55, 0.55, 0.36] }, // dry grass / tan
  { t: 0.82, rgb: [0.79, 0.81, 0.73] }, // bright terrain / haze
  { t: 1.0, rgb: [0.93, 0.96, 0.98] }, // sky-tinted highlight
];

const PORTRAIT_STOPS: ColorStop[] = [
  { t: 0.0, rgb: [0.08, 0.06, 0.06] },
  { t: 0.25, rgb: [0.28, 0.18, 0.15] }, // deep warm shadow (skin-like)
  { t: 0.5, rgb: [0.62, 0.44, 0.36] }, // warm midtone skin
  { t: 0.72, rgb: [0.83, 0.65, 0.55] }, // light skin highlight
  { t: 0.9, rgb: [0.93, 0.82, 0.74] },
  { t: 1.0, rgb: [0.98, 0.93, 0.88] },
];

const MONOCHROME_STOPS: ColorStop[] = [
  { t: 0.0, rgb: [0.0, 0.0, 0.0] },
  { t: 1.0, rgb: [1.0, 1.0, 1.0] },
];

export const PRESETS: Record<PresetId, PresetDefinition> = {
  natural: {
    id: 'natural',
    label: 'Natural',
    description: 'Balanced, restrained photographic approximation.',
    colorStops: NATURAL_STOPS,
    overrides: {
      saturation: 0,
      contrast: 0,
      colorStrength: 100,
      localContrast: 25,
      sharpenAmount: 30,
    },
  },
  neutral: {
    id: 'neutral',
    label: 'Neutral',
    description: 'Minimal color assumptions; closest to a toned grayscale.',
    colorStops: NEUTRAL_STOPS,
    overrides: {
      saturation: -20,
      contrast: 0,
      colorStrength: 60,
      localContrast: 15,
      sharpenAmount: 20,
      sceneHeuristics: false,
    },
  },
  'natural-warm': {
    id: 'natural-warm',
    label: 'Natural Warm',
    description: 'Slightly warmer photographic appearance.',
    colorStops: WARM_STOPS,
    overrides: {
      temperature: 18,
      saturation: 5,
      colorStrength: 100,
      localContrast: 25,
    },
  },
  'natural-cool': {
    id: 'natural-cool',
    label: 'Natural Cool',
    description: 'Slightly cooler, overcast-like appearance.',
    colorStops: COOL_STOPS,
    overrides: {
      temperature: -18,
      saturation: 0,
      colorStrength: 100,
      localContrast: 25,
    },
  },
  'high-contrast': {
    id: 'high-contrast',
    label: 'High Contrast',
    description: 'More local contrast and sharpening for maximum detail.',
    colorStops: HIGH_CONTRAST_STOPS,
    overrides: {
      contrast: 25,
      localContrast: 55,
      sharpenAmount: 60,
      shadowLift: 5,
      highlightRecovery: 15,
    },
  },
  landscape: {
    id: 'landscape',
    label: 'Landscape',
    description: 'Emphasizes natural greens, earth tones, and sky-like blues.',
    colorStops: LANDSCAPE_STOPS,
    overrides: {
      saturation: 12,
      localContrast: 35,
      sceneHeuristics: true,
      colorStrength: 100,
    },
  },
  portrait: {
    id: 'portrait',
    label: 'Portrait / People',
    description: 'Warm, soft tones tuned for skin-like midtones.',
    colorStops: PORTRAIT_STOPS,
    overrides: {
      saturation: 8,
      contrast: -8,
      localContrast: 12,
      sharpenAmount: 15,
      shadowLift: 18,
      highlightRecovery: 25,
      sceneHeuristics: false,
    },
  },
  monochrome: {
    id: 'monochrome',
    label: 'Monochrome Visible',
    description: 'Pure grayscale tonal rendering, no color reconstruction.',
    colorStops: MONOCHROME_STOPS,
    overrides: {
      colorStrength: 0,
      saturation: 0,
      sceneHeuristics: false,
    },
  },
};

export const PRESET_LIST: PresetDefinition[] = Object.values(PRESETS);
