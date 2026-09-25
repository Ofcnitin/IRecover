import type {
  PhotoFinishPreset,
  PhotoFinishCustomSettings,
  CloudinaryTransformDescriptor,
} from '../types/photoFinish';

/**
 * Maps a Photo Finish preset to a restrained, deterministic Cloudinary
 * transformation. Intentionally avoids Cloudinary's generative/AI-powered
 * transformations (e.g. `e_gen_restore`, `e_gen_recolor`, `e_background_removal`,
 * `e_enhance` in its AI mode) -- this stays a controlled photographic
 * finishing layer, not an image generator.
 *
 * Reference: https://cloudinary.com/documentation/transformation_reference
 */
export function buildCloudinaryTransformation(
  preset: PhotoFinishPreset,
  custom: PhotoFinishCustomSettings
): CloudinaryTransformDescriptor | null {
  switch (preset) {
    case 'off':
      return null;

    case 'natural':
      return {
        summary: 'Restrained brightness, color, and contrast balance.',
        transformation: 'e_improve:indoor:0,c_scale',
      };

    case 'natural-auto-color':
      return {
        summary: 'Automatic color balancing with restrained contrast.',
        transformation: 'e_auto_color,c_scale',
      };

    case 'natural-auto-contrast':
      return {
        summary: 'Restrained color with automatic contrast.',
        transformation: 'e_auto_contrast,c_scale',
      };

    case 'natural-auto-color-contrast':
      return {
        summary: 'Automatic color balancing and automatic contrast.',
        transformation: 'e_auto_color/e_auto_contrast,c_scale',
      };

    case 'warm-photograph':
      return {
        summary: 'Subtle warmth with restrained contrast and saturation.',
        transformation: 'e_auto_color,co_rgb:fff4e0,e_tint:15:fff4e0,co_rgb:000000,e_contrast:12,e_saturation:8',
      };

    case 'cool-photograph':
      return {
        summary: 'Subtle cool tone with restrained contrast.',
        transformation: 'e_auto_color,e_tint:15:e0f0ff,e_contrast:12,e_saturation:2',
      };

    case 'film-neutral':
      return {
        summary: 'Controlled contrast and saturation with subtle finishing sharpen.',
        transformation: 'e_contrast:10,e_saturation:-5,e_sharpen:40',
      };

    case 'custom':
      return {
        summary: 'Custom brightness, contrast, saturation, hue, and sharpen.',
        transformation: buildCustomTransformation(custom),
      };
  }
}

function buildCustomTransformation(custom: PhotoFinishCustomSettings): string {
  const parts: string[] = [];
  if (custom.brightness !== 0) parts.push(`e_brightness:${clampInt(custom.brightness, -50, 50)}`);
  if (custom.contrast !== 0) parts.push(`e_contrast:${clampInt(custom.contrast, -50, 50)}`);
  if (custom.saturation !== 0) parts.push(`e_saturation:${clampInt(custom.saturation, -50, 50)}`);
  if (custom.hue !== 0) parts.push(`e_hue:${clampInt(custom.hue, -30, 30)}`);
  if (custom.sharpen > 0) parts.push(`e_sharpen:${clampInt(custom.sharpen, 0, 100)}`);
  return parts.length > 0 ? parts.join(',') : 'e_improve:indoor:0';
}

function clampInt(v: number, lo: number, hi: number): number {
  return Math.round(Math.max(lo, Math.min(hi, v)));
}

export const PHOTO_FINISH_PRESET_LABELS: Record<PhotoFinishPreset, string> = {
  off: 'Off',
  natural: 'Natural',
  'natural-auto-color': 'Natural + Auto Color',
  'natural-auto-contrast': 'Natural + Auto Contrast',
  'natural-auto-color-contrast': 'Natural + Auto Color + Contrast',
  'warm-photograph': 'Warm Photograph',
  'cool-photograph': 'Cool Photograph',
  'film-neutral': 'Film Neutral',
  custom: 'Custom',
};

export const PHOTO_FINISH_PRESET_ORDER: PhotoFinishPreset[] = [
  'off',
  'natural',
  'natural-auto-color',
  'natural-auto-contrast',
  'natural-auto-color-contrast',
  'warm-photograph',
  'cool-photograph',
  'film-neutral',
  'custom',
];
