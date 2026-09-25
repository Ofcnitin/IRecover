export type PhotoFinishPreset =
  | 'off'
  | 'natural'
  | 'natural-auto-color'
  | 'natural-auto-contrast'
  | 'natural-auto-color-contrast'
  | 'warm-photograph'
  | 'cool-photograph'
  | 'film-neutral'
  | 'custom';

export interface PhotoFinishCustomSettings {
  brightness: number; // -50..50
  contrast: number; // -50..50
  saturation: number; // -50..50
  hue: number; // -30..30 degrees
  sharpen: number; // 0..100
}

export interface PhotoFinishSettings {
  preset: PhotoFinishPreset;
  custom: PhotoFinishCustomSettings;
}

export const DEFAULT_PHOTO_FINISH_SETTINGS: PhotoFinishSettings = {
  preset: 'off',
  custom: {
    brightness: 0,
    contrast: 10,
    saturation: 10,
    hue: 0,
    sharpen: 20,
  },
};

/** Whether Photo Finish is currently sending data outside the browser. */
export type PrivacyMode = 'local' | 'cloud';

export type ProcessingMode = 'idle' | 'processing-local' | 'processing-cloud' | 'error';

export interface CloudinaryTransformDescriptor {
  /** Human-readable summary of what will be applied, shown in the UI before sending. */
  summary: string;
  /** The actual Cloudinary transformation string (e.g. "e_auto_color,co_rgb:...'). */
  transformation: string;
}

export type PhotoFinishResult =
  | { status: 'success'; imageDataUrl: string; appliedPreset: PhotoFinishPreset }
  | { status: 'error'; message: string; code: PhotoFinishErrorCode };

export type PhotoFinishErrorCode =
  | 'not-configured'
  | 'network-error'
  | 'validation-error'
  | 'upstream-error'
  | 'timeout'
  | 'unknown';

/** Shape of the JSON our Cloudflare API endpoint returns to the browser. Never contains credentials. */
export interface CloudinaryApiResponse {
  ok: boolean;
  imageUrl?: string;
  imageDataUrl?: string;
  error?: string;
  code?: PhotoFinishErrorCode;
}

export interface ImageProcessingState {
  mode: ProcessingMode;
  privacy: PrivacyMode;
  photoFinishError: string | null;
}
