import type { PhotoFinishSettings, PhotoFinishResult } from '../types/photoFinish';

/**
 * The browser NEVER talks to Cloudinary directly and never sees a Cloudinary
 * API secret. This calls our own same-origin API endpoint (implemented as a
 * Cloudflare Pages Function in /functions/api/photo-finish.ts), which is the
 * only place Cloudinary credentials are used.
 */
const PHOTO_FINISH_ENDPOINT = '/api/photo-finish';

export interface RequestPhotoFinishArgs {
  /** The already-converted local RGB image, as a data URL (PNG). */
  imageDataUrl: string;
  settings: PhotoFinishSettings;
  /** Aborts the request if the user cancels or starts a new one. */
  signal?: AbortSignal;
}

const REQUEST_TIMEOUT_MS = 45000;

export async function requestPhotoFinish({
  imageDataUrl,
  settings,
  signal,
}: RequestPhotoFinishArgs): Promise<PhotoFinishResult> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  const externalAbort = () => controller.abort();
  signal?.addEventListener('abort', externalAbort);

  try {
    const response = await fetch(PHOTO_FINISH_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: imageDataUrl,
        preset: settings.preset,
        custom: settings.custom,
      }),
      signal: controller.signal,
    });

    let payload: any = null;
    try {
      payload = await response.json();
    } catch {
      // Malformed / non-JSON response from upstream.
      return { status: 'error', message: 'Received an unexpected response from the server.', code: 'unknown' };
    }

    if (!response.ok || !payload?.ok) {
      const code = payload?.code ?? 'unknown';
      const message = payload?.error ?? 'Photo Finish failed. You can still use the local result.';
      return { status: 'error', message, code };
    }

    const resultImage: string | undefined = payload.imageDataUrl ?? payload.imageUrl;
    if (!resultImage) {
      return { status: 'error', message: 'Server response did not include a finished image.', code: 'unknown' };
    }

    return { status: 'success', imageDataUrl: resultImage, appliedPreset: settings.preset };
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      return { status: 'error', message: 'Photo Finish timed out or was cancelled.', code: 'timeout' };
    }
    return {
      status: 'error',
      message: 'Could not reach the Photo Finish service. Check your connection and try again.',
      code: 'network-error',
    };
  } finally {
    clearTimeout(timeout);
    signal?.removeEventListener('abort', externalAbort);
  }
}
