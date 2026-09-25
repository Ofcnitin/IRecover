import { signCloudinaryParams, arrayBufferToBase64, parseImageDataUrl } from '../_lib/cloudinaryServer';
import { buildCloudinaryTransformation } from '../../src/services/photoFinishTransforms';
import type { PhotoFinishCustomSettings, PhotoFinishPreset } from '../../src/types/photoFinish';

/**
 * Required server-side environment variables (set in the Cloudflare Pages
 * project settings -- NEVER committed, NEVER prefixed with VITE_ / NEXT_PUBLIC_
 * so they are never bundled into client code):
 *
 *   CLOUDINARY_CLOUD_NAME
 *   CLOUDINARY_API_KEY
 *   CLOUDINARY_API_SECRET
 */
interface Env {
  CLOUDINARY_CLOUD_NAME?: string;
  CLOUDINARY_API_KEY?: string;
  CLOUDINARY_API_SECRET?: string;
}

interface PagesContext {
  request: Request;
  env: Env;
}

const MAX_IMAGE_BYTES = 15 * 1024 * 1024; // 15 MB, after already being processed/downscaled client-side
const UPLOAD_FOLDER = 'irecover-photofinish-tmp';
const ALLOWED_PRESETS: PhotoFinishPreset[] = [
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

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' },
  });
}

function fail(status: number, code: string, error: string): Response {
  // Never include upstream error bodies, stack traces, or credential hints here.
  return json({ ok: false, code, error }, status);
}

export async function onRequestPost(context: PagesContext): Promise<Response> {
  const { request, env } = context;

  let body: any;
  try {
    body = await request.json();
  } catch {
    return fail(400, 'validation-error', 'Request body must be valid JSON.');
  }

  const preset = body?.preset as PhotoFinishPreset;
  if (!ALLOWED_PRESETS.includes(preset)) {
    return fail(400, 'validation-error', 'Unknown or missing Photo Finish preset.');
  }
  if (preset === 'off') {
    return fail(400, 'validation-error', 'Photo Finish is off; nothing to process.');
  }

  const parsed = parseImageDataUrl(body?.image);
  if (!parsed) {
    return fail(400, 'validation-error', 'Image must be a PNG, JPEG, or WebP data URL.');
  }
  if (parsed.byteLength > MAX_IMAGE_BYTES) {
    return fail(
      413,
      'validation-error',
      `Image is too large for Photo Finish (max ${MAX_IMAGE_BYTES / (1024 * 1024)} MB).`
    );
  }

  const custom: PhotoFinishCustomSettings = normalizeCustomSettings(body?.custom);
  const descriptor = buildCloudinaryTransformation(preset, custom);
  if (!descriptor) {
    return fail(400, 'validation-error', 'This preset does not produce a transformation.');
  }

  const cloudName = env.CLOUDINARY_CLOUD_NAME;
  const apiKey = env.CLOUDINARY_API_KEY;
  const apiSecret = env.CLOUDINARY_API_SECRET;

  if (!cloudName || !apiKey || !apiSecret) {
    // Explicitly NOT a crash / fake success -- a clear, honest "not configured"
    // state the frontend can show and gracefully fall back from.
    return fail(
      503,
      'not-configured',
      'Photo Finish is unavailable: Cloudinary credentials are not configured on the server.'
    );
  }

  let uploadedPublicId: string | null = null;

  try {
    // 1. Signed upload of the already-locally-converted RGB image to a
    //    dedicated temporary folder.
    const timestamp = Math.floor(Date.now() / 1000);
    const uploadSignature = await signCloudinaryParams({ folder: UPLOAD_FOLDER, timestamp }, apiSecret);

    const uploadBody = new URLSearchParams();
    uploadBody.set('file', `data:${parsed.mime};base64,${parsed.base64}`);
    uploadBody.set('api_key', apiKey);
    uploadBody.set('timestamp', String(timestamp));
    uploadBody.set('folder', UPLOAD_FOLDER);
    uploadBody.set('signature', uploadSignature);

    const uploadResponse = await fetchWithTimeout(
      `https://api.cloudinary.com/v1_1/${encodeURIComponent(cloudName)}/image/upload`,
      { method: 'POST', body: uploadBody },
      20000
    );

    if (!uploadResponse.ok) {
      return fail(502, 'upstream-error', 'Cloudinary rejected the upload.');
    }
    const uploadJson: any = await uploadResponse.json();
    uploadedPublicId = uploadJson.public_id ?? null;
    const secureUrl: string | undefined = uploadJson.secure_url;
    const format: string | undefined = uploadJson.format;

    if (!secureUrl || !uploadedPublicId || !format) {
      return fail(502, 'upstream-error', 'Cloudinary returned an unexpected upload response.');
    }

    // 2. Build the transformed delivery URL and fetch the finished bytes.
    const transformedUrl = insertTransformation(secureUrl, descriptor.transformation);
    const transformedResponse = await fetchWithTimeout(transformedUrl, {}, 20000);
    if (!transformedResponse.ok) {
      return fail(502, 'upstream-error', 'Cloudinary could not apply the requested finishing transformation.');
    }
    const transformedBuffer = await transformedResponse.arrayBuffer();
    const base64Result = arrayBufferToBase64(transformedBuffer);
    const resultMime = format === 'jpg' ? 'image/jpeg' : `image/${format}`;

    return json({
      ok: true,
      imageDataUrl: `data:${resultMime};base64,${base64Result}`,
    });
  } catch {
    return fail(504, 'timeout', 'Photo Finish timed out or failed unexpectedly. Your local result is still available.');
  } finally {
    // 3. Best-effort cleanup: this asset was only ever meant to exist long
    //    enough to apply the transformation. We do not retain user images.
    if (uploadedPublicId) {
      try {
        await destroyCloudinaryAsset(cloudName, apiKey, apiSecret, uploadedPublicId);
      } catch {
        // Cleanup failures should never surface as a user-facing error --
        // the finishing result has already been returned above.
      }
    }
  }
}

export async function onRequestGet(): Promise<Response> {
  return fail(405, 'validation-error', 'Use POST for Photo Finish requests.');
}

function normalizeCustomSettings(input: any): PhotoFinishCustomSettings {
  const clamp = (v: unknown, lo: number, hi: number, fallback: number) => {
    const n = typeof v === 'number' && Number.isFinite(v) ? v : fallback;
    return Math.max(lo, Math.min(hi, n));
  };
  return {
    brightness: clamp(input?.brightness, -50, 50, 0),
    contrast: clamp(input?.contrast, -50, 50, 10),
    saturation: clamp(input?.saturation, -50, 50, 10),
    hue: clamp(input?.hue, -30, 30, 0),
    sharpen: clamp(input?.sharpen, 0, 100, 20),
  };
}

function insertTransformation(secureUrl: string, transformation: string): string {
  const marker = '/upload/';
  const idx = secureUrl.indexOf(marker);
  if (idx === -1) return secureUrl;
  const insertAt = idx + marker.length;
  return `${secureUrl.slice(0, insertAt)}${transformation}/${secureUrl.slice(insertAt)}`;
}

async function destroyCloudinaryAsset(
  cloudName: string,
  apiKey: string,
  apiSecret: string,
  publicId: string
): Promise<void> {
  const timestamp = Math.floor(Date.now() / 1000);
  const signature = await signCloudinaryParams({ public_id: publicId, timestamp }, apiSecret);
  const body = new URLSearchParams();
  body.set('public_id', publicId);
  body.set('api_key', apiKey);
  body.set('timestamp', String(timestamp));
  body.set('signature', signature);
  await fetchWithTimeout(
    `https://api.cloudinary.com/v1_1/${encodeURIComponent(cloudName)}/image/destroy`,
    { method: 'POST', body },
    10000
  );
}

async function fetchWithTimeout(url: string, init: RequestInit, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}
