import { parseImageDataUrl } from '../_lib/cloudinaryServer';
import { callGeminiVision } from '../_lib/geminiServer';
import { reverseGeocode } from '../_lib/nominatimServer';
import { stripJsonFences, validateGeminiResponse } from '../../src/utils/geographicValidation';
import type { GeographicAnalysisApiResponse } from '../../src/types/geographic';

/**
 * Required server-side environment variables (Cloudflare Pages project
 * settings -- never committed, never prefixed with VITE_):
 *
 *   GEMINI_API_KEY
 *
 * Optional (only needed if reverse geocoding is requested):
 *   none extra -- Nominatim's public endpoint needs no key, only a
 *   descriptive User-Agent, which this function supplies itself.
 */
interface Env {
  GEMINI_API_KEY?: string;
}

interface PagesContext {
  request: Request;
  env: Env;
}

// Deliberately small: this is a downscaled "analysis" copy of the image,
// never the full-resolution original. See src/services/geographicAnalysisApi.ts
// for where that resizing happens before the request is ever sent.
const MAX_IMAGE_BYTES = 6 * 1024 * 1024;
const MAX_BODY_BYTES = 8 * 1024 * 1024;

function json(body: GeographicAnalysisApiResponse, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' },
  });
}

function fail(status: number, code: GeographicAnalysisApiResponse['code'], error: string): Response {
  return json({ ok: false, code, error }, status);
}

function isFiniteNumber(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

export async function onRequestPost(context: PagesContext): Promise<Response> {
  const { request, env } = context;

  const contentLength = Number(request.headers.get('content-length') ?? 0);
  if (contentLength > MAX_BODY_BYTES) {
    return fail(413, 'validation-error', 'Request body is too large.');
  }

  let body: any;
  try {
    body = await request.json();
  } catch {
    return fail(400, 'validation-error', 'Request body must be valid JSON.');
  }

  const parsedImage = parseImageDataUrl(body?.image);
  if (!parsedImage) {
    return fail(400, 'validation-error', 'Image must be a PNG, JPEG, or WebP data URL.');
  }
  if (parsedImage.byteLength > MAX_IMAGE_BYTES) {
    return fail(413, 'validation-error', `Image is too large for analysis (max ${MAX_IMAGE_BYTES / (1024 * 1024)} MB).`);
  }

  const localSummary = typeof body?.localSummary === 'string' ? body.localSummary.slice(0, 4000) : '';
  if (!localSummary) {
    return fail(400, 'validation-error', 'Missing local analysis summary.');
  }

  // GPS is optional context, and only ever comes from file metadata that
  // the browser already extracted -- this endpoint never derives it itself.
  const gpsRequested = body?.gps && isFiniteNumber(body.gps.latitude) && isFiniteNumber(body.gps.longitude);
  const latitude = gpsRequested ? (body.gps.latitude as number) : null;
  const longitude = gpsRequested ? (body.gps.longitude as number) : null;
  if (gpsRequested && (Math.abs(latitude!) > 90 || Math.abs(longitude!) > 180)) {
    return fail(400, 'validation-error', 'GPS coordinates out of range.');
  }

  const wantsLocation = Boolean(body?.requestLocation) && gpsRequested;

  const result: GeographicAnalysisApiResponse = { ok: true };

  // -- Reverse geocoding (Case 1: GPS exists) -----------------------------
  if (wantsLocation) {
    try {
      const location = await reverseGeocode(latitude!, longitude!, {
        userAgent: 'IRecover/1.0 (https://irecover.app; geographic-analysis feature)',
      });
      result.location = location; // may be null if Nominatim had nothing/failed -- never fabricated
    } catch {
      result.location = null;
    }
  } else {
    result.location = null;
  }

  // -- Gemini interpretation (optional layer) ------------------------------
  const apiKey = env.GEMINI_API_KEY;
  if (!apiKey) {
    result.geminiUnavailableReason = 'Geographic interpretation is unavailable: Gemini is not configured on the server.';
    return json(result);
  }

  const gpsContext =
    gpsRequested
      ? `latitude ${latitude!.toFixed(4)}, longitude ${longitude!.toFixed(4)}` +
        (result.location ? `; reverse-geocoded to ${result.location.displayName}` : '')
      : null;

  const geminiCall = await callGeminiVision(apiKey, {
    imageBase64: parsedImage.base64,
    imageMime: parsedImage.mime,
    localAnalysisSummary: localSummary,
    gpsContext,
  });

  if (!geminiCall.ok || !geminiCall.rawText) {
    result.geminiUnavailableReason =
      'Local geographic measurements are available. Natural-language interpretation is temporarily unavailable.';
    return json(result);
  }

  let parsedJson: unknown;
  try {
    parsedJson = JSON.parse(stripJsonFences(geminiCall.rawText));
  } catch {
    return fail(502, 'invalid-response', 'Gemini returned a response that could not be parsed.');
  }

  const validated = validateGeminiResponse(parsedJson);
  if (!validated.ok || !validated.value) {
    return fail(502, 'invalid-response', 'Gemini returned a response that did not match the expected structure.');
  }

  result.gemini = validated.value;
  return json(result);
}

export async function onRequestGet(): Promise<Response> {
  return fail(405, 'validation-error', 'Use POST for Geographic Analysis requests.');
}
