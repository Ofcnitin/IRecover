import type { GeographicAnalysisApiResponse, LocalGeographicAnalysis } from '../types/geographic';

/**
 * The browser NEVER talks to Gemini or Nominatim directly and never sees
 * the Gemini API key. This calls our own same-origin endpoint (a
 * Cloudflare Pages Function at /functions/api/geographic-analysis.ts).
 */
const GEOGRAPHIC_ANALYSIS_ENDPOINT = '/api/geographic-analysis';
const REQUEST_TIMEOUT_MS = 30000;

/** Turns the local measurable analysis into a short plain-text summary Gemini is given as grounding context. */
export function summarizeLocalAnalysis(local: LocalGeographicAnalysis): string {
  const topCover = local.landCover
    .slice(0, 4)
    .map((f) => `${f.label} ~${f.estimatedCoveragePercent}% (${f.confidence} confidence)`)
    .join('; ');
  const ndvi = local.vegetation.ndvi.available
    ? `NDVI mean ${local.vegetation.ndvi.mean}`
    : 'NDVI not available (no spectral bands)';
  return [
    `Estimated land cover: ${topCover || 'none confidently classified'}.`,
    `Vegetation: ${local.vegetation.description} ${ndvi}.`,
    `Water: ${local.water.description}`,
    `Terrain: ${local.terrain.description} (${local.terrain.confidence} confidence).`,
    `Built-up: ${local.builtUp.description}`,
  ].join(' ');
}

export interface RequestGeographicAnalysisArgs {
  /** A downscaled analysis-resolution image, as a data URL. */
  imageDataUrl: string;
  local: LocalGeographicAnalysis;
  gps: { latitude: number; longitude: number } | null;
  requestLocation: boolean;
  signal?: AbortSignal;
}

export interface GeographicAnalysisClientResult {
  ok: boolean;
  response?: GeographicAnalysisApiResponse;
  errorMessage?: string;
}

export async function requestGeographicAnalysis({
  imageDataUrl,
  local,
  gps,
  requestLocation,
  signal,
}: RequestGeographicAnalysisArgs): Promise<GeographicAnalysisClientResult> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  const externalAbort = () => controller.abort();
  signal?.addEventListener('abort', externalAbort);

  try {
    const response = await fetch(GEOGRAPHIC_ANALYSIS_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: imageDataUrl,
        localSummary: summarizeLocalAnalysis(local),
        gps,
        requestLocation,
      }),
      signal: controller.signal,
    });

    let payload: GeographicAnalysisApiResponse | null = null;
    try {
      payload = await response.json();
    } catch {
      return { ok: false, errorMessage: 'Received an unexpected response from the server.' };
    }

    if (!response.ok && !payload) {
      return { ok: false, errorMessage: 'Geographic Analysis request failed.' };
    }

    return { ok: true, response: payload ?? undefined };
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      return { ok: false, errorMessage: 'Geographic Analysis timed out or was cancelled.' };
    }
    return { ok: false, errorMessage: 'Could not reach the Geographic Analysis service. Check your connection.' };
  } finally {
    clearTimeout(timeout);
    signal?.removeEventListener('abort', externalAbort);
  }
}
