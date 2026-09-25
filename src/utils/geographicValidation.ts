import type {
  GeminiFeatureInterpretation,
  GeminiGeographicResponse,
  GeminiLocationGuess,
  LandCoverType,
} from '../types/geographic';

/**
 * Validates and sanitizes an arbitrary parsed-JSON value against the
 * GeminiGeographicResponse shape. Never trust model output blindly --
 * this rejects anything that doesn't match, and clamps/normalizes the
 * fields that do, so a malformed or adversarial response can never
 * reach the UI unvalidated.
 */

const VALID_FEATURE_TYPES: (LandCoverType | 'other')[] = [
  'vegetation',
  'water',
  'bare-soil',
  'rock',
  'snow-ice',
  'urban',
  'agriculture',
  'open-terrain',
  'unknown',
  'other',
];

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function clampConfidence(v: unknown): number {
  const n = typeof v === 'number' && Number.isFinite(v) ? v : 0;
  return Math.max(0, Math.min(1, n));
}

function sanitizeString(v: unknown, maxLen = 800): string | null {
  if (typeof v !== 'string') return null;
  const trimmed = v.trim();
  if (!trimmed) return null;
  return trimmed.slice(0, maxLen);
}

function sanitizeFeature(v: unknown): GeminiFeatureInterpretation | null {
  if (!isPlainObject(v)) return null;
  const description = sanitizeString(v.description);
  if (!description) return null;
  const rawType = typeof v.type === 'string' ? v.type : 'other';
  const type = (VALID_FEATURE_TYPES.includes(rawType as any) ? rawType : 'other') as
    | LandCoverType
    | 'other';
  const coverage =
    typeof v.estimatedCoverage === 'number' && Number.isFinite(v.estimatedCoverage)
      ? Math.max(0, Math.min(100, v.estimatedCoverage))
      : null;
  return {
    type,
    description,
    estimatedCoverage: coverage,
    confidence: clampConfidence(v.confidence),
  };
}

function sanitizeLocation(v: unknown): GeminiLocationGuess | null {
  if (v === null) return null;
  if (!isPlainObject(v)) return null;
  const note = sanitizeString(v.note) ?? 'Location could not be determined from the image alone.';
  return {
    country: sanitizeString(v.country, 200),
    region: sanitizeString(v.region, 200),
    note,
  };
}

export interface GeminiValidationResult {
  ok: boolean;
  value?: GeminiGeographicResponse;
  error?: string;
}

/** Strips ```json fences etc. that chat models sometimes wrap structured output in. */
export function stripJsonFences(text: string): string {
  const trimmed = text.trim();
  const fenced = /^```(?:json)?\s*([\s\S]*?)\s*```$/i.exec(trimmed);
  return fenced ? fenced[1].trim() : trimmed;
}

export function validateGeminiResponse(raw: unknown): GeminiValidationResult {
  if (!isPlainObject(raw)) {
    return { ok: false, error: 'Response was not a JSON object.' };
  }

  const sceneRaw = raw.scene;
  if (!isPlainObject(sceneRaw)) {
    return { ok: false, error: 'Missing "scene" object.' };
  }
  const description = sanitizeString(sceneRaw.description, 1200);
  if (!description) {
    return { ok: false, error: 'Missing "scene.description".' };
  }

  const featuresRaw = Array.isArray(raw.features) ? raw.features : [];
  const features = featuresRaw
    .map(sanitizeFeature)
    .filter((f): f is GeminiFeatureInterpretation => f !== null)
    .slice(0, 20);

  const limitationsRaw = Array.isArray(raw.limitations) ? raw.limitations : [];
  const limitations = limitationsRaw
    .map((l) => sanitizeString(l, 400))
    .filter((l): l is string => l !== null)
    .slice(0, 10);

  if (limitations.length === 0) {
    limitations.push('Exact geographic identity cannot be determined from the image alone.');
  }

  let location: GeminiLocationGuess | null;
  try {
    location = sanitizeLocation(raw.location);
  } catch {
    location = null;
  }

  const value: GeminiGeographicResponse = {
    scene: { description, confidence: clampConfidence(sceneRaw.confidence) },
    features,
    location,
    limitations,
  };

  return { ok: true, value };
}
