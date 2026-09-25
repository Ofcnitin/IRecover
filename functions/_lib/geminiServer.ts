/**
 * Server-only Gemini integration for Geographic Analysis. Runs entirely in
 * the Cloudflare Workers runtime under /functions -- never bundled into
 * the browser build. GEMINI_API_KEY must never be exposed to the client.
 *
 * Auth: Gemini's current REST API takes the key on the `x-goog-api-key`
 * request header (the documented `?key=` query-string form still works
 * but leaks the key into logs/URLs, so we avoid it here).
 */

// Pinned to a specific, current, cost-effective multimodal model. If Google
// renames/retires this model, update GEMINI_MODEL -- check
// https://ai.google.dev/gemini-api/docs/models for the current lineup.
const GEMINI_MODEL = 'gemini-2.5-flash';
const GEMINI_ENDPOINT = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`;

export interface GeminiRequestContext {
  imageBase64: string;
  imageMime: string;
  /** Plain-language summary of the LOCAL measurable analysis, given to Gemini as grounding context. */
  localAnalysisSummary: string;
  /** GPS-derived context, if any -- given to Gemini so it doesn't need to (and must not) guess coordinates. */
  gpsContext: string | null;
}

function buildPrompt(ctx: GeminiRequestContext): string {
  return [
    'You are analyzing a geographic/landscape image for IRecover, an infrared-to-visible image conversion tool.',
    'Describe ONLY features that are visible in the image or supported by the measurements below.',
    'Do NOT invent or guess specific geographic locations, place names, rivers, lakes, cities, countries, roads, mountains, or landmarks.',
    'Do NOT infer exact geographic identity solely from visual appearance.',
    'Clearly distinguish observations (what is visually apparent) from interpretations (your inference).',
    'If you are not confident about something, say so and lower the confidence value rather than guessing.',
    '',
    'Local measurable analysis already computed (use this as grounding context, do not contradict it without reason):',
    ctx.localAnalysisSummary,
    '',
    ctx.gpsContext
      ? `Known GPS-derived context (from file metadata, not from your inference): ${ctx.gpsContext}`
      : 'No GPS metadata was available for this image. Do not guess a location.',
    '',
    'Respond with ONLY a single JSON object (no markdown fences, no commentary) matching exactly this shape:',
    JSON.stringify(
      {
        scene: { description: 'string, 1-3 sentences', confidence: '0-1 number' },
        features: [
          {
            type: 'vegetation | water | bare-soil | rock | snow-ice | urban | agriculture | open-terrain | unknown',
            description: 'string',
            estimatedCoverage: '0-100 number or null',
            confidence: '0-1 number',
          },
        ],
        location: null,
        limitations: ['string'],
      },
      null,
      2
    ),
    '',
    'If location is not known with reasonable confidence from the provided GPS context, return location: null. Never derive location from visual appearance alone.',
  ].join('\n');
}

export interface GeminiCallResult {
  ok: boolean;
  rawText?: string;
  errorStatus?: number;
  errorMessage?: string;
}

export async function callGeminiVision(
  apiKey: string,
  ctx: GeminiRequestContext,
  timeoutMs = 25000
): Promise<GeminiCallResult> {
  const prompt = buildPrompt(ctx);

  const body = {
    contents: [
      {
        role: 'user',
        parts: [
          { text: prompt },
          {
            inline_data: {
              mime_type: ctx.imageMime,
              data: ctx.imageBase64,
            },
          },
        ],
      },
    ],
    generationConfig: {
      temperature: 0.2,
      maxOutputTokens: 1024,
      responseMimeType: 'application/json',
    },
  };

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(GEMINI_ENDPOINT, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-goog-api-key': apiKey,
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    if (!response.ok) {
      // Never forward the upstream error body to the client -- it may
      // contain account/billing details or other sensitive information.
      return { ok: false, errorStatus: response.status, errorMessage: 'Gemini request failed.' };
    }

    const json: any = await response.json();
    const text: string | undefined = json?.candidates?.[0]?.content?.parts?.find((p: any) => typeof p?.text === 'string')
      ?.text;

    if (!text) {
      return { ok: false, errorMessage: 'Gemini returned no usable content.' };
    }

    return { ok: true, rawText: text };
  } catch (err) {
    const aborted = err instanceof Error && err.name === 'AbortError';
    return { ok: false, errorMessage: aborted ? 'Gemini request timed out.' : 'Gemini request failed unexpectedly.' };
  } finally {
    clearTimeout(timer);
  }
}
