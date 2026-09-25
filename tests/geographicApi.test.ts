import { describe, it, expect, vi, afterEach } from 'vitest';
import { onRequestPost } from '../functions/api/geographic-analysis';

const TINY_PNG_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=';
const TINY_PNG_DATA_URL = `data:image/png;base64,${TINY_PNG_BASE64}`;

function makeRequest(body: unknown, headers: Record<string, string> = {}): Request {
  return new Request('https://example.com/api/geographic-analysis', {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...headers },
    body: JSON.stringify(body),
  });
}

describe('POST /api/geographic-analysis request validation', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('rejects a missing/invalid image payload without calling Gemini', async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);
    const res = await onRequestPost({
      request: makeRequest({ image: 'not-an-image', localSummary: 'x' }),
      env: { GEMINI_API_KEY: 'test-key' },
    } as any);
    expect(res.status).toBe(400);
    const json = await res.json();
    expect(json.ok).toBe(false);
    expect(json.code).toBe('validation-error');
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('rejects a request with no local analysis summary', async () => {
    const res = await onRequestPost({
      request: makeRequest({ image: TINY_PNG_DATA_URL }),
      env: { GEMINI_API_KEY: 'test-key' },
    } as any);
    expect(res.status).toBe(400);
    const json = await res.json();
    expect(json.code).toBe('validation-error');
  });

  it('rejects out-of-range GPS coordinates', async () => {
    const res = await onRequestPost({
      request: makeRequest({
        image: TINY_PNG_DATA_URL,
        localSummary: 'summary',
        gps: { latitude: 999, longitude: 0 },
        requestLocation: true,
      }),
      env: { GEMINI_API_KEY: 'test-key' },
    } as any);
    expect(res.status).toBe(400);
  });

  it('never sends anything upstream when the body exceeds the content-length limit', async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);
    const res = await onRequestPost({
      request: makeRequest(
        { image: TINY_PNG_DATA_URL, localSummary: 'x' },
        { 'content-length': String(50 * 1024 * 1024) }
      ),
      env: { GEMINI_API_KEY: 'test-key' },
    } as any);
    expect(res.status).toBe(413);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('reports Gemini as unavailable (not an error) when GEMINI_API_KEY is unset -- local analysis flow is unaffected', async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);
    const res = await onRequestPost({
      request: makeRequest({ image: TINY_PNG_DATA_URL, localSummary: 'Vegetation ~40%.' }),
      env: {},
    } as any);
    expect(res.status).toBe(200);
    const json = await res.json();
    expect(json.ok).toBe(true);
    expect(json.gemini).toBeUndefined();
    expect(json.geminiUnavailableReason).toMatch(/not configured/i);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('does not request reverse geocoding when requestLocation is false, even with valid GPS', async () => {
    const fetchSpy = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetchSpy);
    await onRequestPost({
      request: makeRequest({
        image: TINY_PNG_DATA_URL,
        localSummary: 'x',
        gps: { latitude: 10, longitude: 10 },
        requestLocation: false,
      }),
      env: {}, // no Gemini key either, so no calls should happen at all
    } as any);
    // Only assert no Nominatim call was made (no fetch to nominatim.openstreetmap.org).
    const calledUrls = fetchSpy.mock.calls.map((c: any[]) => String(c[0]));
    expect(calledUrls.some((u) => u.includes('nominatim'))).toBe(false);
  });

  it('returns invalid-response when Gemini returns malformed JSON', async () => {
    const fetchSpy = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          candidates: [{ content: { parts: [{ text: 'not valid json {{{' }] } }],
        }),
        { status: 200 }
      )
    );
    vi.stubGlobal('fetch', fetchSpy);
    const res = await onRequestPost({
      request: makeRequest({ image: TINY_PNG_DATA_URL, localSummary: 'x' }),
      env: { GEMINI_API_KEY: 'test-key' },
    } as any);
    expect(res.status).toBe(502);
    const json = await res.json();
    expect(json.code).toBe('invalid-response');
  });

  it('returns a validated, sanitized result when Gemini responds with well-formed JSON', async () => {
    const fetchSpy = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          candidates: [
            {
              content: {
                parts: [
                  {
                    text: JSON.stringify({
                      scene: { description: 'A green valley.', confidence: 0.8 },
                      features: [{ type: 'vegetation', description: 'Trees', estimatedCoverage: 60, confidence: 0.9 }],
                      location: null,
                      limitations: ['Estimates only.'],
                    }),
                  },
                ],
              },
            },
          ],
        }),
        { status: 200 }
      )
    );
    vi.stubGlobal('fetch', fetchSpy);
    const res = await onRequestPost({
      request: makeRequest({ image: TINY_PNG_DATA_URL, localSummary: 'x' }),
      env: { GEMINI_API_KEY: 'test-key' },
    } as any);
    expect(res.status).toBe(200);
    const json = await res.json();
    expect(json.ok).toBe(true);
    expect(json.gemini.scene.description).toBe('A green valley.');
    expect(json.gemini.features[0].type).toBe('vegetation');
  });

  it('treats an upstream Gemini HTTP failure as "unavailable", not a hard error', async () => {
    const fetchSpy = vi.fn().mockResolvedValue(new Response('forbidden', { status: 403 }));
    vi.stubGlobal('fetch', fetchSpy);
    const res = await onRequestPost({
      request: makeRequest({ image: TINY_PNG_DATA_URL, localSummary: 'x' }),
      env: { GEMINI_API_KEY: 'test-key' },
    } as any);
    expect(res.status).toBe(200);
    const json = await res.json();
    expect(json.ok).toBe(true);
    expect(json.geminiUnavailableReason).toBeTruthy();
  });

  it('never forwards the raw upstream error body to the client', async () => {
    const fetchSpy = vi.fn().mockResolvedValue(new Response('SECRET-INTERNAL-DETAIL', { status: 500 }));
    vi.stubGlobal('fetch', fetchSpy);
    const res = await onRequestPost({
      request: makeRequest({ image: TINY_PNG_DATA_URL, localSummary: 'x' }),
      env: { GEMINI_API_KEY: 'test-key' },
    } as any);
    const text = await res.text();
    expect(text).not.toContain('SECRET-INTERNAL-DETAIL');
  });
});

describe('GET /api/geographic-analysis', () => {
  it('rejects non-POST requests', async () => {
    const { onRequestGet } = await import('../functions/api/geographic-analysis');
    const res = await onRequestGet();
    expect(res.status).toBe(405);
  });
});
