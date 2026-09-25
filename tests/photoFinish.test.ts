import { describe, it, expect, vi, afterEach } from 'vitest';
import { buildCloudinaryTransformation, PHOTO_FINISH_PRESET_ORDER } from '../src/services/photoFinishTransforms';
import { DEFAULT_PHOTO_FINISH_SETTINGS } from '../src/types/photoFinish';
import { sha1Hex, signCloudinaryParams, parseImageDataUrl } from '../functions/_lib/cloudinaryServer';
import { onRequestPost } from '../functions/api/photo-finish';

const TINY_PNG_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=';
const TINY_PNG_DATA_URL = `data:image/png;base64,${TINY_PNG_BASE64}`;

describe('buildCloudinaryTransformation', () => {
  it('returns null for "off" (no Cloudinary call should ever be made)', () => {
    expect(buildCloudinaryTransformation('off', DEFAULT_PHOTO_FINISH_SETTINGS.custom)).toBeNull();
  });

  it('produces a non-empty, deterministic transformation string for every other preset', () => {
    for (const preset of PHOTO_FINISH_PRESET_ORDER) {
      if (preset === 'off') continue;
      const a = buildCloudinaryTransformation(preset, DEFAULT_PHOTO_FINISH_SETTINGS.custom);
      const b = buildCloudinaryTransformation(preset, DEFAULT_PHOTO_FINISH_SETTINGS.custom);
      expect(a).not.toBeNull();
      expect(a!.transformation.length).toBeGreaterThan(0);
      expect(a).toEqual(b); // deterministic
    }
  });

  it('never includes generative/AI transformation codes', () => {
    const forbidden = ['gen_restore', 'gen_recolor', 'gen_background_removal', 'background_removal', 'e_enhance'];
    for (const preset of PHOTO_FINISH_PRESET_ORDER) {
      const descriptor = buildCloudinaryTransformation(preset, DEFAULT_PHOTO_FINISH_SETTINGS.custom);
      if (!descriptor) continue;
      for (const term of forbidden) {
        expect(descriptor.transformation).not.toContain(term);
      }
    }
  });

  it('custom preset reflects clamped brightness/contrast/saturation/hue/sharpen', () => {
    const descriptor = buildCloudinaryTransformation('custom', {
      brightness: 20,
      contrast: -10,
      saturation: 0,
      hue: 15,
      sharpen: 50,
    });
    expect(descriptor?.transformation).toContain('e_brightness:20');
    expect(descriptor?.transformation).toContain('e_contrast:-10');
    expect(descriptor?.transformation).not.toContain('e_saturation'); // 0 means omitted
    expect(descriptor?.transformation).toContain('e_hue:15');
    expect(descriptor?.transformation).toContain('e_sharpen:50');
  });
});

describe('parseImageDataUrl', () => {
  it('accepts a valid PNG data URL', () => {
    const parsed = parseImageDataUrl(TINY_PNG_DATA_URL);
    expect(parsed).not.toBeNull();
    expect(parsed?.mime).toBe('image/png');
    expect(parsed?.byteLength).toBeGreaterThan(0);
  });

  it('rejects non-data-URL strings', () => {
    expect(parseImageDataUrl('https://example.com/image.png')).toBeNull();
  });

  it('rejects unsupported mime types', () => {
    expect(parseImageDataUrl('data:image/gif;base64,AAAA')).toBeNull();
  });

  it('rejects non-string input', () => {
    expect(parseImageDataUrl(undefined)).toBeNull();
    expect(parseImageDataUrl(12345)).toBeNull();
  });
});

describe('Cloudinary signing', () => {
  it('sha1Hex matches the standard FIPS 180 test vector for "abc"', async () => {
    const hash = await sha1Hex('abc');
    expect(hash).toHaveLength(40);
    expect(hash).toBe('a9993e364706816aba3e25717850c26c9cd0d89d');
  });

  it('signCloudinaryParams produces a stable 40-character hex signature', async () => {
    const sig = await signCloudinaryParams({ timestamp: 1700000000, folder: 'irecover-photofinish-tmp' }, 'secret');
    expect(sig).toMatch(/^[a-f0-9]{40}$/);
    const sig2 = await signCloudinaryParams({ timestamp: 1700000000, folder: 'irecover-photofinish-tmp' }, 'secret');
    expect(sig).toBe(sig2);
  });

  it('produces different signatures for different secrets', async () => {
    const a = await signCloudinaryParams({ timestamp: 1, foo: 'bar' }, 'secret-a');
    const b = await signCloudinaryParams({ timestamp: 1, foo: 'bar' }, 'secret-b');
    expect(a).not.toBe(b);
  });
});

describe('POST /api/photo-finish request validation', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function makeRequest(body: unknown): Request {
    return new Request('https://example.com/api/photo-finish', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
  }

  it('rejects a missing/unknown preset without calling Cloudinary', async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);
    const res = await onRequestPost({
      request: makeRequest({ image: TINY_PNG_DATA_URL, preset: 'not-a-real-preset' }),
      env: { CLOUDINARY_CLOUD_NAME: 'x', CLOUDINARY_API_KEY: 'y', CLOUDINARY_API_SECRET: 'z' },
    } as any);
    expect(res.status).toBe(400);
    const json = await res.json();
    expect(json.ok).toBe(false);
    expect(json.code).toBe('validation-error');
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('rejects preset "off" (nothing should ever be sent to Cloudinary for off)', async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);
    const res = await onRequestPost({
      request: makeRequest({ image: TINY_PNG_DATA_URL, preset: 'off' }),
      env: { CLOUDINARY_CLOUD_NAME: 'x', CLOUDINARY_API_KEY: 'y', CLOUDINARY_API_SECRET: 'z' },
    } as any);
    expect(res.status).toBe(400);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('rejects an invalid image payload', async () => {
    const res = await onRequestPost({
      request: makeRequest({ image: 'not-an-image', preset: 'natural' }),
      env: { CLOUDINARY_CLOUD_NAME: 'x', CLOUDINARY_API_KEY: 'y', CLOUDINARY_API_SECRET: 'z' },
    } as any);
    expect(res.status).toBe(400);
    const json = await res.json();
    expect(json.code).toBe('validation-error');
  });

  it('reports "not-configured" and never calls Cloudinary when credentials are missing (local fallback case)', async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);
    const res = await onRequestPost({
      request: makeRequest({ image: TINY_PNG_DATA_URL, preset: 'natural' }),
      env: {},
    } as any);
    expect(res.status).toBe(503);
    const json = await res.json();
    expect(json.ok).toBe(false);
    expect(json.code).toBe('not-configured');
    expect(json.error).not.toMatch(/secret|api_key|apikey/i);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('never leaks the configured API secret in any response body', async () => {
    const fetchSpy = vi.fn().mockResolvedValue(new Response('not json', { status: 500 }));
    vi.stubGlobal('fetch', fetchSpy);
    const res = await onRequestPost({
      request: makeRequest({ image: TINY_PNG_DATA_URL, preset: 'natural' }),
      env: {
        CLOUDINARY_CLOUD_NAME: 'demo-cloud',
        CLOUDINARY_API_KEY: 'demo-key',
        CLOUDINARY_API_SECRET: 'super-secret-value',
      },
    } as any);
    const text = await res.text();
    expect(text).not.toContain('super-secret-value');
  });

  it('completes a full successful round trip against a mocked Cloudinary, and cleans up the temporary asset', async () => {
    const calls: string[] = [];
    const fetchSpy = vi.fn(async (url: string) => {
      calls.push(url);
      if (url.includes('/image/upload')) {
        return new Response(
          JSON.stringify({
            public_id: 'irecover-photofinish-tmp/abc123',
            secure_url: 'https://res.cloudinary.com/demo-cloud/image/upload/v1/irecover-photofinish-tmp/abc123.png',
            format: 'png',
            version: 1,
          }),
          { status: 200 }
        );
      }
      if (url.includes('/image/destroy')) {
        return new Response(JSON.stringify({ result: 'ok' }), { status: 200 });
      }
      // The transformed delivery URL fetch.
      const bytes = Uint8Array.from(atob(TINY_PNG_BASE64), (c) => c.charCodeAt(0));
      return new Response(bytes, { status: 200, headers: { 'content-type': 'image/png' } });
    });
    vi.stubGlobal('fetch', fetchSpy);

    const res = await onRequestPost({
      request: makeRequest({ image: TINY_PNG_DATA_URL, preset: 'natural' }),
      env: {
        CLOUDINARY_CLOUD_NAME: 'demo-cloud',
        CLOUDINARY_API_KEY: 'demo-key',
        CLOUDINARY_API_SECRET: 'demo-secret',
      },
    } as any);

    expect(res.status).toBe(200);
    const json = await res.json();
    expect(json.ok).toBe(true);
    expect(json.imageDataUrl).toMatch(/^data:image\/png;base64,/);

    // Upload, transformed-fetch, and destroy should all have happened.
    expect(calls.some((u) => u.includes('/image/upload'))).toBe(true);
    expect(calls.some((u) => u.includes('/image/destroy'))).toBe(true);
    expect(calls.some((u) => u.includes('e_improve') || u.includes('c_scale'))).toBe(true);
  });
});
