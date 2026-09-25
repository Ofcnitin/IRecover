/**
 * Server-only utilities for the Photo Finish API function. Everything here
 * runs in the Cloudflare Workers runtime (Web Crypto, fetch, etc), never in
 * the browser bundle -- this file lives under /functions, which is not
 * built into the Vite client bundle.
 */

/** SHA-1 hex digest, used for Cloudinary's request-signing scheme. */
export async function sha1Hex(input: string): Promise<string> {
  const bytes = new TextEncoder().encode(input);
  const digest = await crypto.subtle.digest('SHA-1', bytes);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

/**
 * Cloudinary signs requests as SHA1(sorted "key=value&key=value..." for all
 * params except file/api_key/signature/resource_type, with the API secret
 * appended directly -- no separator). See:
 * https://cloudinary.com/documentation/authentication_signatures
 */
export async function signCloudinaryParams(
  params: Record<string, string | number>,
  apiSecret: string
): Promise<string> {
  const sorted = Object.keys(params)
    .sort()
    .map((key) => `${key}=${params[key]}`)
    .join('&');
  return sha1Hex(`${sorted}${apiSecret}`);
}

/** Converts an ArrayBuffer to a base64 string without blowing the call stack on large images. */
export function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = '';
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
}

export interface ParsedDataUrl {
  mime: string;
  base64: string;
  byteLength: number;
}

const DATA_URL_PATTERN = /^data:(image\/(?:png|jpeg|jpg|webp));base64,([a-zA-Z0-9+/=]+)$/;

/** Validates and parses a `data:image/...;base64,...` string without trusting the caller. */
export function parseImageDataUrl(dataUrl: unknown): ParsedDataUrl | null {
  if (typeof dataUrl !== 'string') return null;
  const match = DATA_URL_PATTERN.exec(dataUrl.trim());
  if (!match) return null;
  const [, mime, base64] = match;
  // Rough byte-length estimate from base64 length, without decoding the whole thing.
  const padding = base64.endsWith('==') ? 2 : base64.endsWith('=') ? 1 : 0;
  const byteLength = Math.floor((base64.length * 3) / 4) - padding;
  return { mime: mime === 'image/jpg' ? 'image/jpeg' : mime, base64, byteLength };
}
