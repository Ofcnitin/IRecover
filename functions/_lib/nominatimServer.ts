/**
 * Server-only OpenStreetMap Nominatim reverse-geocoding client.
 *
 * This lives server-side (rather than being called directly from the
 * browser) for two reasons:
 *  1. Nominatim's usage policy requires a valid identifying
 *     User-Agent/Referer, which browsers do not let JS set on fetch().
 *  2. It lets us enforce the "max 1 request/second" policy centrally and
 *     cache/reuse results instead of hammering the public endpoint from
 *     many browsers independently.
 *
 * If IRecover ever needs high-volume geocoding, swap the implementation
 * behind `reverseGeocode` for a self-hosted Nominatim instance or a paid
 * provider -- callers don't need to change.
 */

import type { ReverseGeocodeResult } from '../../src/types/geographic';

const NOMINATIM_ENDPOINT = 'https://nominatim.openstreetmap.org/reverse';
const MIN_INTERVAL_MS = 1100; // stay under Nominatim's 1 req/sec policy with margin
const ATTRIBUTION = '\u00a9 OpenStreetMap contributors';

// Best-effort, per-isolate rate limiting. Cloudflare may spin up multiple
// isolates, so this is a courtesy limiter, not a hard guarantee -- the
// explicit-trigger-only design elsewhere (Geographic Analysis requires an
// explicit "Analyze" click) is what actually keeps volume low.
let lastRequestAt = 0;
const cache = new Map<string, ReverseGeocodeResult>();

function cacheKey(lat: number, lon: number): string {
  // Round to ~1km precision for cache reuse without losing meaningful accuracy.
  return `${lat.toFixed(2)},${lon.toFixed(2)}`;
}

async function throttle(): Promise<void> {
  const now = Date.now();
  const wait = lastRequestAt + MIN_INTERVAL_MS - now;
  if (wait > 0) {
    await new Promise((resolve) => setTimeout(resolve, wait));
  }
  lastRequestAt = Date.now();
}

export interface ReverseGeocodeOptions {
  /** A real contact/identifying string, required by Nominatim's usage policy. */
  userAgent: string;
}

export async function reverseGeocode(
  latitude: number,
  longitude: number,
  options: ReverseGeocodeOptions
): Promise<ReverseGeocodeResult | null> {
  const key = cacheKey(latitude, longitude);
  const cached = cache.get(key);
  if (cached) return cached;

  await throttle();

  const url = new URL(NOMINATIM_ENDPOINT);
  url.searchParams.set('format', 'jsonv2');
  url.searchParams.set('lat', String(latitude));
  url.searchParams.set('lon', String(longitude));
  url.searchParams.set('zoom', '10');
  url.searchParams.set('addressdetails', '1');

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 10000);

  try {
    const response = await fetch(url.toString(), {
      headers: {
        'User-Agent': options.userAgent,
        'Accept-Language': 'en',
      },
      signal: controller.signal,
    });
    if (!response.ok) return null;

    const json: any = await response.json();
    const address = json?.address ?? {};

    const result: ReverseGeocodeResult = {
      displayName: typeof json?.display_name === 'string' ? json.display_name : 'Unknown location',
      country: typeof address.country === 'string' ? address.country : null,
      region: typeof (address.state ?? address.region) === 'string' ? address.state ?? address.region : null,
      district:
        typeof (address.county ?? address.state_district) === 'string'
          ? address.county ?? address.state_district
          : null,
      locality:
        typeof (address.city ?? address.town ?? address.village) === 'string'
          ? address.city ?? address.town ?? address.village
          : null,
      attribution: ATTRIBUTION,
    };

    cache.set(key, result);
    return result;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}
