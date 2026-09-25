/**
 * Color-space conversions used throughout the pipeline.
 * All RGB values in this module are normalized to [0, 1] unless stated otherwise.
 */

export function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

export function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

/** sRGB (gamma-encoded, 0..1) -> linear light (0..1) */
export function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/** linear light (0..1) -> sRGB (gamma-encoded, 0..1) */
export function linearToSrgb(c: number): number {
  const v = c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(Math.max(c, 0), 1 / 2.4) - 0.055;
  return clamp01(v);
}

/** Rec. 709 relative luminance from linear RGB. */
export function linearLuminance(r: number, g: number, b: number): number {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** Perceptual luminance (used for display-referred intensity) from gamma-encoded RGB. */
export function perceptualLuminance(r: number, g: number, b: number): number {
  // r,g,b in 0..1, gamma encoded. Standard broadcast-style weights.
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

// ---------- HSL ----------

export function rgbToHsl(r: number, g: number, b: number): [number, number, number] {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l];
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h: number;
  switch (max) {
    case r:
      h = (g - b) / d + (g < b ? 6 : 0);
      break;
    case g:
      h = (b - r) / d + 2;
      break;
    default:
      h = (r - g) / d + 4;
  }
  h /= 6;
  return [h, s, l];
}

function hue2rgb(p: number, q: number, t: number): number {
  let tt = t;
  if (tt < 0) tt += 1;
  if (tt > 1) tt -= 1;
  if (tt < 1 / 6) return p + (q - p) * 6 * tt;
  if (tt < 1 / 2) return q;
  if (tt < 2 / 3) return p + (q - p) * (2 / 3 - tt) * 6;
  return p;
}

export function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  if (s === 0) return [l, l, l];
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const r = hue2rgb(p, q, h + 1 / 3);
  const g = hue2rgb(p, q, h);
  const b = hue2rgb(p, q, h - 1 / 3);
  return [r, g, b];
}

// ---------- OKLab (Björn Ottosson's formulation) ----------
// Operates on LINEAR sRGB in, out.

export function linearRgbToOklab(r: number, g: number, b: number): [number, number, number] {
  const l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
  const m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
  const s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

  const l_ = Math.cbrt(l);
  const m_ = Math.cbrt(m);
  const s_ = Math.cbrt(s);

  return [
    0.2104542553 * l_ + 0.793617785 * m_ - 0.0040720468 * s_,
    1.9779984951 * l_ - 2.428592205 * m_ + 0.4505937099 * s_,
    0.0259040371 * l_ + 0.7827717662 * m_ - 0.808675766 * s_,
  ];
}

export function oklabToLinearRgb(L: number, a: number, b: number): [number, number, number] {
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.291485548 * b;

  const l = l_ * l_ * l_;
  const m = m_ * m_ * m_;
  const s = s_ * s_ * s_;

  return [
    +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
}

/** Convenience: gamma sRGB (0..1) -> OKLab */
export function srgbToOklab(r: number, g: number, b: number): [number, number, number] {
  return linearRgbToOklab(srgbToLinear(r), srgbToLinear(g), srgbToLinear(b));
}

/** Convenience: OKLab -> gamma sRGB (0..1), clamped into range. */
export function oklabToSrgb(L: number, a: number, b: number): [number, number, number] {
  const [lr, lg, lb] = oklabToLinearRgb(L, a, b);
  return [linearToSrgb(lr), linearToSrgb(lg), linearToSrgb(lb)];
}

/** Linearly interpolate two colors in OKLab space for perceptually smooth gradients. */
export function lerpOklab(
  c1: [number, number, number],
  c2: [number, number, number],
  t: number
): [number, number, number] {
  const [L1, a1, b1] = srgbToOklab(c1[0], c1[1], c1[2]);
  const [L2, a2, b2] = srgbToOklab(c2[0], c2[1], c2[2]);
  const L = L1 + (L2 - L1) * t;
  const a = a1 + (a2 - a1) * t;
  const b = b1 + (b2 - b1) * t;
  return oklabToSrgb(L, a, b);
}
