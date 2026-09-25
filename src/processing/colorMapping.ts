import type { ColorStop } from '../types/processing';
import { clamp01, lerpOklab, rgbToHsl, hslToRgb } from './colorSpace';

/**
 * Samples a sorted list of color stops (t in 0..1) at position t, blending
 * neighboring stops in OKLab for perceptually smooth, banding-free gradients.
 */
export function sampleRamp(stops: ColorStop[], t: number): [number, number, number] {
  const clamped = clamp01(t);
  if (stops.length === 0) return [clamped, clamped, clamped];
  if (clamped <= stops[0].t) return stops[0].rgb;
  if (clamped >= stops[stops.length - 1].t) return stops[stops.length - 1].rgb;

  for (let i = 0; i < stops.length - 1; i++) {
    const a = stops[i];
    const b = stops[i + 1];
    if (clamped >= a.t && clamped <= b.t) {
      const span = Math.max(1e-6, b.t - a.t);
      const localT = (clamped - a.t) / span;
      return lerpOklab(a.rgb, b.rgb, localT);
    }
  }
  return stops[stops.length - 1].rgb;
}

export interface SceneMaps {
  /** 0..1 confidence that a pixel is sky-like (bright, low-texture, upper image). */
  sky?: Float32Array;
  /** 0..1 confidence that a pixel is vegetation-like (mid intensity, textured). */
  vegetation?: Float32Array;
}

export interface ColorMapOptions {
  colorStrength: number; // 0..100
  saturation: number; // -100..100
  temperature: number; // -100..100
  hueBias: number; // -180..180
  scene?: SceneMaps;
}

/**
 * Converts a tone-mapped intensity buffer into an RGBA buffer using the
 * preset's OKLab color ramp, then applies global saturation / white-balance
 * (temperature) / hue-bias adjustments and optional scene tinting.
 */
export function mapIntensityToRgb(
  intensity: Float32Array,
  width: number,
  height: number,
  stops: ColorStop[],
  options: ColorMapOptions
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(width * height * 4);
  const colorStrength = clamp01(options.colorStrength / 100);
  const satAdjust = options.saturation / 100; // -1..1
  const hueBiasRad = (options.hueBias / 360) % 1;
  // Temperature: warm shifts R up/B down, cool the opposite. Small, tasteful magnitude.
  const tempAmt = (options.temperature / 100) * 0.12;

  for (let i = 0, p = 0; i < intensity.length; i++, p += 4) {
    const I = intensity[i];
    let [r, g, b] = sampleRamp(stops, I);

    // Blend toward neutral gray by (1 - colorStrength) so users can dial
    // back toward a monochrome "visible-style" rendering.
    if (colorStrength < 1) {
      r = r * colorStrength + I * (1 - colorStrength);
      g = g * colorStrength + I * (1 - colorStrength);
      b = b * colorStrength + I * (1 - colorStrength);
    }

    // Optional scene tinting: nudges hue subtly, never overrides luminance.
    if (options.scene?.sky && options.scene.sky[i] > 0.15) {
      const w = options.scene.sky[i] * 0.35;
      b = clamp01(b + w * 0.10);
      r = clamp01(r - w * 0.04);
    }
    if (options.scene?.vegetation && options.scene.vegetation[i] > 0.15) {
      const w = options.scene.vegetation[i] * 0.35;
      g = clamp01(g + w * 0.08);
      r = clamp01(r - w * 0.03);
    }

    // White balance (temperature).
    r = clamp01(r + tempAmt);
    b = clamp01(b - tempAmt);

    // Saturation + hue bias, done in HSL for intuitive control.
    if (satAdjust !== 0 || hueBiasRad !== 0) {
      const [h, s, l] = rgbToHsl(r, g, b);
      let newH = h + hueBiasRad;
      newH -= Math.floor(newH);
      const newS = clamp01(satAdjust >= 0 ? s + (1 - s) * satAdjust : s * (1 + satAdjust));
      [r, g, b] = hslToRgb(newH, newS, l);
    }

    out[p] = r * 255;
    out[p + 1] = g * 255;
    out[p + 2] = b * 255;
    out[p + 3] = 255;
  }

  return out;
}
