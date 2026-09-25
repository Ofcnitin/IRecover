import { boxBlur } from './contrast';
import { clamp01 } from './colorSpace';

export interface SharpenParams {
  amount: number; // 0..200 (%)
  radius: number; // 0.3..5 px
  threshold: number; // 0..255, in 8-bit terms
}

/**
 * Classic unsharp mask on a normalized [0,1] channel: blur, subtract to get
 * a "detail" signal, then add a scaled copy of that detail back onto the
 * original. A threshold avoids amplifying flat-field noise, and the amount
 * is capped to discourage halo artifacts.
 */
export function unsharpMask(
  channel: Float32Array,
  width: number,
  height: number,
  params: SharpenParams
): Float32Array {
  if (params.amount <= 0) return channel;
  const radius = Math.max(0.5, params.radius);
  const blurred = boxBlur(channel, width, height, radius);
  const amount = clamp01(params.amount / 200) * 2; // 0..2
  const threshold = params.threshold / 255;

  const out = new Float32Array(channel.length);
  for (let i = 0; i < channel.length; i++) {
    const detail = channel[i] - blurred[i];
    const applied = Math.abs(detail) >= threshold ? detail * amount : 0;
    out[i] = clamp01(channel[i] + applied);
  }
  return out;
}
