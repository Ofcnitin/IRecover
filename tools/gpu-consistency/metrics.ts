/**
 * Error metrics for comparing a CPU reference array against a GPU
 * (or any other) implementation's output. Pure functions, no DOM/GPU
 * dependency, so they're unit-testable on their own (see
 * tests/gpuConsistencyMetrics.test.ts) independently of whether a real
 * GPU is available to actually exercise the shaders.
 */

export interface ErrorMetrics {
  /** Mean absolute error, in the array's native units (e.g. 0..1 for linear/intensity, 0..255 for 8-bit RGBA). */
  mae: number;
  /** Maximum absolute error observed anywhere in the array. */
  maxError: number;
  /** Index of the maximum error, for debugging (pixel = floor(index / channels)). */
  maxErrorIndex: number;
  /** Fraction (0..1) of elements whose absolute error exceeds `tolerance`. */
  fractionOverTolerance: number;
  /** Count of elements whose absolute error exceeds `tolerance`. */
  countOverTolerance: number;
  /** Total elements compared. */
  count: number;
}

/**
 * Compares two equal-length numeric arrays element-by-element.
 * `tolerance` is the per-element absolute-error threshold used for
 * `fractionOverTolerance` -- pick it in the same units as the arrays
 * (e.g. ~1/255 for 8-bit-equivalent precision on a 0..1 buffer).
 */
export function compareArrays(
  reference: ArrayLike<number>,
  actual: ArrayLike<number>,
  tolerance: number
): ErrorMetrics {
  if (reference.length !== actual.length) {
    throw new Error(`compareArrays: length mismatch (${reference.length} vs ${actual.length}).`);
  }
  const n = reference.length;
  let sumAbs = 0;
  let maxError = 0;
  let maxErrorIndex = -1;
  let countOverTolerance = 0;

  for (let i = 0; i < n; i++) {
    const err = Math.abs(actual[i] - reference[i]);
    sumAbs += err;
    if (err > maxError) {
      maxError = err;
      maxErrorIndex = i;
    }
    if (err > tolerance) countOverTolerance++;
  }

  return {
    mae: n > 0 ? sumAbs / n : 0,
    maxError,
    maxErrorIndex,
    fractionOverTolerance: n > 0 ? countOverTolerance / n : 0,
    countOverTolerance,
    count: n,
  };
}

/** Per-channel breakdown for interleaved RGBA (or RGB) buffers, plus the combined metric across all channels. */
export interface ChannelErrorMetrics {
  combined: ErrorMetrics;
  perChannel: ErrorMetrics[]; // one entry per channel, in channel order
}

export function compareInterleavedChannels(
  reference: ArrayLike<number>,
  actual: ArrayLike<number>,
  channels: number,
  tolerance: number,
  channelsToCompare?: number // defaults to `channels` -- e.g. pass 3 to skip alpha on RGBA data
): ChannelErrorMetrics {
  const compareChannels = channelsToCompare ?? channels;
  const n = reference.length;
  if (n !== actual.length) {
    throw new Error(`compareInterleavedChannels: length mismatch (${n} vs ${actual.length}).`);
  }
  if (n % channels !== 0) {
    throw new Error(`compareInterleavedChannels: length ${n} is not a multiple of channels=${channels}.`);
  }

  const perChannelRef: number[][] = Array.from({ length: compareChannels }, () => []);
  const perChannelActual: number[][] = Array.from({ length: compareChannels }, () => []);
  for (let i = 0; i < n; i += channels) {
    for (let c = 0; c < compareChannels; c++) {
      perChannelRef[c].push(reference[i + c]);
      perChannelActual[c].push(actual[i + c]);
    }
  }

  const perChannel = perChannelRef.map((refCh, c) => compareArrays(refCh, perChannelActual[c], tolerance));

  // Combined: flatten only the compared channels (so e.g. alpha, if excluded, doesn't dilute the combined metric).
  const combinedRef: number[] = [];
  const combinedActual: number[] = [];
  for (let i = 0; i < n; i += channels) {
    for (let c = 0; c < compareChannels; c++) {
      combinedRef.push(reference[i + c]);
      combinedActual.push(actual[i + c]);
    }
  }

  return { combined: compareArrays(combinedRef, combinedActual, tolerance), perChannel };
}

/** Formats an ErrorMetrics for a one-line console report. */
export function formatMetrics(label: string, m: ErrorMetrics, tolerance: number, scale = 1): string {
  const pct = (m.fractionOverTolerance * 100).toFixed(3);
  return `${label}: MAE=${(m.mae * scale).toFixed(4)} max=${(m.maxError * scale).toFixed(4)} ` +
    `over-tol(${(tolerance * scale).toFixed(4)})=${pct}% (${m.countOverTolerance}/${m.count})`;
}
