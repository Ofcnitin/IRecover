import { useMemo } from 'react';
import type { HistogramData } from '../types/image';

interface HistogramProps {
  input: HistogramData | null;
  output: HistogramData | null;
  blackPoint: number;
  whitePoint: number;
  /** When true, renders without its own panel wrapper/heading (used inside a collapsible section). */
  compact?: boolean;
}

const WIDTH = 256;
const HEIGHT = 96;

export default function Histogram({ input, output, blackPoint, whitePoint, compact }: HistogramProps) {
  const inputPath = useMemo(() => (input ? buildPath(input.luminance) : ''), [input]);
  const outputPath = useMemo(() => (output ? buildPath(output.luminance) : ''), [output]);

  const body = !input ? (
    <p className="info-note">Load an image to see its intensity distribution.</p>
  ) : (
    <>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="histogram-svg"
        role="img"
        aria-label="Histogram of image intensity: input in gray, output in blue"
        preserveAspectRatio="none"
      >
        <rect
          x={(blackPoint / 255) * WIDTH}
          width={Math.max(0, ((whitePoint - blackPoint) / 255) * WIDTH)}
          y={0}
          height={HEIGHT}
          className="histogram-svg__range"
        />
        <path d={inputPath} className="histogram-svg__input" />
        {output && <path d={outputPath} className="histogram-svg__output" />}
      </svg>
      <div className="histogram-legend">
        <span>
          <i className="legend-dot legend-dot--input" /> Input
        </span>
        {output && (
          <span>
            <i className="legend-dot legend-dot--output" /> Output
          </span>
        )}
        <span className="histogram-legend__stats">
          min {input.min} &middot; max {input.max} &middot; median {input.median}
        </span>
      </div>
    </>
  );

  if (compact) {
    return <div className="histogram-compact">{body}</div>;
  }

  return (
    <section className="panel" aria-labelledby="histogram-heading">
      <h2 id="histogram-heading" className="panel__title">
        Histogram
      </h2>
      {body}
    </section>
  );
}

function buildPath(buckets: number[]): string {
  if (buckets.length === 0) return '';
  const max = Math.max(1, ...buckets);
  // Use a square-root scale so low-count tails remain visible.
  const scaled = buckets.map((v) => Math.sqrt(v / max));
  let d = `M0 ${HEIGHT}`;
  for (let i = 0; i < scaled.length; i++) {
    const x = (i / (scaled.length - 1)) * WIDTH;
    const y = HEIGHT - scaled[i] * HEIGHT;
    d += ` L${x.toFixed(1)} ${y.toFixed(1)}`;
  }
  d += ` L${WIDTH} ${HEIGHT} Z`;
  return d;
}
