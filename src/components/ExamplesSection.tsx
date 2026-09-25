import { SAMPLE_PATTERNS, type SamplePatternId } from '../utils/samplePatterns';

interface ExamplesSectionProps {
  onSampleSelected: (id: SamplePatternId) => void;
}

export default function ExamplesSection({ onSampleSelected }: ExamplesSectionProps) {
  return (
    <section id="examples" className="info-section">
      <h2 className="info-section__title">Examples</h2>
      <p className="info-section__lead">
        IRecover ships with a handful of locally generated test patterns rather than downloaded
        photos, so there's always something to try IRecover on without needing a real IR camera
        or raising any copyright questions. Each demonstrates a different tonal characteristic
        (smooth gradients, high-contrast scenes, NIR-like brightness patterns, and noise).
      </p>
      <div className="examples-grid">
        {SAMPLE_PATTERNS.map((s) => (
          <button key={s.id} type="button" className="examples-grid__item" onClick={() => onSampleSelected(s.id)}>
            <span className="examples-grid__label">{s.label}</span>
            <span className="examples-grid__cta">Try this sample &rarr;</span>
          </button>
        ))}
      </div>
      <p className="info-section__note">
        Have your own infrared or near-infrared photo? Use the upload panel above -- PNG, JPEG,
        and WebP are all supported.
      </p>
    </section>
  );
}
