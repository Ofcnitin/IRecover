const STEPS = [
  { title: 'Analyze', desc: 'Read the image and estimate basic channel statistics -- no object recognition, just numbers.' },
  { title: 'Normalize', desc: 'Stretch the useful intensity range using percentile-based black/white points.' },
  { title: 'Map', desc: 'Convert intensity into a restrained, photographic RGB ramp -- no rainbow thermal palette.' },
  { title: 'Tone', desc: 'Apply exposure, contrast, gamma, shadow lift, and highlight roll-off.' },
  { title: 'Refine', desc: 'Local contrast, noise reduction, and sharpening bring back detail and clarity.' },
  { title: 'Photo Finish (optional)', desc: 'Send the already-converted image to Cloudinary for restrained, controlled finishing.' },
];

export default function HowItWorksSection() {
  return (
    <section id="how-it-works" className="info-section">
      <h2 className="info-section__title">How It Works</h2>
      <p className="info-section__lead">
        Infrared cameras capture wavelengths outside, or partly outside, the range human eyes can
        see. IRecover estimates a plausible visible RGB appearance from the available image
        information using a deterministic, multi-stage pipeline -- there is no machine-learning
        model in the core conversion.
      </p>
      <ol className="how-it-works-steps">
        {STEPS.map((step, i) => (
          <li key={step.title} className="how-it-works-steps__item">
            <span className="how-it-works-steps__index">{i + 1}</span>
            <div>
              <div className="how-it-works-steps__title">{step.title}</div>
              <div className="how-it-works-steps__desc">{step.desc}</div>
            </div>
          </li>
        ))}
      </ol>
      <p className="info-section__note">
        The result is an approximation, not a reconstruction of the original visible-light
        photograph -- infrared imagery simply does not contain all of the information a visible
        photo would.
      </p>
    </section>
  );
}
