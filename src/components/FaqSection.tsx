const FAQ_ITEMS: { q: string; a: string }[] = [
  { q: 'What is IRecover?', a: 'A browser-based tool that converts infrared / near-infrared images into a visible-light-style RGB approximation using deterministic image processing.' },
  { q: 'What is infrared imaging?', a: 'Photography using wavelengths of light longer than visible red light, captured by cameras sensitive to those wavelengths instead of (or in addition to) visible light.' },
  { q: 'What is NIR?', a: 'Near-infrared: light just beyond the visible spectrum (roughly 700-1100 nm). NIR still measures reflected light, unlike thermal imaging, which measures emitted heat.' },
  { q: 'Can IRecover recover the original colors?', a: 'No. IRecover approximates a plausible visible appearance from the available infrared information. It cannot reconstruct the true original visible-light colors, because that information was never captured.' },
  { q: 'Is the processing private?', a: 'By default, yes -- your image is processed entirely in your browser and never uploaded. If you explicitly enable Photo Finish, the already-converted image is sent to Cloudinary for final finishing.' },
  { q: 'Does IRecover use AI?', a: 'The core IR-to-RGB conversion uses deterministic image processing rather than a machine-learning model. Optional Photo Finish currently uses controlled image transformations rather than generative image reconstruction.' },
  { q: 'What is Photo Finish?', a: 'An optional final-stage enhancement that sends your already-converted RGB image to Cloudinary for restrained, photographic-style finishing (color balance, contrast, warmth). It never replaces the core local conversion.' },
  { q: 'Why does Photo Finish send an image externally?', a: 'Photo Finish uses Cloudinary\'s image transformation service, which runs on Cloudinary\'s servers. This only happens if you explicitly turn Photo Finish on.' },
  { q: 'What image formats are supported?', a: 'Upload: PNG, JPEG, and WebP. Export: PNG, JPEG, and WebP.' },
  { q: 'Where is my image processed?', a: 'In your browser, on your device, by default. Only the Photo Finish stage (when enabled) involves an external service.' },
];

export default function FaqSection() {
  return (
    <section id="faq" className="info-section">
      <h2 className="info-section__title">FAQ</h2>
      <div className="faq-list">
        {FAQ_ITEMS.map((item) => (
          <details key={item.q} className="faq-item">
            <summary className="faq-item__q">{item.q}</summary>
            <p className="faq-item__a">{item.a}</p>
          </details>
        ))}
      </div>
    </section>
  );
}
