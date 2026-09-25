export default function AboutSection() {
  return (
    <section id="about" className="info-section">
      <h2 className="info-section__title">About IRecover</h2>
      <p className="info-section__lead">
        IRecover converts infrared and near-infrared imagery into a natural-looking visible-light
        RGB approximation. The core conversion is classical, deterministic image processing --
        color-space math, tone curves, local contrast, and noise/sharpening filters -- not a
        machine-learning model.
      </p>
      <p>
        By default, everything runs locally in your browser: your original image is never
        uploaded anywhere. An optional Photo Finish stage can send the already-converted image to
        Cloudinary for additional, restrained photographic finishing -- but only when you turn it
        on.
      </p>
      <p className="info-section__note">
        Visible colors are approximated from infrared data and may differ from the original
        scene.
      </p>
    </section>
  );
}
