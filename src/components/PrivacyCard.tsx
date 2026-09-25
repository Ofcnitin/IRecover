import { ShieldIcon } from './icons';
import type { PrivacyMode } from '../types/photoFinish';

interface PrivacyCardProps {
  privacyMode: PrivacyMode;
}

export default function PrivacyCard({ privacyMode }: PrivacyCardProps) {
  const isLocal = privacyMode === 'local';
  return (
    <section className="panel privacy-card" aria-labelledby="privacy-card-heading">
      <h2 id="privacy-card-heading" className="panel__title">
        <ShieldIcon size={15} /> Your Privacy Matters
      </h2>
      <p className="info-note">
        Your original image is processed locally in your browser by default. Nothing is uploaded
        unless you explicitly turn on Photo Finish.
      </p>
      <p className="info-note">
        {isLocal
          ? 'Cloud processing is currently off -- everything stays on this device.'
          : 'Photo Finish is on: the already-converted result (not your original) is sent to Cloudinary for final finishing.'}
      </p>
    </section>
  );
}
