import type { LoadedImage } from '../types/image';
import { sanitizeDisplayName } from '../utils/fileValidation';
import { ImageIcon } from './icons';

interface ImageInfoProps {
  image: LoadedImage;
}

function formatBytes(bytes: number): string {
  if (bytes <= 0) return 'Generated sample';
  const units = ['B', 'KB', 'MB', 'GB'];
  let v = bytes;
  let u = 0;
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024;
    u++;
  }
  return `${v.toFixed(v < 10 && u > 0 ? 1 : 0)} ${units[u]}`;
}

function detectedLabel(likelyType: LoadedImage['detected']['likelyType']): string {
  switch (likelyType) {
    case 'nir-grayscale':
      return 'Grayscale (NIR-like)';
    case 'thermal-grayscale':
      return 'Grayscale (thermal-like)';
    case 'false-color-thermal':
      return 'False-color (multi-channel)';
    default:
      return 'Unclassified';
  }
}

export default function ImageInfo({ image }: ImageInfoProps) {
  return (
    <section className="panel" aria-labelledby="image-info-heading">
      <h2 id="image-info-heading" className="panel__title">
        <ImageIcon size={15} /> Image Information
      </h2>
      <dl className="info-rows">
        <div className="info-rows__row">
          <dt>Filename</dt>
          <dd title={sanitizeDisplayName(image.fileName)}>{sanitizeDisplayName(image.fileName)}</dd>
        </div>
        <div className="info-rows__row">
          <dt>Dimensions</dt>
          <dd>
            {image.width} &times; {image.height}
          </dd>
        </div>
        <div className="info-rows__row">
          <dt>File Size</dt>
          <dd>{formatBytes(image.fileSize)}</dd>
        </div>
        <div className="info-rows__row">
          <dt>Image Type</dt>
          <dd>{detectedLabel(image.detected.likelyType)}</dd>
        </div>
        <div className="info-rows__row">
          <dt>Bit Depth</dt>
          <dd>8-bit</dd>
        </div>
        <div className="info-rows__row">
          <dt>Color Profile</dt>
          <dd>sRGB (assumed)</dd>
        </div>
      </dl>
      <p className="info-note">{image.detected.confidenceNote}</p>
    </section>
  );
}
