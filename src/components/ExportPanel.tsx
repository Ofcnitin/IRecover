import { useState } from 'react';
import type { ExportFormat } from '../utils/imageIO';
import { DownloadIcon, SpinnerIcon } from './icons';

interface ExportPanelProps {
  onExportConverted: (format: ExportFormat, quality: number) => Promise<void>;
  onExportOriginal: (format: ExportFormat, quality: number) => Promise<void>;
  onExportPhotoFinish?: (format: ExportFormat, quality: number) => Promise<void>;
  hasPhotoFinishResult: boolean;
  isFullResBusy: boolean;
  disabled: boolean;
}

const FORMATS: { id: ExportFormat; label: string }[] = [
  { id: 'png', label: 'PNG' },
  { id: 'jpeg', label: 'JPEG' },
  { id: 'webp', label: 'WebP' },
];

export default function ExportPanel({
  onExportConverted,
  onExportOriginal,
  onExportPhotoFinish,
  hasPhotoFinishResult,
  isFullResBusy,
  disabled,
}: ExportPanelProps) {
  const [format, setFormat] = useState<ExportFormat>('png');
  const [quality] = useState(0.92);
  const [busy, setBusy] = useState<'converted' | 'original' | 'photo-finish' | null>(null);

  const runExport = async (which: 'converted' | 'original' | 'photo-finish') => {
    setBusy(which);
    try {
      if (which === 'converted') await onExportConverted(format, quality);
      else if (which === 'original') await onExportOriginal(format, quality);
      else if (onExportPhotoFinish) await onExportPhotoFinish(format, quality);
    } finally {
      setBusy(null);
    }
  };

  const primaryLabel = isFullResBusy || busy === 'converted' ? 'Rendering full resolution\u2026' : 'Download Image';

  return (
    <section className="panel export-panel" aria-labelledby="export-heading">
      <h2 id="export-heading" className="visually-hidden">
        Export
      </h2>

      {hasPhotoFinishResult ? (
        <div className="export-choice-row">
          <button
            type="button"
            className="btn btn--primary btn--block"
            disabled={disabled || busy !== null}
            onClick={() => runExport('photo-finish')}
          >
            {busy === 'photo-finish' ? <SpinnerIcon size={15} /> : <DownloadIcon size={15} />} Download Final
          </button>
          <button
            type="button"
            className="btn btn--secondary btn--block"
            disabled={disabled || busy !== null}
            onClick={() => runExport('converted')}
          >
            {busy === 'converted' ? <SpinnerIcon size={15} /> : <DownloadIcon size={15} />} Download IRecover RGB
          </button>
        </div>
      ) : (
        <button
          type="button"
          className="btn btn--primary btn--block download-btn"
          disabled={disabled || busy !== null}
          onClick={() => runExport('converted')}
        >
          {busy === 'converted' || isFullResBusy ? <SpinnerIcon size={15} /> : <DownloadIcon size={15} />}
          {primaryLabel}
        </button>
      )}

      <div className="format-chips" role="radiogroup" aria-label="Export format">
        {FORMATS.map((f) => (
          <button
            key={f.id}
            type="button"
            role="radio"
            aria-checked={format === f.id}
            className={`format-chip ${format === f.id ? 'format-chip--active' : ''}`}
            onClick={() => setFormat(f.id)}
          >
            {f.label}
          </button>
        ))}
      </div>

      <button
        type="button"
        className="link-btn"
        disabled={disabled || busy !== null}
        onClick={() => runExport('original')}
      >
        {busy === 'original' ? 'Preparing\u2026' : 'Download original IR image'}
      </button>
    </section>
  );
}
