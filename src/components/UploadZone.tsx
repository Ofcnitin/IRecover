import { useCallback, useMemo, useRef, useState } from 'react';
import { validateImageFile } from '../utils/fileValidation';
import { SAMPLE_PATTERNS, generateSamplePattern, type SamplePatternId } from '../utils/samplePatterns';
import { UploadCloudIcon } from './icons';

interface UploadZoneProps {
  onFileSelected: (file: File) => void;
  onSampleSelected: (id: SamplePatternId) => void;
  errorMessage: string | null;
}

export default function UploadZone({ onFileSelected, onSampleSelected, errorMessage }: UploadZoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const thumbnails = useMemo(() => {
    const map: Record<string, string> = {};
    for (const s of SAMPLE_PATTERNS) {
      try {
        const loaded = generateSamplePattern(s.id, 96, 72);
        const canvas = document.createElement('canvas');
        canvas.width = loaded.width;
        canvas.height = loaded.height;
        canvas.getContext('2d')?.putImageData(loaded.data, 0, 0);
        map[s.id] = canvas.toDataURL('image/png');
      } catch {
        map[s.id] = '';
      }
    }
    return map;
  }, []);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0];
      if (!file) return;
      onFileSelected(file);
    },
    [onFileSelected]
  );

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragOver(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  const onPaste = useCallback(
    (e: React.ClipboardEvent<HTMLDivElement>) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          const file = item.getAsFile();
          if (file) {
            handleFiles(makeFileList(file));
            e.preventDefault();
            return;
          }
        }
      }
    },
    [handleFiles]
  );

  return (
    <section className="panel upload-panel" aria-labelledby="upload-heading">
      <h2 id="upload-heading" className="panel__title">
        <UploadCloudIcon size={15} /> Upload IR Image
      </h2>

      <div
        className={`upload-zone ${isDragOver ? 'upload-zone--drag' : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={onDrop}
        onPaste={onPaste}
        tabIndex={0}
        role="button"
        aria-label="Upload an infrared image by dropping it here, pasting, or choosing a file"
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click();
        }}
        onClick={() => inputRef.current?.click()}
      >
        <div className="upload-zone__icon" aria-hidden="true">
          <UploadCloudIcon size={30} />
        </div>
        <p className="upload-zone__title">Drop an IR image here</p>
        <p className="upload-zone__subtitle">or click to browse</p>
        <button
          type="button"
          className="btn btn--primary btn--block"
          onClick={(e) => {
            e.stopPropagation();
            inputRef.current?.click();
          }}
        >
          <UploadCloudIcon size={15} /> Choose Image
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          className="visually-hidden"
          onChange={(e) => handleFiles(e.target.files)}
          aria-label="Choose an image file"
        />
        <p className="upload-zone__formats">Supports: JPG, PNG, WebP</p>
        <p className="upload-zone__formats upload-zone__formats--muted">Max size: 60 MB</p>
      </div>

      {errorMessage && (
        <p className="upload-zone__error" role="alert">
          {errorMessage}
        </p>
      )}

      <div className="sample-section">
        <span className="sample-section__label">Or try a sample</span>
        <div className="sample-grid">
          {SAMPLE_PATTERNS.map((s) => (
            <button
              key={s.id}
              type="button"
              className="sample-tile"
              onClick={() => onSampleSelected(s.id)}
              title={`Try the ${s.label} test pattern`}
            >
              {thumbnails[s.id] ? (
                <img src={thumbnails[s.id]} alt="" className="sample-tile__img" />
              ) : (
                <span className="sample-tile__img sample-tile__img--fallback" />
              )}
              <span className="sample-tile__label">{s.label}</span>
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}

function makeFileList(file: File): FileList {
  const dt = new DataTransfer();
  dt.items.add(file);
  return dt.files;
}
