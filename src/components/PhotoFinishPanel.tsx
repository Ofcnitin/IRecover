import { PHOTO_FINISH_PRESET_LABELS, PHOTO_FINISH_PRESET_ORDER } from '../services/photoFinishTransforms';
import type { PhotoFinishSettings, PhotoFinishPreset } from '../types/photoFinish';
import type { PhotoFinishStatus } from '../hooks/usePhotoFinish';
import { SparkleIcon, SpinnerIcon, AlertIcon, CheckIcon } from './icons';

interface PhotoFinishPanelProps {
  enabled: boolean;
  settings: PhotoFinishSettings;
  status: PhotoFinishStatus;
  errorMessage: string | null;
  hasResult: boolean;
  onToggle: (enabled: boolean) => void;
  onPresetChange: (preset: PhotoFinishPreset) => void;
  onCustomChange: (partial: Partial<PhotoFinishSettings['custom']>) => void;
  onApply: () => void;
  onUseLocalInstead: () => void;
  disabled: boolean;
}

export default function PhotoFinishPanel({
  enabled,
  settings,
  status,
  errorMessage,
  hasResult,
  onToggle,
  onPresetChange,
  onCustomChange,
  onApply,
  onUseLocalInstead,
  disabled,
}: PhotoFinishPanelProps) {
  return (
    <section className="panel photo-finish-panel" aria-labelledby="photo-finish-heading">
      <div className="panel__header-row">
        <h2 id="photo-finish-heading" className="panel__title">
          <SparkleIcon size={15} /> Photo Finish
        </h2>
        <button
          type="button"
          role="switch"
          aria-checked={enabled}
          aria-label="Toggle Photo Finish"
          className={`toggle-switch ${enabled ? 'toggle-switch--on' : ''}`}
          onClick={() => onToggle(!enabled)}
          disabled={disabled}
        >
          <span className="toggle-switch__thumb" />
        </button>
      </div>

      <p className="info-note">
        {enabled
          ? 'Final finishing is processed through Cloudinary. The already-converted image (not your original) is sent.'
          : 'Everything stays in your browser.'}
      </p>

      {enabled && (
        <>
          <div className="field">
            <label htmlFor="photo-finish-preset">Finish style</label>
            <select
              id="photo-finish-preset"
              value={settings.preset === 'off' ? 'natural' : settings.preset}
              onChange={(e) => onPresetChange(e.target.value as PhotoFinishPreset)}
              disabled={disabled}
            >
              {PHOTO_FINISH_PRESET_ORDER.filter((p) => p !== 'off').map((p) => (
                <option key={p} value={p}>
                  {PHOTO_FINISH_PRESET_LABELS[p]}
                </option>
              ))}
            </select>
          </div>

          {settings.preset === 'custom' && (
            <div className="photo-finish-custom">
              <CustomSlider label="Brightness" value={settings.custom.brightness} min={-50} max={50} onChange={(v) => onCustomChange({ brightness: v })} />
              <CustomSlider label="Contrast" value={settings.custom.contrast} min={-50} max={50} onChange={(v) => onCustomChange({ contrast: v })} />
              <CustomSlider label="Saturation" value={settings.custom.saturation} min={-50} max={50} onChange={(v) => onCustomChange({ saturation: v })} />
              <CustomSlider label="Hue" value={settings.custom.hue} min={-30} max={30} unit="°" onChange={(v) => onCustomChange({ hue: v })} />
              <CustomSlider label="Sharpen" value={settings.custom.sharpen} min={0} max={100} onChange={(v) => onCustomChange({ sharpen: v })} />
            </div>
          )}

          <button
            type="button"
            className="btn btn--primary btn--block"
            onClick={onApply}
            disabled={disabled || status === 'processing'}
          >
            {status === 'processing' ? (
              <>
                <SpinnerIcon size={15} /> Processing&hellip;
              </>
            ) : (
              <>Apply Photo Finish</>
            )}
          </button>

          {status === 'success' && hasResult && (
            <p className="photo-finish-status photo-finish-status--success">
              <CheckIcon size={14} /> Photo Finish applied.
            </p>
          )}

          {status === 'error' && errorMessage && (
            <div className="photo-finish-status photo-finish-status--error" role="alert">
              <AlertIcon size={14} />
              <span>{errorMessage}</span>
            </div>
          )}

          {(status === 'error' || hasResult) && (
            <button type="button" className="link-btn" onClick={onUseLocalInstead}>
              Use Local Result Instead
            </button>
          )}
        </>
      )}
    </section>
  );
}

interface CustomSliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  unit?: string;
  onChange: (v: number) => void;
}

function CustomSlider({ label, value, min, max, unit = '', onChange }: CustomSliderProps) {
  const id = `pf-${label.toLowerCase()}`;
  return (
    <div className="slider-row">
      <div className="slider-row__top">
        <label htmlFor={id}>{label}</label>
        <output htmlFor={id}>
          {value}
          {unit}
        </output>
      </div>
      <input id={id} type="range" min={min} max={max} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  );
}
