import { PRESET_LIST } from '../processing/presets';
import type { PresetId } from '../types/processing';
import { GridIcon, CheckIcon } from './icons';

interface PresetSelectorProps {
  value: PresetId;
  onChange: (id: PresetId) => void;
}

export default function PresetSelector({ value, onChange }: PresetSelectorProps) {
  return (
    <section className="panel" aria-labelledby="preset-strip-heading">
      <div className="panel__header-row">
        <h2 id="preset-strip-heading" className="panel__title">
          <GridIcon size={15} /> Conversion Presets
        </h2>
      </div>
      <div className="preset-strip" role="radiogroup" aria-labelledby="preset-strip-heading">
        {PRESET_LIST.map((preset) => (
          <button
            key={preset.id}
            type="button"
            role="radio"
            aria-checked={value === preset.id}
            className={`preset-tile ${value === preset.id ? 'preset-tile--active' : ''}`}
            onClick={() => onChange(preset.id)}
          >
            <span
              className="preset-tile__swatch"
              aria-hidden="true"
              style={{
                background: `linear-gradient(135deg, ${preset.colorStops.map((s) => rgbToCss(s.rgb)).join(', ')})`,
              }}
            >
              {value === preset.id && (
                <span className="preset-tile__check">
                  <CheckIcon size={13} />
                </span>
              )}
            </span>
            <span className="preset-tile__label">{preset.label}</span>
          </button>
        ))}
      </div>
    </section>
  );
}

function rgbToCss([r, g, b]: [number, number, number]): string {
  return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
}
