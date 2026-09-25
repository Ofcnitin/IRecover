import { useState } from 'react';
import type { ProcessingSettings, QualityMode } from '../types/processing';
import type { SceneType, InputInterpretation } from '../types/image';
import type { HistogramData } from '../types/image';
import { PRESET_LIST } from '../processing/presets';
import { SlidersIcon, ChevronRightIcon, ChevronDownIcon, HelpCircleIcon } from './icons';
import Histogram from './Histogram';

interface ControlsPanelProps {
  settings: ProcessingSettings;
  isCustomized: boolean;
  onChange: (partial: Partial<ProcessingSettings>) => void;
  onResetAll: () => void;
  inputHistogram: HistogramData | null;
  outputHistogram: HistogramData | null;
  levelsUsed: { black: number; white: number } | null;
}

interface SliderRowProps {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  onChange: (v: number) => void;
  disabled?: boolean;
}

function SliderRow({ id, label, value, min, max, step = 1, unit = '', onChange, disabled }: SliderRowProps) {
  return (
    <div className={`slider-row ${disabled ? 'slider-row--disabled' : ''}`}>
      <div className="slider-row__top">
        <label htmlFor={id}>{label}</label>
        <output htmlFor={id}>
          {Math.round(value * 100) / 100}
          {unit}
        </output>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

const IMAGE_TYPE_OPTIONS: { value: SceneType; label: string }[] = [
  { value: 'auto', label: 'Auto' },
  { value: 'nir-grayscale', label: 'NIR (Grayscale)' },
  { value: 'thermal-grayscale', label: 'Thermal' },
  { value: 'false-color-thermal', label: 'False Color' },
];

export default function ControlsPanel({
  settings,
  isCustomized,
  onChange,
  onResetAll,
  inputHistogram,
  outputHistogram,
  levelsUsed,
}: ControlsPanelProps) {
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [histogramOpen, setHistogramOpen] = useState(false);
  const [preciseLevels, setPreciseLevels] = useState(!settings.autoLevels);

  const set = <K extends keyof ProcessingSettings>(key: K, value: ProcessingSettings[K]) =>
    onChange({ [key]: value } as Partial<ProcessingSettings>);

  const currentPreset = PRESET_LIST.find((p) => p.id === settings.preset);

  return (
    <section className="panel settings-panel" aria-labelledby="settings-heading">
      <div className="panel__header-row">
        <h2 id="settings-heading" className="panel__title">
          <SlidersIcon size={15} /> Conversion Settings
        </h2>
        <button type="button" className="btn btn--ghost btn--small" onClick={onResetAll}>
          Reset All
        </button>
      </div>

      <div className="field">
        <span className="field__label-row">
          Image Type
          <HelpCircleIcon size={13} className="hint-icon" aria-hidden="true" />
        </span>
        <div className="segmented segmented--wrap" role="radiogroup" aria-label="Image type">
          {IMAGE_TYPE_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              type="button"
              role="radio"
              aria-checked={settings.sceneType === opt.value}
              className={`segmented__btn ${settings.sceneType === opt.value ? 'segmented__btn--active' : ''}`}
              onClick={() => set('sceneType', opt.value)}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <div className="field">
        <span className="field__label-row">
          Preset
          <HelpCircleIcon size={13} className="hint-icon" aria-hidden="true" />
        </span>
        <select
          id="preset-select"
          value={settings.preset}
          onChange={(e) => set('preset', e.target.value as ProcessingSettings['preset'])}
        >
          {PRESET_LIST.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
              {p.id === 'natural' ? ' (Recommended)' : ''}
            </option>
          ))}
        </select>
        {isCustomized && (
          <p className="field__hint field__hint--accent">
            Customized from {currentPreset?.label ?? 'preset'} &mdash; adjust sliders below or reset.
          </p>
        )}
      </div>

      <SliderRow id="exposure" label="Exposure" value={settings.exposure} min={-2} max={2} step={0.05} onChange={(v) => set('exposure', v)} />
      <SliderRow id="contrast" label="Contrast" value={settings.contrast} min={-100} max={100} onChange={(v) => set('contrast', v)} />
      <SliderRow id="gamma" label="Gamma" value={settings.gamma} min={0.2} max={3} step={0.02} onChange={(v) => set('gamma', v)} />
      <SliderRow id="saturation" label="Saturation" value={settings.saturation} min={-100} max={100} onChange={(v) => set('saturation', v)} />
      <SliderRow id="temperature" label="Color Temperature" value={settings.temperature} min={-100} max={100} onChange={(v) => set('temperature', v)} />

      <SliderRow
        id="black-pct"
        label="Black Point"
        value={settings.blackPercentile}
        min={0}
        max={20}
        step={0.5}
        unit="%"
        disabled={preciseLevels}
        onChange={(v) => set('blackPercentile', v)}
      />
      <SliderRow
        id="white-pct"
        label="White Point"
        value={settings.whitePercentile}
        min={80}
        max={100}
        step={0.5}
        unit="%"
        disabled={preciseLevels}
        onChange={(v) => set('whitePercentile', v)}
      />
      {preciseLevels && (
        <p className="field__hint">Precise pixel-value levels are active (see Advanced Settings).</p>
      )}

      <details
        className="collapsible"
        open={advancedOpen}
        onToggle={(e) => setAdvancedOpen((e.target as HTMLDetailsElement).open)}
      >
        <summary className="collapsible__summary">
          {advancedOpen ? <ChevronDownIcon size={15} /> : <ChevronRightIcon size={15} />}
          Advanced Settings
        </summary>
        <div className="collapsible__body">
          <div className="field">
            <label htmlFor="interpretation-select">Channel interpretation</label>
            <select
              id="interpretation-select"
              value={settings.interpretation}
              onChange={(e) => set('interpretation', e.target.value as InputInterpretation)}
            >
              <option value="grayscale-ir">Grayscale IR / NIR</option>
              <option value="false-color-ir">False-color IR (recover intensity)</option>
              <option value="native-rgb">Native RGB (treat as ordinary photo)</option>
            </select>
          </div>

          <SliderRow id="highlight-recovery" label="Highlight recovery" value={settings.highlightRecovery} min={0} max={100} onChange={(v) => set('highlightRecovery', v)} />
          <SliderRow id="shadow-lift" label="Shadow lift" value={settings.shadowLift} min={0} max={100} onChange={(v) => set('shadowLift', v)} />
          <SliderRow id="local-contrast" label="Local contrast" value={settings.localContrast} min={0} max={100} onChange={(v) => set('localContrast', v)} />
          <SliderRow id="color-strength" label="Color strength" value={settings.colorStrength} min={0} max={100} onChange={(v) => set('colorStrength', v)} />
          <SliderRow id="hue-bias" label="Hue bias" value={settings.hueBias} min={-180} max={180} unit="°" onChange={(v) => set('hueBias', v)} />
          <SliderRow id="detail-preservation" label="Detail preservation" value={settings.detailPreservation} min={0} max={100} onChange={(v) => set('detailPreservation', v)} />
          <SliderRow id="sharpen-amount" label="Sharpness" value={settings.sharpenAmount} min={0} max={200} unit="%" onChange={(v) => set('sharpenAmount', v)} />
          <SliderRow id="sharpen-radius" label="Sharpen radius" value={settings.sharpenRadius} min={0.3} max={5} step={0.1} unit=" px" onChange={(v) => set('sharpenRadius', v)} />
          <SliderRow id="sharpen-threshold" label="Sharpen threshold" value={settings.sharpenThreshold} min={0} max={40} onChange={(v) => set('sharpenThreshold', v)} />
          <SliderRow id="noise-reduction" label="Noise reduction" value={settings.noiseReduction} min={0} max={100} onChange={(v) => set('noiseReduction', v)} />

          <div className="field">
            <label htmlFor="noise-method">Noise reduction method</label>
            <select
              id="noise-method"
              value={settings.noiseMethod}
              onChange={(e) => set('noiseMethod', e.target.value as ProcessingSettings['noiseMethod'])}
            >
              <option value="gaussian">Gaussian</option>
              <option value="median">Median</option>
              <option value="bilateral">Bilateral (edge-preserving)</option>
            </select>
          </div>

          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={settings.sceneHeuristics}
              onChange={(e) => set('sceneHeuristics', e.target.checked)}
            />
            Scene-aware tinting (deterministic sky/vegetation heuristics)
          </label>

          <div className="control-divider" />

          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={preciseLevels}
              onChange={(e) => {
                const checked = e.target.checked;
                setPreciseLevels(checked);
                set('autoLevels', !checked);
              }}
            />
            Use precise pixel-value levels (0&ndash;255) instead of percentiles
          </label>
          {preciseLevels && (
            <>
              <SliderRow id="black-point-raw" label="Black point (raw)" value={settings.blackPoint} min={0} max={254} onChange={(v) => set('blackPoint', v)} />
              <SliderRow id="white-point-raw" label="White point (raw)" value={settings.whitePoint} min={1} max={255} onChange={(v) => set('whitePoint', v)} />
            </>
          )}

          <div className="control-divider" />

          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={settings.precisionPipeline}
              onChange={(e) => set('precisionPipeline', e.target.checked)}
            />
            Precision pipeline (white balance, tone &amp; color refinement)
          </label>
          {settings.precisionPipeline && (
            <>
              <SliderRow id="white-balance-strength" label="White balance" value={settings.whiteBalanceStrength} min={0} max={100} onChange={(v) => set('whiteBalanceStrength', v)} />
              <SliderRow id="auto-tone-strength" label="AutoTone" value={settings.autoToneStrength} min={0} max={100} onChange={(v) => set('autoToneStrength', v)} />
              <SliderRow id="color-correction-strength" label="Color correction" value={settings.colorCorrectionStrength} min={0} max={100} onChange={(v) => set('colorCorrectionStrength', v)} />

              <div className="field">
                <label htmlFor="precision-select">Precision</label>
                <select
                  id="precision-select"
                  value={settings.precision}
                  onChange={(e) => set('precision', e.target.value as ProcessingSettings['precision'])}
                >
                  <option value="standard">Standard</option>
                  <option value="high">High</option>
                  <option value="maximum">Maximum</option>
                </select>
              </div>

              <div className="field">
                <label htmlFor="engine-select" className="field__label-row">
                  Processing Engine
                  <HelpCircleIcon size={13} aria-label="Auto picks the fastest backend actually available in your browser; GPU backends fall back to CPU automatically if unavailable." />
                </label>
                <select
                  id="engine-select"
                  value={settings.processingEngine}
                  onChange={(e) => set('processingEngine', e.target.value as ProcessingSettings['processingEngine'])}
                >
                  <option value="auto">Auto</option>
                  <option value="cpu">CPU</option>
                  <option value="webgl2">WebGL2</option>
                  <option value="webgpu">WebGPU</option>
                </select>
              </div>
            </>
          )}
        </div>
      </details>

      <details
        className="collapsible"
        open={histogramOpen}
        onToggle={(e) => setHistogramOpen((e.target as HTMLDetailsElement).open)}
      >
        <summary className="collapsible__summary">
          {histogramOpen ? <ChevronDownIcon size={15} /> : <ChevronRightIcon size={15} />}
          Histogram
        </summary>
        <div className="collapsible__body collapsible__body--tight">
          <Histogram
            input={inputHistogram}
            output={outputHistogram}
            blackPoint={levelsUsed?.black ?? 0}
            whitePoint={levelsUsed?.white ?? 255}
            compact
          />
        </div>
      </details>

      <div className="field field--quality">
        <label htmlFor="quality-select" className="field__label-row">
          Processing Quality
        </label>
        <select
          id="quality-select"
          value={settings.quality}
          onChange={(e) => set('quality', e.target.value as QualityMode)}
        >
          <option value="fast">Fast</option>
          <option value="balanced">Balanced (Recommended)</option>
          <option value="high">High Quality</option>
          <option value="maximum">Maximum</option>
        </select>
      </div>
    </section>
  );
}
