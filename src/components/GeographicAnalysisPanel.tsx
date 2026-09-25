import type { UseGeographicAnalysisApi } from '../hooks/useGeographicAnalysis';
import type { InputImageKind } from '../types/geographic';
import { MapPinIcon, CompassIcon, ChevronDownIcon, ChevronRightIcon, SpinnerIcon, AlertIcon } from './icons';

interface GeographicAnalysisPanelProps {
  api: UseGeographicAnalysisApi;
  disabled: boolean;
}

const INPUT_KIND_LABELS: Record<InputImageKind, string> = {
  rgb: 'Ordinary RGB',
  nir: 'Near-infrared (single band)',
  thermal: 'Thermal',
  'false-color': 'False color',
  multispectral: 'Multispectral (multiple real bands)',
  unknown: 'Unknown',
};

export default function GeographicAnalysisPanel({ api, disabled }: GeographicAnalysisPanelProps) {
  const { isActive, activate, inputKind, setInputKind, metadata, report, isAnalyzing, includeLocation, setIncludeLocation, runAnalysis } =
    api;

  const hasGps = !!metadata?.hasGps;

  return (
    <section className="panel geo-panel" aria-labelledby="geo-heading">
      <button
        type="button"
        className="panel__header-row geo-panel__toggle"
        onClick={() => (isActive ? undefined : activate())}
        aria-expanded={isActive}
        disabled={disabled}
      >
        <h2 id="geo-heading" className="panel__title">
          <CompassIcon size={15} /> Geographic Analysis
        </h2>
        {isActive ? <ChevronDownIcon size={16} /> : <ChevronRightIcon size={16} />}
      </button>

      {!isActive && (
        <p className="info-note">
          Optional: estimate land cover, vegetation, water, and terrain from this image. Not enabled by default.
        </p>
      )}

      {isActive && (
        <div className="geo-panel__body">
          <p className="info-note">
            Local measurements run entirely in your browser. Natural-language interpretation is optional, explicitly
            triggered, and uses IRecover&rsquo;s server endpoint &mdash; nothing is sent automatically.
          </p>

          <div className="field">
            <label htmlFor="geo-input-kind">Image type</label>
            <select
              id="geo-input-kind"
              value={inputKind}
              onChange={(e) => setInputKind(e.target.value as InputImageKind)}
              disabled={disabled}
            >
              {(Object.keys(INPUT_KIND_LABELS) as InputImageKind[]).map((kind) => (
                <option key={kind} value={kind}>
                  {INPUT_KIND_LABELS[kind]}
                </option>
              ))}
            </select>
          </div>

          <div className="geo-panel__gps-row">
            <MapPinIcon size={14} />
            {hasGps && metadata?.latitude !== null && metadata?.longitude !== null ? (
              <span>
                GPS detected: {metadata!.latitude!.toFixed(4)}, {metadata!.longitude!.toFixed(4)}
              </span>
            ) : (
              <span>GPS information not available from this file&rsquo;s metadata.</span>
            )}
          </div>

          {hasGps && (
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={includeLocation}
                onChange={(e) => setIncludeLocation(e.target.checked)}
                disabled={disabled}
              />
              <span>
                Look up approximate location context via OpenStreetMap (sends only the coordinates already in this
                file&rsquo;s metadata)
              </span>
            </label>
          )}

          <button
            type="button"
            className="btn btn--primary btn--block"
            onClick={() => void runAnalysis()}
            disabled={disabled || isAnalyzing}
          >
            {isAnalyzing ? (
              <>
                <SpinnerIcon size={15} /> Analyzing&hellip;
              </>
            ) : (
              <>Analyze Geography</>
            )}
          </button>

          {report && <GeographicReportView report={report} />}
        </div>
      )}
    </section>
  );
}

function GeographicReportView({ report }: { report: NonNullable<UseGeographicAnalysisApi['report']> }) {
  const { local, gemini, location, dataSent } = report;

  return (
    <div className="geo-report">
      {gemini.status === 'success' && (
        <div className="geo-report__section">
          <h3 className="geo-report__heading">Scene</h3>
          <p>{gemini.result.scene.description}</p>
          <ConfidenceTag value={gemini.result.scene.confidence} />
        </div>
      )}

      {gemini.status === 'unavailable' && (
        <p className="info-note info-note--muted">{gemini.reason}</p>
      )}

      {gemini.status === 'error' && (
        <div className="geo-report__status geo-report__status--error" role="alert">
          <AlertIcon size={14} />
          <span>{gemini.message}</span>
        </div>
      )}

      <div className="geo-report__section">
        <h3 className="geo-report__heading">Land Cover (estimated)</h3>
        <table className="geo-table">
          <thead>
            <tr>
              <th>Feature</th>
              <th>Estimate</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {local.landCover.map((f) => (
              <tr key={f.type}>
                <td>{f.label}</td>
                <td>~{f.estimatedCoveragePercent}%</td>
                <td className={`geo-confidence geo-confidence--${f.confidence}`}>{f.confidence}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="geo-report__grid">
        <FeatureCard title="Vegetation" description={local.vegetation.description} confidence={local.vegetation.confidence} />
        <FeatureCard title="Water" description={local.water.description} confidence={local.water.confidence} />
        <FeatureCard title="Terrain" description={local.terrain.description} confidence={local.terrain.confidence} />
        <FeatureCard title="Built-up" description={local.builtUp.description} confidence={local.builtUp.confidence} />
      </div>

      <p className="info-note info-note--muted">
        {local.vegetation.ndvi.available
          ? `NDVI (measured): mean ${local.vegetation.ndvi.mean}`
          : local.vegetation.ndvi.reason}
      </p>

      {location && location !== 'not-requested' && location !== 'unavailable' && (
        <div className="geo-report__section">
          <h3 className="geo-report__heading">Geographic Context</h3>
          <p>{location.displayName}</p>
          <p className="info-note info-note--muted">{location.attribution}</p>
        </div>
      )}
      {location === 'unavailable' && (
        <p className="info-note info-note--muted">Geographic location could not be determined from available metadata.</p>
      )}

      <div className="geo-report__section">
        <h3 className="geo-report__heading">Limitations</h3>
        <ul className="geo-limitations">
          {[...(gemini.status === 'success' ? gemini.result.limitations : []), ...local.limitations].map((l, i) => (
            <li key={i}>{l}</li>
          ))}
        </ul>
      </div>

      <p className="info-note info-note--muted">
        {dataSent === 'none' && 'Nothing left your browser for this analysis.'}
        {dataSent === 'gemini' &&
          'The converted image was sent to IRecover\u2019s secure server endpoint for interpretation via Gemini.'}
        {dataSent === 'gemini-and-location' &&
          'The converted image and this file\u2019s GPS coordinates were sent to IRecover\u2019s secure server endpoint for interpretation and location lookup.'}
      </p>
    </div>
  );
}

function FeatureCard({ title, description, confidence }: { title: string; description: string; confidence: string }) {
  return (
    <div className="geo-feature-card">
      <div className="geo-feature-card__top">
        <span className="geo-feature-card__title">{title}</span>
        <ConfidenceTag value={confidence as any} />
      </div>
      <p>{description}</p>
    </div>
  );
}

function ConfidenceTag({ value }: { value: 'high' | 'medium' | 'low' | number }) {
  if (typeof value === 'number') {
    const level = value > 0.75 ? 'high' : value > 0.4 ? 'medium' : 'low';
    return <span className={`geo-confidence geo-confidence--${level}`}>{Math.round(value * 100)}% confidence</span>;
  }
  return <span className={`geo-confidence geo-confidence--${value}`}>{value} confidence</span>;
}
