import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Header from './components/Header';
import Disclaimer from './components/Disclaimer';
import UploadZone from './components/UploadZone';
import ImageInfo from './components/ImageInfo';
import PrivacyCard from './components/PrivacyCard';
import PresetSelector from './components/PresetSelector';
import ControlsPanel from './components/ControlsPanel';
import ImageWorkspace from './components/ImageWorkspace';
import ExportPanel from './components/ExportPanel';
import PhotoFinishPanel from './components/PhotoFinishPanel';
import GeographicAnalysisPanel from './components/GeographicAnalysisPanel';
import HowItWorksSection from './components/HowItWorksSection';
import ExamplesSection from './components/ExamplesSection';
import AboutSection from './components/AboutSection';
import FaqSection from './components/FaqSection';
import { useImageProcessor } from './hooks/useImageProcessor';
import { usePhotoFinish } from './hooks/usePhotoFinish';
import { useGeographicAnalysis } from './hooks/useGeographicAnalysis';
import { PRESETS } from './processing/presets';
import { DEFAULT_SETTINGS, type ProcessingSettings } from './types/processing';
import type { LoadedImage } from './types/image';
import {
  loadImageFile,
  downscaleImageData,
  exportImageData,
  exportDataUrl,
  imageDataToPngDataUrl,
  buildOutputFileName,
  PREVIEW_MAX_DIMENSION,
  PHOTO_FINISH_MAX_DIMENSION,
  MAX_SAFE_DIMENSION,
  type ExportFormat,
} from './utils/imageIO';
import { validateImageFile } from './utils/fileValidation';
import { generateSamplePattern, type SamplePatternId } from './utils/samplePatterns';
import type { PrivacyMode } from './types/photoFinish';
import './App.css';

const CUSTOMIZATION_KEYS: (keyof ProcessingSettings)[] = [
  'exposure',
  'contrast',
  'gamma',
  'saturation',
  'temperature',
  'blackPercentile',
  'whitePercentile',
  'highlightRecovery',
  'shadowLift',
  'localContrast',
  'colorStrength',
  'hueBias',
  'detailPreservation',
  'sharpenAmount',
  'sharpenRadius',
  'sharpenThreshold',
  'noiseReduction',
  'sceneHeuristics',
];

function computeIsCustomized(settings: ProcessingSettings): boolean {
  const base = { ...DEFAULT_SETTINGS, ...PRESETS[settings.preset].overrides };
  return CUSTOMIZATION_KEYS.some((key) => settings[key] !== base[key]);
}

function getTheme(): 'light' | 'dark' {
  if (typeof window === 'undefined') return 'light';
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export default function App() {
  const [theme, setTheme] = useState<'light' | 'dark'>(getTheme);
  const [image, setImage] = useState<LoadedImage | null>(null);
  const [previewSource, setPreviewSource] = useState<ImageData | null>(null);
  const [settings, setSettings] = useState<ProcessingSettings>(DEFAULT_SETTINGS);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [isFullResBusy, setIsFullResBusy] = useState(false);
  const [largeImageNotice, setLargeImageNotice] = useState<string | null>(null);

  const { result, isProcessing, error: processingError, process, processImmediate } = useImageProcessor();
  const photoFinish = usePhotoFinish();
  const geographicAnalysis = useGeographicAnalysis(image, result?.output ?? null);

  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const applyImage = useCallback((loaded: LoadedImage) => {
    setImage(loaded);
    setUploadError(null);
    photoFinish.useLocalInstead();
    geographicAnalysis.reset();

    const longest = Math.max(loaded.width, loaded.height);
    if (longest > MAX_SAFE_DIMENSION) {
      setLargeImageNotice(
        `This image is very large (${loaded.width}\u00d7${loaded.height}). An optimized version will be used for preview and export to keep the browser responsive.`
      );
    } else if (longest > PREVIEW_MAX_DIMENSION) {
      setLargeImageNotice(
        `This image (${loaded.width}\u00d7${loaded.height}) is being previewed at a reduced size for smooth editing. Downloads are still rendered at full resolution.`
      );
    } else {
      setLargeImageNotice(null);
    }

    const preview = downscaleImageData(loaded.data, PREVIEW_MAX_DIMENSION);
    setPreviewSource(preview);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onFileSelected = useCallback(
    async (file: File) => {
      const validation = validateImageFile(file);
      if (!validation.ok) {
        setUploadError(validation.error ?? 'This file could not be used.');
        return;
      }
      try {
        const loaded = await loadImageFile(file);
        applyImage(loaded);
      } catch (err) {
        setUploadError(err instanceof Error ? err.message : 'Failed to read this image.');
      }
    },
    [applyImage]
  );

  const onSampleSelected = useCallback(
    (id: SamplePatternId) => {
      const loaded = generateSamplePattern(id);
      applyImage(loaded);
    },
    [applyImage]
  );

  useEffect(() => {
    if (previewSource) {
      process(previewSource, settings);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewSource, settings]);

  const updateSettings = useCallback(
    (partial: Partial<ProcessingSettings>) => {
      setSettings((prev) => ({ ...prev, ...partial }));
      photoFinish.useLocalInstead();
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  const onPresetChange = useCallback(
    (presetId: ProcessingSettings['preset']) => {
      const preset = PRESETS[presetId];
      setSettings((prev) => ({ ...prev, preset: presetId, ...preset.overrides }));
      photoFinish.useLocalInstead();
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  const resetAll = useCallback(() => {
    setSettings((prev) => ({ ...DEFAULT_SETTINGS, preset: prev.preset, ...PRESETS[prev.preset].overrides }));
    photoFinish.useLocalInstead();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const exportConverted = useCallback(
    async (format: ExportFormat, quality: number) => {
      if (!image) return;
      setIsFullResBusy(true);
      try {
        const source = downscaleImageData(image.data, MAX_SAFE_DIMENSION);
        const fullResult = await processImmediate(source, settingsRef.current);
        const fileName = buildOutputFileName(image.fileName, 'visible', format === 'jpeg' ? 'jpg' : format);
        await exportImageData(fullResult.output, format, quality, fileName);
      } finally {
        setIsFullResBusy(false);
      }
    },
    [image, processImmediate]
  );

  const exportOriginal = useCallback(
    async (format: ExportFormat, quality: number) => {
      if (!image) return;
      const fileName = buildOutputFileName(image.fileName, 'original', format === 'jpeg' ? 'jpg' : format);
      await exportImageData(image.data, format, quality, fileName);
    },
    [image]
  );

  const exportPhotoFinishResult = useCallback(
    async (format: ExportFormat, quality: number) => {
      if (!image || !photoFinish.resultDataUrl) return;
      const fileName = buildOutputFileName(image.fileName, 'photo-finish', format === 'jpeg' ? 'jpg' : format);
      await exportDataUrl(photoFinish.resultDataUrl, format, quality, fileName);
    },
    [image, photoFinish.resultDataUrl]
  );

  const preparePhotoFinishInput = useCallback(async (): Promise<string> => {
    if (!image) throw new Error('No image loaded.');
    const source = downscaleImageData(image.data, PHOTO_FINISH_MAX_DIMENSION);
    const localResult = await processImmediate(source, settingsRef.current);
    return imageDataToPngDataUrl(localResult.output);
  }, [image, processImmediate]);

  const onApplyPhotoFinish = useCallback(() => {
    void photoFinish.apply(preparePhotoFinishInput);
  }, [photoFinish, preparePhotoFinishInput]);

  const privacyMode: PrivacyMode = photoFinish.enabled && photoFinish.settings.preset !== 'off' ? 'cloud' : 'local';

  const workspaceOriginal = previewSource;
  const workspaceConverted = result?.output ?? null;
  const isCustomized = useMemo(() => computeIsCustomized(settings), [settings]);
  const currentPresetLabel = PRESETS[settings.preset].label;

  const errorToShow = uploadError ?? processingError;

  return (
    <div className="app-shell">
      <Header theme={theme} onToggleTheme={() => setTheme((t) => (t === 'light' ? 'dark' : 'light'))} privacyMode={privacyMode} />
      <Disclaimer />

      {errorToShow && (
        <div className="global-error" role="alert">
          {errorToShow}
        </div>
      )}

      <main id="convert">
        {!image ? (
          <div className="landing-upload">
            <UploadZone onFileSelected={onFileSelected} onSampleSelected={onSampleSelected} errorMessage={uploadError} />
          </div>
        ) : (
          <div className="workbench">
            <aside className="workbench__left">
              <UploadZone onFileSelected={onFileSelected} onSampleSelected={onSampleSelected} errorMessage={uploadError} />
              <ImageInfo image={image} />
              <PrivacyCard privacyMode={privacyMode} />
            </aside>

            <div className="workbench__center">
              <ImageWorkspace
                original={workspaceOriginal}
                converted={workspaceConverted}
                isProcessing={isProcessing}
                presetLabel={currentPresetLabel}
              />

              {largeImageNotice && (
                <div className="notice" role="status">
                  <span>{largeImageNotice}</span>
                </div>
              )}

              <PresetSelector value={settings.preset} onChange={onPresetChange} />
            </div>

            <aside className="workbench__right">
              <ControlsPanel
                settings={settings}
                isCustomized={isCustomized}
                onChange={updateSettings}
                onResetAll={resetAll}
                inputHistogram={result?.inputHistogram ?? null}
                outputHistogram={result?.outputHistogram ?? null}
                levelsUsed={result?.levelsUsed ?? null}
              />

              <PhotoFinishPanel
                enabled={photoFinish.enabled}
                settings={photoFinish.settings}
                status={photoFinish.status}
                errorMessage={photoFinish.errorMessage}
                hasResult={!!photoFinish.resultDataUrl}
                onToggle={photoFinish.setEnabled}
                onPresetChange={photoFinish.setPreset}
                onCustomChange={photoFinish.setCustom}
                onApply={onApplyPhotoFinish}
                onUseLocalInstead={photoFinish.useLocalInstead}
                disabled={!image}
              />

              <GeographicAnalysisPanel api={geographicAnalysis} disabled={!image || !result} />

              <ExportPanel
                onExportConverted={exportConverted}
                onExportOriginal={exportOriginal}
                onExportPhotoFinish={exportPhotoFinishResult}
                hasPhotoFinishResult={!!photoFinish.resultDataUrl}
                isFullResBusy={isFullResBusy}
                disabled={!image}
              />
            </aside>
          </div>
        )}
      </main>

      <ExamplesSection onSampleSelected={onSampleSelected} />
      <HowItWorksSection />
      <AboutSection />
      <FaqSection />

      <footer className="app-footer">
        <div className="app-footer__brand">
          <span className="brand-mark brand-mark--small" aria-hidden="true">
            <span className="brand-mark__dot brand-mark__dot--a" />
            <span className="brand-mark__dot brand-mark__dot--b" />
          </span>
          <div>
            <div className="app-footer__name">IRecover</div>
            <div className="app-footer__tagline">A clearer view of a different world.</div>
          </div>
        </div>
        <div className="app-footer__links">
          <span>Privacy</span>
          <span>Open Source</span>
          <span>Contact</span>
        </div>
      </footer>
    </div>
  );
}
