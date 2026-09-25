import { useCallback, useMemo, useRef, useState } from 'react';
import { analyzeGeography } from '../processing/geographicAnalysis';
import { requestGeographicAnalysis } from '../services/geographicAnalysisApi';
import { downscaleImageData, imageDataToPngDataUrl, GEOGRAPHIC_ANALYSIS_MAX_DIMENSION } from '../utils/imageIO';
import type { LoadedImage } from '../types/image';
import type {
  GeminiAnalysisStatus,
  GeographicAnalysisReport,
  ImageGeographicMetadata,
  InputImageKind,
} from '../types/geographic';

const EMPTY_METADATA = (width: number, height: number, format: string): ImageGeographicMetadata => ({
  width,
  height,
  format,
  hasGps: false,
  latitude: null,
  longitude: null,
  cameraMake: null,
  cameraModel: null,
  captureDate: null,
  metadataAvailable: false,
});

function guessInputKind(image: LoadedImage): InputImageKind {
  switch (image.detected.likelyType) {
    case 'nir-grayscale':
      return 'nir';
    case 'thermal-grayscale':
      return 'thermal';
    case 'false-color-thermal':
      return 'false-color';
    default:
      return 'unknown';
  }
}

export interface UseGeographicAnalysisApi {
  /** Whether the panel has ever been expanded/triggered -- keeps local analysis lazy. */
  isActive: boolean;
  activate: () => void;
  inputKind: InputImageKind;
  setInputKind: (kind: InputImageKind) => void;
  metadata: ImageGeographicMetadata | null;
  report: GeographicAnalysisReport | null;
  isAnalyzing: boolean;
  includeLocation: boolean;
  setIncludeLocation: (v: boolean) => void;
  /** Explicitly triggers the (optional) Gemini + reverse-geocoding request. Never called automatically. */
  runAnalysis: () => Promise<void>;
  reset: () => void;
}

export function useGeographicAnalysis(
  image: LoadedImage | null,
  convertedRgb: ImageData | null
): UseGeographicAnalysisApi {
  const [isActive, setIsActive] = useState(false);
  const [inputKindOverride, setInputKindOverride] = useState<InputImageKind | null>(null);
  const [report, setReport] = useState<GeographicAnalysisReport | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [includeLocation, setIncludeLocation] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const inputKind = useMemo<InputImageKind>(
    () => inputKindOverride ?? (image ? guessInputKind(image) : 'unknown'),
    [inputKindOverride, image]
  );

  const metadata = useMemo<ImageGeographicMetadata | null>(() => {
    if (!image) return null;
    return image.geoMetadata ?? EMPTY_METADATA(image.width, image.height, image.mimeType);
  }, [image]);

  const activate = useCallback(() => setIsActive(true), []);

  const setInputKind = useCallback((kind: InputImageKind) => {
    setInputKindOverride(kind);
    // Changing the declared input type changes NDVI eligibility etc, so any
    // previous report (especially a Gemini result grounded in the old
    // local summary) is now stale.
    setReport(null);
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setReport(null);
    setIsAnalyzing(false);
  }, []);

  const runAnalysis = useCallback(async () => {
    if (!image || !convertedRgb || !metadata) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setIsAnalyzing(true);

    try {
      // Local, deterministic analysis always runs -- it never leaves the browser.
      // Real NDVI band data isn't available from a single-channel IR/false-color
      // capture, so ndviBands is intentionally omitted rather than fabricated.
      const local = analyzeGeography({ rgb: convertedRgb, inputType: inputKind, ndviBands: null });

      const gps =
        metadata.hasGps && metadata.latitude !== null && metadata.longitude !== null
          ? { latitude: metadata.latitude, longitude: metadata.longitude }
          : null;
      const requestLocation = includeLocation && gps !== null;

      let gemini: GeminiAnalysisStatus = { status: 'idle' };
      let location: GeographicAnalysisReport['location'] = gps ? 'not-requested' : 'unavailable';
      let dataSent: GeographicAnalysisReport['dataSent'] = 'none';

      // Sending anything to Gemini is itself optional and explicit --
      // this function only ever runs from a direct user "Analyze" action.
      const analysisImage = downscaleImageData(convertedRgb, GEOGRAPHIC_ANALYSIS_MAX_DIMENSION);
      const imageDataUrl = imageDataToPngDataUrl(analysisImage);

      const clientResult = await requestGeographicAnalysis({
        imageDataUrl,
        local,
        gps,
        requestLocation,
        signal: controller.signal,
      });

      if (!clientResult.ok) {
        gemini = { status: 'error', code: 'unknown', message: clientResult.errorMessage ?? 'Request failed.' };
      } else {
        const payload = clientResult.response;
        dataSent = requestLocation ? 'gemini-and-location' : 'gemini';
        if (payload?.gemini) {
          gemini = { status: 'success', result: payload.gemini };
        } else if (payload?.geminiUnavailableReason) {
          gemini = { status: 'unavailable', reason: payload.geminiUnavailableReason };
        } else if (!payload?.ok) {
          gemini = {
            status: 'error',
            code: payload?.code ?? 'unknown',
            message: payload?.error ?? 'Geographic interpretation failed.',
          };
        }
        if (requestLocation) {
          location = payload?.location ?? 'unavailable';
        }
      }

      setReport({ metadata, local, gemini, location, dataSent });
    } finally {
      setIsAnalyzing(false);
    }
  }, [image, convertedRgb, metadata, inputKind, includeLocation]);

  return {
    isActive,
    activate,
    inputKind,
    setInputKind,
    metadata,
    report,
    isAnalyzing,
    includeLocation,
    setIncludeLocation,
    runAnalysis,
    reset,
  };
}
