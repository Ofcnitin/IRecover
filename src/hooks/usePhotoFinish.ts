import { useCallback, useRef, useState } from 'react';
import { requestPhotoFinish } from '../services/cloudinary';
import { DEFAULT_PHOTO_FINISH_SETTINGS } from '../types/photoFinish';
import type { PhotoFinishSettings, PhotoFinishPreset, PhotoFinishCustomSettings } from '../types/photoFinish';

export type PhotoFinishStatus = 'idle' | 'processing' | 'success' | 'error';

export interface UsePhotoFinishApi {
  enabled: boolean;
  settings: PhotoFinishSettings;
  status: PhotoFinishStatus;
  resultDataUrl: string | null;
  errorMessage: string | null;
  setEnabled: (enabled: boolean) => void;
  setPreset: (preset: PhotoFinishPreset) => void;
  setCustom: (custom: Partial<PhotoFinishCustomSettings>) => void;
  /** Sends the given local RGB PNG data URL to the Photo Finish endpoint. */
  apply: (getImageDataUrl: () => Promise<string> | string) => Promise<void>;
  /** Discards any Photo Finish result/error and falls back to the local result. */
  useLocalInstead: () => void;
}

export function usePhotoFinish(): UsePhotoFinishApi {
  const [enabled, setEnabledState] = useState(false);
  const [settings, setSettings] = useState<PhotoFinishSettings>(DEFAULT_PHOTO_FINISH_SETTINGS);
  const [status, setStatus] = useState<PhotoFinishStatus>('idle');
  const [resultDataUrl, setResultDataUrl] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const setEnabled = useCallback((next: boolean) => {
    setEnabledState(next);
    if (!next) {
      abortRef.current?.abort();
      setStatus('idle');
      setResultDataUrl(null);
      setErrorMessage(null);
      setSettings((prev) => ({ ...prev, preset: 'off' }));
    } else {
      setSettings((prev) => (prev.preset === 'off' ? { ...prev, preset: 'natural' } : prev));
    }
  }, []);

  const setPreset = useCallback((preset: PhotoFinishPreset) => {
    setSettings((prev) => ({ ...prev, preset }));
    // Switching presets never auto-triggers a network call -- the user
    // must press Apply. Clear any stale result so it's not mistaken for
    // the result of the newly selected preset.
    setResultDataUrl(null);
    setStatus('idle');
    setErrorMessage(null);
  }, []);

  const setCustom = useCallback((custom: Partial<PhotoFinishCustomSettings>) => {
    setSettings((prev) => ({ ...prev, custom: { ...prev.custom, ...custom } }));
  }, []);

  const apply = useCallback(
    async (getImageDataUrl: () => Promise<string> | string) => {
      if (!enabled || settings.preset === 'off') return;
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setStatus('processing');
      setErrorMessage(null);

      try {
        const imageDataUrl = await getImageDataUrl();
        const result = await requestPhotoFinish({ imageDataUrl, settings, signal: controller.signal });
        if (result.status === 'success') {
          setResultDataUrl(result.imageDataUrl);
          setStatus('success');
        } else {
          setResultDataUrl(null);
          setStatus('error');
          setErrorMessage(result.message);
        }
      } catch (err) {
        setResultDataUrl(null);
        setStatus('error');
        setErrorMessage(err instanceof Error ? err.message : 'Photo Finish failed unexpectedly.');
      }
    },
    [enabled, settings]
  );

  const useLocalInstead = useCallback(() => {
    abortRef.current?.abort();
    setResultDataUrl(null);
    setStatus('idle');
    setErrorMessage(null);
  }, []);

  return { enabled, settings, status, resultDataUrl, errorMessage, setEnabled, setPreset, setCustom, apply, useLocalInstead };
}
