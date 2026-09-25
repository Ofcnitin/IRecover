import { useEffect, useRef, useState, useCallback } from 'react';
import type { ProcessingSettings } from '../types/processing';
import type { HistogramData } from '../types/image';
import { debounce } from '../utils/debounce';
import type { PrecisionDiagnostics } from '../processing/pipeline';
import type { ExecutionReport } from '../processing/executionReport';

export interface ProcessResult {
  output: ImageData;
  inputHistogram: HistogramData;
  outputHistogram: HistogramData;
  levelsUsed: { black: number; white: number };
  precision?: PrecisionDiagnostics;
  /** What actually produced the pixels (webgpu | webgl2 | cpu) + fallback audit trail. */
  execution?: ExecutionReport;
  durationMs: number;
}

interface UseImageProcessorApi {
  result: ProcessResult | null;
  isProcessing: boolean;
  error: string | null;
  /** Request processing of `source` with `settings`. Debounced automatically. */
  process: (source: ImageData, settings: ProcessingSettings) => void;
  /** Process immediately, bypassing debounce (used for full-resolution export). */
  processImmediate: (source: ImageData, settings: ProcessingSettings) => Promise<ProcessResult>;
}

/**
 * Wraps the pixel-processing Web Worker so the main thread / UI never
 * blocks on the actual image math. Requests are debounced while the user is
 * actively dragging a slider, and stale responses (superseded by a newer
 * request) are discarded.
 */
export function useImageProcessor(debounceMs = 60): UseImageProcessorApi {
  const workerRef = useRef<Worker | null>(null);
  const requestIdRef = useRef(0);
  const latestHandledId = useRef(0);
  const pendingResolvers = useRef(new Map<number, (r: ProcessResult) => void>());
  const silentRequestIds = useRef(new Set<number>());

  const [result, setResult] = useState<ProcessResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const worker = new Worker(new URL('../workers/imageProcessor.worker.ts', import.meta.url), {
      type: 'module',
    });
    workerRef.current = worker;

    worker.onmessage = (ev: MessageEvent<any>) => {
      const msg = ev.data;
      setIsProcessing(false);

      if (msg.error) {
        setError(msg.error);
        return;
      }

      const output = new ImageData(new Uint8ClampedArray(msg.buffer), msg.width, msg.height);
      const processed: ProcessResult = {
        output,
        inputHistogram: msg.inputHistogram,
        outputHistogram: msg.outputHistogram,
        levelsUsed: msg.levelsUsed,
        precision: msg.precision,
        execution: msg.execution,
        durationMs: msg.durationMs,
      };

      const resolver = pendingResolvers.current.get(msg.requestId);
      if (resolver) {
        resolver(processed);
        pendingResolvers.current.delete(msg.requestId);
      }

      const wasSilent = silentRequestIds.current.delete(msg.requestId);
      if (!wasSilent && msg.requestId >= latestHandledId.current) {
        latestHandledId.current = msg.requestId;
        setError(null);
        setResult(processed);
      }
    };

    worker.onerror = (ev) => {
      setIsProcessing(false);
      setError(ev.message || 'The image processor encountered an unexpected error.');
    };

    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  const send = useCallback((source: ImageData, settings: ProcessingSettings, silent = false): number => {
    const id = ++requestIdRef.current;
    const worker = workerRef.current;
    if (silent) silentRequestIds.current.add(id);
    if (!worker) return id;
    setIsProcessing(true);
    // Copy the buffer since ImageData from React state may be reused/read elsewhere.
    const bufferCopy = source.data.buffer.slice(0);
    worker.postMessage(
      {
        requestId: id,
        width: source.width,
        height: source.height,
        buffer: bufferCopy,
        settings,
      },
      [bufferCopy]
    );
    return id;
  }, []);

  const debouncedSend = useRef(
    debounce((source: ImageData, settings: ProcessingSettings) => {
      send(source, settings);
    }, debounceMs)
  );

  // Keep the debounced function pointing at the latest `send` (it captures
  // workerRef via closure, which is stable, so this is mostly future-proofing).
  useEffect(() => {
    debouncedSend.current = debounce((source: ImageData, settings: ProcessingSettings) => {
      send(source, settings);
    }, debounceMs);
  }, [send, debounceMs]);

  const process = useCallback(
    (source: ImageData, settings: ProcessingSettings) => {
      debouncedSend.current(source, settings);
    },
    []
  );

  const processImmediate = useCallback(
    (source: ImageData, settings: ProcessingSettings): Promise<ProcessResult> => {
      return new Promise((resolve, reject) => {
        const id = send(source, settings, true);
        pendingResolvers.current.set(id, resolve);
        setTimeout(() => {
          if (pendingResolvers.current.has(id)) {
            pendingResolvers.current.delete(id);
            silentRequestIds.current.delete(id);
            reject(new Error('Processing timed out.'));
          }
        }, 30000);
      });
    },
    [send]
  );

  return { result, isProcessing, error, process, processImmediate };
}
