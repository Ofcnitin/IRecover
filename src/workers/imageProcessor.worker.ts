import type { runPipeline } from '../processing/pipeline';
import { runPipelineAsync } from '../processing/pipelineAsync';
import type { ProcessingSettings } from '../types/processing';

export interface WorkerRequest {
  requestId: number;
  width: number;
  height: number;
  buffer: ArrayBuffer; // RGBA Uint8ClampedArray backing buffer, transferred
  settings: ProcessingSettings;
}

export interface WorkerResponse {
  requestId: number;
  width: number;
  height: number;
  buffer: ArrayBuffer; // transferred back
  inputHistogram: ReturnType<typeof runPipeline>['inputHistogram'];
  outputHistogram: ReturnType<typeof runPipeline>['outputHistogram'];
  levelsUsed: ReturnType<typeof runPipeline>['levelsUsed'];
  precision?: ReturnType<typeof runPipeline>['precision'];
  /** What actually produced the pixels (webgpu | webgl2 | cpu) + fallback audit trail. */
  execution?: ReturnType<typeof runPipeline>['execution'];
  durationMs: number;
  error?: string;
}

// The pipeline is now async (WebGPU readback), but requests must still be
// processed strictly one at a time -- as the previous synchronous handler
// implicitly guaranteed -- so two runs never contend for GPU memory.
let queue: Promise<void> = Promise.resolve();
self.onmessage = (ev: MessageEvent<WorkerRequest>) => {
  queue = queue.then(() => handle(ev));
};

async function handle(ev: MessageEvent<WorkerRequest>): Promise<void> {
  const { requestId, width, height, buffer, settings } = ev.data;
  const start = performance.now();
  try {
    const data = new ImageData(new Uint8ClampedArray(buffer), width, height);
    // WebGPU first (async), then the synchronous WebGL2 -> CPU chain.
    const result = await runPipelineAsync(data, settings);
    const outBuffer = result.output.data.buffer;
    const response: WorkerResponse = {
      requestId,
      width,
      height,
      buffer: outBuffer,
      inputHistogram: result.inputHistogram,
      outputHistogram: result.outputHistogram,
      levelsUsed: result.levelsUsed,
      precision: result.precision,
      execution: result.execution,
      durationMs: performance.now() - start,
    };
    self.postMessage(response, [outBuffer]);
  } catch (err) {
    const response: WorkerResponse = {
      requestId,
      width,
      height,
      buffer,
      inputHistogram: { luminance: [], min: 0, max: 0, mean: 0, median: 0, percentiles: {} },
      outputHistogram: { luminance: [], min: 0, max: 0, mean: 0, median: 0, percentiles: {} },
      levelsUsed: { black: 0, white: 255 },
      durationMs: performance.now() - start,
      error: err instanceof Error ? err.message : 'Unknown processing error',
    };
    self.postMessage(response);
  }
}

export {};
