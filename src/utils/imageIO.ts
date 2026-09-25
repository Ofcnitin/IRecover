import type { LoadedImage } from '../types/image';
import { analyzeImage } from '../processing/imageAnalysis';
import { extractImageMetadata } from './exifMetadata';

/** Maximum dimension (longest side) processed at "preview" resolution while dragging sliders. */
export const PREVIEW_MAX_DIMENSION = 1200;

/** Maximum dimension sent to Photo Finish -- keeps uploads fast without a dozen tiny requests. */
export const PHOTO_FINISH_MAX_DIMENSION = 1600;

/** Safety ceiling: images larger than this (longest side) are downscaled even for "full" export. */
export const MAX_SAFE_DIMENSION = 6000;

/** Maximum dimension sent to Geographic Analysis (Gemini) -- deliberately small to control cost and payload size. */
export const GEOGRAPHIC_ANALYSIS_MAX_DIMENSION = 1024;

export async function loadImageFile(file: File): Promise<LoadedImage> {
  const bitmap = await createImageBitmap(file);
  const canvas = document.createElement('canvas');
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Canvas 2D context is unavailable in this browser.');
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();

  let data: ImageData;
  try {
    data = ctx.getImageData(0, 0, canvas.width, canvas.height);
  } catch (err) {
    throw new Error(
      'Could not read pixel data from this image (it may be corrupted or in an unsupported color profile).'
    );
  }

  // Metadata extraction never blocks or fails image loading -- it's a
  // best-effort read of EXIF/eXIf bytes, separate from pixel decoding.
  const geoMetadata = await extractImageMetadata(file, canvas.width, canvas.height).catch(() => undefined);

  return {
    fileName: file.name,
    fileSize: file.size,
    mimeType: file.type || 'unknown',
    width: canvas.width,
    height: canvas.height,
    data,
    detected: analyzeImage(data),
    geoMetadata,
  };
}

/** Downscales ImageData so its longest side is at most maxDimension. Returns the same object if already small enough. */
export function downscaleImageData(data: ImageData, maxDimension: number): ImageData {
  const longest = Math.max(data.width, data.height);
  if (longest <= maxDimension) return data;

  const scale = maxDimension / longest;
  const newWidth = Math.max(1, Math.round(data.width * scale));
  const newHeight = Math.max(1, Math.round(data.height * scale));

  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = data.width;
  srcCanvas.height = data.height;
  const srcCtx = srcCanvas.getContext('2d')!;
  srcCtx.putImageData(data, 0, 0);

  const dstCanvas = document.createElement('canvas');
  dstCanvas.width = newWidth;
  dstCanvas.height = newHeight;
  const dstCtx = dstCanvas.getContext('2d', { willReadFrequently: true })!;
  dstCtx.imageSmoothingEnabled = true;
  dstCtx.imageSmoothingQuality = 'high';
  dstCtx.drawImage(srcCanvas, 0, 0, newWidth, newHeight);

  return dstCtx.getImageData(0, 0, newWidth, newHeight);
}

export function imageDataToCanvas(data: ImageData): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = data.width;
  canvas.height = data.height;
  const ctx = canvas.getContext('2d')!;
  ctx.putImageData(data, 0, 0);
  return canvas;
}

export type ExportFormat = 'png' | 'jpeg' | 'webp';

function formatToMime(format: ExportFormat): string {
  return format === 'png' ? 'image/png' : format === 'jpeg' ? 'image/jpeg' : 'image/webp';
}

async function downloadBlob(blob: Blob, fileName: string): Promise<void> {
  const url = URL.createObjectURL(blob);
  try {
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    a.remove();
  } finally {
    // Release the object URL after the download has had a chance to start.
    setTimeout(() => URL.revokeObjectURL(url), 4000);
  }
}

export async function exportImageData(
  data: ImageData,
  format: ExportFormat,
  quality: number,
  fileName: string
): Promise<void> {
  const canvas = imageDataToCanvas(data);
  const mime = formatToMime(format);
  const blob: Blob | null = await new Promise((resolve) =>
    canvas.toBlob(resolve, mime, format === 'png' ? undefined : quality)
  );
  if (!blob) throw new Error('Failed to encode the image for export.');
  await downloadBlob(blob, fileName);
}

/** Converts an ImageData buffer to a lossless PNG data URL (used to hand off local results to Photo Finish). */
export function imageDataToPngDataUrl(data: ImageData): string {
  return imageDataToCanvas(data).toDataURL('image/png');
}

/** Downloads an existing data URL (e.g. a Photo Finish result) in the requested export format. */
export async function exportDataUrl(
  dataUrl: string,
  format: ExportFormat,
  quality: number,
  fileName: string
): Promise<void> {
  const img = new Image();
  const loaded = new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error('Failed to decode the finished image for export.'));
  });
  img.src = dataUrl;
  await loaded;

  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0);

  const mime = formatToMime(format);
  const blob: Blob | null = await new Promise((resolve) =>
    canvas.toBlob(resolve, mime, format === 'png' ? undefined : quality)
  );
  if (!blob) throw new Error('Failed to encode the image for export.');
  await downloadBlob(blob, fileName);
}

export function buildOutputFileName(originalName: string, suffix: string, extension: string): string {
  const dot = originalName.lastIndexOf('.');
  const base = dot > 0 ? originalName.slice(0, dot) : originalName;
  const safeBase = base.replace(/[^a-zA-Z0-9-_ ]/g, '').trim() || 'image';
  return `${safeBase}-${suffix}.${extension}`;
}
