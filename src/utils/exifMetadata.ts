import type { ImageGeographicMetadata } from '../types/geographic';

/**
 * Minimal, dependency-free EXIF parser. Reads only what IRecover needs
 * (GPS position, camera make/model, capture date) directly from the
 * TIFF/EXIF byte structure. Never invents values -- if the relevant tag
 * isn't present, the corresponding field stays null.
 *
 * Supports:
 *  - JPEG: APP1 "Exif\0\0" segment
 *  - PNG: the "eXIf" ancillary chunk (same TIFF structure inside)
 * Any other format (or a JPEG/PNG with no EXIF) yields metadataAvailable: false.
 */

interface TiffReader {
  bytes: DataView;
  littleEndian: boolean;
  tiffStart: number;
}

const TAG_GPS_IFD_POINTER = 0x8825;
const TAG_MAKE = 0x010f;
const TAG_MODEL = 0x0110;
const TAG_DATETIME_ORIGINAL = 0x9003;
const TAG_DATETIME = 0x0132;

const TAG_GPS_LAT_REF = 0x0001;
const TAG_GPS_LAT = 0x0002;
const TAG_GPS_LON_REF = 0x0003;
const TAG_GPS_LON = 0x0004;

function findJpegExifSegment(view: DataView): { start: number } | null {
  if (view.byteLength < 4 || view.getUint16(0) !== 0xffd8) return null;
  let offset = 2;
  while (offset + 4 <= view.byteLength) {
    const marker = view.getUint16(offset);
    if (marker === 0xffda) break; // start of scan -- no more markers to check
    if ((marker & 0xff00) !== 0xff00) break;
    const length = view.getUint16(offset + 2);
    if (marker === 0xffe1) {
      // APP1 -- check for "Exif\0\0"
      const sigStart = offset + 4;
      if (
        sigStart + 6 <= view.byteLength &&
        view.getUint8(sigStart) === 0x45 && // E
        view.getUint8(sigStart + 1) === 0x78 && // x
        view.getUint8(sigStart + 2) === 0x69 && // i
        view.getUint8(sigStart + 3) === 0x66 && // f
        view.getUint8(sigStart + 4) === 0x00 &&
        view.getUint8(sigStart + 5) === 0x00
      ) {
        return { start: sigStart + 6 };
      }
    }
    offset += 2 + length;
  }
  return null;
}

function findPngExifChunk(view: DataView): { start: number; length: number } | null {
  if (view.byteLength < 8) return null;
  const PNG_SIG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  for (let i = 0; i < 8; i++) {
    if (view.getUint8(i) !== PNG_SIG[i]) return null;
  }
  let offset = 8;
  while (offset + 8 <= view.byteLength) {
    const length = view.getUint32(offset);
    const type =
      String.fromCharCode(view.getUint8(offset + 4)) +
      String.fromCharCode(view.getUint8(offset + 5)) +
      String.fromCharCode(view.getUint8(offset + 6)) +
      String.fromCharCode(view.getUint8(offset + 7));
    const dataStart = offset + 8;
    if (type === 'eXIf') {
      return { start: dataStart, length };
    }
    if (type === 'IDAT') break; // eXIf must appear before image data
    offset = dataStart + length + 4; // + CRC
  }
  return null;
}

function readTiffHeader(view: DataView, tiffStart: number): TiffReader | null {
  if (tiffStart + 8 > view.byteLength) return null;
  const byteOrder = view.getUint16(tiffStart);
  const littleEndian = byteOrder === 0x4949;
  if (!littleEndian && byteOrder !== 0x4d4d) return null;
  const magic = view.getUint16(tiffStart + 2, littleEndian);
  if (magic !== 42) return null;
  return { bytes: view, littleEndian, tiffStart };
}

interface IfdEntry {
  tag: number;
  type: number;
  count: number;
  valueOffsetOrValue: number;
  entryOffset: number;
}

function readIfd(reader: TiffReader, ifdOffset: number): IfdEntry[] {
  const { bytes, littleEndian, tiffStart } = reader;
  const abs = tiffStart + ifdOffset;
  if (abs + 2 > bytes.byteLength) return [];
  const count = bytes.getUint16(abs, littleEndian);
  const entries: IfdEntry[] = [];
  for (let i = 0; i < count; i++) {
    const entryOffset = abs + 2 + i * 12;
    if (entryOffset + 12 > bytes.byteLength) break;
    entries.push({
      tag: bytes.getUint16(entryOffset, littleEndian),
      type: bytes.getUint16(entryOffset + 2, littleEndian),
      count: bytes.getUint32(entryOffset + 4, littleEndian),
      valueOffsetOrValue: bytes.getUint32(entryOffset + 8, littleEndian),
      entryOffset,
    });
  }
  return entries;
}

function typeSize(type: number): number {
  switch (type) {
    case 1: // BYTE
    case 2: // ASCII
    case 7: // UNDEFINED
      return 1;
    case 3: // SHORT
      return 2;
    case 4: // LONG
    case 9: // SLONG
      return 4;
    case 5: // RATIONAL
    case 10: // SRATIONAL
      return 8;
    default:
      return 4;
  }
}

function readAscii(reader: TiffReader, entry: IfdEntry): string | null {
  const size = typeSize(entry.type) * entry.count;
  const { bytes, tiffStart } = reader;
  const dataOffset = size <= 4 ? entry.entryOffset + 8 : tiffStart + entry.valueOffsetOrValue;
  if (dataOffset + entry.count > bytes.byteLength) return null;
  let str = '';
  for (let i = 0; i < entry.count; i++) {
    const code = bytes.getUint8(dataOffset + i);
    if (code === 0) break;
    str += String.fromCharCode(code);
  }
  return str.trim() || null;
}

function readRationalArray(reader: TiffReader, entry: IfdEntry): number[] {
  const { bytes, littleEndian, tiffStart } = reader;
  const dataOffset = tiffStart + entry.valueOffsetOrValue;
  const values: number[] = [];
  for (let i = 0; i < entry.count; i++) {
    const off = dataOffset + i * 8;
    if (off + 8 > bytes.byteLength) break;
    const numerator = bytes.getUint32(off, littleEndian);
    const denominator = bytes.getUint32(off + 4, littleEndian);
    values.push(denominator === 0 ? 0 : numerator / denominator);
  }
  return values;
}

function dmsToDecimal(dms: number[], ref: string | null): number | null {
  if (dms.length !== 3) return null;
  const [d, m, s] = dms;
  let decimal = d + m / 60 + s / 3600;
  if (ref === 'S' || ref === 'W') decimal = -decimal;
  return Math.round(decimal * 1e6) / 1e6;
}

function parseExifFromTiff(view: DataView, tiffStart: number): Partial<ImageGeographicMetadata> {
  const reader = readTiffHeader(view, tiffStart);
  if (!reader) return {};

  const ifd0Offset = view.getUint32(tiffStart + 4, reader.littleEndian);
  const ifd0 = readIfd(reader, ifd0Offset);

  let cameraMake: string | null = null;
  let cameraModel: string | null = null;
  let captureDate: string | null = null;
  let gpsIfdOffset: number | null = null;

  for (const entry of ifd0) {
    if (entry.tag === TAG_MAKE) cameraMake = readAscii(reader, entry);
    else if (entry.tag === TAG_MODEL) cameraModel = readAscii(reader, entry);
    else if (entry.tag === TAG_DATETIME || entry.tag === TAG_DATETIME_ORIGINAL) {
      captureDate = readAscii(reader, entry) ?? captureDate;
    } else if (entry.tag === TAG_GPS_IFD_POINTER) {
      gpsIfdOffset = entry.valueOffsetOrValue;
    }
  }

  let latitude: number | null = null;
  let longitude: number | null = null;

  if (gpsIfdOffset !== null) {
    const gpsIfd = readIfd(reader, gpsIfdOffset);
    let latRef: string | null = null;
    let lonRef: string | null = null;
    let latDms: number[] = [];
    let lonDms: number[] = [];
    for (const entry of gpsIfd) {
      if (entry.tag === TAG_GPS_LAT_REF) latRef = readAscii(reader, entry);
      else if (entry.tag === TAG_GPS_LON_REF) lonRef = readAscii(reader, entry);
      else if (entry.tag === TAG_GPS_LAT) latDms = readRationalArray(reader, entry);
      else if (entry.tag === TAG_GPS_LON) lonDms = readRationalArray(reader, entry);
    }
    latitude = dmsToDecimal(latDms, latRef);
    longitude = dmsToDecimal(lonDms, lonRef);
  }

  return { cameraMake, cameraModel, captureDate, latitude, longitude, hasGps: latitude !== null && longitude !== null };
}

/**
 * Extracts whatever geographic-relevant metadata is actually present in
 * the file. Returns metadataAvailable: false rather than guessing when
 * no EXIF/eXIf block was found -- this is intentionally conservative.
 */
export async function extractImageMetadata(
  file: File,
  width: number,
  height: number
): Promise<ImageGeographicMetadata> {
  const base: ImageGeographicMetadata = {
    width,
    height,
    format: file.type || 'unknown',
    hasGps: false,
    latitude: null,
    longitude: null,
    cameraMake: null,
    cameraModel: null,
    captureDate: null,
    metadataAvailable: false,
  };

  try {
    // Only read the first portion of the file -- EXIF/eXIf always appears
    // near the start, and we don't need the rest for metadata purposes.
    const headSize = Math.min(file.size, 2 * 1024 * 1024);
    const buffer = await file.slice(0, headSize).arrayBuffer();
    const view = new DataView(buffer);

    let tiffStart: number | null = null;

    if (file.type === 'image/jpeg' || file.type === 'image/jpg') {
      const segment = findJpegExifSegment(view);
      if (segment) tiffStart = segment.start;
    } else if (file.type === 'image/png') {
      const chunk = findPngExifChunk(view);
      if (chunk) tiffStart = chunk.start;
    }

    if (tiffStart === null) return base;

    const parsed = parseExifFromTiff(view, tiffStart);
    return {
      ...base,
      ...parsed,
      metadataAvailable: true,
    } as ImageGeographicMetadata;
  } catch {
    // Corrupt/unreadable metadata should never break the app -- fall back
    // to "no metadata available" rather than throwing.
    return base;
  }
}
