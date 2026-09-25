import { describe, it, expect } from 'vitest';
import { extractImageMetadata } from '../src/utils/exifMetadata';

/**
 * Builds a minimal, synthetic JPEG byte stream containing just enough
 * structure (SOI + APP1/Exif/TIFF + GPS IFD) for the parser to exercise
 * its real code paths, without needing a real photo fixture on disk.
 */
function buildJpegWithGps(lat: number, lon: number, latRef: 'N' | 'S', lonRef: 'E' | 'W'): Uint8Array {
  // -- Build the TIFF/EXIF body (big-endian "MM") --
  const parts: number[] = [];
  const push16 = (v: number) => parts.push((v >> 8) & 0xff, v & 0xff);
  const push32 = (v: number) => parts.push((v >> 24) & 0xff, (v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff);

  // TIFF header: byte order MM, magic 42, offset to IFD0 = 8
  push16(0x4d4d);
  push16(42);
  push32(8);

  // IFD0: 1 entry -> GPS IFD pointer (tag 0x8825), pointing right after IFD0.
  const ifd0Offset = 8;
  const ifd0EntryCount = 1;
  const ifd0Size = 2 + ifd0EntryCount * 12 + 4; // count + entries + next-IFD offset
  const gpsIfdOffset = ifd0Offset + ifd0Size;

  push16(ifd0EntryCount);
  push16(0x8825); // GPSInfo tag
  push16(4); // type LONG
  push32(1); // count
  push32(gpsIfdOffset); // value (offset to GPS IFD)
  push32(0); // next IFD offset (none)

  // GPS IFD: LatRef(ASCII,2, inline), Lat(RATIONAL,3, out-of-line),
  // LonRef(ASCII,2, inline), Lon(RATIONAL,3, out-of-line).
  // Per TIFF spec, a value whose total size is <=4 bytes is stored directly
  // in the entry's value field, NOT as an out-of-line pointer -- so the two
  // ASCII ref fields (1 char + null = 2 bytes) go inline.
  const gpsEntryCount = 4;
  const gpsIfdHeaderSize = 2 + gpsEntryCount * 12 + 4;
  let dataCursor = gpsIfdOffset + gpsIfdHeaderSize;

  const latRationalOffset = dataCursor;
  dataCursor += 24; // 3 rationals * 8 bytes
  const lonRationalOffset = dataCursor;
  dataCursor += 24;

  push16(gpsEntryCount);

  // GPSLatitudeRef -- inline ASCII value (padded to 4 bytes)
  push16(0x0001);
  push16(2); // ASCII
  push32(2); // count (1 char + null)
  parts.push(latRef.charCodeAt(0), 0, 0, 0);

  // GPSLatitude -- out-of-line RATIONAL[3]
  push16(0x0002);
  push16(5); // RATIONAL
  push32(3);
  push32(latRationalOffset);

  // GPSLongitudeRef -- inline ASCII value (padded to 4 bytes)
  push16(0x0003);
  push16(2);
  push32(2);
  parts.push(lonRef.charCodeAt(0), 0, 0, 0);

  // GPSLongitude -- out-of-line RATIONAL[3]
  push16(0x0004);
  push16(5);
  push32(3);
  push32(lonRationalOffset);

  push32(0); // next IFD offset (none)

  // -- Out-of-line data (only the two RATIONAL[3] arrays) --
  const latDeg = Math.floor(lat);
  const latMin = Math.floor((lat - latDeg) * 60);
  const latSec = ((lat - latDeg) * 60 - latMin) * 60;
  const lonDeg = Math.floor(lon);
  const lonMin = Math.floor((lon - lonDeg) * 60);
  const lonSec = ((lon - lonDeg) * 60 - lonMin) * 60;

  const pushRational = (value: number) => {
    // Represent as value*1000/1000 for simplicity, integer-safe.
    push32(Math.round(value * 1000));
    push32(1000);
  };

  // These pushes must land exactly at latRationalOffset/lonRationalOffset,
  // which holds because parts is built strictly sequentially up to here.
  pushRational(latDeg);
  pushRational(latMin);
  pushRational(latSec);
  pushRational(lonDeg);
  pushRational(lonMin);
  pushRational(lonSec);

  const tiffBytes = parts;

  // -- Wrap in JPEG APP1/Exif segment + SOI/EOI --
  const exifHeader = [0x45, 0x78, 0x69, 0x66, 0x00, 0x00]; // "Exif\0\0"
  const app1Body = [...exifHeader, ...tiffBytes];
  const app1Length = app1Body.length + 2; // includes the length field itself

  const jpeg: number[] = [
    0xff,
    0xd8, // SOI
    0xff,
    0xe1, // APP1 marker
    (app1Length >> 8) & 0xff,
    app1Length & 0xff,
    ...app1Body,
    0xff,
    0xd9, // EOI
  ];

  return new Uint8Array(jpeg);
}

describe('extractImageMetadata', () => {
  it('reports metadataAvailable: false for a file with no EXIF', async () => {
    const file = new File([new Uint8Array([0xff, 0xd8, 0xff, 0xd9])], 'plain.jpg', { type: 'image/jpeg' });
    const result = await extractImageMetadata(file, 10, 10);
    expect(result.metadataAvailable).toBe(false);
    expect(result.hasGps).toBe(false);
    expect(result.latitude).toBeNull();
  });

  it('extracts GPS latitude/longitude from a synthetic EXIF GPS block', async () => {
    const bytes = buildJpegWithGps(37.7749, 122.4194, 'N', 'W');
    const file = new File([bytes], 'geo.jpg', { type: 'image/jpeg' });
    const result = await extractImageMetadata(file, 100, 100);
    expect(result.metadataAvailable).toBe(true);
    expect(result.hasGps).toBe(true);
    expect(result.latitude).toBeCloseTo(37.7749, 1);
    // West longitude must come out negative.
    expect(result.longitude).toBeLessThan(0);
    expect(result.longitude).toBeCloseTo(-122.4194, 1);
  });

  it('never throws on a corrupt/truncated file', async () => {
    const file = new File([new Uint8Array([0xff, 0xd8, 0xff, 0xe1, 0xff])], 'broken.jpg', { type: 'image/jpeg' });
    await expect(extractImageMetadata(file, 5, 5)).resolves.toBeDefined();
  });

  it('returns metadataAvailable: false for a non-image mime type', async () => {
    const file = new File([new Uint8Array([1, 2, 3])], 'notes.txt', { type: 'text/plain' });
    const result = await extractImageMetadata(file, 0, 0);
    expect(result.metadataAvailable).toBe(false);
  });
});
