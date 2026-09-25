/**
 * Types for IRecover's Geographic Analysis feature.
 *
 * This is a SEPARATE, OPTIONAL capability from the core IR -> RGB
 * conversion pipeline. It never feeds back into the conversion itself.
 *
 * Everything here distinguishes OBSERVED/MEASURED values (computed
 * directly from pixels or file metadata) from INTERPRETED/ESTIMATED
 * values (heuristic classification, or Gemini's natural-language read).
 * Nothing in this module should ever be presented to the user as an
 * exact, authoritative measurement.
 */

export type ConfidenceLevel = 'high' | 'medium' | 'low';

/** What kind of capture the analysis believes it is looking at -- purely descriptive, never assumed. */
export type InputImageKind = 'rgb' | 'nir' | 'thermal' | 'false-color' | 'multispectral' | 'unknown';

export type LandCoverType =
  | 'vegetation'
  | 'water'
  | 'bare-soil'
  | 'rock'
  | 'snow-ice'
  | 'urban'
  | 'agriculture'
  | 'open-terrain'
  | 'unknown';

export interface LandCoverEstimate {
  type: LandCoverType;
  label: string;
  /** Estimated percentage of analyzed pixels (0-100). Always an estimate, never exact. */
  estimatedCoveragePercent: number;
  confidence: ConfidenceLevel;
}

export interface VegetationEstimate {
  detected: boolean;
  estimatedCoveragePercent: number | null;
  confidence: ConfidenceLevel;
  description: string;
  /** Only populated when real NIR + Red band data was actually available. */
  ndvi:
    | { available: true; mean: number; sampleCount: number }
    | { available: false; reason: string };
}

export interface WaterEstimate {
  detected: boolean;
  estimatedCoveragePercent: number | null;
  confidence: ConfidenceLevel;
  description: string;
}

export interface TerrainEstimate {
  description: string;
  confidence: ConfidenceLevel;
}

export interface BuiltUpEstimate {
  detected: boolean;
  confidence: ConfidenceLevel;
  description: string;
}

/** Purely local, deterministic, non-ML measurements/heuristics. Runs entirely in the browser. */
export interface LocalGeographicAnalysis {
  inputType: InputImageKind;
  landCover: LandCoverEstimate[];
  vegetation: VegetationEstimate;
  water: WaterEstimate;
  terrain: TerrainEstimate;
  builtUp: BuiltUpEstimate;
  /** Number of pixels actually sampled (large images are subsampled for speed). */
  sampledPixels: number;
  limitations: string[];
}

/** Metadata pulled from the file itself (EXIF/PNG chunks), never inferred visually. */
export interface ImageGeographicMetadata {
  width: number;
  height: number;
  format: string;
  hasGps: boolean;
  latitude: number | null;
  longitude: number | null;
  cameraMake: string | null;
  cameraModel: string | null;
  captureDate: string | null;
  /** True only if EXIF/metadata bytes were actually found and parsed. */
  metadataAvailable: boolean;
}

/** A manually-supplied location, used as context only (Case 2 in the spec). */
export interface ManualLocationContext {
  label: string;
}

export interface ReverseGeocodeResult {
  displayName: string;
  country: string | null;
  region: string | null;
  district: string | null;
  locality: string | null;
  attribution: string;
}

/** Gemini's structured multimodal interpretation. Nothing here is ever treated as ground truth. */
export interface GeminiSceneInterpretation {
  description: string;
  confidence: number;
}

export interface GeminiFeatureInterpretation {
  type: LandCoverType | 'other';
  description: string;
  estimatedCoverage: number | null;
  confidence: number;
}

export interface GeminiLocationGuess {
  country: string | null;
  region: string | null;
  note: string;
}

export interface GeminiGeographicResponse {
  scene: GeminiSceneInterpretation;
  features: GeminiFeatureInterpretation[];
  location: GeminiLocationGuess | null;
  limitations: string[];
}

export type GeographicAnalysisErrorCode =
  | 'not-configured'
  | 'validation-error'
  | 'upstream-error'
  | 'invalid-response'
  | 'timeout'
  | 'unknown';

export type GeminiAnalysisStatus =
  | { status: 'idle' }
  | { status: 'unavailable'; reason: string }
  | { status: 'success'; result: GeminiGeographicResponse }
  | { status: 'error'; code: GeographicAnalysisErrorCode; message: string };

/** What the server endpoint returns to the browser. Never contains credentials or raw upstream errors. */
export interface GeographicAnalysisApiResponse {
  ok: boolean;
  gemini?: GeminiGeographicResponse;
  geminiUnavailableReason?: string;
  location?: ReverseGeocodeResult | null;
  code?: GeographicAnalysisErrorCode;
  error?: string;
}

/** The full, combined report shown in the UI. */
export interface GeographicAnalysisReport {
  metadata: ImageGeographicMetadata;
  local: LocalGeographicAnalysis;
  gemini: GeminiAnalysisStatus;
  location: ReverseGeocodeResult | null | 'unavailable' | 'not-requested';
  /** What actually left the browser for this report, for honest privacy messaging. */
  dataSent: 'none' | 'gemini' | 'gemini-and-location';
}
