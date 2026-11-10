// Wire types of the inference service (schema 1.x) and the review domain model.
// The service contract is defined in src/impactlens/schemas.py and published as
// JSON Schema under /schemas. Keep these in step with it.

export const MODEL_DAMAGE_TYPES = [
  'dent',
  'scratch',
  'crack',
  'glass_shatter',
  'lamp_broken',
  'tire_flat',
] as const;
export type ModelDamageType = (typeof MODEL_DAMAGE_TYPES)[number];

/** Reviewer-only label. A model never returns it. */
export const OTHER_UNCERTAIN = 'other_uncertain' as const;
export const REVIEW_DAMAGE_TYPES = [...MODEL_DAMAGE_TYPES, OTHER_UNCERTAIN] as const;
export type ReviewDamageType = (typeof REVIEW_DAMAGE_TYPES)[number];

export const DAMAGE_LABELS: Record<ReviewDamageType, string> = {
  dent: 'Dent',
  scratch: 'Scratch',
  crack: 'Crack',
  glass_shatter: 'Glass shatter',
  lamp_broken: 'Lamp broken',
  tire_flat: 'Tyre flat',
  other_uncertain: 'Other / uncertain',
};

export const DETECTION_FLAGS = [
  'mask_fragmented',
  'polygon_simplified',
  'touches_image_border',
  'small_region',
] as const;
export type DetectionFlag = (typeof DETECTION_FLAGS)[number];

export const FLAG_DESCRIPTIONS: Record<DetectionFlag, string> = {
  mask_fragmented: 'The mask had several pieces; only the largest is outlined.',
  polygon_simplified: 'The outline was simplified to the service vertex limit.',
  touches_image_border: 'The region reaches the photo edge; damage may continue out of frame.',
  small_region: 'Very small region (< 0.1% of the photo); small regions are the least reliable.',
};

export type Point = [number, number];
/** [x_min, y_min, x_max, y_max], normalized to the EXIF-oriented image. */
export type BBox = [number, number, number, number];

export interface Detection {
  id: string;
  damage_type: ModelDamageType;
  score: number;
  bbox: BBox;
  polygon: Point[];
  part: null;
  severity: 'unknown';
  flags: DetectionFlag[];
  review_status: 'pending_review';
}

export interface PredictResponse {
  schema_version: string;
  request_id: string;
  model_id: string;
  model_revision: string;
  checkpoint_sha256: string;
  image: {
    width: number;
    height: number;
    format: 'JPEG' | 'PNG' | 'WEBP';
    exif_orientation: number | null;
    orientation_applied: boolean;
    sha256: string;
  };
  score_threshold: number;
  outcome: 'detections_returned' | 'no_supported_region_returned';
  message: string;
  detections: Detection[];
  warnings: string[];
  latency_ms: { queue: number; decode: number; inference: number; postprocess: number; total: number };
  limitations: string[];
}

export interface ServiceLimits {
  max_upload_bytes: number;
  max_image_pixels: number;
  min_image_side: number;
  accepted_media_types: string[];
  score_threshold: { default: number; min: number; max: number };
  max_detections: number;
  max_polygon_points: number;
  inference_timeout_s: number;
}

export interface HealthResponse {
  schema_version: string;
  status: 'ok';
  ready: boolean;
  reasons: string[];
  model_id: string | null;
  model_revision: string | null;
  checkpoint_sha256: string | null;
  task: string | null;
  classes: string[];
  device: string | null;
  limits: ServiceLimits;
  service_version: string;
  source_url: string | null;
}

export interface ErrorBody {
  code: string;
  message: string;
  request_id: string;
  reasons?: string[];
  retry_after_s?: number;
}

/** Limits used before a service has been contacted. They mirror the service defaults. */
export const DEFAULT_LIMITS: ServiceLimits = {
  max_upload_bytes: 12 * 1024 * 1024,
  max_image_pixels: 50_000_000,
  min_image_side: 32,
  accepted_media_types: ['image/jpeg', 'image/png', 'image/webp'],
  score_threshold: { default: 0.25, min: 0.05, max: 0.95 },
  max_detections: 50,
  max_polygon_points: 200,
  inference_timeout_s: 60,
};

// ----------------------------------------------------------------- review model
export type Origin = 'illustrative_sample' | 'model_output' | 'human_annotation';
export type ReviewStatus = 'pending_review' | 'confirmed' | 'rejected' | 'needs_more_evidence';
export type VisibleSeverity = 'unknown' | 'minor' | 'moderate' | 'severe';

export const REVIEW_STATUS_LABELS: Record<ReviewStatus, string> = {
  pending_review: 'Pending review',
  confirmed: 'Confirmed by reviewer',
  rejected: 'Rejected by reviewer',
  needs_more_evidence: 'Needs more evidence',
};

export const SEVERITY_LABELS: Record<VisibleSeverity, string> = {
  unknown: 'Unknown',
  minor: 'Minor (reviewer estimate)',
  moderate: 'Moderate (reviewer estimate)',
  severe: 'Severe (reviewer estimate)',
};

export interface Region {
  bbox: BBox;
  polygon: Point[];
}

/** Immutable copy of what the model returned. Never edited. */
export interface ModelPrediction {
  detection_id: string;
  damage_type: ModelDamageType;
  score: number;
  region: Region;
  flags: DetectionFlag[];
  request_id: string;
  model_id: string;
  model_revision: string;
  checkpoint_sha256: string;
  score_threshold: number;
  schema_version: string;
}

export interface Finding {
  id: string;
  photo_id: string;
  origin: Origin;
  /** Present for model output; preserved even after a reviewer changes the label. */
  original: ModelPrediction | null;
  damage_type: ReviewDamageType;
  region: Region;
  /** Reviewer entry. The baseline model does not predict parts. */
  part: string | null;
  /** Reviewer's visual estimate. The baseline model does not predict severity. */
  severity: VisibleSeverity;
  review_status: ReviewStatus;
  notes: string;
  /** Actual workshop quote entered by the reviewer, in INR. Never computed. */
  quote_inr: number | null;
  /** Same physical damage as another finding (e.g. seen in another photo). */
  same_damage_as: string | null;
  created_at: string;
  updated_at: string;
}

export interface ModelRun {
  request_id: string;
  model_id: string;
  model_revision: string;
  checkpoint_sha256: string;
  schema_version: string;
  score_threshold: number;
  outcome: PredictResponse['outcome'];
  message: string;
  warnings: string[];
  limitations: string[];
  latency_ms: PredictResponse['latency_ms'];
  completed_at: string;
}

export interface Photo {
  id: string;
  file_name: string;
  media_type: string;
  bytes: number;
  sha256: string;
  /** Dimensions after EXIF orientation, i.e. as displayed. */
  width: number;
  height: number;
  /** Object URL of the in-memory file. null when restored from storage without the file. */
  object_url: string | null;
  /** Bundled illustration (demo mode) instead of a user file. */
  sample_url?: string;
  model_run: ModelRun | null;
}

export interface VehicleContext {
  make: string;
  model: string;
  year: string;
  body_style: string;
  colour: string;
  notes: string;
}

export interface InspectionCase {
  id: string;
  mode: 'live' | 'illustrative_demo';
  title: string;
  created_at: string;
  updated_at: string;
  vehicle: VehicleContext;
  reviewer_name: string;
  reviewer_notes: string;
  photos: Photo[];
  findings: Finding[];
}

export const MAX_PHOTOS = 6;
