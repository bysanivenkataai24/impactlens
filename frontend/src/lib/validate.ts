// Runtime validation of service responses. A response that fails validation is
// shown as an "invalid response" error; its detections are never displayed.

import {
  DETECTION_FLAGS,
  MODEL_DAMAGE_TYPES,
  type BBox,
  type Detection,
  type ErrorBody,
  type HealthResponse,
  type Point,
  type PredictResponse,
  type ServiceLimits,
} from './types';

export class ValidationError extends Error {
  constructor(public readonly path: string, message: string) {
    super(`${path}: ${message}`);
    this.name = 'ValidationError';
  }
}

type Json = unknown;

const isObject = (value: Json): value is Record<string, Json> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

function obj(value: Json, path: string): Record<string, Json> {
  if (!isObject(value)) throw new ValidationError(path, 'expected an object');
  return value;
}

function str(value: Json, path: string, { max = 2000, pattern }: { max?: number; pattern?: RegExp } = {}): string {
  if (typeof value !== 'string') throw new ValidationError(path, 'expected a string');
  if (value.length > max) throw new ValidationError(path, `longer than ${max} characters`);
  if (pattern && !pattern.test(value)) throw new ValidationError(path, 'unexpected format');
  return value;
}

function num(value: Json, path: string, low = -Infinity, high = Infinity): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new ValidationError(path, 'expected a finite number');
  if (value < low || value > high) throw new ValidationError(path, `outside [${low}, ${high}]`);
  return value;
}

function int(value: Json, path: string, low: number, high: number): number {
  const n = num(value, path, low, high);
  if (!Number.isInteger(n)) throw new ValidationError(path, 'expected an integer');
  return n;
}

function bool(value: Json, path: string): boolean {
  if (typeof value !== 'boolean') throw new ValidationError(path, 'expected a boolean');
  return value;
}

function arr(value: Json, path: string, maxLength: number): Json[] {
  if (!Array.isArray(value)) throw new ValidationError(path, 'expected an array');
  if (value.length > maxLength) throw new ValidationError(path, `more than ${maxLength} items`);
  return value;
}

function oneOf<T extends string>(value: Json, path: string, allowed: readonly T[]): T {
  if (typeof value !== 'string' || !(allowed as readonly string[]).includes(value)) {
    throw new ValidationError(path, `expected one of ${allowed.join(', ')}`);
  }
  return value as T;
}

const SHA256 = /^[0-9a-f]{64}$/;
export const SUPPORTED_SCHEMA_MAJOR = 1;

function schemaVersion(value: Json, path: string): string {
  const version = str(value, path, { max: 20, pattern: /^\d+\.\d+\.\d+$/ });
  if (Number(version.split('.')[0]) !== SUPPORTED_SCHEMA_MAJOR) {
    throw new ValidationError(path, `unsupported schema version ${version} (this UI reads ${SUPPORTED_SCHEMA_MAJOR}.x)`);
  }
  return version;
}

function unit(value: Json, path: string): number {
  return num(value, path, 0, 1);
}

function parseDetection(value: Json, path: string, maxPolygonPoints: number): Detection {
  const d = obj(value, path);
  const bboxRaw = arr(d.bbox, `${path}.bbox`, 4);
  if (bboxRaw.length !== 4) throw new ValidationError(`${path}.bbox`, 'expected 4 numbers');
  const bbox = bboxRaw.map((v, i) => unit(v, `${path}.bbox[${i}]`)) as BBox;
  if (!(bbox[0] < bbox[2] && bbox[1] < bbox[3])) throw new ValidationError(`${path}.bbox`, 'min must be below max');
  const polygonRaw = arr(d.polygon, `${path}.polygon`, maxPolygonPoints);
  if (polygonRaw.length < 3) throw new ValidationError(`${path}.polygon`, 'needs at least 3 points');
  const polygon = polygonRaw.map((p, i) => {
    const pair = arr(p, `${path}.polygon[${i}]`, 2);
    if (pair.length !== 2) throw new ValidationError(`${path}.polygon[${i}]`, 'expected [x, y]');
    return [unit(pair[0], `${path}.polygon[${i}][0]`), unit(pair[1], `${path}.polygon[${i}][1]`)] as Point;
  });
  if (d.part !== null) throw new ValidationError(`${path}.part`, 'expected null (no validated part model in schema 1.x)');
  if (d.severity !== 'unknown') throw new ValidationError(`${path}.severity`, "expected 'unknown' in schema 1.x");
  const flags = arr(d.flags ?? [], `${path}.flags`, 16).map((f, i) => oneOf(f, `${path}.flags[${i}]`, DETECTION_FLAGS));
  return {
    id: str(d.id, `${path}.id`, { max: 64 }),
    damage_type: oneOf(d.damage_type, `${path}.damage_type`, MODEL_DAMAGE_TYPES),
    score: unit(d.score, `${path}.score`),
    bbox,
    polygon,
    part: null,
    severity: 'unknown',
    flags,
    review_status: oneOf(d.review_status, `${path}.review_status`, ['pending_review'] as const),
  };
}

export function parsePredictResponse(value: Json, maxPolygonPoints = 2000): PredictResponse {
  const r = obj(value, 'response');
  const image = obj(r.image, 'image');
  const latency = obj(r.latency_ms, 'latency_ms');
  const detections = arr(r.detections, 'detections', 300).map((d, i) =>
    parseDetection(d, `detections[${i}]`, maxPolygonPoints),
  );
  const outcome = oneOf(r.outcome, 'outcome', ['detections_returned', 'no_supported_region_returned'] as const);
  if ((outcome === 'detections_returned') !== detections.length > 0) {
    throw new ValidationError('outcome', 'does not agree with the number of detections');
  }
  const ids = new Set(detections.map((d) => d.id));
  if (ids.size !== detections.length) throw new ValidationError('detections', 'duplicate detection ids');
  return {
    schema_version: schemaVersion(r.schema_version, 'schema_version'),
    request_id: str(r.request_id, 'request_id', { max: 64 }),
    model_id: str(r.model_id, 'model_id', { max: 128 }),
    model_revision: str(r.model_revision, 'model_revision', { max: 128 }),
    checkpoint_sha256: str(r.checkpoint_sha256, 'checkpoint_sha256', { pattern: SHA256 }),
    image: {
      width: int(image.width, 'image.width', 1, 100_000),
      height: int(image.height, 'image.height', 1, 100_000),
      format: oneOf(image.format, 'image.format', ['JPEG', 'PNG', 'WEBP'] as const),
      exif_orientation: image.exif_orientation === null ? null : int(image.exif_orientation, 'image.exif_orientation', 0, 65535),
      orientation_applied: bool(image.orientation_applied, 'image.orientation_applied'),
      sha256: str(image.sha256, 'image.sha256', { pattern: SHA256 }),
    },
    score_threshold: unit(r.score_threshold, 'score_threshold'),
    outcome,
    message: str(r.message, 'message', { max: 1000 }),
    detections,
    warnings: arr(r.warnings ?? [], 'warnings', 32).map((w, i) => str(w, `warnings[${i}]`, { max: 100 })),
    latency_ms: {
      queue: num(latency.queue, 'latency_ms.queue', 0),
      decode: num(latency.decode, 'latency_ms.decode', 0),
      inference: num(latency.inference, 'latency_ms.inference', 0),
      postprocess: num(latency.postprocess, 'latency_ms.postprocess', 0),
      total: num(latency.total, 'latency_ms.total', 0),
    },
    limitations: arr(r.limitations, 'limitations', 32).map((l, i) => str(l, `limitations[${i}]`, { max: 1000 })),
  };
}

function parseLimits(value: Json): ServiceLimits {
  const l = obj(value, 'limits');
  const t = obj(l.score_threshold, 'limits.score_threshold');
  const limits: ServiceLimits = {
    max_upload_bytes: int(l.max_upload_bytes, 'limits.max_upload_bytes', 1, 2 ** 40),
    max_image_pixels: int(l.max_image_pixels, 'limits.max_image_pixels', 1, 2 ** 40),
    min_image_side: int(l.min_image_side, 'limits.min_image_side', 1, 100_000),
    accepted_media_types: arr(l.accepted_media_types, 'limits.accepted_media_types', 16).map((m, i) =>
      str(m, `limits.accepted_media_types[${i}]`, { max: 64 }),
    ),
    score_threshold: {
      default: unit(t.default, 'limits.score_threshold.default'),
      min: unit(t.min, 'limits.score_threshold.min'),
      max: unit(t.max, 'limits.score_threshold.max'),
    },
    max_detections: int(l.max_detections, 'limits.max_detections', 1, 10_000),
    max_polygon_points: int(l.max_polygon_points, 'limits.max_polygon_points', 3, 100_000),
    inference_timeout_s: num(l.inference_timeout_s, 'limits.inference_timeout_s', 0.1, 3600),
  };
  if (!(limits.score_threshold.min <= limits.score_threshold.default && limits.score_threshold.default <= limits.score_threshold.max)) {
    throw new ValidationError('limits.score_threshold', 'default outside [min, max]');
  }
  return limits;
}

const nullableStr = (value: Json, path: string, max = 256) => (value === null ? null : str(value, path, { max }));

export function parseHealthResponse(value: Json): HealthResponse {
  const r = obj(value, 'response');
  const sourceUrl = nullableStr(r.source_url ?? null, 'source_url', 400);
  return {
    schema_version: schemaVersion(r.schema_version, 'schema_version'),
    status: oneOf(r.status, 'status', ['ok'] as const),
    ready: bool(r.ready, 'ready'),
    reasons: arr(r.reasons, 'reasons', 32).map((v, i) => str(v, `reasons[${i}]`, { max: 64 })),
    model_id: nullableStr(r.model_id, 'model_id', 128),
    model_revision: nullableStr(r.model_revision, 'model_revision', 128),
    checkpoint_sha256: r.checkpoint_sha256 === null ? null : str(r.checkpoint_sha256, 'checkpoint_sha256', { pattern: SHA256 }),
    task: nullableStr(r.task, 'task', 32),
    classes: arr(r.classes, 'classes', 64).map((v, i) => str(v, `classes[${i}]`, { max: 64 })),
    device: nullableStr(r.device, 'device', 32),
    limits: parseLimits(r.limits),
    service_version: str(r.service_version, 'service_version', { max: 32 }),
    // Only https links are rendered as links.
    source_url: sourceUrl && /^https:\/\//.test(sourceUrl) ? sourceUrl : null,
  };
}

/** Our error envelope, or null when the body is something else (e.g. a proxy page). */
export function parseErrorBody(value: Json): ErrorBody | null {
  try {
    const r = obj(value, 'response');
    schemaVersion(r.schema_version, 'schema_version');
    const e = obj(r.error, 'error');
    return {
      code: str(e.code, 'error.code', { max: 64 }),
      message: str(e.message, 'error.message', { max: 1000 }),
      request_id: str(e.request_id, 'error.request_id', { max: 64 }),
      reasons: e.reasons === undefined ? undefined : arr(e.reasons, 'error.reasons', 32).map((v, i) => str(v, `error.reasons[${i}]`, { max: 64 })),
      retry_after_s: e.retry_after_s === undefined ? undefined : int(e.retry_after_s, 'error.retry_after_s', 0, 86_400),
    };
  } catch {
    return null;
  }
}
