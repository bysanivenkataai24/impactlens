import type { PredictResponse } from '../src/lib/types';

export const SHA = 'a'.repeat(64);

export function predictResponse(overrides: Partial<PredictResponse> = {}): PredictResponse {
  return {
    schema_version: '1.0.0',
    request_id: 'req-1',
    model_id: 'impactlens-yolo11n-seg-v1',
    model_revision: 'run-1',
    checkpoint_sha256: SHA,
    image: { width: 400, height: 200, format: 'JPEG', exif_orientation: 6, orientation_applied: true, sha256: 'b'.repeat(64) },
    score_threshold: 0.25,
    outcome: 'detections_returned',
    message: '1 region(s) returned for human review.',
    detections: [
      {
        id: 'det-1',
        damage_type: 'dent',
        score: 0.82,
        bbox: [0.25, 0.25, 0.75, 0.75],
        polygon: [
          [0.25, 0.25],
          [0.75, 0.25],
          [0.75, 0.75],
          [0.25, 0.75],
        ],
        part: null,
        severity: 'unknown',
        flags: ['small_region'],
        review_status: 'pending_review',
      },
    ],
    warnings: [],
    latency_ms: { queue: 0, decode: 2, inference: 80, postprocess: 1, total: 90 },
    limitations: ['Decision support only.'],
    ...overrides,
  };
}

export const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T;
