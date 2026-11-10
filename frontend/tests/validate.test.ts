import { describe, expect, it } from 'vitest';

import { parseErrorBody, parseHealthResponse, parsePredictResponse } from '../src/lib/validate';
import { clone, predictResponse } from './fixtures';

const health = {
  schema_version: '1.0.0',
  status: 'ok',
  ready: true,
  reasons: [],
  model_id: 'm',
  model_revision: 'r',
  checkpoint_sha256: 'c'.repeat(64),
  task: 'segment',
  classes: ['dent'],
  device: 'cpu',
  limits: {
    max_upload_bytes: 1000,
    max_image_pixels: 1000,
    min_image_side: 32,
    accepted_media_types: ['image/jpeg'],
    score_threshold: { default: 0.25, min: 0.05, max: 0.95 },
    max_detections: 50,
    max_polygon_points: 200,
    inference_timeout_s: 60,
  },
  service_version: '0.1.0',
  source_url: 'https://example.org/src',
};

describe('parsePredictResponse', () => {
  it('accepts a valid response', () => {
    const parsed = parsePredictResponse(clone(predictResponse()));
    expect(parsed.detections[0].bbox).toEqual([0.25, 0.25, 0.75, 0.75]);
    expect(parsed.image.width).toBe(400);
  });

  it('accepts a no-detection response', () => {
    const response = predictResponse({ detections: [], outcome: 'no_supported_region_returned', message: 'none' });
    expect(parsePredictResponse(clone(response)).detections).toEqual([]);
  });

  const broken: Array<[string, (r: ReturnType<typeof predictResponse>) => void]> = [
    ['coordinate above 1', (r) => (r.detections[0].bbox[2] = 1.2)],
    ['negative coordinate', (r) => (r.detections[0].polygon[1][0] = -0.01)],
    ['inverted box', (r) => (r.detections[0].bbox = [0.8, 0.2, 0.3, 0.6])],
    ['two-point polygon', (r) => (r.detections[0].polygon = [[0.1, 0.1], [0.2, 0.2]])],
    ['unknown damage type', (r) => ((r.detections[0] as { damage_type: string }).damage_type = 'rust')],
    ['reviewer-only label from a model', (r) => ((r.detections[0] as { damage_type: string }).damage_type = 'other_uncertain')],
    ['score above 1', (r) => (r.detections[0].score = 1.5)],
    ['predicted part', (r) => ((r.detections[0] as { part: unknown }).part = 'door')],
    ['predicted severity', (r) => ((r.detections[0] as { severity: string }).severity = 'severe')],
    ['unknown flag', (r) => ((r.detections[0] as { flags: string[] }).flags = ['looks_bad'])],
    ['future major schema', (r) => (r.schema_version = '2.0.0')],
    ['outcome disagrees with detections', (r) => (r.outcome = 'no_supported_region_returned')],
    ['duplicate ids', (r) => (r.detections = [r.detections[0], { ...r.detections[0] }])],
    ['bad checkpoint hash', (r) => (r.checkpoint_sha256 = 'xyz')],
    ['non-integer width', (r) => (r.image.width = 10.5)],
  ];
  it.each(broken)('rejects %s', (_name, mutate) => {
    const response = clone(predictResponse());
    mutate(response);
    expect(() => parsePredictResponse(response)).toThrow();
  });

  it('rejects non-objects and non-finite values', () => {
    expect(() => parsePredictResponse(null)).toThrow();
    expect(() => parsePredictResponse('{}')).toThrow();
    const response = clone(predictResponse()) as unknown as { detections: Array<{ score: unknown }> };
    response.detections[0].score = 'NaN';
    expect(() => parsePredictResponse(response)).toThrow();
  });

  it('enforces the polygon vertex limit', () => {
    expect(() => parsePredictResponse(clone(predictResponse()), 3)).toThrow(/more than 3/);
  });
});

describe('parseHealthResponse', () => {
  it('parses limits and keeps https source links only', () => {
    expect(parseHealthResponse(clone(health)).source_url).toBe('https://example.org/src');
    expect(parseHealthResponse({ ...clone(health), source_url: 'javascript:alert(1)' }).source_url).toBeNull();
  });

  it('rejects an inconsistent threshold range', () => {
    const bad = clone(health);
    bad.limits.score_threshold.default = 0.99;
    expect(() => parseHealthResponse(bad)).toThrow();
  });
});

describe('parseErrorBody', () => {
  it('reads our envelope and ignores anything else', () => {
    const body = { schema_version: '1.0.0', error: { code: 'rate_limited', message: 'slow down', request_id: 'r', retry_after_s: 7 } };
    expect(parseErrorBody(body)?.retry_after_s).toBe(7);
    expect(parseErrorBody('<html>Bad gateway</html>')).toBeNull();
    expect(parseErrorBody({ detail: 'Not Found' })).toBeNull();
  });
});
