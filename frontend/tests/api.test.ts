import { afterEach, describe, expect, it, vi } from 'vitest';

import { errorFromResponse, fetchHealth, normalizeBaseUrl, predict } from '../src/lib/api';
import { predictResponse } from './fixtures';

const envelope = (code: string, extra: Record<string, unknown> = {}) =>
  JSON.stringify({ schema_version: '1.0.0', error: { code, message: `message for ${code}`, request_id: 'rid', ...extra } });

const json = (body: string, status: number, headers: Record<string, string> = {}) =>
  new Response(body, { status, headers: { 'content-type': 'application/json', ...headers } });

describe('errorFromResponse', () => {
  it.each([
    [401, 'unauthenticated', 'unauthenticated'],
    [429, 'rate_limited', 'rate_limited'],
    [429, 'server_busy', 'busy'],
    [503, 'model_unavailable', 'model_unavailable'],
    [504, 'inference_timeout', 'server_timeout'],
    [413, 'payload_too_large', 'rejected_input'],
    [415, 'unsupported_media_type', 'rejected_input'],
    [422, 'malformed_image', 'rejected_input'],
    [500, 'inference_failed', 'server_error'],
  ])('HTTP %i %s -> %s', async (status, code, kind) => {
    const error = await errorFromResponse(json(envelope(code, { reasons: ['model_not_configured'] }), status));
    expect(error.kind).toBe(kind);
    expect(error.requestId).toBe('rid');
  });

  it('treats a model that is still loading as a cold start, not a failure', async () => {
    const error = await errorFromResponse(json(envelope('model_unavailable', { reasons: ['model_loading'], retry_after_s: 10 }), 503));
    expect(error.kind).toBe('cold_start');
    expect(error.retryAfterS).toBe(10);
  });

  it('treats a proxy page during start-up as a cold start', async () => {
    const error = await errorFromResponse(new Response('<html>Starting</html>', { status: 503 }));
    expect(error.kind).toBe('cold_start');
  });

  it('takes Retry-After from the header when the body lacks it', async () => {
    const error = await errorFromResponse(json(envelope('server_busy'), 429, { 'retry-after': '12' }));
    expect(error.retryAfterS).toBe(12);
  });
});

describe('normalizeBaseUrl', () => {
  it('accepts same-origin, https and localhost http', () => {
    expect(normalizeBaseUrl('')).toEqual({ url: '' });
    expect(normalizeBaseUrl('https://owner-impactlens.hf.space/')).toEqual({ url: 'https://owner-impactlens.hf.space' });
    expect(normalizeBaseUrl('http://localhost:7860')).toEqual({ url: 'http://localhost:7860' });
  });
  it.each(['http://example.org', 'ftp://x.org', 'https://user:pw@x.org', 'https://x.org/?k=1', 'not a url'])('rejects %s', (input) => {
    expect('error' in normalizeBaseUrl(input)).toBe(true);
  });
});

describe('predict', () => {
  afterEach(() => vi.unstubAllGlobals());
  const request = { file: new Blob(['x']), fileName: 'a.jpg', scoreThreshold: 0.3, timeoutMs: 1000, maxPolygonPoints: 200 };
  const config = { baseUrl: 'https://svc.example', apiKey: 'k'.repeat(30) };

  it('sends the key and threshold, and returns validated detections', async () => {
    const fetchMock = vi.fn(async () => json(JSON.stringify(predictResponse()), 200));
    vi.stubGlobal('fetch', fetchMock);
    const result = await predict(config, request);
    expect(result.ok && result.value.detections).toHaveLength(1);
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('https://svc.example/predict');
    expect((init.headers as Record<string, string>).Authorization).toBe(`Bearer ${config.apiKey}`);
    expect((init.body as FormData).get('score_threshold')).toBe('0.300');
    expect(init.credentials).toBe('omit');
  });

  it('never shows detections from an invalid response', async () => {
    const bad = predictResponse();
    bad.detections[0].bbox = [0, 0, 2, 2];
    vi.stubGlobal('fetch', vi.fn(async () => json(JSON.stringify(bad), 200)));
    const result = await predict(config, request);
    expect(result.ok).toBe(false);
    expect(!result.ok && result.error.kind).toBe('invalid_response');
  });

  it('reports a non-JSON success body as invalid', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('<html>ok</html>', { status: 200 })));
    const result = await predict(config, request);
    expect(!result.ok && result.error.kind).toBe('invalid_response');
  });

  it('maps a network failure', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => Promise.reject(new TypeError('Failed to fetch'))));
    const result = await predict(config, request);
    expect(!result.ok && result.error.kind).toBe('network');
  });

  it('times out on the client', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn((_url: string, init: RequestInit) => new Promise((_resolve, reject) => init.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError'))))),
    );
    const result = await predict(config, { ...request, timeoutMs: 20 });
    expect(!result.ok && result.error.kind).toBe('client_timeout');
  });

  it('distinguishes a user cancel from a timeout', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn((_url: string, init: RequestInit) => new Promise((_resolve, reject) => init.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError'))))),
    );
    const controller = new AbortController();
    const pending = predict(config, { ...request, timeoutMs: 5000, signal: controller.signal });
    controller.abort();
    const result = await pending;
    expect(!result.ok && result.error.kind).toBe('cancelled');
  });

  it('refuses to call the service without a key', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    const result = await predict({ ...config, apiKey: '' }, request);
    expect(!result.ok && result.error.kind).toBe('unauthenticated');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('health never sends the key', async () => {
    const fetchMock = vi.fn(async () => new Response('nope', { status: 502 }));
    vi.stubGlobal('fetch', fetchMock);
    const result = await fetchHealth(config);
    expect(!result.ok && result.error.kind).toBe('cold_start');
    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(init.headers).toBeUndefined();
  });
});
