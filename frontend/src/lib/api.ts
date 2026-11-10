// Client for the ImpactLens inference service. Every failure becomes a typed
// UiError; nothing here ever substitutes sample or cached detections.

import { parseErrorBody, parseHealthResponse, parsePredictResponse, ValidationError } from './validate';
import type { HealthResponse, PredictResponse } from './types';

export type UiErrorKind =
  | 'network' // no HTTP response at all (offline, DNS, CORS refusal)
  | 'cold_start' // the host answered but the app is not up yet (sleeping/starting Space)
  | 'unauthenticated'
  | 'rate_limited'
  | 'busy'
  | 'model_unavailable'
  | 'rejected_input' // 413 / 415 / 422
  | 'server_timeout' // 504 from the service: budget exceeded
  | 'client_timeout' // we stopped waiting
  | 'cancelled'
  | 'server_error'
  | 'invalid_response'
  | 'configuration';

export interface UiError {
  kind: UiErrorKind;
  title: string;
  detail: string;
  status?: number;
  code?: string;
  reasons?: string[];
  retryAfterS?: number;
  requestId?: string;
}

export type Result<T> = { ok: true; value: T } | { ok: false; error: UiError };

export const REASON_TEXT: Record<string, string> = {
  model_loading: 'The service is up and still loading its checkpoint. Try again in a moment.',
  api_key_not_configured: 'The service has no API_KEY secret configured.',
  api_key_too_short: 'The service API_KEY is shorter than 24 characters.',
  model_id_not_configured: 'MODEL_ID is not set on the service.',
  model_not_configured: 'No checkpoint is configured (MODEL_PATH or MODEL_REPO / MODEL_FILENAME / MODEL_REVISION).',
  model_source_ambiguous: 'Both MODEL_PATH and MODEL_REPO are set; choose one.',
  model_source_incomplete: 'MODEL_REPO, MODEL_FILENAME and MODEL_REVISION must all be set.',
  model_revision_not_pinned: 'MODEL_REVISION must be a full 40-character commit SHA.',
  checkpoint_not_found: 'The configured checkpoint file does not exist.',
  checkpoint_download_failed: 'The pinned checkpoint could not be downloaded.',
  checkpoint_hash_mismatch: 'The checkpoint SHA-256 does not match MODEL_SHA256; it was not loaded.',
  unsupported_checkpoint_format: 'The checkpoint is not a supported .pt file.',
  unsupported_task: 'The checkpoint is not an instance-segmentation model.',
  class_mapping_mismatch: 'The checkpoint classes are not the ImpactLens damage taxonomy (e.g. generic COCO weights).',
  smoke_checkpoint_refused: 'The checkpoint is a pipeline smoke-test artifact, not a damage model.',
  device_unavailable: 'The configured DEVICE (e.g. CUDA) is not available on this host.',
  model_load_failed: 'The checkpoint could not be loaded; see the service log.',
};

export interface ClientConfig {
  /** '' means the same origin that served this page. */
  baseUrl: string;
  apiKey: string;
}

/** Validate a user-entered service URL. Returns the normalized URL or an error message. */
export function normalizeBaseUrl(input: string): { url: string } | { error: string } {
  const trimmed = input.trim().replace(/\/+$/, '');
  if (trimmed === '') return { url: '' };
  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    return { error: 'Enter a full URL such as https://owner-impactlens.hf.space' };
  }
  const local = ['localhost', '127.0.0.1', '[::1]'].includes(parsed.hostname);
  if (parsed.protocol !== 'https:' && !(parsed.protocol === 'http:' && local)) {
    return { error: 'Use https:// (plain http is accepted only for localhost), so the service key is not sent in clear text.' };
  }
  if (parsed.username || parsed.password || parsed.search || parsed.hash) {
    return { error: 'The URL must not contain credentials, a query string or a fragment.' };
  }
  return { url: `${parsed.origin}${parsed.pathname.replace(/\/+$/, '')}` };
}

async function readJson(response: Response): Promise<unknown> {
  const text = await response.text();
  if (text.length > 5_000_000) throw new ValidationError('response', 'response body is too large');
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

function retryAfter(response: Response, fromBody?: number): number | undefined {
  if (fromBody !== undefined) return fromBody;
  const header = Number(response.headers.get('retry-after'));
  return Number.isFinite(header) && header >= 0 ? header : undefined;
}

/** Map a non-2xx response to a UiError. */
export async function errorFromResponse(response: Response): Promise<UiError> {
  const body = parseErrorBody(await readJson(response));
  const status = response.status;
  const base = { status, code: body?.code, requestId: body?.request_id ?? response.headers.get('x-request-id') ?? undefined };
  if (!body) {
    // Not our envelope: a proxy or host page. On Hugging Face this is typical while a
    // Space is waking up or rebuilding.
    if ([502, 503, 504].includes(status)) {
      return { ...base, kind: 'cold_start', title: 'Service is not up yet', detail: `The host answered HTTP ${status} without an ImpactLens response. A sleeping or restarting service can take a minute or more to start. Try again shortly.` };
    }
    return { ...base, kind: 'server_error', title: `Unexpected HTTP ${status}`, detail: 'The response did not come from an ImpactLens service. Check the service URL.' };
  }
  switch (body.code) {
    case 'unauthenticated':
      return { ...base, kind: 'unauthenticated', title: 'Service key rejected', detail: 'The service did not accept the key. Check the key on the Model connection page.' };
    case 'rate_limited':
      return { ...base, kind: 'rate_limited', title: 'Too many requests', detail: body.message, retryAfterS: retryAfter(response, body.retry_after_s) };
    case 'server_busy':
      return { ...base, kind: 'busy', title: 'Service at capacity', detail: body.message, retryAfterS: retryAfter(response, body.retry_after_s) };
    case 'model_unavailable':
      if (body.reasons?.includes('model_loading')) {
        return { ...base, kind: 'cold_start', title: 'Model still loading', detail: 'The service is starting and is still loading its checkpoint. No prediction was made; try again shortly.', reasons: body.reasons, retryAfterS: retryAfter(response, body.retry_after_s) };
      }
      return { ...base, kind: 'model_unavailable', title: 'No model loaded', detail: 'The service is running but cannot make predictions, so no result was produced.', reasons: body.reasons };
    case 'inference_timeout':
      return { ...base, kind: 'server_timeout', title: 'Inference timed out', detail: `${body.message} The service may still be busy with that photo; wait before retrying.`, retryAfterS: retryAfter(response, body.retry_after_s) };
    case 'payload_too_large':
    case 'image_pixel_limit_exceeded':
    case 'unsupported_media_type':
    case 'malformed_image':
    case 'image_too_small':
    case 'invalid_request':
    case 'request_timeout':
      return { ...base, kind: 'rejected_input', title: 'Photo not accepted', detail: body.message };
    default:
      return { ...base, kind: 'server_error', title: 'Service error', detail: `${body.message} No result was produced.` };
  }
}

interface FetchOptions {
  timeoutMs: number;
  signal?: AbortSignal;
}

async function guardedFetch(url: string, init: RequestInit, { timeoutMs, signal }: FetchOptions): Promise<Result<Response>> {
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const onAbort = () => controller.abort();
  signal?.addEventListener('abort', onAbort);
  try {
    const response = await fetch(url, { ...init, signal: controller.signal, credentials: 'omit', cache: 'no-store', referrerPolicy: 'no-referrer' });
    return { ok: true, value: response };
  } catch {
    if (timedOut) {
      return { ok: false, error: { kind: 'client_timeout', title: 'No answer in time', detail: `Stopped waiting after ${Math.round(timeoutMs / 1000)} s. A service that was asleep can take longer on its first request. If a prediction was already running, the service may still finish it; its result is discarded.` } };
    }
    if (signal?.aborted) {
      return { ok: false, error: { kind: 'cancelled', title: 'Cancelled', detail: 'The request was cancelled. A prediction already running on the service is not interrupted; its result is discarded.' } };
    }
    return { ok: false, error: { kind: 'network', title: 'Service unreachable', detail: 'No response. Check the URL and your connection. If the service is on another origin, it must list this page in ALLOWED_ORIGINS (CORS).' } };
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', onAbort);
  }
}

function endpoint(config: ClientConfig, path: string): string {
  return `${config.baseUrl}${path}`;
}

function authHeaders(config: ClientConfig): Record<string, string> {
  return config.apiKey ? { Authorization: `Bearer ${config.apiKey}` } : {};
}

export async function fetchHealth(config: ClientConfig, timeoutMs = 15_000, signal?: AbortSignal): Promise<Result<HealthResponse>> {
  const result = await guardedFetch(endpoint(config, '/health'), { method: 'GET' }, { timeoutMs, signal });
  if (!result.ok) {
    if (result.error.kind === 'client_timeout') {
      return { ok: false, error: { ...result.error, kind: 'cold_start', title: 'Service is slow to answer', detail: 'The health check did not answer within 15 s. A service that was asleep may be starting; try again in a minute.' } };
    }
    return result;
  }
  const response = result.value;
  if (!response.ok) return { ok: false, error: await errorFromResponse(response) };
  try {
    return { ok: true, value: parseHealthResponse(await readJson(response)) };
  } catch (error) {
    return { ok: false, error: invalid(error) };
  }
}

export async function checkAuth(config: ClientConfig, signal?: AbortSignal): Promise<Result<true>> {
  if (!config.apiKey) return { ok: false, error: { kind: 'unauthenticated', title: 'No service key', detail: 'Enter the service key to run predictions.' } };
  const result = await guardedFetch(endpoint(config, '/auth/check'), { method: 'GET', headers: authHeaders(config) }, { timeoutMs: 15_000, signal });
  if (!result.ok) return result;
  if (!result.value.ok) return { ok: false, error: await errorFromResponse(result.value) };
  return { ok: true, value: true };
}

function invalid(error: unknown): UiError {
  const detail = error instanceof ValidationError ? error.message : 'The response could not be read.';
  return { kind: 'invalid_response', title: 'Invalid response from service', detail: `${detail} Nothing from this response is shown.` };
}

export interface PredictRequest {
  file: Blob;
  fileName: string;
  scoreThreshold: number;
  timeoutMs: number;
  maxPolygonPoints: number;
  signal?: AbortSignal;
}

export async function predict(config: ClientConfig, request: PredictRequest): Promise<Result<PredictResponse>> {
  if (!config.apiKey) return { ok: false, error: { kind: 'unauthenticated', title: 'No service key', detail: 'Enter the service key on the Model connection page first.' } };
  const form = new FormData();
  form.append('image', request.file, request.fileName);
  form.append('score_threshold', request.scoreThreshold.toFixed(3));
  const result = await guardedFetch(
    endpoint(config, '/predict'),
    { method: 'POST', headers: authHeaders(config), body: form },
    { timeoutMs: request.timeoutMs, signal: request.signal },
  );
  if (!result.ok) return result;
  const response = result.value;
  if (!response.ok) return { ok: false, error: await errorFromResponse(response) };
  try {
    return { ok: true, value: parsePredictResponse(await readJson(response), request.maxPolygonPoints) };
  } catch (error) {
    return { ok: false, error: invalid(error) };
  }
}
