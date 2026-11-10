# Security, privacy and operations

## Scope of the current controls

The baseline is built for a **developer pilot**: one operator, a few trusted
reviewers, one shared service key. It is not yet suitable as a broad public
product. What a public product additionally needs is listed at the end.

## Threats and controls

| Threat | Control in the code | Test |
|---|---|---|
| Unauthenticated use of the model | `API_KEY` required (≥ 24 chars) for `/predict` and `/auth/check`; constant-time comparison; missing key → service not ready | `test_predict_requires_a_valid_key`, config tests |
| Key guessing / abuse | Per-client sliding-window rate limit, applied *before* authentication (failed attempts count) | `test_failed_authentication_attempts_are_rate_limited_too` |
| Memory exhaustion via uploads | Declared and **received** byte limits; multipart limited to one file + one field, non-file fields capped at 1 KB; body read timeout; capacity check before the body is read | size / streaming / slow-upload tests |
| Decompression bombs | Pixel budget checked from the header before decoding; Pillow's own bomb limit set to the same value and warnings treated as errors | `test_decompression_bomb_is_rejected_by_pixel_count` |
| Malicious file types | Type detected from magic bytes; only JPEG/PNG/WebP codecs enabled for decoding; first frame only | media-type tests |
| Server-side request forgery | No URL input exists; unknown form fields are rejected | `test_unknown_form_field_is_rejected` |
| Pickle code execution from checkpoints | Only the operator-configured checkpoint is ever loaded; SHA-256 checked against `MODEL_SHA256` **before** deserialization; Hub revisions must be full commit SHAs; visitors can never supply a model | `test_hash_mismatch_is_detected_before_any_model_code_runs` |
| Wrong or test model in production | Task and class-name checks; smoke checkpoints refused; generic COCO refused | integration tests with real weights |
| Leaking internals in errors | Fixed error envelope; exception text only in server logs | `test_inference_error_is_500_without_leaking_details_or_detections` |
| Cross-site use from the browser | CORS off unless `ALLOWED_ORIGINS` lists exact origins (`*` refused); no credentials mode | CORS tests |
| XSS / clickjacking | CSP without `unsafe-inline`/`unsafe-eval`, `frame-ancestors 'none'` (or Hugging Face only inside a Space), `X-Frame-Options`, `nosniff`, `Referrer-Policy: no-referrer`, `Permissions-Policy`; the UI renders all text as text | header tests; Playwright asserts no CSP violations |
| Secrets in source or images | Secrets only from the environment; `.env` ignored by git and Docker; `repr` hides them; no secret at build time; Space packaging aborts on credential-like strings | `test_secrets_never_appear_in_repr_or_public_summary`; image inspected |
| Key exposure in the browser | Key kept only in tab memory (React state), never in storage, URLs or logs; sent only to the configured https/localhost origin | Playwright checks localStorage |
| Excessive load | One model call at a time, bounded queue, per-request time budget | `tests/test_capacity.py` |

**CORS is not authentication.** It only stops *browsers* on other origins from
reading responses. Any script can still call the API; the key and rate limit
are what protect it.

## Data handling

* Uploads are processed **in memory**: the multipart parser is configured never
  to spool to disk, and decoded pixels are discarded after the response.
* EXIF and all other metadata (GPS position, camera serial numbers,
  timestamps) are dropped by decoding to a bare pixel array. The response echoes
  only the image size, format, orientation tag and SHA-256.
* Nothing is persisted server-side. Logs contain request id, byte count, image
  size, detection count, threshold and timings, never image content, keys or
  client addresses (Uvicorn's access log is disabled).
* The browser keeps photos in tab memory. Optional local saving keeps metadata
  and findings only, in that browser; it is not a team database.
* Exported annotated images are re-encoded from canvas pixels, without metadata.
* Faces and number plates are **not** redacted automatically. Do so before
  sharing photos outside the review team.

## Consent and reuse

Do not reuse customer photos for training without written consent that covers
training and evaluation. Record the consent reference in the manifest's
`license` column. Keep the ability to delete a person's images from future
dataset revisions (manifest rows are keyed by content hash).

## Monitoring (what to watch without logging images)

* Availability: `/ready` status, restarts, startup duration, `model_loading` time.
* Latency: `latency_ms` fields (queue, decode, inference, total) p50/p95 per
  hour; compare with the deployed-hardware benchmark.
* Errors: counts by `error.code` (429 `server_busy` means capacity, not abuse).
* Model drift without images: per-class detection rate, score distribution,
  share of zero-detection responses, image size and orientation mix. Alert on
  shifts against the first weeks of the pilot.
* Review outcomes: from exported reports, the share of model findings rejected or
  relabelled per class and per `meta_*` slice. This is the most direct signal of
  real-world error. Feed reviewed, consented cases into the next annotation
  round.

## Before a broad public launch

* Server-side user accounts and short-lived access tokens instead of a shared
  key; per-user quotas; a distributed rate limiter at a gateway.
* If cases must be stored: access-controlled object storage and a database with
  per-user authorization, encryption at rest, retention and deletion policies,
  and an audit log of who viewed or changed which finding.
* Plate/face redaction, a privacy notice and a data-protection review under the
  applicable law (in India, the Digital Personal Data Protection Act, 2023).
* Dependency and image vulnerability scanning in CI; a key-rotation procedure.
