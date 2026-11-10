# Inference API (schema 1.0.0)

Machine-readable contract: [`schemas/`](../schemas) (JSON Schema, generated from
`src/impactlens/schemas.py`; a test fails if they drift) and `GET /openapi.json`.

Interactive docs pages (`/docs`, `/redoc`) are disabled: they load scripts from a
CDN, which the service's Content-Security-Policy forbids.

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | none | Liveness + readiness detail. Always 200 while the process serves. |
| GET | `/ready` | none | Same body; **503** unless real inference is possible. Use for readiness probes. |
| GET | `/auth/check` | key | Lets a client verify its key without sending a photo. |
| POST | `/predict` | key | One photo in, regions for review out. |
| GET | `/` and static files | none | The review UI (when built). |

Authentication: `Authorization: Bearer <API_KEY>` or `X-API-Key: <API_KEY>`.
Comparison is constant-time. For a *private* Hugging Face Space, programmatic
clients must also send their Hugging Face token in `Authorization`, so they
should send the ImpactLens key in `X-API-Key`.

## `GET /health`

```json
{
  "schema_version": "1.0.0",
  "status": "ok",
  "ready": false,
  "reasons": ["model_not_configured"],
  "model_id": null,
  "model_revision": null,
  "checkpoint_sha256": null,
  "task": null,
  "classes": [],
  "device": null,
  "limits": {
    "max_upload_bytes": 12582912, "max_image_pixels": 50000000, "min_image_side": 32,
    "accepted_media_types": ["image/jpeg", "image/png", "image/webp"],
    "score_threshold": {"default": 0.25, "min": 0.05, "max": 0.95},
    "max_detections": 50, "max_polygon_points": 200, "inference_timeout_s": 60.0
  },
  "service_version": "0.1.0",
  "source_url": null
}
```

`ready` is true only when an API key, a model id and a checkpoint are configured,
the checkpoint passed its hash check, is a `segment` model whose class names are
exactly the six supported types, and warm-up inference succeeded.

| Reason | Meaning |
|---|---|
| `model_loading` | Process is up; checkpoint still downloading / loading / warming up. |
| `api_key_not_configured`, `api_key_too_short` | `API_KEY` missing or shorter than 24 characters. |
| `model_id_not_configured` | `MODEL_ID` missing. |
| `model_not_configured`, `model_source_ambiguous`, `model_source_incomplete` | Set exactly one of `MODEL_PATH` or `MODEL_REPO`+`MODEL_FILENAME`+`MODEL_REVISION`. |
| `model_revision_not_pinned` | `MODEL_REVISION` is not a 40-character commit SHA. |
| `checkpoint_not_found`, `checkpoint_download_failed` | File missing / download failed. |
| `checkpoint_hash_mismatch` | SHA-256 differs from `MODEL_SHA256`; file was **not** deserialized. |
| `unsupported_checkpoint_format`, `unsupported_task` | Not a `.pt` segmentation checkpoint. |
| `class_mapping_mismatch` | Classes are not the taxonomy (e.g. generic COCO weights). |
| `smoke_checkpoint_refused` | Artifact of `impactlens smoke`, not a damage model. |
| `device_unavailable` | `DEVICE=cuda` (etc.) requested but not present. No silent CPU fallback. |
| `model_load_failed` | Anything else; details in the server log only. |

## `POST /predict`

`multipart/form-data` with exactly these fields:

| Field | Required | Notes |
|---|---|---|
| `image` | yes | One JPEG, PNG or WebP file. The type is detected from the bytes; name and declared type are ignored. |
| `score_threshold` | no | Finite number within `limits.score_threshold.min..max`. Default `limits.score_threshold.default`. |

No other fields are accepted (there is deliberately no URL field: the server
never fetches remote content).

```bash
curl -sS -X POST "$URL/predict" \
  -H "Authorization: Bearer $API_KEY" \
  -F image=@photo.jpg -F score_threshold=0.3
```

Response (abridged):

```json
{
  "schema_version": "1.0.0",
  "request_id": "4c1f...",
  "model_id": "impactlens-yolo11n-seg-v1",
  "model_revision": "0d6f...40 hex",
  "checkpoint_sha256": "9a1b...64 hex",
  "image": {"width": 3024, "height": 4032, "format": "JPEG", "exif_orientation": 6,
            "orientation_applied": true, "sha256": "..."},
  "score_threshold": 0.3,
  "outcome": "detections_returned",
  "message": "2 region(s) returned for human review.",
  "detections": [
    {
      "id": "det-1",
      "damage_type": "dent",
      "score": 0.71,
      "bbox": [0.412, 0.533, 0.587, 0.648],
      "polygon": [[0.415, 0.541], [0.46, 0.534], "..."],
      "part": null,
      "severity": "unknown",
      "flags": ["small_region"],
      "review_status": "pending_review"
    }
  ],
  "warnings": [],
  "latency_ms": {"queue": 0.0, "decode": 41.2, "inference": 182.5, "postprocess": 3.1, "total": 231.0},
  "limitations": ["Decision support only. ...", "..."]
}
```

### Coordinates

* Normalized to `[0, 1]` against the **EXIF-oriented** image (the one browsers
  display), origin top-left, x right, y down. `image.width/height` are that
  image's pixel size.
* `bbox` = `[x_min, y_min, x_max, y_max]`, strictly `min < max`.
* `polygon` = largest connected outline of the predicted mask, 3–`max_polygon_points`
  vertices, closed ring without repeating the first vertex. All values finite and in range.
* Ordered by `score` descending; at most `max_detections`.

### Fields that are fixed in schema 1.x

* `part` is always `null` and `severity` always `"unknown"`: there is no validated
  part or severity model. Reviewers enter both in the UI.
* `score` is an **uncalibrated model score**, not a probability.
* `review_status` is always `"pending_review"`.

### Flags (geometric facts, not quality judgements)

| Flag | Meaning |
|---|---|
| `mask_fragmented` | Mask had several pieces; only the largest is returned. |
| `polygon_simplified` | Outline exceeded the vertex cap and was simplified. |
| `touches_image_border` | Region reaches the photo edge; damage may continue out of frame. |
| `small_region` | Region < 0.1 % of the photo. |

Response `warnings`: `detection_cap_reached` (lower-scoring regions may be
missing), `invalid_detections_dropped` (the runtime returned non-finite or
degenerate geometry, or a class outside the taxonomy, which was discarded).

### Zero detections

`outcome: "no_supported_region_returned"`, `detections: []`, and the message
*"No supported damage region was returned for this photo at the requested score
threshold. This does not mean the vehicle is undamaged."*

## Errors

Every error uses one envelope, and never contains detections:

```json
{"schema_version": "1.0.0",
 "error": {"code": "server_busy", "message": "...", "request_id": "...", "retry_after_s": 15}}
```

| HTTP | `code` | When |
|---|---|---|
| 401 | `unauthenticated` | Missing / wrong key (`WWW-Authenticate: Bearer`). |
| 408 | `request_timeout` | Upload body not received within `REQUEST_BODY_TIMEOUT_S`. |
| 413 | `payload_too_large` | More than `MAX_UPLOAD_BYTES` (checked on declared length **and** received bytes). |
| 413 | `image_pixel_limit_exceeded` | Header dimensions exceed `MAX_IMAGE_PIXELS` (checked before decompression). |
| 415 | `unsupported_media_type` | Not multipart, or the bytes are not JPEG/PNG/WebP. |
| 422 | `malformed_image` | Empty, truncated or corrupt image. |
| 422 | `image_too_small` | Shorter side below `MIN_IMAGE_SIDE`. |
| 422 | `invalid_request` | Bad threshold, missing/duplicate `image`, unknown field, malformed multipart. |
| 429 | `rate_limited` | Per-client limit (`Retry-After`). |
| 429 | `server_busy` | One inference running and `MAX_PENDING_REQUESTS` waiting (`Retry-After`). |
| 503 | `model_unavailable` | Not ready; `reasons` as in `/health`. `Retry-After` while `model_loading`. |
| 504 | `inference_timeout` | Queue + inference exceeded `INFERENCE_TIMEOUT_S`. |
| 500 | `inference_failed` / `internal_error` | Model or server error; details only in the log. |
| 404 / 405 | `not_found` / `method_not_allowed` | Unknown path / wrong method. |

Checks run in this order: rate limit → authentication → readiness → media type →
declared size → capacity → bounded body read → decode → inference.

## Concurrency, timeouts and cancellation

* One model call runs at a time on a dedicated worker thread; up to
  `MAX_PENDING_REQUESTS` wait; more are refused with 429 before their body is read.
* The time budget covers queueing and inference. On timeout the client gets 504.
* **A timed-out or disconnected request does not stop a model call that is
  already running** (CPU/GPU kernels cannot be interrupted from Python). The
  slot stays occupied until that call returns, so a retry storm cannot stack
  model calls. This is tested in `tests/test_capacity.py`.

## Versioning

`schema_version` is semver. Patch: wording only. Minor: new optional fields or
new codes. Major: anything a strict client could break on. The UI refuses any
major version other than 1.
