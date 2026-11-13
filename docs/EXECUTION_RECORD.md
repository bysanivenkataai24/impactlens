# Execution record (2026-11-10)

What was actually run while building this repository, and what was not.
Environment: cloud development container, 4 vCPU (Intel Xeon @ 2.10 GHz),
15.7 GB RAM, **no GPU**, Python 3.12.3 venv from `uv.lock`, Node 22.23.4,
Docker 29.9.2. Network limited to package registries plus a few allowed hosts.

## Not done, and why

| Item | Status | Missing input |
|---|---|---|
| Training a damage model | **Not done** | Licensed damage data (CarDD requires a signed licence request; commercial use needs authorization) and a training GPU. |
| Evaluation / mAP of a damage model | **Not done** | A trained model and a held-out labelled split. No accuracy number exists for ImpactLens. |
| Benchmark of a damage model on deployment hardware | **Not done** | A trained model and the deployed Space. |
| Public deployment of the full service (Docker Space) | **Not done** | Hugging Face answered 402: Docker Spaces need PRO, which the owner's account does not have; also no checkpoint. |
| Public deployment of the review UI only | **Done 2026-11-13** | Static Space `2cooltoslay/impactlens`, https://2cooltoslay-impactlens.static.hf.space, built with `scripts/prepare_static_space.sh` from commit 64a7ef0. Verified in Chromium against the live URL: demo, photo upload, hand-drawn region, PDF export, no page errors. No API or model behind it. |
| Real download from a pinned Hub revision | **Not verified end to end** | The sandbox network blocks the Hub's file-storage hosts (`cas-server.xethub.hf.co`, `us.aws.cdn.hf.co`). Verified instead: branch revisions are refused before any download; a failed download yields `checkpoint_download_failed` and the service keeps serving. |
| Calibration, quality gate, part/severity/cost models | **Not implemented** (by design) | Validated data and protocols; see EVALUATION.md and EXTENSIONS.md. |
| GitHub Actions CI | Workflow written, **not run** on GitHub | Push access to Actions; each step was run locally instead. |
| GPU inference | **Not tested** | No GPU; the lock file pins CPU wheels. |

## Done, with results

### Python (`uv run pytest -q`)

169 passed, 3 skipped. The skips are the opt-in real-weight integration tests;
two of them were run separately (below). Test groups:

* `test_api_contract.py`: HTTP contract with a **synthetic adapter** (a test
  double that "detects" magenta pixels). Auth, threshold validation, media-type
  sniffing, size and pixel limits (including streamed bodies without
  Content-Length), decompression bombs, EXIF orientations 1–8 in PNG and
  JPEG/WebP, coordinate normalization and clamping, detection and vertex caps,
  flags, error envelopes, rate limiting, security headers, CORS, static UI.
* `test_capacity.py`: real concurrent requests: busy → 429, one model call at a
  time, timeout → 504 while the slot stays held until the call returns, slow
  upload → 408, background model loading.
* `test_units.py`: configuration validation, decoding, post-processing,
  security helpers, model store hash checks, extension gating, CLI overrides.
* `test_data_pipeline.py`: manifest provenance, EXIF size conflicts, exact and
  near-duplicates, no leakage group across splits, frozen test set across
  revisions, clean YOLO export, repeat factors, shipped configs valid.
* `test_schema_files.py`: committed JSON Schemas match the Pydantic models.

These tests do **not** measure model quality.

### Real Ultralytics checkpoints (opt-in integration tests)

* Official COCO `yolo11n-seg.pt` (SHA-256 `55ed65c5…dcbe152`) via the production
  loader → `ready: false`, `class_mapping_mismatch`; `/predict` 503. **Passed.**
* Smoke-test checkpoint → `smoke_checkpoint_refused`. **Passed.**
* Trained damage checkpoint → skipped (none exists).

### Pipeline smoke test (`impactlens smoke`)

On synthetic shapes (63 images, 124 instances, 8 negatives, two views per
"incident", an exact duplicate, a re-encoded near-duplicate, an EXIF-6
portrait), final run on the committed code: manifest 71 rows; 1 exact and 1
near-duplicate removed; split train/val/test images 42/14/13 in 24/8/7 groups
with **0** groups spanning splits; YOLO export; 3-epoch CPU training (≈22 s);
validation; artifact export; the service adapter refused the checkpoint
(`smoke_checkpoint_refused`) and the explicit test override loaded it and
returned instances. Mask AP50-95 on the synthetic validation split was 0.16
after 3 epochs and 0.59 after a 25-epoch run: evidence that the training loop
learns, and **nothing** about vehicle damage. Tiled evaluation (`--tile 160`)
also ran.

### Frontend

* `npm run lint` (ESLint with jsx-a11y strict, react-hooks) and `npm run typecheck`: clean.
* `npm test` (Vitest): 65 passed: response validation, API error mapping,
  timeouts and cancel, findings and origin labels, preserved original
  predictions, quote rules, reducer, report shape, local storage, PDF.
* `npm run e2e` (Playwright 1.63.0, Chromium; desktop 1440×900 and Pixel 7):
  **36 passed** against the real FastAPI service with the synthetic adapter:
  EXIF-6 portrait and landscape overlays aligned within 0.006 of the expected
  normalized box (also after resizing), WebP, invalid file rejected, photo
  switching keeps findings, no-detection wording, wrong key, intercepted
  503/502-style/504/429/413/invalid responses, mismatched image hash refused,
  cancel, manual polygon drawing and numeric rectangle entry, editing type /
  part / severity / status / notes / quote, JSON / annotated JPEG (no EXIF) /
  PDF exports, local save without photo bytes or key and re-attachment with
  hash check, demo mode, axe-core WCAG 2.2 AA scan (no serious/critical
  violations), keyboard path, no horizontal scroll, no CSP violations.
* `npm audit --omit=dev`: 0 vulnerabilities. Full `npm audit`: 7 moderate, all
  in the development-only typescript-eslint/minimatch chain (not shipped).

### Python dependency audit

`pip-audit 2.9.0` on the locked runtime set: no known vulnerabilities; torch and
torchvision `+cpu` builds could not be audited (not on PyPI).

### Docker

* `docker build` of the repository and of the `scripts/prepare_space.sh`
  bundle: both succeed (image 2.21 GB, CPU PyTorch).
* Container runs as UID 1000; image config contains no proxy, CA or secret
  values; Docker health check reports healthy.
* Without configuration: `/health` 200 `ready: false` with reasons, `/ready`
  503, `/predict` 401, UI served with the CSP; with `SPACE_ID` set,
  `frame-ancestors 'self' https://huggingface.co`.
* With COCO weights mounted → `class_mapping_mismatch`; with smoke weights →
  `smoke_checkpoint_refused`.
* Real Ultralytics inference inside the image (smoke checkpoint, explicit test
  override): load + warm-up 3.5 s, warm inference p50 ≈ 59 ms on a 1920×1080
  input at 640 px on this CPU.

### Benchmarks (this sandbox only; files in `docs/benchmarks/`)

| Measurement | Result | Caveat |
|---|---|---|
| `benchmark model`, YOLO11n-seg architecture at 640 px, 1920×1080 input | import 1.1 s, load + warm-up 1.5 s, warm p50 50 ms / p95 58 ms, peak RSS ≈ 410 MB | Smoke checkpoint: architecture timing, not a damage model. Not Hugging Face hardware. |
| `benchmark http`, synthetic adapter, 1.4 MB 1920×1080 JPEG | concurrency 1: p50 52 ms, ≈19 req/s; 2: p50 100 ms; 4: p50 207 ms; 8: 34 of 40 → 429 `server_busy` | Measures service overhead (upload, decode, hashing, contract) and the bounded queue, with no model. |

## Reproduce

```bash
uv sync --frozen && uv run pytest -q
```

```bash
cd frontend && npm ci && npm run lint && npm run typecheck && npm test && npm run e2e
```

```bash
uv run impactlens smoke
```

```bash
docker build -t impactlens . && docker run --rm -p 7860:7860 impactlens
```
