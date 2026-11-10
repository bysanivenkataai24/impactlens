# ImpactLens

Decision support for vehicle inspection. A reviewer uploads photos; an
instance-segmentation model **suggests** visible damage regions (dent, scratch,
crack, glass shatter, broken lamp, flat tyre); the reviewer confirms, corrects,
rejects or adds regions and exports an assessment record.

ImpactLens does not decide claims, fault, roadworthiness, hidden damage, part,
severity or cost. Every model region starts as *pending review*, scores are
uncalibrated, and an empty result never means "undamaged".

## Status

| Part | State |
|---|---|
| Inference service (FastAPI, versioned contract, auth, limits, EXIF handling) | Working, tested |
| Review UI (TypeScript/React: overlays, manual regions, review, JSON/JPEG/PDF export) | Working, tested on desktop and mobile Chromium |
| Data pipeline (manifest, de-duplication, leakage-free group splits, label export) | Working, tested |
| Train / evaluate / robustness / export / benchmark commands | Working; executed end to end on synthetic shapes only |
| Docker image (port 7860, non-root) and Hugging Face Space bundle | Built and run locally |
| **Trained damage model** | **Does not exist yet.** Needs licensed data and a GPU run. |
| **Public deployment** | **Not done.** Needs the owner's Hugging Face account and a trained model. |

Without a model the service runs, reports `ready: false` with reasons, answers
`/predict` with 503, and the UI still supports manual review, export and an
illustrative demo. Nothing ever fills in fake detections.

Details: [docs/EXECUTION_RECORD.md](docs/EXECUTION_RECORD.md).

## Quick start (local)

Requirements: Python 3.12 with [uv](https://docs.astral.sh/uv/), Node 22, optionally Docker.

```bash
uv sync --frozen
```

```bash
cd frontend && npm ci && npm run build && cd ..
```

```bash
API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" uv run impactlens serve
```

Open <http://127.0.0.1:7860>. Without `MODEL_ID` and a checkpoint, the pill
reads "Service up, no model": use **Open illustrative demo** or add photos and
draw regions by hand. With a trained artifact from `impactlens export`:

```bash
set -a && . models/<model_id>/service.env && set +a && API_KEY=<key> uv run impactlens serve
```

Docker (same behaviour, port 7860):

```bash
docker build -t impactlens .
```

```bash
docker run --rm -p 7860:7860 --env-file .env impactlens
```

`.env.example` documents every variable.

## Configuration (environment variables)

| Variable | Purpose |
|---|---|
| `API_KEY` (secret) | Shared developer-pilot key, ≥ 24 characters. |
| `MODEL_ID` | Name reported with every prediction. |
| `MODEL_PATH` **or** `MODEL_REPO` + `MODEL_FILENAME` + `MODEL_REVISION` | Checkpoint source. Hub revisions must be full commit SHAs. |
| `MODEL_SHA256` | Expected checkpoint hash, checked before the file is deserialized. |
| `HF_TOKEN` (secret) | Read token for a private model repository. |
| `DEVICE` | `cpu` (default), `cuda`, `cuda:N`, `mps`. No silent fallback. |
| `ALLOWED_ORIGINS` | Exact browser origins for cross-origin use; empty = same origin only. |
| `DEFAULT_SCORE_THRESHOLD`, `INFERENCE_IMAGE_SIZE`, `NMS_IOU` | From the validation-selected operating point (`service.env`). |
| `MAX_UPLOAD_BYTES`, `MAX_IMAGE_PIXELS`, `INFERENCE_TIMEOUT_S`, `MAX_PENDING_REQUESTS`, `RATE_LIMIT_PER_MINUTE` | Limits. |
| `SOURCE_URL` | Public source link (AGPL-3.0 §13), shown in the UI. |

## Commands

```text
impactlens serve                       run the service (env configuration)
impactlens data manifest|dedup|split|stats|export-yolo
impactlens train --config configs/train/<experiment>.yaml
impactlens evaluate --weights ... --data ... --split val|test
impactlens robustness --weights ... --data ... --score-threshold ...
impactlens export --run runs/<exp>/<run> --model-id <id>
impactlens benchmark model|http ...
impactlens smoke                       pipeline mechanics on synthetic shapes (not a validation)
impactlens schema [--check]            write / verify the JSON Schema files
```

Full workflow: [docs/TRAINING.md](docs/TRAINING.md).

## Tests

```bash
uv run pytest -q
```

```bash
cd frontend && npm run lint && npm run typecheck && npm test && npm run e2e
```

Contract and browser tests use a **synthetic adapter** (a test double that
"detects" magenta pixels) and say nothing about model quality. Opt-in
real-weight tests: set `IMPACTLENS_TEST_COCO_WEIGHTS`,
`IMPACTLENS_TEST_SMOKE_WEIGHTS` or `IMPACTLENS_TEST_WEIGHTS` +
`IMPACTLENS_TEST_IMAGES` (see `tests/test_integration_weights.py`).

## Repository layout

```text
src/impactlens/
  adapters/      model runtimes behind one protocol (Ultralytics YOLO segmentation)
  service/       FastAPI app, bounded inference runtime, security middleware
  data/          manifest, de-duplication, group splits, YOLO export, smoke data
  training/      train, evaluate, robustness, export, benchmark, smoke
  schemas.py     versioned response contract (-> schemas/*.json)
  imaging.py     safe decoding (sniffing, pixel budget, EXIF orientation)
  postprocess.py normalization, validation and caps for detections
  extensions.py  evidence-gated interfaces for part / severity / cost
frontend/        TypeScript/React review UI, Vitest and Playwright tests
configs/train/   baseline, ablation and alternative experiment configs
deploy/, scripts/ Hugging Face Space card and packaging script
docs/            guides, cards, execution record, measured benchmarks
```

## Documentation

* [API contract](docs/API.md) · [Deployment (Hugging Face Spaces and other hosts)](docs/DEPLOYMENT.md)
* [Model card](docs/MODEL_CARD.md) · [Dataset card](docs/DATASET_CARD.md) · [Annotation guide](docs/ANNOTATION_GUIDE.md)
* [Training](docs/TRAINING.md) · [Evaluation and release criteria](docs/EVALUATION.md)
* [Security, privacy and operations](docs/SECURITY_PRIVACY.md) · [Extensions: parts, severity, cost](docs/EXTENSIONS.md)
* [Licensing](docs/LICENSING.md) · [Execution record](docs/EXECUTION_RECORD.md)

## Next action to enable real predictions

1. Request CarDD through its licensing form (and, for any commercial or public
   use, the PIC Lab's written authorization), and collect consented intact-vehicle
   and local photos as negatives and evaluation data.
2. Build the manifest, split and export it, then run
   `impactlens train --config configs/train/baseline_yolo11n_640.yaml` on a GPU.
3. Evaluate on validation, export the artifact, publish it to a pinned Hub
   revision and follow [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Licence

AGPL-3.0 (see `LICENSE` and [docs/LICENSING.md](docs/LICENSING.md)): the model
runtime, Ultralytics, is AGPL-3.0. Dataset licences, notably CarDD's, impose
further restrictions on data and trained weights.
