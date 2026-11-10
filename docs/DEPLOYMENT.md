# Deployment guide

ImpactLens ships as **one container**: the FastAPI service serves the review UI
and the API on port 7860. The same image runs on Hugging Face Docker Spaces or
any container host; only environment variables change.

> Status (2026-11-10): no public deployment was made from this repository. That
> needs the owner's Hugging Face account, a plan that allows Docker Spaces and a
> trained ImpactLens checkpoint, none of which were available to the build. The
> image was built and run locally; see [EXECUTION_RECORD.md](EXECUTION_RECORD.md).

## 0. Before you start

| Need | Why |
|---|---|
| A trained, evaluated checkpoint (`models/<id>/` from `impactlens export`) | Without it the service runs but `/predict` is 503. Generic COCO weights are refused. |
| A Hugging Face account on a plan that can create Docker Spaces | Per the Spaces docs (checked 2026-11-10), creating a Docker Space requires a paid plan (PRO for personal accounts, Team/Enterprise for organizations). CPU Basic hardware itself has no hourly cost. |
| Written permission covering public use of models trained on licensed data | CarDD's licence restricts commercial use and redistribution; see [DATASET_CARD.md](DATASET_CARD.md). |
| `hf` CLI (installed with this project: `uv run hf ...`) and `hf auth login` | Commands below. |

**Free hardware sleeps** when unused and the first request after that waits for
a cold start (the UI shows this). There is no free always-on hosting; paid
hardware can disable sleep (`hf spaces settings --sleep-time -1`).

## 1. Publish the checkpoint to a pinned model repository

```bash
uv run hf repos create <owner>/impactlens-damage-model --type model --private
```

```bash
uv run hf upload <owner>/impactlens-damage-model models/<model_id>/<model_id>.pt <model_id>.pt --commit-message "Add <model_id>"
```

```bash
uv run hf upload <owner>/impactlens-damage-model models/<model_id>/model_manifest.json model_manifest.json
```

Read the commit SHA that now contains the file. This is `MODEL_REVISION`:

```bash
uv run hf models info <owner>/impactlens-damage-model --expand sha --format json
```

`MODEL_SHA256` is in `models/<model_id>/SHA256SUMS`. The service hashes the file
and refuses it on mismatch **before** deserializing it (a `.pt` file is a pickle).

For a private model repository, create a *fine-grained, read-only* token limited
to that repository (Hugging Face settings → Access Tokens). It becomes the
`HF_TOKEN` secret.

## 2. Create the Space

```bash
uv run hf repos create <owner>/impactlens --type space --sdk docker --public
```

Use `--protected` instead of `--public` to keep the source private while the app
stays reachable. ImpactLens is AGPL-3.0, so users of the running app must still
be able to obtain its source: set `SOURCE_URL` (below) to a public copy.

Space URL: `https://<owner>-impactlens.hf.space` (the `SPACE_HOST` the platform
injects; lower-cased, `/` replaced by `-`).

## 3. Secrets and variables

Secrets (encrypted; never in files, images, the UI or logs). `--secrets NAME`
reads the value from your shell environment, so it stays out of shell history:

```bash
export API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

```bash
uv run hf spaces secrets add <owner>/impactlens --secrets API_KEY
```

```bash
uv run hf spaces secrets add <owner>/impactlens --secrets HF_TOKEN
```

Variables (public configuration). Write them to a file and upload it:

```bash
cat > space.env <<'EOF'
MODEL_ID=<model_id>
MODEL_REPO=<owner>/impactlens-damage-model
MODEL_FILENAME=<model_id>.pt
MODEL_REVISION=<40-character commit sha from step 1>
MODEL_SHA256=<64-character sha256 from SHA256SUMS>
DEVICE=cpu
DEFAULT_SCORE_THRESHOLD=<value from models/<model_id>/service.env>
INFERENCE_IMAGE_SIZE=<value from models/<model_id>/service.env>
SOURCE_URL=https://github.com/<owner>/impactlens
EOF
```

```bash
uv run hf spaces variables add <owner>/impactlens --env-file space.env
```

Leave `ALLOWED_ORIGINS` empty when the UI is used from the Space itself.

## 4. Choose hardware

```bash
uv run hf spaces hardware
```

```bash
uv run hf spaces settings <owner>/impactlens --hardware cpu-basic
```

CPU Basic is 2 vCPU / 16 GB. The baseline (YOLO11n-seg, CPU PyTorch) runs there;
measure before deciding (step 8). The lock file pins **CPU** PyTorch wheels; a
GPU deployment needs a CUDA build (see "Other hosts").

## 5. Upload and build

```bash
scripts/prepare_space.sh build/space
```

```bash
uv run hf upload spaces/<owner>/impactlens build/space . --commit-message "Deploy $(git rev-parse --short HEAD)"
```

`prepare_space.sh` copies only what the Docker build uses and writes the Space
card (`README.md` with `sdk: docker`, `app_port: 7860`) from
`deploy/huggingface/SPACE_README.md`. It refuses to run into a non-empty
directory and aborts if anything looks like a credential.

```bash
uv run hf spaces wait <owner>/impactlens --timeout 30m
```

```bash
uv run hf spaces logs <owner>/impactlens
```

## 6. Verify readiness

```bash
export URL=https://<owner>-impactlens.hf.space
```

```bash
curl -sS "$URL/health"
```

Check: `"ready": true`, `model_id`, `model_revision` and `checkpoint_sha256`
equal what you published, `classes` are the six damage types. Right after a
restart `reasons` may be `["model_loading"]` for a while; anything else is a
configuration problem (see the table in [API.md](API.md#get-health)).

## 7. Run a real test photo and connect the UI

```bash
curl -sS -X POST "$URL/predict" -H "Authorization: Bearer $API_KEY" -F image=@/path/to/real-vehicle-photo.jpg
```

Then open `$URL`, go to **Model connection**, leave the URL empty (same
origin), paste the key and press **Check connection**. Upload a photo, press
**Run model on this photo**, review the findings and export a report.

The key lives only in that tab's memory. Share it only with pilot reviewers;
rotate it by changing the secret (the Space restarts).

## 8. Measure on the deployed hardware

```bash
uv run impactlens benchmark http --url "$URL" --image /path/to/real-vehicle-photo.jpg --requests 30 --concurrency 1 2 4 --out runs/bench/space-cpu-basic.json
```

It reads the key from `API_KEY` (never from the command line). Set acceptance
targets (p95 latency, error rate at expected concurrency) from this baseline;
see [EVALUATION.md](EVALUATION.md#serving-benchmarks).

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Space stuck "Building"/`BUILD_ERROR` | `hf spaces logs <space>`; a missing file usually means the Space was not assembled with `prepare_space.sh`. |
| Space "Starting" for a long time, then `RUNTIME_ERROR` | `startup_duration_timeout` is 30 m in the card. The service itself listens within seconds; check logs for a crash at import. |
| UI loads, pill says "Service up, no model" | `/health` `reasons` tell you which variable is missing or wrong. |
| `model_revision_not_pinned` | Use the full 40-character commit SHA, not `main` or a tag. |
| `checkpoint_download_failed` | Wrong repo/filename/revision, private repo without `HF_TOKEN`, or token lacks read access to that repo. |
| `checkpoint_hash_mismatch` | The file at that revision is not the one you exported. Do not "fix" by removing `MODEL_SHA256`. |
| `class_mapping_mismatch` | You pointed at generic COCO weights or a model trained on other classes. Train with `impactlens train`. |
| `smoke_checkpoint_refused` | You deployed the output of `impactlens smoke`. It is not a model. |
| `device_unavailable` | `DEVICE=cuda` on CPU hardware, or a CPU-only PyTorch build on a GPU host. |
| 401 in the UI | Key mismatch; re-enter it. Remember it is forgotten on reload. |
| "Service unreachable" when the UI is on another site | That origin is not in `ALLOWED_ORIGINS` (CORS), or the URL is wrong. CORS only controls browsers; it is not authentication. |
| 413 for phone photos | Raise `MAX_UPLOAD_BYTES` / `MAX_IMAGE_PIXELS` deliberately, or resize on the client. |
| Overlays misaligned | Should not happen (the UI hides overlays if the browser's oriented size differs from the service's). Report the browser version. |
| UI blank inside the Space page iframe | `FRAME_ANCESTORS` overridden to `'none'`; leave it unset on Spaces. |

## Other container hosts

The business logic is host-independent. For Cloud Run, Fly.io, ECS, Kubernetes
or a VM:

```bash
docker build -t impactlens .
```

```bash
docker run --rm -p 7860:7860 --env-file .env impactlens
```

* Liveness probe: `GET /health`. Readiness probe: `GET /ready`.
* One process per container (`workers=1`); scale with replicas. The rate limiter
  and queue are per replica and in memory.
* Behind a load balancer, set `TRUSTED_PROXY_HOPS` to the number of proxies whose
  `X-Forwarded-For` you trust, otherwise all clients share one rate-limit bucket.
* Terminate TLS in front of the container.
* Model from inside the image: put the file under `models/` before `docker
  build` and set `MODEL_PATH=/app/models/<file>.pt` plus `MODEL_SHA256`.
* GPU: change `[tool.uv.sources]` in `pyproject.toml` to a CUDA wheel index
  (e.g. `https://download.pytorch.org/whl/cu128`), run `uv lock`, re-run the
  tests, build on a CUDA base image, set `DEVICE=cuda`. Untested here (no GPU).

## Building behind a TLS-intercepting proxy

Corporate networks that re-sign TLS need their CA during the build only:

```bash
docker build --secret id=extra_ca,src=/path/to/corporate-ca.pem -t impactlens .
```

The secret is mounted for the download steps and never written to a layer. It
is optional and ignored on Hugging Face.

## Rollback

Every deployment is a Space commit and every model a pinned revision. To roll
back, set `MODEL_REVISION`/`MODEL_SHA256` to the previous values (the Space
restarts), or revert the Space repository to the previous commit.
