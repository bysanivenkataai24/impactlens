# Training workflow

Training runs on a training machine (ideally one CUDA GPU), never inside the
HTTP service. Every step is a CLI command; every hyper-parameter is in a YAML
file under `configs/train/`; every run directory records its configuration,
dataset revision and environment.

## Environment

```bash
uv sync --frozen
```

Pinned by `uv.lock`: Python 3.12, ultralytics 8.4.208, torch 2.15.0 and
torchvision 0.29.2 (**CPU** wheels), numpy 2.5.4, opencv-python-headless
5.0.0.93. For GPU training, re-lock against a CUDA wheel index (see
[DEPLOYMENT.md](DEPLOYMENT.md#other-container-hosts)) and record the CUDA
version; `environment.json` in each run captures what was actually used.

## 0. Smoke test (mechanics only)

```bash
uv run impactlens smoke --epochs 3
```

Generates abstract shapes (not vehicles), runs manifest → dedup → group split →
label export → train → evaluate → export → adapter load, and checks that the
service adapter refuses the resulting checkpoint. It takes about 20 s on a
4-vCPU CPU. **Its metrics measure nothing about vehicle damage.**

## 1. Build the manifest

```bash
uv run impactlens data manifest --data-root data/raw --images cardd/images --coco cardd/annotations/instances_all.json --source cardd --license "CarDD licence; consent ref <id>; research use only" --annotation-revision cardd-r1 --image-metadata data/raw/cardd_groups.csv --out data/manifest-v1.csv
```

Add further sources (consented local photos, negatives) to the same manifest
with one command per source, each with its own `--source`, `--license` and
`--annotation-revision`; negatives use `--negatives-dir`. Unknown annotation
categories stop the build; drop them explicitly with `--ignore-category`.

## 2. Remove duplicates, then split by group

```bash
uv run impactlens data dedup --manifest data/manifest-v1.csv
```

Review `data/manifest-v1.dedup.json` (near-duplicate pairs and duplicates whose
annotations differ) before continuing.

```bash
uv run impactlens data split --manifest data/manifest-v1.csv --fractions 0.70 0.15 0.15 --seed 20261110
```

Without incident/vehicle ids the command stops; `--allow-ungrouped` accepts the
risk explicitly and the report says so. For later revisions add
`--previous data/manifest-v0.csv` to keep the test set frozen.

```bash
uv run impactlens data stats --manifest data/manifest-v1.csv
```

Read the warnings. Report these counts, not the split percentages.

## 3. Export the training dataset

```bash
uv run impactlens data export-yolo --manifest data/manifest-v1.csv --data-root data/raw --out datasets/impactlens-v1 --rfs-threshold 0.3
```

Images are EXIF-oriented, re-encoded without metadata and named by content
hash. `dataset_revision.json` and `image_index.csv` are written next to
`dataset.yaml`. `--rfs-threshold` additionally writes `dataset_rfs.yaml` for the
repeat-factor-sampling ablation (validation and test are unaffected).

## 4. Train the baseline

```bash
uv run impactlens train --config configs/train/baseline_yolo11n_640.yaml
```

Baseline (from the config, all explicit): YOLO11n-seg from the hashed COCO
checkpoint; seed 20261110; 640 px; batch 16; AdamW, lr0 1e-3, cosine decay to
lr0 × 0.01 after 3 warm-up epochs; weight decay 5e-4; up to 150 epochs with
early stopping after 30 epochs without validation improvement; device `0`.
Training-only augmentation that moves images and masks together: HSV jitter,
±3° rotation, ±10 % translation, ±50 % scale, horizontal flip, mosaic (off for
the last 15 epochs). No vertical flip (inverted vehicles are not a deployment
case) and no copy-paste (pasted damage on arbitrary surfaces is implausible
evidence).

Override for a quick CPU check (still not a model):

```bash
uv run impactlens train --config configs/train/baseline_yolo11n_640.yaml --set device=cpu --set amp=false --set workers=2 --set epochs=1 --set fraction=0.05
```

Outputs under `runs/<experiment>/<utc>-<config digest>/`: `config.resolved.yaml`,
`dataset_revision.json`, `environment.json`, `results.csv` (per-epoch losses
and validation metrics), `weights/best.pt`, `weights/last.pt` (both tagged with
provenance) and `train_summary.json`.

Experiment tracking: the run directory is the reproducible log. To add an
external tracker set `tracker: tensorboard` (or `mlflow`, `wandb`, ...) in the
config; it is off by default.

## 5. Evaluate on validation and select the operating point

```bash
uv run impactlens evaluate --weights runs/baseline_yolo11n_640/<run>/weights/best.pt --data datasets/impactlens-v1/dataset.yaml --split val --out runs/baseline_yolo11n_640/<run>/eval --device 0
```

Writes `evaluation_val.json`, `per_image_val.csv` and `operating_point.json`
(threshold selected on validation). See [EVALUATION.md](EVALUATION.md).

## 6. Ablations and alternatives

Run each on the **same dataset revision** and compare on validation only:

| Config | Question |
|---|---|
| `ablation_yolo11n_1024.yaml` | Does 1024 px raise recall for small scratches/cracks enough to pay for latency? |
| `evaluate --tile <px>` on the baseline | Does tiled inference help small regions? Check duplicate regions along tile borders and boundary truncation in `per_image_val.csv` before adopting; the service does not tile. |
| `ablation_yolo11n_640_clspw.yaml` | Does inverse-frequency class weighting help rare classes without hurting frequent ones? |
| `ablation_yolo11n_640_rfs.yaml` | Does repeat-factor sampling help rare classes? |
| `alternative_yolo26n_640.yaml`, `alternative_yolo11s_640.yaml` | Is another architecture or size better on this data? Benchmark latency too. |

Report per-class AP with instance counts, recall by region size, negative-image
false-positive rate and latency for each. Differences smaller than the variation
between seeds are not differences: re-run the leading candidates with 2–3 seeds.

## 7. Robustness, then the one final test

```bash
uv run impactlens robustness --weights <best.pt> --data datasets/impactlens-v1/dataset.yaml --score-threshold <from operating_point.json> --out <run>/robustness --device 0
```

Freeze the model and threshold, then evaluate the test split once:

```bash
uv run impactlens evaluate --weights <best.pt> --data datasets/impactlens-v1/dataset.yaml --split test --final-test --score-threshold <from operating_point.json> --out <run>/eval --device 0
```

The command refuses without `--final-test` and an explicit validation-chosen
threshold, and appends to `datasets/impactlens-v1/test_evaluations.jsonl`.

## 8. Export the deployable artifact

```bash
uv run impactlens export --run runs/baseline_yolo11n_640/<run> --model-id impactlens-yolo11n-seg-v1 --evaluation <run>/eval/evaluation_val.json --operating-point <run>/eval/operating_point.json
```

Produces `models/impactlens-yolo11n-seg-v1/` with the checkpoint,
`model_manifest.json`, `SHA256SUMS` and `service.env`. Model ids are immutable.
Then follow [DEPLOYMENT.md](DEPLOYMENT.md).

## Reproducibility notes

* Seeds and `deterministic: true` are set, but some CUDA kernels are not
  bit-reproducible; expect small run-to-run differences on GPU.
* Ultralytics downloads its initialization weights from GitHub release assets
  if they are missing; the config's `init_checkpoint_sha256` is checked before
  the file is loaded.
* Ultralytics usage analytics are disabled for every command.
