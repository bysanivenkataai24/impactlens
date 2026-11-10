# Model card: ImpactLens damage segmentation

## Status

**No trained ImpactLens damage model exists yet.** Nothing in this repository
has been trained on vehicle damage data, evaluated on it or benchmarked as a
damage detector. This card is filled in from `models/<id>/model_manifest.json`
and `evaluation_val.json` / `evaluation_test.json` when the first model is
trained. Until then the service correctly reports `ready: false`.

What exists and was verified (see [EXECUTION_RECORD.md](EXECUTION_RECORD.md)):

* the training, evaluation and export pipeline, executed end to end on
  synthetic shapes (`impactlens smoke`) to prove the mechanics;
* the inference adapter, run on real Ultralytics checkpoints inside the
  production image;
* refusal of the two checkpoints that are most likely to be deployed by mistake:
  generic COCO weights (`class_mapping_mismatch`) and smoke-test weights
  (`smoke_checkpoint_refused`).

## Intended use

Decision support for a trained inspector looking at photographs of a vehicle:
propose regions that may show one of six visible damage types, for the inspector
to confirm, correct, reject or supplement.

## Out of scope

Not for, and not evaluated for: automatic claim approval or denial; fault or
liability; roadworthiness or mechanical condition; hidden or structural damage;
repair-versus-replace decisions; cost estimation; identifying people or
vehicles; any decision made without a human reviewing the photo.

## Model

| Item | Value |
|---|---|
| Task | Instance segmentation (masks + boxes) |
| Classes | `dent`, `scratch`, `crack`, `glass_shatter`, `lamp_broken`, `tire_flat` (taxonomy of CarDD). Reviewers may also use `other_uncertain`; the model never outputs it. |
| Baseline architecture | Ultralytics YOLO11n-seg (≈2.8 M parameters) |
| Initialization | Official COCO checkpoint `yolo11n-seg.pt`, release asset `v8.4.0`, SHA-256 `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`, 80 COCO classes, AGPL-3.0. COCO weights are an initialization, **not** a damage detector. |
| Alternatives to compare | YOLO26n-seg (`361fbfab…71e73f`), YOLO11s-seg (`1caa81c0…ff257c3`); see `configs/train/` |
| Runtime | ultralytics 8.4.208, torch 2.15.0 (CPU wheels in the image), Python 3.12 |
| Input | RGB, EXIF-oriented, letterboxed to the trained image size (baseline 640) |
| Output | Per region: class, uncalibrated score, normalized box and polygon |
| Not predicted | Vehicle part, severity, cost (always `null` / `unknown`) |

## Training data

Not yet available. Candidate: CarDD (research dataset, restrictive licence:
prior consent required for research, authorization required for any commercial
use including testing commercial systems, no redistribution). See
[DATASET_CARD.md](DATASET_CARD.md) for the licence analysis, known gaps and the
data that must be collected for an Indian deployment (intact vehicles,
local body styles, local conditions).

## Evaluation

Not performed. When performed, report from `impactlens evaluate` (protocol in
[EVALUATION.md](EVALUATION.md)):

| Metric (split: test, frozen model, threshold chosen on val) | Value |
|---|---|
| Mask AP50-95 / AP50 / AP75 | — |
| Box AP50-95 / AP50 | — |
| Per-class mask AP50-95 (with instance counts) | — |
| Instance precision / recall at the operating threshold | — |
| Recall by region size (< 0.1 %, 0.1–1 %, ≥ 1 % of image) | — |
| Image-level false-positive rate on intact vehicles | — |
| Image-level false-positive rate on out-of-domain images | — |
| Largest cross-class confusions | — |
| Robustness deltas (blur, glare, JPEG, rain, low light, occlusion, partial view) | — |
| Dataset revision / split counts | — |

## Operating threshold

Selected on the validation split only (`operating_point.json`, default rule:
maximize mean per-class mask F1). It is a ranking cut-off on an **uncalibrated
score**; it is not a probability. No calibration method has been applied or
evaluated, so the UI labels values as "model score".

## Known and expected limitations

* Photographs cannot show hidden or structural damage.
* Reflections, decals, dirt, shadows, panel seams, unusual paint and old
  repairs are expected false-positive sources; annotate them as hard negatives.
* Small scratches and cracks are the hardest; compare 1024 px or tiling before
  relying on recall for them.
* Domain shift: CarDD-style imagery may not represent Indian streets, body
  styles, two/three-wheelers, monsoon conditions or phone cameras in use.
* No vehicle-presence or image-quality gate is implemented. None has been
  validated, so none is claimed; out-of-domain inputs may still produce regions.
* The same damage seen in several photos produces several findings.

## Ethical and privacy notes

Photos can contain faces, number plates and location metadata. The service
discards metadata and stores nothing; it does not redact faces or plates.
Customer photos must not be reused for training without consent.

## Licence

A model trained with Ultralytics from Ultralytics pretrained weights is subject
to AGPL-3.0 (or an Ultralytics Enterprise licence). The training data's licence
also governs use of the trained weights. See [LICENSING.md](LICENSING.md).
