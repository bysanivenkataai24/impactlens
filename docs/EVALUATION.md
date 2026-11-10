# Evaluation, uncertainty and release criteria

No damage model has been evaluated yet. This document is the protocol; numbers
go into [MODEL_CARD.md](MODEL_CARD.md) once measured.

## What `impactlens evaluate` measures

**Ranking metrics** (Ultralytics validator, score threshold 0.001, NMS IoU as in
the service):

* box and mask AP averaged over IoU 0.50:0.95, AP at each IoU step, AP50, AP75;
* per-class AP with the number of ground-truth instances. Classes without
  ground truth in the split are reported as `not_evaluated`, never as a number;
* precision, recall and F1 as functions of the score threshold;
* the confusion matrix at 0.001.

**Operating-point analysis**, at one threshold, through the same adapter the
service uses (so it measures what a reviewer sees):

* instance precision and recall (greedy, score-ordered matching, same class,
  mask IoU ≥ 0.5), overall and per class;
* recall by region size: < 0.1 %, 0.1–1 %, ≥ 1 % of the image area;
* cross-class confusions: predictions overlapping (IoU ≥ 0.5) an unmatched
  ground truth of another class;
* image-level false-positive rate on negative images, per negative type
  (intact vehicles, reflections, decals, out-of-domain roads, documents,
  animals, empty scenes …);
* the same counts broken down by every `meta_*` column of the manifest
  (lighting, colour, angle, camera, weather, region …), plus a per-image CSV for
  failure inspection.

## Rules

* Thresholds, ablation winners and any other choice are made on **validation**.
* The test split is used once per frozen model, needs `--final-test` and the
  validation-selected threshold, and every use is logged.
* Report counts next to every rate. A per-class number from 12 instances is an
  anecdote.
* Compare models on the same dataset revision only (`dataset_revision` is in
  every output).

## Scores and calibration

Detection scores are **not** calibrated probabilities. No calibration has been
implemented or evaluated, so everything user-facing says "model score
(uncalibrated)". If probabilistic language is ever needed:

1. Hold out a calibration split from validation (group-aware, never test).
2. Fit per-class temperature or isotonic calibration of the "this region is a
   true positive at IoU ≥ 0.5" event.
3. Report reliability diagrams and expected calibration error on the test
   split, per class, with bin counts.
4. Re-calibrate whenever the model or the data distribution changes.

Until then, do not describe a score as a likelihood or a confidence percentage.

## Uncertainty and abstention

What exists:

* every finding is `pending_review` until a person decides;
* zero detections are reported as "no supported region returned", never as
  "undamaged";
* geometric flags (`small_region`, `touches_image_border`, `mask_fragmented`,
  `polygon_simplified`) and warnings (`detection_cap_reached`).

What does **not** exist: a vehicle-presence or image-quality gate. A heuristic
(blur variance, exposure histogram) could be added, but it must be validated on
held-out labelled examples (including the out-of-domain negatives) before it is
called a gate, and its abstention rate and misses must be reported. None has
been, so none is claimed; out-of-domain photos may produce regions, which is
why the negative-image false-positive rate is a release metric.

## Robustness

`impactlens robustness` re-evaluates the validation split under synthetic
Gaussian and motion blur, JPEG quality 15, glare, low light with noise, rain
streaks, occlusion patches and 70 % partial-view crops (labels clipped; regions
less than 25 % visible are dropped). Synthetic corruptions show sensitivity;
confirm on real photos taken in rain, at night, through glass and with
budget phone cameras.

## Release criteria

Targets are set **after** the first baseline is measured on the real validation
split and agreed with the people who will review findings. Fill in and sign off
before any pilot:

| Criterion | Target | Measured (test) |
|---|---|---|
| Mask AP50 per class with ≥ 30 test instances | ≥ baseline − agreed margin | |
| Recall at the operating point, per class | agreed per class | |
| Recall for regions < 0.1 % of the image | reported, with a minimum agreed | |
| Image-level false-positive rate on intact vehicles | ≤ agreed | |
| Image-level false-positive rate on out-of-domain images | ≤ agreed | |
| Worst `meta_*` slice vs overall recall | gap ≤ agreed | |
| Robustness: largest AP drop across corruptions | reported | |
| Annotator agreement (ceiling) | reported | |
| p95 end-to-end latency on deployed hardware at expected concurrency | ≤ agreed | |
| Error rate (5xx) under load | ≤ agreed | |

Failing a criterion means "not released", not "release with a note".

## Serving benchmarks

`impactlens benchmark model` (in-process: cold start, warm latency, memory) and
`impactlens benchmark http` (end-to-end p50/p95, throughput and status codes at
several concurrencies) measure; they never estimate. Run the HTTP benchmark
against the **deployed** service with real photos, record the hardware, and set
targets from that. The numbers in `docs/benchmarks/` were measured in the
development sandbox (4 vCPU, no GPU) with a smoke checkpoint or a synthetic
adapter and are not transferable to Hugging Face hardware.
