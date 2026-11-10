# Dataset card

## Status

No vehicle-damage dataset is included in, or was used by, this repository. The
only data ever processed here is procedurally generated smoke-test imagery
(abstract shapes, CC0, `impactlens.data.synthetic`), which contains no vehicles.

## Candidate: CarDD

Source: <https://cardd-ustc.github.io/> and its licence
<https://cardd-ustc.github.io/docs/CarDD_license.pdf> (both read 2026-11-10).

What the project page states: 4,000 high-resolution car damage images, over
9,000 annotated instances, six categories (dent, scratch, crack, glass shatter,
lamp broken, tire flat), for detection and segmentation. Download is by
licensing form: the link is sent after a signed form is emailed to the authors.

Licence conditions (summary of the PDF, not legal advice):

| Condition | Consequence for ImpactLens |
|---|---|
| Dataset is the property of the PIC Lab (CAS); users acquire no rights | Nothing from CarDD may be committed to this repository or bundled in an image. |
| Research use requires **prior consent** of the PIC Lab | Sign and send the licensing form before downloading. |
| **Any commercial use requires prior authorization**, explicitly including "testing commercial systems" | A product pilot, a paid workshop deployment or a public demo of a commercial product needs written authorization first. |
| No transfer, distribution or broadcast to third parties without authorization | Do not mirror the data or share splits; ask whether publishing *trained weights* is permitted before uploading them to a public repository. |
| Publications must cite the paper | Citation below. |

Public discoverability of copies elsewhere does not grant rights; do not use
re-uploads by other parties.

```bibtex
@article{CarDD,
  author={Wang, Xinkuang and Li, Wenjing and Wu, Zhongcheng},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  title={CarDD: A New Dataset for Vision-Based Car Damage Detection},
  year={2023}, volume={24}, number={7}, pages={7202-7214},
  doi={10.1109/TITS.2023.3258480}
}
```

### Gaps to expect (verify once the data is obtained)

| Gap | Why it matters | Action |
|---|---|---|
| Every image shows damage; no intact vehicles | False positives on undamaged cars cannot be measured or trained against | Collect consented intact-vehicle photos as negatives |
| No part labels | Part cannot be predicted | Keep `part` a reviewer entry (see [EXTENSIONS.md](EXTENSIONS.md)) |
| No severity rubric | Severity cannot be predicted; area is not severity | Keep `severity` a reviewer entry |
| No repair estimates or invoices | Cost cannot be modelled | Keep reviewer-entered quotes only |
| Incident / vehicle identifiers not described | Views of one car may leak across splits | Use near-duplicate linking and, if needed, manual grouping; record the residual risk (`--allow-ungrouped`) |
| Geographic and camera coverage not described | Indian roads, body styles, two/three-wheelers, monsoon conditions may be under-represented | Collect a consented local evaluation set before any Indian deployment |

## Data the pipeline expects

* Images: JPEG, PNG or WebP.
* Annotations: COCO instance-segmentation JSON with polygon `segmentation`
  (RLE "crowd" annotations are skipped and counted). Category names are
  normalized (`"glass shatter"` → `glass_shatter`), so CarDD names map directly.
  Annotation coordinates must refer to the **EXIF-oriented** image; images whose
  annotated size matches only the raw sensor orientation are excluded and
  reported, never silently rotated.
* Optional metadata CSV keyed by `file_name`: `incident_id`, `vehicle_group_id`
  and any `meta_*` columns (e.g. `meta_lighting`, `meta_colour`, `meta_angle`,
  `meta_camera`, `meta_weather`, `meta_region`, `meta_body_style`). These drive
  leakage-safe splitting and the failure-analysis slices in evaluation.
* Negatives: a directory of unannotated images whose first sub-folder names the
  negative type, e.g. `negatives/intact_vehicle/`, `negatives/reflection/`,
  `negatives/out_of_domain_document/`, `negatives/out_of_domain_animal/`,
  `negatives/empty_scene/`, `negatives/road/`.

## Manifest

One CSV row per image (`impactlens data manifest`), the single record of what is
in which split and why. Core columns:

| Column | Meaning |
|---|---|
| `image_id` | `<source>/<file stem>`; unique |
| `source`, `license` | Provenance and the terms the image was obtained under (required) |
| `annotation_revision` | Version of the annotation export |
| `image_path`, `annotation_file`, `coco_image_id` | Where the data lives (relative to `--data-root`) |
| `sha256`, `phash` | Exact and perceptual fingerprints (dHash + mean colour) |
| `width`, `height`, `exif_orientation` | EXIF-oriented size |
| `incident_id`, `vehicle_group_id`, `group_source` | Grouping ids and whether they were provided |
| `leakage_group` | Connected component of incident, vehicle and duplicate links |
| `n_instances`, `class_counts`, `is_negative` | Annotation summary |
| `status`, `duplicate_of` | `kept`, `dropped_exact_duplicate`, `dropped_near_duplicate`, `excluded_orientation_conflict`, `excluded_size_mismatch` |
| `split` | `train`, `val`, `test` (empty if not kept) |

The dataset revision id is a hash over image ids, content hashes, annotation
revision, status, split and class counts; every training run records it.

## Split policy

1. De-duplicate **before** splitting: exact SHA-256 matches and near-duplicates
   (dHash Hamming distance ≤ 4 and mean colour within 12 levels; review
   `manifest.dedup.json`). Dropped images stay in the manifest for audit and
   still link groups.
2. Build leakage groups: images sharing an incident, a vehicle or a duplicate
   relation form one group, across sources.
3. Assign whole groups to train / val / test with iterative stratification on
   class presence (rarest class first), default 70 / 15 / 15. Fractions are a
   target, not a guarantee: `impactlens data stats` reports the real class and
   group counts and warns when val/test have fewer than 30 instances of a class,
   none of a class, or no negatives.
4. Freeze the test set: later revisions pass `--previous <old manifest>` so
   every earlier test image stays in test and no earlier train/val image moves
   into test. Every use of the test split is appended to `test_evaluations.jsonl`.

## Collection and consent requirements for new data

* Written consent from the vehicle owner (and photographer) that covers model
  training and evaluation; record the consent reference in `license`.
* Strip location metadata on export (the pipeline re-encodes images without
  metadata); blur faces and number plates if the images will be viewed outside
  the annotation team.
* Coverage targets to report (not to assume): body styles common in India
  (hatchback, compact sedan, SUV/MUV, pickup), colours (including white/silver,
  where scratches are hard to see), lighting (day, dusk, night, sodium street
  light), weather (dry, rain, wet surface), viewing angle and distance (close-up
  plus a wider context view of every damage), cameras (several phone models), and
  damage scale (hairline scratch to panel-wide dent).
