# Annotation guide (revision 1)

Goal: instance masks of **visible** damage, consistent enough that two
annotators produce the same instances. When in doubt, follow the rule, not the
intuition, and raise the case for adjudication.

## General rules

* Annotate in the EXIF-oriented image (as any photo viewer shows it).
* One instance = one contiguous damaged region of one type. Draw a polygon
  around the visible extent; follow the edge within about 2 px at 100 % zoom.
* Minimum size: annotate if you can identify the type at 100 % zoom. Smaller →
  leave it unannotated and tick `needs_closer_photo` in the image notes.
* Damage cut by the image border: annotate the visible part.
* Occluded damage: annotate visible pieces; if one damage is split by an
  occluder (wiper, person), use one instance with several polygons.
* Unsure whether it is damage at all → do **not** label it as a damage class;
  add the image to the adjudication queue.
* Unsure of the type but sure it is damage → category `uncertain` (exported with
  `--ignore-category uncertain`, so it never becomes a training label).

## Classes

| Class | Annotate | Do not annotate |
|---|---|---|
| `dent` | Deformation of a body panel, with or without paint damage. Outline the deformed area (where reflections bend). | Panel seams and design creases; reflections that look wavy because of the environment. |
| `scratch` | Paint/clear-coat damage without deformation: lines, scuffs, abrasions. One instance per connected scratch area. | Dirt streaks, bird droppings, decals, tape, water marks (check the context view). |
| `crack` | Fractures in plastic or painted parts (bumpers, trims, mirror housings). | Cracks in glass (→ `glass_shatter`); panel gaps. |
| `glass_shatter` | Broken, cracked or chipped glass of windscreen, windows or mirror glass. The cracked area, not the whole pane. | Reflections, raindrops, dirt on glass. Lamp lenses (→ `lamp_broken`). |
| `lamp_broken` | Head-, tail-, fog- or indicator lamp with cracked/missing lens or broken housing. Outline the damaged lamp area. | Lamps that only look dark because they are off; condensation. |
| `tire_flat` | Visibly deflated tyre (bulging sidewall at the contact patch, rim close to the ground). Outline the tyre. | Normal tyre bulge; low-profile tyres; tyres cut off by the frame without visible deflation. |

## Overlapping damage

* Different types on the same spot: annotate **both**, each with its own full
  extent (a dent with a scratch across it = one `dent` + one `scratch`).
  Masks may overlap. During training Ultralytics (8.4.208, `overlap_mask:
  true`) merges instance masks into one map with smaller instances drawn over
  larger ones: a small scratch stays intact, and the dent's training target
  loses the overlapped pixels. If that matters for a class pair, compare
  `overlap_mask: false` in an ablation.
* Crack in glass: always `glass_shatter`, never also `crack`.
* Broken lamp with cracked surrounding bumper: `lamp_broken` for the lamp,
  `crack` for the bumper.
* Many scratches close together: one instance if they touch or are less than
  roughly a finger-width apart; otherwise separate.

## Hard negatives (leave unannotated, keep the image)

Images that show these and **no** damage are valuable negatives; put them in
`negatives/<type>/`:

| Type folder | Examples |
|---|---|
| `reflection` | Trees, buildings or lights reflected on glossy paint or glass |
| `decal` | Stickers, wraps, logos, racing stripes, number-plate frames |
| `dirt` | Mud splashes, dust, bird droppings, water spots |
| `shadow` | Hard shadows across panels |
| `panel_seam` | Door and bonnet gaps, design creases, trim lines |
| `unusual_paint` | Matte, metallic flake, two-tone, faded paint |
| `old_repair` | Visible repaint or filler that is not current damage (if your protocol counts these as non-damage; record the decision) |
| `intact_vehicle` | Undamaged vehicles, many angles and colours |
| `out_of_domain_*` | Roads, documents, animals, empty scenes, interiors |

## Image-level metadata to record

`incident_id` and `vehicle_group_id` (required for leakage-free splits; the same
vehicle photographed on two days shares `vehicle_group_id`), and where known
`meta_lighting`, `meta_weather`, `meta_colour`, `meta_body_style`, `meta_angle`
(front, rear, left, right, close-up), `meta_camera`, `meta_image_quality`
(sharp, blurred, glare, low light). These become evaluation slices.

## Quality assurance

1. **Calibration round:** all annotators label the same 50 images; discuss every
   disagreement and update this guide (bump the revision).
2. **Double annotation:** at least 10 % of images (stratified by class and
   source) are annotated independently by two people.
3. **Agreement**, computed per batch on the double-annotated images:
   * instance agreement: match instances at mask IoU ≥ 0.5 (same procedure as
     `impactlens.training.evaluate.match_image`), report F1 between annotators
     per class;
   * class agreement on matched instances: Cohen's kappa;
   * boundary agreement: mean IoU of matched masks.
   Agreement sets a realistic ceiling for model metrics; a class with low
   agreement needs clearer rules before more data.
4. **Adjudication:** a senior annotator resolves disagreements and every
   `uncertain` / queued item; decisions are recorded with the image id and, if
   general, added to this guide.
5. **Review sampling:** a reviewer checks 5 % of single-annotated images per
   batch; error rate above the agreed threshold → re-review the batch.
6. **Versioning:** every export gets a new `annotation_revision`; never edit an
   exported revision in place.
