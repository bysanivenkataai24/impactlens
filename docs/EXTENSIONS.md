# Extensions: parts, severity and repair cost

None of these exists. The baseline returns `part: null` and `severity:
"unknown"` and never estimates cost; reviewers enter all three. The interfaces
in `src/impactlens/extensions.py` define how they could be added without
fabricating outputs: each default returns "not available", and
`require_evidence()` refuses to enable an implementation that has no held-out
`ValidationEvidence`.

## Affected part

* **Data:** a separately annotated parts dataset (part masks for every view),
  ideally on the same images as the damage masks.
* **Association policy:** overlap alone does not identify the damaged
  component. A scratch can cross a door seam; perspective can make a wing dent
  overlap the door mask. Define the rule (e.g. largest overlap above a minimum
  share, "multiple parts" when ambiguous, "unknown" below a threshold) and
  validate it against human part labels for each damage instance.
* **Report:** per-part accuracy of the *association*, not just part
  segmentation mIoU; the "unknown/multiple" rate.
* **Output:** `PartAssignment(part, score, policy)`; schema minor version bump
  to allow non-null `part` with a `part_source` field.

## Visible severity

* **Rubric first:** written definitions per class (e.g. dent depth relative to
  panel curvature, scratch through clear coat vs. primer vs. metal), with
  reference photos.
* **Data:** severity labels made with the rubric, double-annotated; report
  inter-annotator agreement (weighted kappa). If annotators cannot agree, a
  model cannot be validated against them.
* **Never** derive severity from mask or box area: area in pixels depends on
  distance, focal length and viewpoint, not on physical damage.
* **Output:** a label with its evidence record; the UI would show "model
  estimate" separately from the reviewer's entry, which stays authoritative.

## Repair cost

* Keep it a **separate service** from visual detection, with its own data,
  versioning and evaluation.
* **Data:** actual estimates or invoices joined to findings, with make, model,
  year, region, part availability, labour rates, repair operation (repair /
  replace / paint), taxes and date.
* **Output:** `CostInterval(currency, low, high, assumptions, evidence)`, an
  interval with stated assumptions, evaluated for coverage on held-out
  invoices; or nothing. Never a single number derived from a score or an area.
* Until then: reviewer-entered workshop quotes in INR, summed only as a
  provisional partial subtotal of confirmed, de-duplicated findings.

## Language-model summaries

A language model may turn **validated, structured** findings into prose for a
reviewer. It must not add damage, numbers, probabilities, severity, cost or
repair advice that is not in the structured input, and its output must be
labelled as a summary of the review record.
