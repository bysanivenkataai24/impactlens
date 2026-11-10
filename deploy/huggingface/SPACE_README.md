---
title: ImpactLens
emoji: 🔍
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: agpl-3.0
short_description: Vehicle damage region review (decision support only)
startup_duration_timeout: 30m
---

# ImpactLens

Decision support for vehicle inspection: suggested damage regions for human
review, manual annotation and report export. Not a claim decision, repair
estimate or roadworthiness assessment.

- Every model region is pending human review. Scores are uncalibrated.
- An empty result does not mean the vehicle is undamaged.
- Photos are processed in memory and are not stored by the service.
- Source code (AGPL-3.0): see the `SOURCE_URL` link in the app footer.

Model and data status, limitations and licences: see `docs/MODEL_CARD.md` and
`docs/DATASET_CARD.md` in the source repository.
