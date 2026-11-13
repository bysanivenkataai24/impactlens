---
title: ImpactLens (review UI)
emoji: 🔍
colorFrom: blue
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: agpl-3.0
short_description: Vehicle damage review UI (no model attached)
---

# ImpactLens: review interface only

This Space hosts the ImpactLens **inspection interface** as static files. There
is **no inference service and no model** behind it, so "Run model" reports that
no service is connected. Everything else runs in your browser:

- the illustrative demo (a drawing with hand-made sample findings);
- adding up to six photos, drawing damage regions by hand, and recording type,
  part, visible severity, review status, notes and workshop quotes (INR);
- JSON report, annotated image and PDF export.

Photos never leave your browser: there is no server to send them to. Optional
"Save in this browser" stores findings without photo bytes in local storage.

Decision support only: not a claim decision, repair estimate or roadworthiness
assessment.

Source code (AGPL-3.0) and the full service with model support:
https://github.com/bysanivenkataai24/impactlens
