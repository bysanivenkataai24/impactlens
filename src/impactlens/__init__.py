"""ImpactLens: decision-support vehicle damage segmentation.

The package is split so that the pieces can be reasoned about separately:

* :mod:`impactlens.adapters`  - model runtimes behind one small protocol (no web code).
* :mod:`impactlens.service`   - the FastAPI inference service (no model-library imports).
* :mod:`impactlens.data`      - manifest, de-duplication, group-aware splits, label export.
* :mod:`impactlens.training`  - train / evaluate / export / benchmark commands.
"""

__version__ = "0.1.0"
