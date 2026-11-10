"""Extension interfaces for part association, visible severity and repair cost.

None of these capabilities exists in the baseline, and nothing here fakes one.
Each interface demands an evidence record (what it was validated on, with what
result) before an implementation may be registered, and the shipped default
for every interface returns "not available".

Why each capability needs its own evidence:

* **Part association.** A damage mask overlapping a part mask does not always
  identify the damaged component (a scratch can cross a door seam; a dent on a
  wing may be photographed through a door). This needs a separately labelled
  parts dataset *and* a validated overlap/association policy.
* **Visible severity.** Needs a written rubric, annotations made with it,
  inter-annotator agreement and a held-out evaluation. Mask or box area is not
  physical severity: it depends on distance, focal length and viewpoint.
* **Repair cost.** Needs real estimates or invoices and features such as make,
  model, year, region, part availability, labour rates, repair operation and
  taxes. A model score or an image area is never mapped to rupees. A cost
  service must return an interval with its assumptions, or nothing.

A language model may later summarise *validated, structured* findings for a
reviewer, but must not add damage, numbers, probabilities or repair advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .schemas import Detection


@dataclass(frozen=True)
class ValidationEvidence:
    """Minimum record required before an extension may be enabled."""

    dataset_revision: str
    split: str
    metric: str
    value: float
    sample_size: int
    protocol: str
    #: Where the full evaluation report lives (a path or URL).
    report: str

    def __post_init__(self) -> None:
        if self.split not in {"val", "test"}:
            raise ValueError("evidence must come from a held-out split")
        if self.sample_size <= 0:
            raise ValueError("evidence needs a positive sample size")
        if not (self.protocol and self.report and self.dataset_revision):
            raise ValueError("evidence must name its protocol, report and dataset revision")


@dataclass(frozen=True)
class PartAssignment:
    part: str
    #: Uncalibrated association score of the part model, if any.
    score: float | None
    policy: str


@dataclass(frozen=True)
class CostInterval:
    currency: str
    low: float
    high: float
    assumptions: tuple[str, ...]
    evidence: ValidationEvidence

    def __post_init__(self) -> None:
        if not (0 <= self.low <= self.high):
            raise ValueError("a cost interval needs 0 <= low <= high")
        if not self.assumptions:
            raise ValueError("a cost interval must state its assumptions")


@dataclass(frozen=True)
class VehicleFacts:
    """Reviewer-entered context a cost service would need (never inferred from photos)."""

    make: str | None = None
    model: str | None = None
    year: int | None = None
    region: str | None = None
    extra: dict[str, str] = field(default_factory=dict)


class PartAssociator(Protocol):
    evidence: ValidationEvidence | None

    def assign(self, detection: Detection, image_rgb) -> PartAssignment | None: ...


class SeverityEstimator(Protocol):
    evidence: ValidationEvidence | None

    def estimate(self, detection: Detection, image_rgb) -> str: ...


class RepairCostService(Protocol):
    evidence: ValidationEvidence | None

    def estimate(self, findings: list[dict], vehicle: VehicleFacts) -> CostInterval | None: ...


class NoPartModel:
    evidence = None

    def assign(self, detection: Detection, image_rgb) -> PartAssignment | None:
        return None


class NoSeverityModel:
    evidence = None

    def estimate(self, detection: Detection, image_rgb) -> str:
        return "unknown"


class NoCostModel:
    """Repair cost is kept separate from visual detection and is not offered."""

    evidence = None

    def estimate(self, findings: list[dict], vehicle: VehicleFacts) -> CostInterval | None:
        return None


def require_evidence(
    extension: PartAssociator | SeverityEstimator | RepairCostService,
) -> ValidationEvidence:
    """Gate for wiring an extension into the service. Raises without evidence."""
    evidence = getattr(extension, "evidence", None)
    if not isinstance(evidence, ValidationEvidence):
        raise RuntimeError(
            f"{type(extension).__name__} has no validation evidence; it cannot be enabled. "
            "See docs/EXTENSIONS.md."
        )
    return evidence
