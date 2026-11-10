"""Versioned wire contract of the inference service.

``SCHEMA_VERSION`` follows semantic versioning:

* patch - documentation or description changes only;
* minor - new optional fields, new flag / warning / reason codes;
* major - anything a strict client could break on (removed or renamed fields,
  changed coordinate conventions, changed types).

The JSON Schema generated from these models is committed under ``schemas/`` and
a test fails when the two drift apart.

Coordinate convention (all detections):
    normalized to ``[0, 1]`` against the **EXIF-oriented** image, origin at the
    top-left corner, x to the right, y downwards. ``bbox`` is
    ``[x_min, y_min, x_max, y_max]``. ``polygon`` is a closed ring given without
    repeating the first vertex.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .taxonomy import MODEL_DAMAGE_TYPES

SCHEMA_VERSION = "1.0.0"

DamageType = Literal["dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"]
assert set(DamageType.__args__) == set(MODEL_DAMAGE_TYPES)  # keep the Literal in sync

Unit = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]

#: Documented per-detection flags. They are geometric facts about the returned
#: region, not quality judgements and not validated classifiers.
DetectionFlag = Literal[
    "mask_fragmented",
    "polygon_simplified",
    "touches_image_border",
    "small_region",
]

DETECTION_FLAG_DOCS: dict[str, str] = {
    "mask_fragmented": (
        "The predicted mask had several disconnected pieces. Only the largest piece is "
        "returned as the polygon."
    ),
    "polygon_simplified": (
        "The mask outline had more vertices than the service limit and was simplified."
    ),
    "touches_image_border": (
        "The region reaches the edge of the photo, so the damage may continue outside the frame."
    ),
    "small_region": (
        "The region covers less than 0.1% of the photo. Small regions are typically the "
        "least reliable detections."
    ),
}

#: Documented response-level warnings.
ResponseWarning = Literal[
    "detection_cap_reached",
    "invalid_detections_dropped",
]

#: Reasons ``/health`` may give for ``ready: false``.
NotReadyReason = Literal[
    "api_key_not_configured",
    "api_key_too_short",
    "model_id_not_configured",
    "model_not_configured",
    "model_source_ambiguous",
    "model_source_incomplete",
    "model_revision_not_pinned",
    "checkpoint_not_found",
    "checkpoint_download_failed",
    "checkpoint_hash_mismatch",
    "unsupported_checkpoint_format",
    "unsupported_task",
    "class_mapping_mismatch",
    "smoke_checkpoint_refused",
    "device_unavailable",
    "model_load_failed",
]

#: Sent with every prediction. The wording is part of the contract's intent:
#: clients should display it, not paraphrase it into something stronger.
LIMITATIONS: tuple[str, ...] = (
    "Decision support only. Every returned region is a suggestion pending human review.",
    "The model sees one photograph. It cannot establish mechanical condition, "
    "roadworthiness, hidden or structural damage, fault, claim eligibility or repair cost.",
    "An empty result means no supported region was returned for this photo at this "
    "threshold. It does not mean the vehicle is undamaged.",
    "'score' is an uncalibrated model score, not a probability that damage is present.",
    "Only these visible damage types are supported: dent, scratch, crack, glass_shatter, "
    "lamp_broken, tire_flat. Other damage is not detected.",
    "Affected part and severity are not predicted: 'part' is null and 'severity' is "
    "'unknown' until a reviewer enters them.",
    "The same physical damage can appear in several photos. Detections are not "
    "de-duplicated across views and must not be summed into a damage count or a cost.",
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Detection(_Strict):
    id: str = Field(description="Stable within one response, e.g. 'det-1'. Ordered by score.")
    damage_type: DamageType
    score: Unit = Field(
        description="Uncalibrated model score in [0, 1]. Not a probability of damage."
    )
    bbox: Annotated[list[Unit], Field(min_length=4, max_length=4)] = Field(
        description="[x_min, y_min, x_max, y_max], normalized to the EXIF-oriented image."
    )
    polygon: Annotated[
        list[Annotated[list[Unit], Field(min_length=2, max_length=2)]],
        Field(min_length=3, max_length=2000),
    ] = Field(description="Mask outline as [x, y] pairs, normalized like bbox.")
    part: None = Field(
        default=None,
        description="Always null in schema 1.x: the baseline has no validated part model.",
    )
    severity: Literal["unknown"] = Field(
        default="unknown",
        description="Always 'unknown' in schema 1.x: the baseline has no validated severity model.",
    )
    flags: list[DetectionFlag] = Field(default_factory=list)
    review_status: Literal["pending_review"] = "pending_review"


class ImageInfo(_Strict):
    width: int = Field(gt=0, description="Width in pixels after EXIF orientation is applied.")
    height: int = Field(gt=0, description="Height in pixels after EXIF orientation is applied.")
    format: Literal["JPEG", "PNG", "WEBP"]
    exif_orientation: int | None = Field(
        description="EXIF orientation tag found in the upload (1-8), or null if absent."
    )
    orientation_applied: bool = Field(
        description="True when the pixels were rotated/mirrored to honour the EXIF tag."
    )
    sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="SHA-256 of the uploaded bytes as received."
    )


class LatencyMs(_Strict):
    queue: float = Field(ge=0, description="Time spent waiting for an inference slot.")
    decode: float = Field(ge=0)
    inference: float = Field(ge=0, description="Model call, including pre/post-processing.")
    postprocess: float = Field(ge=0)
    total: float = Field(ge=0, description="Server-side handling time for this request.")


class PredictResponse(_Strict):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    request_id: str
    model_id: str
    model_revision: str
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image: ImageInfo
    score_threshold: Unit
    outcome: Literal["detections_returned", "no_supported_region_returned"]
    message: str
    detections: list[Detection]
    warnings: list[ResponseWarning] = Field(default_factory=list)
    latency_ms: LatencyMs
    limitations: list[str]


class ScoreThresholdLimits(_Strict):
    default: Unit
    min: Unit
    max: Unit


class ServiceLimits(_Strict):
    max_upload_bytes: int
    max_image_pixels: int
    min_image_side: int
    accepted_media_types: list[str]
    score_threshold: ScoreThresholdLimits
    max_detections: int
    max_polygon_points: int
    inference_timeout_s: float


class HealthResponse(_Strict):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    status: Literal["ok"] = "ok"
    ready: bool = Field(description="True only when /predict can run real inference.")
    reasons: list[NotReadyReason] = Field(
        default_factory=list, description="Why the service is not ready. Empty when ready."
    )
    model_id: str | None
    model_revision: str | None
    checkpoint_sha256: str | None
    task: str | None
    classes: list[str]
    device: str | None
    limits: ServiceLimits
    service_version: str
    source_url: str | None = Field(
        description="Where the source code of this deployment can be obtained (AGPL-3.0)."
    )


class AuthCheckResponse(_Strict):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    authenticated: Literal[True] = True


ErrorCode = Literal[
    "unauthenticated",
    "rate_limited",
    "server_busy",
    "model_unavailable",
    "unsupported_media_type",
    "payload_too_large",
    "image_pixel_limit_exceeded",
    "malformed_image",
    "image_too_small",
    "invalid_request",
    "request_timeout",
    "inference_timeout",
    "inference_failed",
    "not_found",
    "method_not_allowed",
    "internal_error",
]


class ErrorBody(_Strict):
    code: ErrorCode
    message: str
    request_id: str
    reasons: list[NotReadyReason] | None = None
    retry_after_s: int | None = None


class ErrorResponse(_Strict):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    error: ErrorBody
