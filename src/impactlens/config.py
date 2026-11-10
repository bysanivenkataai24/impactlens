"""Environment-based configuration for the inference service.

Two kinds of problems are kept apart on purpose:

* A *malformed* value (``MAX_UPLOAD_BYTES=banana``) raises :class:`ConfigError`
  and the process refuses to start. That is a deployment bug.
* A *missing* requirement (no API key, no checkpoint) does not stop the process.
  The service starts, ``/health`` reports ``ready: false`` with reason codes and
  ``/predict`` answers 503. The review UI stays usable for manual work.

Secrets (``API_KEY``, ``HF_TOKEN``) are read here and nowhere else, and
:meth:`Settings.public_summary` never includes them.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

MIN_API_KEY_LENGTH = 24
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ORIGIN = re.compile(r"^https?://[A-Za-z0-9.\-]+(:\d{1,5})?$")
_DEVICE = re.compile(r"^(cpu|mps|cuda(:\d+)?)$")
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-/]{0,95}$")


class ConfigError(ValueError):
    """A configuration value is present but unusable."""


def _clean(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _int(env: Mapping[str, str], name: str, default: int, *, low: int, high: int) -> int:
    raw = _clean(env, name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if not low <= value <= high:
        raise ConfigError(f"{name} must be between {low} and {high}, got {value}")
    return value


def _float(env: Mapping[str, str], name: str, default: float, *, low: float, high: float) -> float:
    raw = _clean(env, name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if not math.isfinite(value) or not low <= value <= high:
        raise ConfigError(f"{name} must be a finite number between {low} and {high}, got {raw!r}")
    return value


def _origins(env: Mapping[str, str]) -> tuple[str, ...]:
    raw = _clean(env, "ALLOWED_ORIGINS")
    if raw is None:
        return ()
    origins: list[str] = []
    for item in raw.split(","):
        origin = item.strip().rstrip("/")
        if not origin:
            continue
        if origin == "*":
            raise ConfigError(
                "ALLOWED_ORIGINS='*' is refused. List the exact browser origins that may call "
                "this service (for example https://owner-space.hf.space)."
            )
        if not _ORIGIN.match(origin):
            raise ConfigError(
                f"ALLOWED_ORIGINS entry {origin!r} is not an origin (scheme://host[:port])"
            )
        origins.append(origin)
    return tuple(dict.fromkeys(origins))


@dataclass(frozen=True)
class Settings:
    """Validated service configuration. Build it with :meth:`from_env`."""

    # --- identity / auth -------------------------------------------------
    api_key: str | None = field(default=None, repr=False)
    model_id: str | None = None
    # --- checkpoint source (exactly one of path / repo) -------------------
    model_path: Path | None = None
    model_repo: str | None = None
    model_filename: str | None = None
    model_revision: str | None = None
    model_sha256: str | None = None
    hf_token: str | None = field(default=None, repr=False)
    device: str = "cpu"
    # --- inference parameters --------------------------------------------
    inference_image_size: int | None = None
    nms_iou: float = 0.7
    default_score_threshold: float = 0.25
    min_score_threshold: float = 0.05
    max_score_threshold: float = 0.95
    max_detections: int = 50
    max_polygon_points: int = 200
    # --- request limits ---------------------------------------------------
    max_upload_bytes: int = 12 * 1024 * 1024
    max_image_pixels: int = 50_000_000
    min_image_side: int = 32
    request_body_timeout_s: float = 30.0
    inference_timeout_s: float = 60.0
    max_pending_requests: int = 4
    rate_limit_per_minute: int = 30
    trusted_proxy_hops: int = 0
    # --- web ----------------------------------------------------------------
    allowed_origins: tuple[str, ...] = ()
    frame_ancestors: str = "'none'"
    frontend_dist: Path | None = None
    source_url: str | None = None
    log_level: str = "INFO"
    #: Load the checkpoint after the server starts listening (True) or before (False).
    model_load_in_background: bool = True

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env

        device = (_clean(env, "DEVICE") or "cpu").lower()
        if not _DEVICE.match(device):
            raise ConfigError(f"DEVICE must be cpu, mps, cuda or cuda:<index>, got {device!r}")

        model_sha256 = _clean(env, "MODEL_SHA256")
        if model_sha256 is not None:
            model_sha256 = model_sha256.lower()
            if not _SHA256.match(model_sha256):
                raise ConfigError("MODEL_SHA256 must be 64 lowercase hexadecimal characters")

        model_id = _clean(env, "MODEL_ID")
        if model_id is not None and not _MODEL_ID.match(model_id):
            raise ConfigError(
                "MODEL_ID may contain letters, digits, '.', '_', '-', '/' (max 96 characters)"
            )

        low = _float(env, "MIN_SCORE_THRESHOLD", 0.05, low=0.001, high=0.99)
        high = _float(env, "MAX_SCORE_THRESHOLD", 0.95, low=0.01, high=0.999)
        default = _float(env, "DEFAULT_SCORE_THRESHOLD", 0.25, low=0.001, high=0.999)
        if not low < high:
            raise ConfigError("MIN_SCORE_THRESHOLD must be lower than MAX_SCORE_THRESHOLD")
        if not low <= default <= high:
            raise ConfigError("DEFAULT_SCORE_THRESHOLD must lie within the min/max thresholds")

        image_size_raw = _clean(env, "INFERENCE_IMAGE_SIZE")
        image_size = None
        if image_size_raw is not None:
            image_size = _int(env, "INFERENCE_IMAGE_SIZE", 640, low=160, high=2048)
            if image_size % 32:
                raise ConfigError("INFERENCE_IMAGE_SIZE must be a multiple of 32")

        # Hugging Face injects SPACE_ID inside a Space. The Space page embeds the app in
        # an iframe served from huggingface.co, so that single ancestor is allowed there.
        frame_default = "'self' https://huggingface.co" if _clean(env, "SPACE_ID") else "'none'"
        frame_ancestors = _clean(env, "FRAME_ANCESTORS") or frame_default
        if any(ch in frame_ancestors for ch in ";,\r\n"):
            raise ConfigError("FRAME_ANCESTORS must be a space-separated CSP source list")

        source_url = _clean(env, "SOURCE_URL")
        if source_url is not None and not re.match(r"^https://[^\s<>\"']{3,300}$", source_url):
            raise ConfigError("SOURCE_URL must be an https:// URL")

        log_level = (_clean(env, "LOG_LEVEL") or "INFO").upper()
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ConfigError("LOG_LEVEL must be DEBUG, INFO, WARNING or ERROR")

        load_mode = (_clean(env, "MODEL_LOAD_MODE") or "background").lower()
        if load_mode not in {"background", "blocking"}:
            raise ConfigError("MODEL_LOAD_MODE must be 'background' or 'blocking'")

        model_path = _clean(env, "MODEL_PATH")
        frontend_dist = _clean(env, "FRONTEND_DIST")

        return cls(
            api_key=_clean(env, "API_KEY"),
            model_id=model_id,
            model_path=Path(model_path) if model_path else None,
            model_repo=_clean(env, "MODEL_REPO"),
            model_filename=_clean(env, "MODEL_FILENAME"),
            model_revision=_clean(env, "MODEL_REVISION"),
            model_sha256=model_sha256,
            hf_token=_clean(env, "HF_TOKEN"),
            device=device,
            inference_image_size=image_size,
            nms_iou=_float(env, "NMS_IOU", 0.7, low=0.1, high=0.95),
            default_score_threshold=default,
            min_score_threshold=low,
            max_score_threshold=high,
            max_detections=_int(env, "MAX_DETECTIONS", 50, low=1, high=300),
            max_polygon_points=_int(env, "MAX_POLYGON_POINTS", 200, low=8, high=2000),
            max_upload_bytes=_int(
                env, "MAX_UPLOAD_BYTES", 12 * 1024 * 1024, low=64 * 1024, high=64 * 1024 * 1024
            ),
            max_image_pixels=_int(
                env, "MAX_IMAGE_PIXELS", 50_000_000, low=65_536, high=200_000_000
            ),
            min_image_side=_int(env, "MIN_IMAGE_SIDE", 32, low=8, high=1024),
            request_body_timeout_s=_float(env, "REQUEST_BODY_TIMEOUT_S", 30.0, low=1.0, high=600.0),
            inference_timeout_s=_float(env, "INFERENCE_TIMEOUT_S", 60.0, low=1.0, high=600.0),
            max_pending_requests=_int(env, "MAX_PENDING_REQUESTS", 4, low=0, high=256),
            rate_limit_per_minute=_int(env, "RATE_LIMIT_PER_MINUTE", 30, low=1, high=100_000),
            trusted_proxy_hops=_int(env, "TRUSTED_PROXY_HOPS", 0, low=0, high=8),
            allowed_origins=_origins(env),
            frame_ancestors=frame_ancestors,
            frontend_dist=Path(frontend_dist) if frontend_dist else None,
            source_url=source_url,
            log_level=log_level,
            model_load_in_background=load_mode == "background",
        )

    # ------------------------------------------------------------------
    def configuration_problems(self) -> list[tuple[str, str]]:
        """Missing or unsafe *requirements* as ``(reason_code, explanation)`` pairs.

        These make the service not-ready instead of crashing it. The explanations
        name environment variables but never echo secret values.
        """
        problems: list[tuple[str, str]] = []
        if not self.api_key:
            problems.append(("api_key_not_configured", "API_KEY is not set."))
        elif len(self.api_key) < MIN_API_KEY_LENGTH:
            problems.append(
                (
                    "api_key_too_short",
                    f"API_KEY must be at least {MIN_API_KEY_LENGTH} characters of random data.",
                )
            )
        if not self.model_id:
            problems.append(("model_id_not_configured", "MODEL_ID is not set."))

        has_path = self.model_path is not None
        has_repo = any((self.model_repo, self.model_filename))
        if has_path and has_repo:
            problems.append(
                (
                    "model_source_ambiguous",
                    "Set either MODEL_PATH or MODEL_REPO/MODEL_FILENAME/MODEL_REVISION, not both.",
                )
            )
        elif not has_path and not has_repo:
            problems.append(
                (
                    "model_not_configured",
                    "No checkpoint configured. Set MODEL_PATH, or MODEL_REPO, MODEL_FILENAME "
                    "and MODEL_REVISION.",
                )
            )
        elif has_repo:
            if not (self.model_repo and self.model_filename and self.model_revision):
                problems.append(
                    (
                        "model_source_incomplete",
                        "MODEL_REPO, MODEL_FILENAME and MODEL_REVISION must all be set.",
                    )
                )
            elif not _COMMIT_SHA.match(self.model_revision):
                problems.append(
                    (
                        "model_revision_not_pinned",
                        "MODEL_REVISION must be a full 40-character commit SHA, not a branch "
                        "or tag, so the deployed checkpoint cannot change silently.",
                    )
                )
        return problems

    def public_summary(self) -> dict[str, object]:
        """Non-secret limits a client needs to validate its own requests."""
        return {
            "max_upload_bytes": self.max_upload_bytes,
            "max_image_pixels": self.max_image_pixels,
            "min_image_side": self.min_image_side,
            "accepted_media_types": ["image/jpeg", "image/png", "image/webp"],
            "score_threshold": {
                "default": self.default_score_threshold,
                "min": self.min_score_threshold,
                "max": self.max_score_threshold,
            },
            "max_detections": self.max_detections,
            "max_polygon_points": self.max_polygon_points,
            "inference_timeout_s": self.inference_timeout_s,
        }
