"""Resolve the configured checkpoint to a local file and fingerprint it.

The SHA-256 is computed, and compared with ``MODEL_SHA256`` when that is set,
*before* any model library opens the file. PyTorch ``.pt`` checkpoints are
pickles; refusing an unexpected file before it is deserialized is the point.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from .adapters.base import ModelUnavailable
from .config import Settings

log = logging.getLogger("impactlens.model_store")


@dataclass(frozen=True)
class ResolvedCheckpoint:
    path: Path
    sha256: str
    #: Pinned commit for hub checkpoints; operator label or content hash for local files.
    revision: str
    source: str  # "local" | "huggingface"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _download_from_hub(settings: Settings) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover - broken installation
        raise ModelUnavailable("checkpoint_download_failed", "huggingface_hub is missing") from exc
    try:
        local = hf_hub_download(
            repo_id=settings.model_repo,
            filename=settings.model_filename,
            revision=settings.model_revision,
            token=settings.hf_token or False,  # False: never pick up an ambient token
        )
    except Exception as exc:
        # The exception text can contain request URLs; log the type only.
        log.error("checkpoint download failed: %s", type(exc).__name__)
        raise ModelUnavailable(
            "checkpoint_download_failed",
            f"{type(exc).__name__} while downloading the pinned checkpoint",
        ) from exc
    return Path(local)


def resolve_checkpoint(settings: Settings) -> ResolvedCheckpoint:
    """Locate (or download) the checkpoint and verify its hash. Raises ModelUnavailable."""
    if settings.model_path is not None:
        path, source = settings.model_path, "local"
        if not path.is_file():
            raise ModelUnavailable("checkpoint_not_found", "MODEL_PATH does not point to a file")
    else:
        path, source = _download_from_hub(settings), "huggingface"

    digest = sha256_file(path)
    if settings.model_sha256 and digest != settings.model_sha256:
        raise ModelUnavailable(
            "checkpoint_hash_mismatch",
            "checkpoint SHA-256 differs from MODEL_SHA256; refusing to load it",
        )
    revision = settings.model_revision or f"sha256:{digest[:12]}"
    return ResolvedCheckpoint(path=path, sha256=digest, revision=revision, source=source)
