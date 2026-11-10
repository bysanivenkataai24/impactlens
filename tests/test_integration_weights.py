"""Opt-in tests that load REAL Ultralytics checkpoints. Skipped unless configured.

IMPACTLENS_TEST_WEIGHTS=<path to a trained ImpactLens damage checkpoint>
    Loads it through the production loader, runs /predict on real photos from
    IMPACTLENS_TEST_IMAGES (a directory) and checks the contract holds. This
    proves the integration works; it does not measure accuracy - use
    `impactlens evaluate` for that.

IMPACTLENS_TEST_COCO_WEIGHTS=<path to an official yolo11n-seg.pt>
    Proves that generic COCO weights are refused: they are an initialization,
    not a damage detector.

IMPACTLENS_TEST_SMOKE_WEIGHTS=<path to a checkpoint from `impactlens smoke`>
    Proves that smoke-test checkpoints are refused by the service.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from impactlens.config import Settings
from impactlens.schemas import PredictResponse
from impactlens.service import create_app

from .conftest import API_KEY

pytestmark = pytest.mark.integration


def _path(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"{variable} is not set")
    path = Path(value)
    if not path.is_file():
        pytest.fail(f"{variable} points to a missing file: {path}")
    return path


def _settings(weights: Path) -> Settings:
    return Settings(api_key=API_KEY, model_id="integration-test", model_path=weights)


def test_coco_checkpoint_is_refused_by_class_mapping():
    with TestClient(create_app(_settings(_path("IMPACTLENS_TEST_COCO_WEIGHTS")))) as client:
        health = client.get("/health").json()
        assert health["ready"] is False and health["reasons"] == ["class_mapping_mismatch"]
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {API_KEY}"},
            files={"image": ("a.png", b"\x89PNG\r\n\x1a\n", "image/png")},
        )
        assert response.status_code == 503


def test_smoke_checkpoint_is_refused():
    with TestClient(create_app(_settings(_path("IMPACTLENS_TEST_SMOKE_WEIGHTS")))) as client:
        assert client.get("/health").json()["reasons"] == ["smoke_checkpoint_refused"]


def test_trained_checkpoint_serves_contract_conforming_predictions():
    weights = _path("IMPACTLENS_TEST_WEIGHTS")
    image_dir = Path(os.environ.get("IMPACTLENS_TEST_IMAGES", ""))
    photos = (
        sorted(
            p for p in image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if image_dir.is_dir()
        else []
    )
    if not photos:
        pytest.skip("IMPACTLENS_TEST_IMAGES must name a directory of real photos")
    with TestClient(create_app(_settings(weights))) as client:
        health = client.get("/health").json()
        assert health["ready"] is True, health["reasons"]
        for photo in photos[:10]:
            response = client.post(
                "/predict",
                headers={"Authorization": f"Bearer {API_KEY}"},
                files={"image": (photo.name, photo.read_bytes(), "application/octet-stream")},
            )
            assert response.status_code == 200, response.text
            body = PredictResponse.model_validate(response.json())
            assert body.checkpoint_sha256 == health["checkpoint_sha256"]
