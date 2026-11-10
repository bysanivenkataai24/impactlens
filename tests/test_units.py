"""Unit tests for configuration, decoding, post-processing, security and the model store."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from impactlens.adapters.base import ModelUnavailable, RawInstance
from impactlens.config import MIN_API_KEY_LENGTH, ConfigError, Settings
from impactlens.imaging import ImageRejected, decode_image, sniff_format
from impactlens.model_store import resolve_checkpoint, sha256_file
from impactlens.postprocess import build_detections, polygon_area, simplify_polygon
from impactlens.service.runtime import build_model_state
from impactlens.service.security import (
    SlidingWindowRateLimiter,
    api_key_matches,
    content_security_policy,
)
from impactlens.taxonomy import MODEL_DAMAGE_TYPES, class_mapping_problem, normalize_class_name

from .conftest import encode

GOOD_KEY = "k" * MIN_API_KEY_LENGTH
SHA = "a" * 40


# ------------------------------------------------------------------- config
def test_defaults_are_safe():
    settings = Settings.from_env({})
    assert settings.device == "cpu"
    assert settings.allowed_origins == ()
    assert settings.frame_ancestors == "'none'"
    assert settings.max_upload_bytes == 12 * 1024 * 1024
    assert {code for code, _ in settings.configuration_problems()} == {
        "api_key_not_configured",
        "model_id_not_configured",
        "model_not_configured",
    }


def test_complete_local_configuration_has_no_problems(tmp_path):
    env = {"API_KEY": GOOD_KEY, "MODEL_ID": "impactlens/yolo11n-seg-v1", "MODEL_PATH": str(tmp_path / "m.pt")}
    assert Settings.from_env(env).configuration_problems() == []


def test_hub_configuration_requires_a_pinned_commit():
    base = {"API_KEY": GOOD_KEY, "MODEL_ID": "m", "MODEL_REPO": "owner/repo", "MODEL_FILENAME": "best.pt"}
    for revision in ("main", "v1.0", SHA[:12], SHA.upper()):
        codes = [c for c, _ in Settings.from_env({**base, "MODEL_REVISION": revision}).configuration_problems()]
        assert codes == ["model_revision_not_pinned"], revision
    assert Settings.from_env({**base, "MODEL_REVISION": SHA}).configuration_problems() == []
    assert [c for c, _ in Settings.from_env(base).configuration_problems()] == ["model_source_incomplete"]


def test_two_checkpoint_sources_are_ambiguous():
    env = {"API_KEY": GOOD_KEY, "MODEL_ID": "m", "MODEL_PATH": "a.pt", "MODEL_REPO": "o/r"}
    assert [c for c, _ in Settings.from_env(env).configuration_problems()] == ["model_source_ambiguous"]


@pytest.mark.parametrize(
    "env",
    [
        {"MAX_UPLOAD_BYTES": "banana"},
        {"MAX_UPLOAD_BYTES": "10"},
        {"DEVICE": "gpu"},
        {"DEVICE": "cuda:x"},
        {"ALLOWED_ORIGINS": "*"},
        {"ALLOWED_ORIGINS": "https://ok.example,*"},
        {"ALLOWED_ORIGINS": "ui.example"},
        {"ALLOWED_ORIGINS": "https://ui.example/path"},
        {"MODEL_SHA256": "xyz"},
        {"DEFAULT_SCORE_THRESHOLD": "nan"},
        {"DEFAULT_SCORE_THRESHOLD": "0.99"},
        {"MIN_SCORE_THRESHOLD": "0.9", "MAX_SCORE_THRESHOLD": "0.5"},
        {"INFERENCE_IMAGE_SIZE": "650"},
        {"FRAME_ANCESTORS": "'none'; script-src *"},
        {"SOURCE_URL": "javascript:alert(1)"},
        {"MODEL_ID": "bad id with spaces"},
        {"LOG_LEVEL": "LOUD"},
        {"TRUSTED_PROXY_HOPS": "-1"},
    ],
)
def test_malformed_values_fail_fast(env):
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_origins_are_normalized_and_deduplicated():
    env = {"ALLOWED_ORIGINS": " https://a.example/ , http://localhost:5173,https://a.example"}
    assert Settings.from_env(env).allowed_origins == ("https://a.example", "http://localhost:5173")


def test_space_id_allows_only_the_hugging_face_frame():
    assert Settings.from_env({"SPACE_ID": "owner/space"}).frame_ancestors == "'self' https://huggingface.co"
    explicit = Settings.from_env({"SPACE_ID": "owner/space", "FRAME_ANCESTORS": "'none'"})
    assert explicit.frame_ancestors == "'none'"


def test_secrets_never_appear_in_repr_or_public_summary():
    settings = Settings.from_env({"API_KEY": "super-secret-key-value-1234567890", "HF_TOKEN": "hf_secrettoken"})
    for text in (repr(settings), str(settings.public_summary()), str(settings.configuration_problems())):
        assert "super-secret" not in text and "hf_secrettoken" not in text


def test_blank_values_count_as_unset():
    settings = Settings.from_env({"API_KEY": "   ", "MODEL_PATH": "", "DEVICE": " "})
    assert settings.api_key is None and settings.model_path is None and settings.device == "cpu"


# ----------------------------------------------------------------- taxonomy
def test_class_name_normalization():
    assert normalize_class_name("Glass Shatter") == "glass_shatter"
    assert normalize_class_name(" lamp-broken ") == "lamp_broken"
    assert normalize_class_name("tire  flat") == "tire_flat"


def test_cardd_style_names_are_accepted():
    cardd = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
    assert class_mapping_problem(cardd) is None
    assert class_mapping_problem(cardd[:5]) is not None
    assert class_mapping_problem([*cardd, "dent"]) is not None
    assert "missing" in class_mapping_problem(["person", "car"])


# ------------------------------------------------------------------ imaging
def test_sniffing_uses_content_only():
    assert sniff_format(encode(Image.new("RGB", (40, 40)), "JPEG")) == "JPEG"
    assert sniff_format(encode(Image.new("RGB", (40, 40)), "PNG")) == "PNG"
    assert sniff_format(encode(Image.new("RGB", (40, 40)), "WEBP")) == "WEBP"
    for other in (b"", b"GIF89a", b"RIFF\0\0\0\0WAVEfmt ", b"\x00\x00\x00\x18ftypheic", b"II*\x00"):
        assert sniff_format(other) is None


def test_decode_returns_contiguous_rgb_and_hash():
    data = encode(Image.new("RGB", (80, 60), (10, 20, 30)), "PNG")
    decoded = decode_image(data, max_pixels=10_000)
    assert decoded.rgb.shape == (60, 80, 3) and decoded.rgb.dtype == np.uint8
    assert decoded.rgb.flags["C_CONTIGUOUS"] and tuple(decoded.rgb[0, 0]) == (10, 20, 30)
    assert (decoded.width, decoded.height, decoded.format) == (80, 60, "PNG")
    assert decoded.sha256 == hashlib.sha256(data).hexdigest()
    assert decoded.exif_orientation is None and decoded.orientation_applied is False


def test_pixel_budget_is_checked_from_the_header():
    data = encode(Image.new("RGB", (200, 200)), "PNG")
    with pytest.raises(ImageRejected) as caught:
        decode_image(data, max_pixels=39_999)
    assert (caught.value.status, caught.value.code) == (413, "image_pixel_limit_exceeded")
    assert decode_image(data, max_pixels=40_000).width == 200


def test_unknown_exif_orientation_values_are_ignored():
    exif = Image.Exif()
    exif[0x0112] = 99
    decoded = decode_image(encode(Image.new("RGB", (80, 60)), "PNG", exif=exif), max_pixels=10_000)
    assert (decoded.width, decoded.height) == (80, 60) and decoded.orientation_applied is False


def test_animated_image_uses_first_frame_only():
    frames = [Image.new("RGB", (64, 64), color) for color in ((255, 0, 0), (0, 255, 0))]
    buffer = io.BytesIO()
    frames[0].save(buffer, "WEBP", save_all=True, append_images=frames[1:], lossless=True)
    decoded = decode_image(buffer.getvalue(), max_pixels=10_000)
    assert tuple(decoded.rgb[0, 0]) == (255, 0, 0)


# -------------------------------------------------------------- postprocess
def _square(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def test_normalization_uses_width_for_x_and_height_for_y():
    instance = RawInstance("dent", 0.5, (100, 50, 300, 150), _square(100, 50, 300, 150))
    (detection,), warnings = build_detections(
        [instance], width=400, height=200, max_detections=10, max_polygon_points=50
    )
    assert detection.bbox == [0.25, 0.25, 0.75, 0.75] and warnings == []
    assert detection.polygon[2] == [0.75, 0.75]


def test_closing_vertex_and_repeats_are_removed():
    ring = np.array([[10, 10], [10, 10], [50, 10], [50, 50], [10, 50], [10, 10]], dtype=np.float64)
    (detection,), _ = build_detections(
        [RawInstance("dent", 0.5, (10, 10, 50, 50), ring)],
        width=100, height=100, max_detections=10, max_polygon_points=50,
    )
    assert len(detection.polygon) == 4


def test_simplify_keeps_shape_within_cap():
    angles = np.linspace(0, 2 * np.pi, 3000, endpoint=False)
    circle = np.stack([100 + 80 * np.cos(angles), 100 + 80 * np.sin(angles)], axis=1)
    simplified, changed = simplify_polygon(circle, 40)
    assert changed and 3 <= len(simplified) <= 40
    assert polygon_area(simplified) == pytest.approx(np.pi * 80**2, rel=0.05)
    untouched, changed = simplify_polygon(circle[:30], 40)
    assert not changed and len(untouched) == 30


def test_build_detections_rejects_impossible_dimensions():
    with pytest.raises(ValueError):
        build_detections([], width=0, height=10, max_detections=1, max_polygon_points=8)


# ----------------------------------------------------------------- security
def test_api_key_comparison():
    assert api_key_matches("abc", "abc")
    assert not api_key_matches("abc", "abd")
    assert not api_key_matches(None, "abc") and not api_key_matches("abc", None)
    assert not api_key_matches("", "") and not api_key_matches("é", "e")


def test_sliding_window_limiter_with_a_fake_clock():
    now = [0.0]
    limiter = SlidingWindowRateLimiter(2, 60.0, clock=lambda: now[0])
    assert limiter.hit("a") is None and limiter.hit("a") is None
    assert limiter.hit("a") == 60
    assert limiter.hit("b") is None  # keys are independent
    now[0] = 45.0
    assert limiter.hit("a") == 15
    now[0] = 60.1
    assert limiter.hit("a") is None  # the first hit left the window


def test_limiter_memory_is_bounded():
    now = [0.0]
    limiter = SlidingWindowRateLimiter(5, 60.0, max_keys=100, clock=lambda: now[0])
    for i in range(1000):
        limiter.hit(f"client-{i}")
    assert len(limiter._hits) <= 100


def test_csp_has_no_unsafe_sources():
    csp = content_security_policy("'none'")
    assert "unsafe" not in csp and "default-src 'self'" in csp and csp.endswith("frame-ancestors 'none'")


def test_forwarded_header_is_ignored_unless_proxies_are_trusted():
    from starlette.requests import Request

    from impactlens.service.security import client_address

    def request(xff: str) -> Request:
        return Request(
            {"type": "http", "client": ("10.0.0.9", 1234), "headers": [(b"x-forwarded-for", xff.encode())]}
        )

    assert client_address(request("1.2.3.4"), 0) == "10.0.0.9"
    assert client_address(request("6.6.6.6, 1.2.3.4"), 1) == "1.2.3.4"  # spoofed prefix ignored
    assert client_address(request("6.6.6.6, 1.2.3.4, 10.1.1.1"), 2) == "1.2.3.4"
    assert client_address(request(""), 1) == "10.0.0.9"


# -------------------------------------------------------------- model store
def test_sha256_and_default_revision(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not really a checkpoint")
    digest = hashlib.sha256(b"not really a checkpoint").hexdigest()
    assert sha256_file(checkpoint) == digest
    resolved = resolve_checkpoint(Settings(model_path=checkpoint))
    assert (resolved.sha256, resolved.revision, resolved.source) == (digest, f"sha256:{digest[:12]}", "local")
    labelled = resolve_checkpoint(Settings(model_path=checkpoint, model_revision="run-2026-11-10"))
    assert labelled.revision == "run-2026-11-10"


def test_hash_mismatch_is_detected_before_any_model_code_runs(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ModelUnavailable) as caught:
        resolve_checkpoint(Settings(model_path=checkpoint, model_sha256="b" * 64))
    assert caught.value.code == "checkpoint_hash_mismatch"


def test_missing_checkpoint_file():
    with pytest.raises(ModelUnavailable) as caught:
        resolve_checkpoint(Settings(model_path=Path("/nonexistent/model.pt")))
    assert caught.value.code == "checkpoint_not_found"


def test_default_loader_reports_missing_file_as_not_ready(tmp_path):
    from impactlens.service.runtime import default_model_loader

    settings = Settings(api_key=GOOD_KEY, model_id="m", model_path=tmp_path / "absent.pt")
    state = build_model_state(settings, default_model_loader)
    assert state.ready is False and state.reasons == ["checkpoint_not_found"]
    assert state.adapter is None


def test_default_loader_refuses_a_file_that_is_not_a_checkpoint(tmp_path):
    """A random file with a .pt name must make the service not-ready, not crash it."""
    from impactlens.service.runtime import default_model_loader

    fake = tmp_path / "fake.pt"
    fake.write_bytes(b"this is not a torch checkpoint")
    state = build_model_state(Settings(api_key=GOOD_KEY, model_id="m", model_path=fake), default_model_loader)
    assert state.ready is False and state.reasons == ["model_load_failed"]


def test_default_loader_refuses_unsupported_file_types(tmp_path):
    from impactlens.service.runtime import default_model_loader

    other = tmp_path / "model.onnx"
    other.write_bytes(b"x")
    state = build_model_state(Settings(api_key=GOOD_KEY, model_id="m", model_path=other), default_model_loader)
    assert state.reasons == ["unsupported_checkpoint_format"]


def test_taxonomy_is_exactly_the_six_supported_classes():
    assert MODEL_DAMAGE_TYPES == ("dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat")
