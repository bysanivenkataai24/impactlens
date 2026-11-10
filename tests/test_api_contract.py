"""HTTP contract tests with a SYNTHETIC adapter (see conftest.py).

These verify request validation, the response contract, coordinate handling and
error behaviour. They do not, and cannot, say anything about model quality.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from impactlens.adapters.base import ModelUnavailable, RawInstance
from impactlens.schemas import LIMITATIONS, PredictResponse
from impactlens.service import create_app
from impactlens.taxonomy import MODEL_DAMAGE_TYPES

from .conftest import (
    API_KEY,
    MAGENTA,
    SYNTHETIC_SHA256,
    SyntheticAdapter,
    canvas_with_block,
    encode,
    make_settings,
    post_image,
    synthetic_loader,
)

pytestmark = pytest.mark.contract


def app_client(adapter=None, loader=None, **settings) -> TestClient:
    adapter = adapter or SyntheticAdapter()
    app = create_app(make_settings(**settings), model_loader=loader or synthetic_loader(adapter))
    return TestClient(app)


def error_code(response) -> str:
    body = response.json()
    assert body["schema_version"] == "1.0.0"
    assert body["error"]["request_id"] == response.headers["x-request-id"]
    return body["error"]["code"]


# --------------------------------------------------------------------- health
def test_health_reports_ready_model_identity_and_limits(client):
    body = client.get("/health").json()
    assert body["ready"] is True and body["reasons"] == []
    assert body["model_id"] == "synthetic-test-double"
    assert body["model_revision"] == "test"
    assert body["checkpoint_sha256"] == SYNTHETIC_SHA256
    assert body["task"] == "segment"
    assert body["classes"] == list(MODEL_DAMAGE_TYPES)
    assert body["limits"]["accepted_media_types"] == ["image/jpeg", "image/png", "image/webp"]
    assert body["limits"]["score_threshold"] == {"default": 0.25, "min": 0.05, "max": 0.95}
    assert client.get("/ready").status_code == 200


def test_health_never_contains_the_api_key(client):
    assert API_KEY not in client.get("/health").text


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"api_key": None}, "api_key_not_configured"),
        ({"api_key": "short"}, "api_key_too_short"),
        ({"model_id": None}, "model_id_not_configured"),
    ],
)
def test_missing_required_configuration_makes_service_not_ready(overrides, reason, auth):
    with app_client(**overrides) as client:
        body = client.get("/health").json()
        assert body["ready"] is False and reason in body["reasons"]
        assert client.get("/ready").status_code == 503
        # A request can never be served while required configuration is missing.
        response = post_image(client, canvas_with_block(64, 64, (8, 8, 32, 32)), auth)
        assert response.status_code in (401, 503)
        assert "detections" not in response.json()


def test_no_checkpoint_configured_is_not_ready_and_loader_is_not_called(auth):
    def loader(_settings):
        raise AssertionError("loader must not run without a configured checkpoint")

    with app_client(loader=loader, model_path=None) as client:
        body = client.get("/health").json()
        assert body["ready"] is False and body["reasons"] == ["model_not_configured"]
        assert body["model_revision"] is None and body["checkpoint_sha256"] is None
        response = post_image(client, canvas_with_block(64, 64, (8, 8, 32, 32)), auth)
        assert response.status_code == 503
        assert error_code(response) == "model_unavailable"
        assert response.json()["error"]["reasons"] == ["model_not_configured"]


@pytest.mark.parametrize("code", ["checkpoint_not_found", "checkpoint_hash_mismatch"])
def test_unavailable_model_returns_503_and_never_detections(code, auth):
    def loader(_settings):
        raise ModelUnavailable(code, "detail that must stay in the server log /srv/models/x.pt")

    with app_client(loader=loader) as client:
        health = client.get("/health").json()
        assert health["ready"] is False and health["reasons"] == [code]
        response = post_image(client, canvas_with_block(64, 64, (8, 8, 32, 32)), auth)
        assert response.status_code == 503
        assert error_code(response) == "model_unavailable"
        assert "detections" not in response.json()
        assert "/srv/models" not in response.text and "/srv/models" not in client.get("/health").text


def test_unexpected_loader_crash_is_reported_as_model_load_failed(auth):
    def loader(_settings):
        raise RuntimeError("boom")

    with app_client(loader=loader) as client:
        assert client.get("/health").json()["reasons"] == ["model_load_failed"]
        assert post_image(client, b"x", auth).status_code == 503


def test_checkpoint_with_unsupported_classes_is_refused(auth):
    coco_like = SyntheticAdapter(class_names=("person", "bicycle", "car"))
    with app_client(adapter=coco_like) as client:
        health = client.get("/health").json()
        assert health["ready"] is False and health["reasons"] == ["class_mapping_mismatch"]
        response = post_image(client, canvas_with_block(64, 64, (8, 8, 32, 32)), auth)
        assert response.status_code == 503 and coco_like.calls == 0


def test_checkpoint_with_extra_class_is_refused():
    extra = SyntheticAdapter(class_names=(*MODEL_DAMAGE_TYPES, "rust"))
    with app_client(adapter=extra) as client:
        assert client.get("/health").json()["reasons"] == ["class_mapping_mismatch"]


def test_non_segmentation_checkpoint_is_refused():
    with app_client(adapter=SyntheticAdapter(task="detect")) as client:
        assert client.get("/health").json()["reasons"] == ["unsupported_task"]


def test_class_names_are_matched_by_name_not_by_index():
    shuffled = SyntheticAdapter(class_names=tuple(reversed(MODEL_DAMAGE_TYPES)))
    with app_client(adapter=shuffled) as client:
        assert client.get("/health").json()["ready"] is True


# ----------------------------------------------------------------------- auth
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong-key-wrong-key-wrong-key"},
        {"Authorization": f"Basic {API_KEY}"},
        {"Authorization": "Bearer "},
        {"X-API-Key": API_KEY + "x"},
    ],
)
def test_predict_requires_a_valid_key(client, adapter, headers):
    response = post_image(client, canvas_with_block(64, 64, (8, 8, 32, 32)), headers)
    assert response.status_code == 401
    assert error_code(response) == "unauthenticated"
    assert response.headers["www-authenticate"].startswith("Bearer")
    assert adapter.calls == 0


@pytest.mark.parametrize(
    "headers", [{"Authorization": f"Bearer {API_KEY}"}, {"X-API-Key": API_KEY}]
)
def test_both_key_headers_are_accepted(client, headers):
    assert post_image(client, canvas_with_block(64, 64, (8, 8, 32, 32)), headers).status_code == 200
    assert client.get("/auth/check", headers=headers).json()["authenticated"] is True


def test_auth_check_rejects_missing_key(client):
    assert client.get("/auth/check").status_code == 401


# ------------------------------------------------------------------- success
def test_predict_returns_the_documented_contract(client, auth):
    data = canvas_with_block(400, 200, (100, 50, 300, 150))
    response = post_image(client, data, auth)
    assert response.status_code == 200
    body = response.json()
    PredictResponse.model_validate(body)  # strict: unknown or missing fields fail

    assert body["schema_version"] == "1.0.0"
    assert body["request_id"] == response.headers["x-request-id"]
    assert body["model_id"] == "synthetic-test-double"
    assert body["model_revision"] == "test"
    assert body["checkpoint_sha256"] == SYNTHETIC_SHA256
    assert body["image"]["width"] == 400 and body["image"]["height"] == 200
    assert body["image"]["format"] == "PNG" and body["image"]["exif_orientation"] is None
    assert body["score_threshold"] == 0.25
    assert body["outcome"] == "detections_returned"
    assert body["limitations"] == list(LIMITATIONS)
    assert set(body["latency_ms"]) == {"queue", "decode", "inference", "postprocess", "total"}

    (detection,) = body["detections"]
    assert detection["id"] == "det-1"
    assert detection["damage_type"] == "dent"
    assert detection["score"] == 0.9
    assert detection["bbox"] == [0.25, 0.25, 0.75, 0.75]
    assert detection["polygon"] == [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]
    assert detection["part"] is None
    assert detection["severity"] == "unknown"
    assert detection["review_status"] == "pending_review"
    assert response.headers["cache-control"] == "no-store"


def test_image_sha256_is_the_hash_of_the_uploaded_bytes(client, auth):
    import hashlib

    data = canvas_with_block(64, 64, (8, 8, 32, 32))
    assert post_image(client, data, auth).json()["image"]["sha256"] == hashlib.sha256(data).hexdigest()


def test_no_detection_is_not_reported_as_undamaged(client, auth):
    plain = encode(Image.new("RGB", (320, 240), (128, 128, 128)), "PNG")
    body = post_image(client, plain, auth).json()
    assert body["detections"] == []
    assert body["outcome"] == "no_supported_region_returned"
    assert "does not mean the vehicle is undamaged" in body["message"]
    assert "undamaged" not in body["outcome"] and "no_damage" not in body["outcome"]


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_all_three_formats_are_decoded(client, auth, fmt):
    kwargs = {"lossless": True} if fmt == "WEBP" else {}
    body = post_image(client, canvas_with_block(320, 160, (80, 40, 240, 120), fmt, **kwargs), auth).json()
    assert body["image"]["format"] == fmt
    assert body["detections"][0]["bbox"] == pytest.approx([0.25, 0.25, 0.75, 0.75], abs=0.02)


# ----------------------------------------------------------------- threshold
@pytest.mark.parametrize("value", ["abc", "nan", "inf", "-inf", "0", "0.049", "0.951", "1.5", "", "0.5,0.6"])
def test_invalid_threshold_is_rejected(client, auth, adapter, value):
    response = post_image(client, canvas_with_block(64, 64, (8, 8, 32, 32)), auth, score_threshold=value)
    assert response.status_code == 422
    assert error_code(response) == "invalid_request"
    assert adapter.calls == 0


@pytest.mark.parametrize("value", ["0.05", "0.5", "0.95", " 0.3 "])
def test_valid_threshold_is_echoed(client, auth, value):
    body = post_image(client, canvas_with_block(64, 64, (8, 8, 32, 32)), auth, score_threshold=value).json()
    assert body["score_threshold"] == pytest.approx(float(value))


def test_threshold_is_passed_to_the_adapter(client, auth):
    # The synthetic adapter returns nothing above 0.9.
    data = canvas_with_block(64, 64, (8, 8, 32, 32))
    assert post_image(client, data, auth, score_threshold="0.95").json()["detections"] == []


# ------------------------------------------------------------ media / format
GIF = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"


@pytest.mark.parametrize(
    "payload",
    [GIF, b"%PDF-1.7 not an image", b"<svg xmlns='http://www.w3.org/2000/svg'/>", b"BM" + b"\0" * 64, b"hello"],
    ids=["gif", "pdf", "svg", "bmp", "text"],
)
def test_unsupported_content_is_415_regardless_of_declared_type(client, auth, adapter, payload):
    response = client.post(
        "/predict", headers=auth, files={"image": ("photo.jpg", payload, "image/jpeg")}
    )
    assert response.status_code == 415
    assert error_code(response) == "unsupported_media_type"
    assert adapter.calls == 0


def test_non_multipart_request_is_415(client, auth):
    response = client.post("/predict", headers=auth, json={"image": "aGVsbG8="})
    assert response.status_code == 415
    raw = client.post(
        "/predict",
        headers={**auth, "Content-Type": "image/jpeg"},
        content=canvas_with_block(64, 64, (8, 8, 32, 32), "JPEG"),
    )
    assert raw.status_code == 415


def test_truncated_and_corrupt_images_are_422(client, auth, adapter):
    jpeg = canvas_with_block(256, 256, (10, 10, 100, 100), "JPEG")
    png = canvas_with_block(256, 256, (10, 10, 100, 100), "PNG")
    for payload in (jpeg[: len(jpeg) // 3], png[:40], b"\xff\xd8\xff" + b"\x00" * 200, b"\x89PNG\r\n\x1a\n" + b"junk" * 50):
        response = post_image(client, payload, auth)
        assert response.status_code == 422, response.text
        assert error_code(response) == "malformed_image"
    assert adapter.calls == 0


def test_empty_file_is_422(client, auth):
    response = post_image(client, b"", auth)
    assert response.status_code == 422 and error_code(response) == "malformed_image"


def test_tiny_image_is_422(client, auth):
    response = post_image(client, encode(Image.new("RGB", (16, 200)), "PNG"), auth)
    assert response.status_code == 422 and error_code(response) == "image_too_small"


def test_missing_image_field_is_422(client, auth):
    response = client.post("/predict", headers=auth, files={"photo": ("a.png", b"x", "image/png")})
    assert response.status_code == 422 and error_code(response) == "invalid_request"
    only_text = client.post("/predict", headers=auth, files={"image": (None, "not a file")})
    assert only_text.status_code == 422


def test_two_files_are_rejected(client, auth):
    data = canvas_with_block(64, 64, (8, 8, 32, 32))
    response = client.post(
        "/predict",
        headers=auth,
        files=[("image", ("a.png", data, "image/png")), ("image", ("b.png", data, "image/png"))],
    )
    assert response.status_code == 422


def test_unknown_form_field_is_rejected(client, auth):
    data = canvas_with_block(64, 64, (8, 8, 32, 32))
    response = post_image(client, data, auth, image_url="http://169.254.169.254/latest/meta-data")
    assert response.status_code == 422 and error_code(response) == "invalid_request"


# --------------------------------------------------------------------- limits
def _multipart(data: bytes, boundary: str = "testboundary") -> tuple[bytes, str]:
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="p.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def test_oversized_upload_is_413_before_any_decoding(auth):
    adapter = SyntheticAdapter()
    with app_client(adapter=adapter, max_upload_bytes=64 * 1024) as client:
        rng = np.random.default_rng(0)
        noisy = encode(Image.fromarray(rng.integers(0, 255, (400, 400, 3), dtype=np.uint8)), "PNG")
        assert len(noisy) > 64 * 1024 + 16 * 1024
        response = post_image(client, noisy, auth)
        assert response.status_code == 413 and error_code(response) == "payload_too_large"
        assert adapter.calls == 0


def test_size_limit_is_enforced_on_streamed_bytes_without_content_length(auth):
    adapter = SyntheticAdapter()
    with app_client(adapter=adapter, max_upload_bytes=64 * 1024) as client:
        body, content_type = _multipart(b"\x89PNG\r\n\x1a\n" + b"\0" * (200 * 1024))

        def chunks():
            for start in range(0, len(body), 8192):
                yield body[start : start + 8192]

        response = client.post(
            "/predict", headers={**auth, "Content-Type": content_type}, content=chunks()
        )
        assert "content-length" not in response.request.headers
        assert response.status_code == 413 and error_code(response) == "payload_too_large"
        assert adapter.calls == 0


def test_file_just_over_the_limit_but_inside_the_overhead_allowance_is_413(auth):
    with app_client(max_upload_bytes=64 * 1024) as client:
        payload = b"\x89PNG\r\n\x1a\n" + b"\0" * (64 * 1024)  # limit + 8 bytes
        response = post_image(client, payload, auth)
        assert response.status_code == 413


def test_decompression_bomb_is_rejected_by_pixel_count(auth):
    adapter = SyntheticAdapter()
    with app_client(adapter=adapter, max_image_pixels=1_000_000) as client:
        bomb = encode(Image.new("RGB", (3000, 3000), (10, 10, 10)), "PNG")
        assert len(bomb) < 100_000  # small on the wire, 9 MP when decoded
        response = post_image(client, bomb, auth)
        assert response.status_code == 413
        assert error_code(response) == "image_pixel_limit_exceeded"
        assert adapter.calls == 0


# ---------------------------------------------------------------- orientation
def _stored_for_display(display: np.ndarray, orientation: int) -> np.ndarray:
    """Pixels a camera would store so that EXIF ``orientation`` yields ``display``.

    Written from the EXIF specification with plain numpy, independently of the
    Pillow helper the service uses. 2, 3, 4, 5 and 7 are their own inverse.
    """
    if orientation == 1:
        return display
    if orientation == 2:  # mirrored horizontally
        return display[:, ::-1]
    if orientation == 3:  # rotated 180
        return display[::-1, ::-1]
    if orientation == 4:  # mirrored vertically
        return display[::-1, :]
    if orientation == 5:  # transposed
        return display.transpose(1, 0, 2)
    if orientation == 6:  # viewer rotates 90 CW, so stored is display rotated 90 CCW
        return np.rot90(display, k=1)
    if orientation == 7:  # transverse
        return display[::-1, ::-1].transpose(1, 0, 2)
    if orientation == 8:  # viewer rotates 90 CCW, so stored is display rotated 90 CW
        return np.rot90(display, k=-1)
    raise ValueError(orientation)


@pytest.mark.parametrize("orientation", range(1, 9))
def test_coordinates_refer_to_the_exif_oriented_image(client, auth, orientation):
    # What the reviewer sees: a 300x500 portrait with a block near the top-left.
    width, height, box = 300, 500, (30, 50, 120, 150)
    display = np.full((height, width, 3), 128, dtype=np.uint8)
    display[box[1] : box[3], box[0] : box[2]] = MAGENTA
    stored = np.ascontiguousarray(_stored_for_display(display, orientation))

    exif = Image.Exif()
    exif[0x0112] = orientation
    data = encode(Image.fromarray(stored), "PNG", exif=exif)

    body = post_image(client, data, auth).json()
    assert (body["image"]["width"], body["image"]["height"]) == (width, height)
    assert body["image"]["exif_orientation"] == orientation
    assert body["image"]["orientation_applied"] is (orientation != 1)
    expected = [box[0] / width, box[1] / height, box[2] / width, box[3] / height]
    assert body["detections"][0]["bbox"] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("fmt", ["JPEG", "WEBP"])
def test_phone_style_rotated_upload_in_lossy_formats(client, auth, fmt):
    width, height, box = 300, 500, (30, 50, 120, 150)
    display = np.full((height, width, 3), 128, dtype=np.uint8)
    display[box[1] : box[3], box[0] : box[2]] = MAGENTA
    exif = Image.Exif()
    exif[0x0112] = 6
    data = encode(Image.fromarray(np.ascontiguousarray(_stored_for_display(display, 6))), fmt, exif=exif)
    body = post_image(client, data, auth).json()
    assert (body["image"]["width"], body["image"]["height"]) == (width, height)
    expected = [box[0] / width, box[1] / height, box[2] / width, box[3] / height]
    assert body["detections"][0]["bbox"] == pytest.approx(expected, abs=0.02)


def test_modes_other_than_rgb_are_converted(client, auth):
    for mode in ("L", "RGBA", "P", "CMYK"):
        image = Image.new("RGB", (64, 64), (128, 128, 128)).convert(mode)
        fmt = "JPEG" if mode == "CMYK" else "PNG"
        assert post_image(client, encode(image, fmt), auth).status_code == 200, mode


# ------------------------------------------------- adapter output validation
def _poly(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def test_invalid_adapter_output_is_dropped_and_coordinates_are_clamped(auth):
    adapter = SyntheticAdapter()
    adapter.override = lambda _img: [
        RawInstance("dent", 0.8, (-50.0, -20.0, 60.0, 40.0), _poly(-50, -20, 60, 40)),  # clamp
        RawInstance("scratch", float("nan"), (1, 1, 20, 20), _poly(1, 1, 20, 20)),
        RawInstance("crack", 0.7, (1.0, 1.0, float("inf"), 20.0), _poly(1, 1, 20, 20)),
        RawInstance("crack", 0.7, (1, 1, 20, 20), np.array([[1, 1], [float("nan"), 2], [3, 3]])),
        RawInstance("rust", 0.9, (1, 1, 20, 20), _poly(1, 1, 20, 20)),  # outside taxonomy
        RawInstance("dent", 1.7, (1, 1, 20, 20), _poly(1, 1, 20, 20)),  # impossible score
        RawInstance("dent", 0.6, (30, 30, 30, 50), _poly(30, 30, 30, 50)),  # zero-area box
        RawInstance("dent", 0.6, (30, 30, 50, 50), np.zeros((0, 2))),  # empty mask
        RawInstance("dent", 0.6, (30, 30, 50, 50), np.array([[30, 30], [50, 50]])),  # 2 points
    ]
    with app_client(adapter=adapter) as client:
        response = post_image(client, encode(Image.new("RGB", (100, 80)), "PNG"), auth)
        body = response.json()
        PredictResponse.model_validate(body)
        assert [d["damage_type"] for d in body["detections"]] == ["dent"]
        assert body["detections"][0]["bbox"] == [0.0, 0.0, 0.6, 0.5]
        assert body["warnings"] == ["invalid_detections_dropped"]
        flat = [v for d in body["detections"] for p in d["polygon"] for v in p]
        assert all(0.0 <= v <= 1.0 for v in flat)


def test_detection_count_is_capped_and_ordered_by_score(auth):
    adapter = SyntheticAdapter()
    adapter.override = lambda _img: [
        RawInstance("scratch", 0.10 + i * 0.01, (i, i, i + 10, i + 10), _poly(i, i, i + 10, i + 10))
        for i in range(60)
    ]
    with app_client(adapter=adapter, max_detections=5) as client:
        body = post_image(client, encode(Image.new("RGB", (200, 200)), "PNG"), auth).json()
        scores = [d["score"] for d in body["detections"]]
        assert len(scores) == 5 and scores == sorted(scores, reverse=True)
        assert [d["id"] for d in body["detections"]] == [f"det-{i}" for i in range(1, 6)]
        assert "detection_cap_reached" in body["warnings"]


def test_polygon_vertex_count_is_capped(auth):
    angles = np.linspace(0, 2 * np.pi, 5000, endpoint=False)
    circle = np.stack([500 + 300 * np.cos(angles), 500 + 300 * np.sin(angles)], axis=1)
    adapter = SyntheticAdapter()
    adapter.override = lambda _img: [RawInstance("dent", 0.9, (200, 200, 800, 800), circle)]
    with app_client(adapter=adapter, max_polygon_points=64) as client:
        body = post_image(client, encode(Image.new("RGB", (1000, 1000)), "PNG"), auth).json()
        detection = body["detections"][0]
        assert 3 <= len(detection["polygon"]) <= 64
        assert "polygon_simplified" in detection["flags"]


def test_documented_flags(auth):
    adapter = SyntheticAdapter()
    adapter.override = lambda _img: [
        RawInstance("dent", 0.9, (0, 100, 300, 400), _poly(0, 100, 300, 400), fragment_count=3),
        RawInstance("scratch", 0.8, (500, 500, 510, 510), _poly(500, 500, 510, 510)),
        RawInstance("crack", 0.7, (300, 300, 700, 700), _poly(300, 300, 700, 700)),
    ]
    with app_client(adapter=adapter) as client:
        body = post_image(client, encode(Image.new("RGB", (1000, 1000)), "PNG"), auth).json()
        flags = {d["damage_type"]: d["flags"] for d in body["detections"]}
        assert flags == {
            "dent": ["mask_fragmented", "touches_image_border"],
            "scratch": ["small_region"],
            "crack": [],
        }


# -------------------------------------------------------------------- errors
def test_inference_error_is_500_without_leaking_details_or_detections(auth):
    adapter = SyntheticAdapter()
    adapter.fail_with = RuntimeError("CUDA error at /srv/secret/path line 12")
    with app_client(adapter=adapter) as client:
        response = post_image(client, canvas_with_block(64, 64, (8, 8, 32, 32)), auth)
        assert response.status_code == 500
        assert error_code(response) == "inference_failed"
        assert "secret" not in response.text and "CUDA" not in response.text
        assert "detections" not in response.json()
        # The service keeps working afterwards.
        adapter.fail_with = None
        assert post_image(client, canvas_with_block(64, 64, (8, 8, 32, 32)), auth).status_code == 200


def test_rate_limit_returns_429_with_retry_after(auth):
    with app_client(rate_limit_per_minute=3) as client:
        data = canvas_with_block(64, 64, (8, 8, 32, 32))
        assert [post_image(client, data, auth).status_code for _ in range(3)] == [200, 200, 200]
        limited = post_image(client, data, auth)
        assert limited.status_code == 429
        assert error_code(limited) == "rate_limited"
        assert int(limited.headers["retry-after"]) >= 1
        assert limited.json()["error"]["retry_after_s"] == int(limited.headers["retry-after"])
        assert client.get("/health").status_code == 200  # health is not rate limited


def test_failed_authentication_attempts_are_rate_limited_too():
    with app_client(rate_limit_per_minute=2) as client:
        wrong = {"Authorization": "Bearer nope-nope-nope-nope-nope-nope"}
        statuses = [client.get("/auth/check", headers=wrong).status_code for _ in range(3)]
        assert statuses == [401, 401, 429]


def test_unknown_route_and_wrong_method_use_the_error_envelope(client):
    missing = client.get("/does-not-exist")
    assert missing.status_code == 404 and error_code(missing) == "not_found"
    wrong = client.get("/predict")
    assert wrong.status_code == 405 and error_code(wrong) == "method_not_allowed"


# ---------------------------------------------------------- headers and CORS
def test_security_headers_on_every_kind_of_response(client, auth):
    responses = [
        client.get("/health"),
        client.get("/does-not-exist"),
        post_image(client, b"nope", auth),
    ]
    for response in responses:
        headers = response.headers
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["referrer-policy"] == "no-referrer"
        assert headers["x-frame-options"] == "DENY"
        csp = headers["content-security-policy"]
        assert "script-src 'self'" in csp and "frame-ancestors 'none'" in csp
        assert "'unsafe-inline'" not in csp and "'unsafe-eval'" not in csp
        assert "object-src 'none'" in csp
        assert len(headers["x-request-id"]) == 32
        assert "server" not in headers or "uvicorn" not in headers["server"].lower()


def test_request_ids_are_unique_and_server_generated(client):
    first = client.get("/health", headers={"X-Request-ID": "attacker\r\nchosen"})
    second = client.get("/health")
    assert first.headers["x-request-id"] != second.headers["x-request-id"]
    assert "attacker" not in first.headers["x-request-id"]


def test_frame_ancestors_setting_controls_embedding():
    with app_client(frame_ancestors="'self' https://huggingface.co") as client:
        headers = client.get("/health").headers
        assert "frame-ancestors 'self' https://huggingface.co" in headers["content-security-policy"]
        assert "x-frame-options" not in headers


def test_cors_is_off_unless_origins_are_configured(client):
    response = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in response.headers


def test_cors_allows_only_listed_origins(auth):
    with app_client(allowed_origins=("https://ui.example",)) as client:
        ok = client.get("/health", headers={"Origin": "https://ui.example"})
        assert ok.headers["access-control-allow-origin"] == "https://ui.example"
        assert "access-control-allow-credentials" not in ok.headers
        other = client.get("/health", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in other.headers

        preflight = client.options(
            "/predict",
            headers={
                "Origin": "https://ui.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "https://ui.example"

        # Errors must be readable by the allowed UI as well.
        denied = client.post("/predict", headers={"Origin": "https://ui.example"})
        assert denied.status_code == 401
        assert denied.headers["access-control-allow-origin"] == "https://ui.example"


def test_cors_does_not_replace_authentication():
    with app_client(allowed_origins=("https://ui.example",)) as client:
        response = client.post(
            "/predict",
            headers={"Origin": "https://ui.example"},
            files={"image": ("a.png", canvas_with_block(64, 64, (8, 8, 32, 32)), "image/png")},
        )
        assert response.status_code == 401


# ---------------------------------------------------------------- static UI
def test_static_ui_is_served_and_api_routes_keep_priority(tmp_path, auth):
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><title>ImpactLens</title>")
    (tmp_path / "assets" / "app-abc123.js").write_text("console.log('ui')")
    with app_client(frontend_dist=tmp_path) as client:
        index = client.get("/")
        assert index.status_code == 200 and "ImpactLens" in index.text
        assert index.headers["cache-control"] == "no-cache"
        asset = client.get("/assets/app-abc123.js")
        assert asset.headers["cache-control"] == "public, max-age=31536000, immutable"
        assert client.get("/health").json()["status"] == "ok"
        assert post_image(client, canvas_with_block(64, 64, (8, 8, 32, 32)), auth).status_code == 200
        assert client.get("/../pyproject.toml").status_code == 404
        assert client.get("/assets/%2e%2e/%2e%2e/pyproject.toml").status_code == 404


def test_interactive_docs_are_disabled_but_openapi_is_published(client):
    assert client.get("/docs").status_code == 404
    spec = client.get("/openapi.json").json()
    assert {"/health", "/predict", "/ready", "/auth/check"} <= set(spec["paths"])
    assert "multipart/form-data" in spec["paths"]["/predict"]["post"]["requestBody"]["content"]


def test_upload_is_not_mutated_and_adapter_receives_rgb(auth):
    seen = {}
    adapter = SyntheticAdapter()

    def capture(image):
        seen["shape"], seen["dtype"], seen["pixel"] = image.shape, image.dtype, tuple(image[0, 0])
        return []

    adapter.override = capture
    with app_client(adapter=adapter) as client:
        red = encode(Image.new("RGB", (50, 40), (255, 0, 0)), "PNG")
        assert post_image(client, red, auth).status_code == 200
    assert seen == {"shape": (40, 50, 3), "dtype": np.uint8, "pixel": (255, 0, 0)}


def test_metadata_is_not_echoed_back(client, auth):
    exif = Image.Exif()
    exif[0x010F] = "SecretCameraMaker"
    exif[0x0112] = 1
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (128, 128, 128)).save(buffer, "JPEG", exif=exif)
    response = post_image(client, buffer.getvalue(), auth)
    assert response.status_code == 200 and "SecretCameraMaker" not in response.text
