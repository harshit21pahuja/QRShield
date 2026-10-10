"""
QRShield — Prototype Integration & Unit Tests

Validates:
1. QR decoding (OpenCV QRCodeDetector)
2. URL feature extraction consistency (29 features)
3. URL model loading & inference (RandomForestClassifier)
4. Image model loading & inference (MobileNetV2 + Grad-CAM)
5. Fusion calculation (55/45 soft-voting formula)
6. Calibration loading (IsotonicRegression)
7. /health endpoint readiness
8. /predict with a SAFE synthetic/local QR image
9. /predict with invalid image bytes
10. /predict with undecodable non-QR image

CRITICAL SAFETY:
All test URLs are inert strings. ZERO network requests are made.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import qrcode
from PIL import Image
from starlette.testclient import TestClient

from backend.main import app
from src.features.url_features import FEATURE_NAMES, extract_url_features
from src.inference import (
    fusion_engine,
    image_engine,
    qr_decoder,
    url_engine,
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def safe_qr_bytes() -> bytes:
    """Generate a safe, inert synthetic QR code in memory."""
    qr = qrcode.QRCode(box_size=8, border=4)
    qr.add_data("https://en.wikipedia.org/wiki/QR_code")
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def non_qr_bytes() -> bytes:
    """Generate a valid image that contains no QR code."""
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Test 1: QR Decoding ──────────────────────────────────────────
def test_qr_decoding(safe_qr_bytes: bytes):
    result = qr_decoder.decode(safe_qr_bytes)
    assert result.success is True
    assert result.decoded_url == "https://en.wikipedia.org/wiki/QR_code"
    assert result.is_http is True
    assert result.error is None


# ── Test 2: URL Feature Extraction Consistency ───────────────────
def test_url_feature_extraction_consistency():
    test_url = "https://example.com/login?token=abc-123_456"
    features = extract_url_features(test_url)

    assert isinstance(features, dict)
    assert len(features) == 29
    assert sorted(list(features.keys())) == sorted(FEATURE_NAMES)

    # Check key feature values
    assert features["has_https"] == 1.0
    assert features["has_ip_hostname"] == 0.0
    assert features["num_dots"] >= 1.0
    assert features["url_length"] == float(len(test_url))


# ── Test 3: URL Model Loading & Inference ────────────────────────
def test_url_model_loading():
    assert url_engine.model is not None
    assert getattr(url_engine.model, "n_features_in_", None) == 29

    test_url = "https://en.wikipedia.org/wiki/QR_code"
    pred = url_engine.predict(test_url)

    assert "url_probability" in pred
    assert 0.0 <= pred["url_probability"] <= 1.0
    assert pred["prediction"] in ("BENIGN", "MALICIOUS")
    assert isinstance(pred["explanation"], list)
    assert pred["urls_contacted"] is False


# ── Test 4: Image Model Loading & Inference ──────────────────────
def test_image_model_loading(safe_qr_bytes: bytes):
    assert image_engine.model is not None

    pred = image_engine.predict(safe_qr_bytes, generate_gradcam=True)

    assert "image_probability" in pred
    assert 0.0 <= pred["image_probability"] <= 1.0
    assert pred["prediction"] in ("BENIGN", "MALICIOUS")
    assert pred["gradcam"] is not None
    assert pred["gradcam"]["available"] is True
    assert "data:image/png;base64," in pred["gradcam"]["data_url"]


# ── Test 5: Fusion Calculation (55/45 Formula) ───────────────────
def test_fusion_calculation():
    url_prob = 0.80
    img_prob = 0.40

    fusion_res = fusion_engine.compute_fusion(url_prob, img_prob)

    expected_raw = round(0.55 * url_prob + 0.45 * img_prob, 6)
    assert fusion_res["fusion_probability"] == expected_raw
    assert 0.0 <= fusion_res["calibrated_probability"] <= 1.0
    assert fusion_res["weights"]["url_weight"] == pytest.approx(0.55)
    assert fusion_res["weights"]["image_weight"] == pytest.approx(0.45)


# ── Test 6: Calibration Loading ──────────────────────────────────
def test_calibration_loading():
    assert fusion_engine.calibrator is not None
    cal_pred = fusion_engine.calibrator.predict([0.5])
    assert len(cal_pred) == 1
    assert 0.0 <= cal_pred[0] <= 1.0


# ── Test 7: GET /health Endpoint ─────────────────────────────────
def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_ready"] is True
    assert data["components"]["url_model"]["loaded"] is True
    assert data["components"]["image_model"]["loaded"] is True
    assert data["components"]["fusion"]["loaded"] is True


# ── Test 8: POST /predict with Safe QR ───────────────────────────
def test_predict_safe_qr(client: TestClient, safe_qr_bytes: bytes):
    response = client.post(
        "/predict",
        files={"file": ("safe_qr.png", safe_qr_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["decoded_url"] == "https://en.wikipedia.org/wiki/QR_code"
    assert 0.0 <= data["url_probability"] <= 1.0
    assert 0.0 <= data["image_probability"] <= 1.0
    assert 0.0 <= data["calibrated_probability"] <= 1.0
    assert data["prediction"] in ("BENIGN", "MALICIOUS")
    assert "explanation" in data
    assert "url" in data["explanation"]
    assert "image" in data["explanation"]
    assert data["explanation"]["gradcam"]["available"] is True


# ── Test 9: POST /predict with Invalid Image ─────────────────────
def test_predict_invalid_image(client: TestClient):
    # Empty payload
    response = client.post(
        "/predict",
        files={"file": ("corrupt.png", b"", "image/png")},
    )
    assert response.status_code == 400

    # Garbage bytes
    response_garbage = client.post(
        "/predict",
        files={"file": ("garbage.png", b"not-a-valid-image-stream", "image/png")},
    )
    assert response_garbage.status_code == 422
    assert response_garbage.json()["status"] == "error"


# ── Test 10: POST /predict with Undecodable QR ───────────────────
def test_predict_undecodable_qr(client: TestClient, non_qr_bytes: bytes):
    response = client.post(
        "/predict",
        files={"file": ("no_qr.png", non_qr_bytes, "image/png")},
    )
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert "No QR code could be decoded" in data["error"]
