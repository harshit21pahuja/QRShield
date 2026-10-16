"""
QRShield — Fast Mocked API Unit Tests

Validates API behavior, schemas, error states, and security edge cases
WITHOUT loading heavy machine learning models or making network requests.
Executes in under 2 seconds for high-velocity CI and local testing.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from starlette.testclient import TestClient

from backend.config import get_settings
from backend.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def valid_png_bytes() -> bytes:
    """Generate a valid 50x50 in-memory PNG."""
    img = Image.new("RGB", (50, 50), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def mock_predict_payload() -> dict:
    """Mocked output of fusion_engine.predict_pipeline."""
    return {
        "status": "ok",
        "decoded_url": "https://example.com/test-path",
        "is_http": True,
        "url_probability": 0.123456,
        "image_probability": 0.054321,
        "fusion_probability": 0.092345,
        "calibrated_probability": 0.081234,
        "prediction": "BENIGN",
        "risk_level": "LOW",
        "risk_label": "Minimal Threat",
        "threshold": 0.543407,
        "weights": {"url_weight": 0.55, "image_weight": 0.45},
        "explanation": {
            "url": [
                {
                    "feature": "url_length",
                    "value": 29.0,
                    "importance_weight": 0.045,
                    "impact": "benign_indicator",
                    "finding": "Standard URL length (29 chars).",
                    "description": "Short, typical URL paths correlate with legitimate websites.",
                }
            ],
            "image": [
                {
                    "component": "Visual Density & Embedding",
                    "finding": "Visual features consistent with standard QR codes.",
                }
            ],
            "gradcam": {
                "available": True,
                "data_url": "data:image/png;base64,mockbase64data",
                "focus_region": "Data Matrix",
            },
        },
        "safety_notice": "Static local analysis only. Destination URL was NOT contacted, crawled, or visited.",
    }


# ── 1. Root & Discovery Endpoint ─────────────────────────────────────────────

def test_root_endpoint_schema(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data
    assert "documentation" in data
    assert data["documentation"] == "/docs"
    assert "safety" in data


# ── 2. Health & Readiness Checks ─────────────────────────────────────────────

def test_health_endpoint_healthy(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "models_ready" in data
    assert "components" in data
    assert "url_model" in data["components"]
    assert "image_model" in data["components"]
    assert "fusion" in data["components"]
    assert "safety_guarantee" in data


def test_health_endpoint_degraded(client: TestClient):
    with patch("backend.main.url_engine.model", None):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["models_ready"] is False
        assert data["components"]["url_model"]["loaded"] is False


def test_readiness_probe_ready(client: TestClient):
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.text == "READY"


def test_readiness_probe_not_ready(client: TestClient):
    with patch("backend.main.url_engine.model", None):
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.text == "MODELS_NOT_READY"


# ── 3. Prediction & Input Validation ─────────────────────────────────────────

def test_predict_success_mocked(client: TestClient, valid_png_bytes: bytes, mock_predict_payload: dict):
    with patch("backend.main.fusion_engine.predict_pipeline", return_value=mock_predict_payload):
        response = client.post(
            "/predict",
            files={"file": ("test_qr.png", valid_png_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["decoded_url"] == "https://example.com/test-path"
        assert data["prediction"] == "BENIGN"
        assert data["risk_level"] == "LOW"
        assert data["calibrated_probability"] == pytest.approx(0.081234)
        assert data["explanation"]["gradcam"]["available"] is True


def test_predict_missing_filename(client: TestClient, valid_png_bytes: bytes):
    response = client.post(
        "/predict",
        files={"file": ("", valid_png_bytes, "image/png")},
    )
    assert response.status_code in (400, 422)


def test_predict_unsupported_file_extension(client: TestClient):
    response = client.post(
        "/predict",
        files={"file": ("malware.exe", b"MZexecutable", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_predict_empty_file(client: TestClient):
    response = client.post(
        "/predict",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_predict_corrupt_image(client: TestClient):
    response = client.post(
        "/predict",
        files={"file": ("corrupt.png", b"not-an-image-data-stream", "image/png")},
    )
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert "corrupt" in data["error"].lower() or "not a recognized image" in data["error"].lower()


def test_predict_oversized_file(client: TestClient, valid_png_bytes: bytes):
    settings = get_settings()
    # Temporarily set upload limit to 50 bytes to test rejection
    with patch.object(settings, "max_upload_size_bytes", 50):
        response = client.post(
            "/predict",
            files={"file": ("oversized.png", valid_png_bytes, "image/png")},
        )
        assert response.status_code == 413
        assert "exceeds maximum upload size limit" in response.json()["detail"]


def test_predict_internal_pipeline_error_sanitized(client: TestClient, valid_png_bytes: bytes):
    with patch("backend.main.fusion_engine.predict_pipeline", side_effect=RuntimeError("Secret internal path /var/root/fail")):
        with patch.object(get_settings(), "debug", False):
            response = client.post(
                "/predict",
                files={"file": ("test.png", valid_png_bytes, "image/png")},
            )
            assert response.status_code == 500
            data = response.json()
            assert data["status"] == "error"
            # Ensure internal path is NOT leaked when debug is False
            assert "/var/root/fail" not in data["error"]
            assert "Internal error while analyzing QR code" in data["error"]


def test_predict_pipeline_returns_status_error(client: TestClient, valid_png_bytes: bytes):
    error_result = {
        "status": "error",
        "error": "No QR code could be decoded from the provided image.",
        "decoded_url": None,
    }
    with patch("backend.main.fusion_engine.predict_pipeline", return_value=error_result):
        response = client.post(
            "/predict",
            files={"file": ("no_qr.png", valid_png_bytes, "image/png")},
        )
        assert response.status_code == 422
        assert response.json()["status"] == "error"
        assert "No QR code could be decoded" in response.json()["error"]
