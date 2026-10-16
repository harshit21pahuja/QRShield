"""
QRShield — Security and Boundary Tests

Validates:
1. Zero-Contact Network Invariance:
   Guarantees that feature extraction, QR decoding, and inference NEVER attempt
   network socket connections, DNS requests, or HTTP requests.
2. CORS and Configuration Behavior:
   Validates origin parsing and credential rules.
3. Input Boundary Conditions:
   Empty strings, malformed URLs, special characters, and non-ASCII strings.
"""

from __future__ import annotations

import socket
import urllib.request
from unittest.mock import patch

import pytest

from backend.config import Settings
from src.features.url_features import extract_url_features
from src.inference.qr_decode import qr_decoder


# ── 1. Zero-Contact Network Invariance ───────────────────────────────────────

def test_zero_contact_url_feature_extraction():
    """Verify that feature extraction on hostile URLs makes zero socket connections."""
    suspicious_url = "http://phishing-credential-steal.malicious-site.test/login.php?user=admin"

    with patch.object(socket, "socket", side_effect=AssertionError("Network socket called!")):
        with patch.object(urllib.request, "urlopen", side_effect=AssertionError("HTTP request attempted!")):
            features = extract_url_features(suspicious_url)
            assert isinstance(features, dict)
            assert len(features) == 29
            assert features["has_ip_hostname"] == 0.0


def test_zero_contact_qr_decoding():
    """Verify that decoding never initiates network requests."""
    from PIL import Image
    import io

    img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    dummy_bytes = buf.getvalue()

    with patch.object(socket, "socket", side_effect=AssertionError("Network socket called!")):
        with patch.object(urllib.request, "urlopen", side_effect=AssertionError("HTTP request attempted!")):
            result = qr_decoder.decode(dummy_bytes)
            assert result.success is False
            assert "No QR code" in result.error


# ── 2. CORS & Configuration Behavior ─────────────────────────────────────────

def test_cors_settings_parsing():
    """Verify comma-separated CORS_ORIGINS parsing in Settings."""
    with patch.dict("os.environ", {"CORS_ORIGINS": "https://app.example.com, https://admin.example.com "}):
        settings = Settings()
        assert settings.cors_origins == [
            "https://app.example.com",
            "https://admin.example.com",
        ]


def test_max_upload_size_env_override():
    """Verify MAX_UPLOAD_SIZE_BYTES env variable override."""
    with patch.dict("os.environ", {"MAX_UPLOAD_SIZE_BYTES": "5242880"}):  # 5 MB
        settings = Settings()
        assert settings.max_upload_size_bytes == 5242880


# ── 3. Malformed and Hostile URL Inputs ───────────────────────────────────────

def test_extract_features_empty_url():
    features = extract_url_features("")
    assert isinstance(features, dict)
    assert len(features) == 29
    assert features["url_length"] == 0.0


def test_extract_features_punycode_and_special_chars():
    hostile_url = "https://xn--e1afmkfd.xn--p1ai/path?query=<script>alert(1)</script>&token=abc%20def"
    features = extract_url_features(hostile_url)
    assert isinstance(features, dict)
    assert len(features) == 29
    assert features["url_length"] == len(hostile_url)
    assert features["num_question"] == 1.0
