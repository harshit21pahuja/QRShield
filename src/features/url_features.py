"""
QRShield — URL Feature Extraction

Extracts 29 lexical and structural features from a raw URL string.
All analysis is STATIC — no network requests are made.

This module is the single source of truth for URL features.
It is used by:
  - training pipeline
  - evaluation pipeline
  - API inference
  - explanation system

SECURITY: This module NEVER visits, fetches, or resolves any URL.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Dict, List, Optional
from urllib.parse import urlparse, parse_qs

from src.config import SUSPICIOUS_TOKENS


# ── Feature name registry ─────────────────────────────────────
FEATURE_NAMES: List[str] = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "num_dots",
    "num_subdomains",
    "digit_count",
    "digit_ratio",
    "alpha_ratio",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_at",
    "num_percent",
    "num_question",
    "num_equals",
    "num_ampersand",
    "num_colon",
    "num_special",
    "special_ratio",
    "query_param_count",
    "has_https",
    "has_ip_hostname",
    "has_port",
    "hostname_hyphen_count",
    "suspicious_token_count",
    "url_entropy",
    "hostname_entropy",
    "path_entropy",
]


def _shannon_entropy(text: str) -> float:
    """Calculate Shannon entropy of a string.

    Returns 0.0 for empty strings.
    """
    if not text:
        return 0.0
    length = len(text)
    counts = Counter(text)
    entropy = -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )
    return round(entropy, 6)


def _is_ip_address(hostname: str) -> bool:
    """Check if hostname is an IPv4 or IPv6 address (no DNS resolution)."""
    # IPv4 pattern
    ipv4_pattern = re.compile(
        r"^(\d{1,3}\.){3}\d{1,3}$"
    )
    if ipv4_pattern.match(hostname):
        parts = hostname.split(".")
        if all(0 <= int(p) <= 255 for p in parts):
            return True
    # IPv6 — bracketed or raw hex groups
    if ":" in hostname:
        cleaned = hostname.strip("[]")
        ipv6_pattern = re.compile(r"^([0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}$")
        if ipv6_pattern.match(cleaned):
            return True
    return False


def _count_special_characters(text: str) -> int:
    """Count characters that are not alphanumeric."""
    return sum(1 for ch in text if ch not in string.ascii_letters + string.digits)


def _count_suspicious_tokens(url_lower: str) -> int:
    """Count how many known suspicious tokens appear in the URL."""
    return sum(1 for token in SUSPICIOUS_TOKENS if token in url_lower)


def extract_url_features(url: str) -> Dict[str, float]:
    """Extract all lexical/structural features from a URL.

    Args:
        url: Raw URL string. Treated as inert data — NEVER visited.

    Returns:
        Dictionary mapping feature name to numeric value.
        For malformed URLs, returns safe defaults (mostly zeros).
    """
    features: Dict[str, float] = {}

    # Fallback for empty / None
    if not url or not isinstance(url, str):
        return {name: 0.0 for name in FEATURE_NAMES}

    url = url.strip()
    url_lower = url.lower()

    # Parse URL safely
    try:
        parsed = urlparse(url)
    except Exception:
        return {name: 0.0 for name in FEATURE_NAMES}

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    # ── Basic length features ─────────────────────────────────
    features["url_length"] = float(len(url))
    features["hostname_length"] = float(len(hostname))
    features["path_length"] = float(len(path))
    features["query_length"] = float(len(query))

    # ── Dot / subdomain features ──────────────────────────────
    features["num_dots"] = float(url.count("."))
    hostname_parts = hostname.split(".") if hostname else []
    # subdomains = parts minus TLD minus SLD (at least 0)
    features["num_subdomains"] = float(max(0, len(hostname_parts) - 2))

    # ── Character-type features ───────────────────────────────
    digit_count = sum(1 for ch in url if ch.isdigit())
    alpha_count = sum(1 for ch in url if ch.isalpha())
    url_len = max(len(url), 1)  # avoid division by zero

    features["digit_count"] = float(digit_count)
    features["digit_ratio"] = round(digit_count / url_len, 6)
    features["alpha_ratio"] = round(alpha_count / url_len, 6)

    # ── Punctuation / symbol counts ───────────────────────────
    features["num_hyphens"] = float(url.count("-"))
    features["num_underscores"] = float(url.count("_"))
    features["num_slashes"] = float(url.count("/"))
    features["num_at"] = float(url.count("@"))
    features["num_percent"] = float(url.count("%"))
    features["num_question"] = float(url.count("?"))
    features["num_equals"] = float(url.count("="))
    features["num_ampersand"] = float(url.count("&"))
    features["num_colon"] = float(url.count(":"))

    special_count = _count_special_characters(url)
    features["num_special"] = float(special_count)
    features["special_ratio"] = round(special_count / url_len, 6)

    # ── Query features ────────────────────────────────────────
    try:
        params = parse_qs(query)
        features["query_param_count"] = float(len(params))
    except Exception:
        features["query_param_count"] = 0.0

    # ── Protocol / security ───────────────────────────────────
    features["has_https"] = 1.0 if parsed.scheme.lower() == "https" else 0.0

    # ── IP-based hostname ─────────────────────────────────────
    features["has_ip_hostname"] = 1.0 if _is_ip_address(hostname) else 0.0

    # ── Port ──────────────────────────────────────────────────
    features["has_port"] = 1.0 if parsed.port is not None else 0.0

    # ── Hostname hyphen count ─────────────────────────────────
    features["hostname_hyphen_count"] = float(hostname.count("-"))

    # ── Suspicious tokens ─────────────────────────────────────
    features["suspicious_token_count"] = float(
        _count_suspicious_tokens(url_lower)
    )

    # ── Entropy ───────────────────────────────────────────────
    features["url_entropy"] = _shannon_entropy(url)
    features["hostname_entropy"] = _shannon_entropy(hostname)
    features["path_entropy"] = _shannon_entropy(path)

    return features


def extract_features_batch(urls: List[str]) -> List[Dict[str, float]]:
    """Extract features for a list of URLs.

    Args:
        urls: List of raw URL strings.

    Returns:
        List of feature dictionaries, one per URL.
    """
    return [extract_url_features(url) for url in urls]
