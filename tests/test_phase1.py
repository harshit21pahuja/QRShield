"""
QRShield — Unit Tests

Tests for:
  - URL parsing and feature extraction
  - Empty/malformed URL handling
  - HTTPS detection
  - IP address detection
  - Shannon entropy
  - Suspicious token extraction
  - Domain extraction
  - Label normalization
  - Duplicate handling
  - Grouped splitting and leakage detection
  - Prediction response format
  - Synthetic dataset
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.url_features import (
    extract_url_features,
    extract_features_batch,
    _shannon_entropy,
    _is_ip_address,
    _count_suspicious_tokens,
    FEATURE_NAMES,
)
from src.data.loader import (
    normalize_labels,
    extract_domain,
    add_domain_column,
    clean_dataset,
)
from src.data.splitter import domain_grouped_split, _verify_no_overlap
from src.data.synthetic import generate_smoke_dataset


# ═══════════════════════════════════════════════════════════════
# URL Feature Extraction Tests
# ═══════════════════════════════════════════════════════════════

class TestURLFeatureExtraction:
    """Tests for URL feature extraction."""

    def test_basic_url(self) -> None:
        """A normal HTTPS URL should produce valid features."""
        features = extract_url_features("https://www.google.com/search?q=test")
        assert isinstance(features, dict)
        assert set(features.keys()) == set(FEATURE_NAMES)
        assert features["has_https"] == 1.0
        assert features["url_length"] > 0
        assert features["hostname_length"] > 0

    def test_all_features_present(self) -> None:
        """Every feature name should be in the output."""
        features = extract_url_features("http://example.com")
        for name in FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"

    def test_feature_count(self) -> None:
        """Should produce exactly 29 features."""
        features = extract_url_features("https://test.com")
        assert len(features) == 29

    def test_empty_url(self) -> None:
        """Empty string should return all zeros."""
        features = extract_url_features("")
        assert all(v == 0.0 for v in features.values())

    def test_none_url(self) -> None:
        """None should return all zeros."""
        features = extract_url_features(None)  # type: ignore
        assert all(v == 0.0 for v in features.values())

    def test_malformed_url(self) -> None:
        """Malformed URL should not crash."""
        features = extract_url_features("not a valid url at all")
        assert isinstance(features, dict)
        assert len(features) == 29

    def test_https_detection(self) -> None:
        """HTTPS should be detected correctly."""
        assert extract_url_features("https://secure.com")["has_https"] == 1.0
        assert extract_url_features("http://insecure.com")["has_https"] == 0.0

    def test_ip_hostname_detection(self) -> None:
        """IP-based hostname should be detected."""
        f1 = extract_url_features("http://192.168.1.1/login")
        assert f1["has_ip_hostname"] == 1.0

        f2 = extract_url_features("https://www.google.com")
        assert f2["has_ip_hostname"] == 0.0

    def test_port_detection(self) -> None:
        """Port in URL should be detected."""
        f1 = extract_url_features("http://example.com:8080/path")
        assert f1["has_port"] == 1.0

        f2 = extract_url_features("https://example.com/path")
        assert f2["has_port"] == 0.0

    def test_url_length(self) -> None:
        """URL length should match actual string length."""
        url = "https://www.example.com/very/long/path?with=params"
        features = extract_url_features(url)
        assert features["url_length"] == float(len(url))

    def test_digit_ratio(self) -> None:
        """Digit ratio should be between 0 and 1."""
        features = extract_url_features("http://192.168.1.1:8080/12345")
        assert 0.0 <= features["digit_ratio"] <= 1.0
        assert features["digit_count"] > 0

    def test_alpha_ratio(self) -> None:
        """Alpha ratio should be between 0 and 1."""
        features = extract_url_features("https://www.example.com")
        assert 0.0 <= features["alpha_ratio"] <= 1.0

    def test_special_characters(self) -> None:
        """Special characters should be counted."""
        url = "http://bad.com/path?a=1&b=2&c=3"
        features = extract_url_features(url)
        assert features["num_ampersand"] == 2.0
        assert features["num_equals"] == 3.0
        assert features["num_question"] == 1.0

    def test_query_param_count(self) -> None:
        """Query parameters should be counted."""
        features = extract_url_features("http://example.com/path?a=1&b=2&c=3")
        assert features["query_param_count"] == 3.0

    def test_suspicious_tokens(self) -> None:
        """Suspicious tokens should be counted."""
        url = "http://login-verify-account-secure.com/password/update"
        features = extract_url_features(url)
        assert features["suspicious_token_count"] >= 3  # login, verify, account, secure, password, update

    def test_subdomains(self) -> None:
        """Subdomain count should be correct."""
        # www.sub.example.com → 2 subdomains (www, sub)
        features = extract_url_features("http://www.sub.example.com")
        assert features["num_subdomains"] >= 1

    def test_batch_extraction(self) -> None:
        """Batch extraction should return list of dicts."""
        urls = ["https://google.com", "http://evil.com/phish"]
        results = extract_features_batch(urls)
        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)

    def test_at_symbol(self) -> None:
        """@ in URL should be counted (common phishing indicator)."""
        features = extract_url_features("http://evil.com@legit.com/path")
        assert features["num_at"] == 1.0

    def test_hyphens_in_hostname(self) -> None:
        """Hostname hyphens should be counted separately."""
        features = extract_url_features("http://pay-pal-secure-login.com/signin")
        assert features["hostname_hyphen_count"] >= 3


# ═══════════════════════════════════════════════════════════════
# Entropy Tests
# ═══════════════════════════════════════════════════════════════

class TestEntropy:
    """Tests for Shannon entropy calculation."""

    def test_empty_string(self) -> None:
        """Empty string entropy should be 0."""
        assert _shannon_entropy("") == 0.0

    def test_single_char(self) -> None:
        """Single character repeated has 0 entropy."""
        assert _shannon_entropy("aaaa") == 0.0

    def test_high_entropy(self) -> None:
        """Diverse characters should have higher entropy."""
        low = _shannon_entropy("aaa")
        high = _shannon_entropy("abc123!@#")
        assert high > low

    def test_known_value(self) -> None:
        """Two equally likely chars should have entropy 1.0."""
        assert abs(_shannon_entropy("ab") - 1.0) < 0.001


# ═══════════════════════════════════════════════════════════════
# IP Address Detection Tests
# ═══════════════════════════════════════════════════════════════

class TestIPDetection:
    """Tests for IP address detection."""

    def test_valid_ipv4(self) -> None:
        assert _is_ip_address("192.168.1.1") is True
        assert _is_ip_address("10.0.0.1") is True
        assert _is_ip_address("255.255.255.255") is True

    def test_invalid_ipv4(self) -> None:
        assert _is_ip_address("256.1.1.1") is False
        assert _is_ip_address("www.google.com") is False

    def test_empty(self) -> None:
        assert _is_ip_address("") is False


# ═══════════════════════════════════════════════════════════════
# Suspicious Token Tests
# ═══════════════════════════════════════════════════════════════

class TestSuspiciousTokens:
    """Tests for suspicious token counting."""

    def test_no_tokens(self) -> None:
        assert _count_suspicious_tokens("https://www.example.com") == 0

    def test_multiple_tokens(self) -> None:
        count = _count_suspicious_tokens("http://login-verify-password.com")
        assert count >= 3

    def test_case_insensitive(self) -> None:
        """Input is expected to be lowercased before calling."""
        count = _count_suspicious_tokens("http://login.com")
        assert count >= 1


# ═══════════════════════════════════════════════════════════════
# Domain Extraction Tests
# ═══════════════════════════════════════════════════════════════

class TestDomainExtraction:
    """Tests for registrable domain extraction."""

    def test_basic_domain(self) -> None:
        domain = extract_domain("https://www.google.com/path")
        assert "google" in domain.lower()

    def test_subdomain(self) -> None:
        domain = extract_domain("https://mail.google.com/inbox")
        # Should extract google.com, not mail.google.com
        assert "google" in domain.lower()

    def test_empty(self) -> None:
        assert extract_domain("") == "unknown"

    def test_none(self) -> None:
        assert extract_domain(None) == "unknown"  # type: ignore

    def test_ip_url(self) -> None:
        domain = extract_domain("http://192.168.1.1/path")
        # Should return something, not crash
        assert isinstance(domain, str)
        assert len(domain) > 0


# ═══════════════════════════════════════════════════════════════
# Label Normalization Tests
# ═══════════════════════════════════════════════════════════════

class TestLabelNormalization:
    """Tests for label normalization."""

    def test_already_binary(self) -> None:
        df = pd.DataFrame({"url": ["http://a.com", "http://b.com"], "label": [0, 1]})
        result = normalize_labels(df)
        assert list(result["label"]) == [0, 1]

    def test_string_labels(self) -> None:
        df = pd.DataFrame({
            "url": ["http://a.com", "http://b.com"],
            "label": ["benign", "phishing"],
        })
        result = normalize_labels(df)
        assert list(result["label"]) == [0, 1]

    def test_unknown_labels_raise(self) -> None:
        df = pd.DataFrame({
            "url": ["http://a.com"],
            "label": ["unknown_category"],
        })
        with pytest.raises(ValueError, match="Unknown label"):
            normalize_labels(df)


# ═══════════════════════════════════════════════════════════════
# Dataset Cleaning Tests
# ═══════════════════════════════════════════════════════════════

class TestDatasetCleaning:
    """Tests for dataset cleaning."""

    def test_remove_duplicates(self) -> None:
        df = pd.DataFrame({
            "url": ["http://a.com", "http://a.com", "http://b.com"],
            "label": [0, 0, 1],
        })
        cleaned = clean_dataset(df, min_url_length=5)
        assert len(cleaned) == 2

    def test_remove_short_urls(self) -> None:
        df = pd.DataFrame({
            "url": ["http://a.com/path", "ab"],
            "label": [0, 1],
        })
        cleaned = clean_dataset(df, min_url_length=10)
        assert len(cleaned) == 1

    def test_remove_missing(self) -> None:
        df = pd.DataFrame({
            "url": ["http://a.com/path", None],
            "label": [0, 1],
        })
        cleaned = clean_dataset(df, min_url_length=5)
        assert len(cleaned) == 1


# ═══════════════════════════════════════════════════════════════
# Domain-Grouped Splitting Tests
# ═══════════════════════════════════════════════════════════════

class TestDomainGroupedSplit:
    """Tests for leakage-safe domain-grouped splitting."""

    def _make_test_df(self) -> pd.DataFrame:
        """Create a small test DataFrame with multiple domains."""
        urls = []
        labels = []
        domains = []
        # Create 10 domains with 5 URLs each
        for i in range(10):
            domain = f"domain{i}.com"
            for j in range(5):
                label = 0 if i < 5 else 1  # first 5 domains benign, rest malicious
                urls.append(f"http://{domain}/page{j}")
                labels.append(label)
                domains.append(domain)
        return pd.DataFrame({"url": urls, "label": labels, "domain": domains})

    def test_zero_overlap(self) -> None:
        """Train, val, test must have zero domain overlap."""
        df = self._make_test_df()
        train, val, test = domain_grouped_split(df, seed=42)

        train_domains = set(train["domain"].unique())
        val_domains = set(val["domain"].unique())
        test_domains = set(test["domain"].unique())

        assert len(train_domains & val_domains) == 0
        assert len(train_domains & test_domains) == 0
        assert len(val_domains & test_domains) == 0

    def test_all_samples_present(self) -> None:
        """All samples should end up in exactly one split."""
        df = self._make_test_df()
        train, val, test = domain_grouped_split(df, seed=42)
        assert len(train) + len(val) + len(test) == len(df)

    def test_leakage_detection(self) -> None:
        """Verification function should raise on overlap."""
        with pytest.raises(AssertionError, match="LEAKAGE"):
            _verify_no_overlap({"a.com", "b.com"}, {"b.com", "c.com"}, {"d.com"})


# ═══════════════════════════════════════════════════════════════
# Synthetic Dataset Tests
# ═══════════════════════════════════════════════════════════════

class TestSyntheticDataset:
    """Tests for synthetic smoke-test dataset."""

    def test_structure(self) -> None:
        df = generate_smoke_dataset()
        assert "url" in df.columns
        assert "label" in df.columns
        assert len(df) == 70

    def test_balance(self) -> None:
        df = generate_smoke_dataset()
        assert (df["label"] == 0).sum() == 35
        assert (df["label"] == 1).sum() == 35

    def test_deterministic(self) -> None:
        df1 = generate_smoke_dataset(seed=42)
        df2 = generate_smoke_dataset(seed=42)
        assert df1.equals(df2)

    def test_urls_non_empty(self) -> None:
        df = generate_smoke_dataset()
        assert all(len(url) > 0 for url in df["url"])


# ═══════════════════════════════════════════════════════════════
# Integration: Feature Extraction on Synthetic Data
# ═══════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests using synthetic data."""

    def test_full_feature_extraction(self) -> None:
        """Extract features from all synthetic URLs without error."""
        df = generate_smoke_dataset()
        for url in df["url"]:
            features = extract_url_features(url)
            assert len(features) == 29
            assert all(isinstance(v, (int, float)) for v in features.values())

    def test_malicious_vs_benign_features(self) -> None:
        """Malicious URLs should generally have different feature distributions."""
        df = generate_smoke_dataset()
        benign = df[df["label"] == 0]
        malicious = df[df["label"] == 1]

        benign_features = extract_features_batch(benign["url"].tolist())
        malicious_features = extract_features_batch(malicious["url"].tolist())

        # Malicious URLs in our synthetic set use HTTP, not HTTPS
        avg_https_benign = np.mean([f["has_https"] for f in benign_features])
        avg_https_malicious = np.mean([f["has_https"] for f in malicious_features])
        assert avg_https_benign > avg_https_malicious

        # Malicious URLs should have more suspicious tokens
        avg_tokens_benign = np.mean([f["suspicious_token_count"] for f in benign_features])
        avg_tokens_malicious = np.mean([f["suspicious_token_count"] for f in malicious_features])
        assert avg_tokens_malicious > avg_tokens_benign
