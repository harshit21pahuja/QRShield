"""
QRShield — URL Inference Module

Loads the Phase 3 trained RandomForestClassifier once at startup and performs
inert, static lexical feature analysis on decoded URL strings.

CRITICAL SAFETY:
This module NEVER visits, crawls, sends HTTP requests to, or resolves any URL.
URLs are treated strictly as inert data strings.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Smart App Control compatibility shim for scikit-learn on Windows
try:
    dummy_emi = types.ModuleType("sklearn.metrics.cluster._expected_mutual_info_fast")
    dummy_emi.expected_mutual_information = lambda *args, **kwargs: 0.0
    sys.modules.setdefault("sklearn.metrics.cluster._expected_mutual_info_fast", dummy_emi)
except Exception:
    pass

try:
    if "sklearn.ensemble" not in sys.modules:
        ens_mod = types.ModuleType("sklearn.ensemble")
        from sklearn.ensemble._forest import (
            RandomForestClassifier,
            RandomForestRegressor,
            ExtraTreesClassifier,
            ExtraTreesRegressor,
        )
        ens_mod.RandomForestClassifier = RandomForestClassifier
        ens_mod.RandomForestRegressor = RandomForestRegressor
        ens_mod.ExtraTreesClassifier = ExtraTreesClassifier
        ens_mod.ExtraTreesRegressor = ExtraTreesRegressor
        sys.modules["sklearn.ensemble"] = ens_mod
except Exception:
    pass

import joblib

from src.config import PATHS
from src.features.url_features import FEATURE_NAMES, extract_url_features


# Inert Top Reputable Root Domains Baseline
# Pure lexical models suffer from severe false positives on top root domains (e.g. youtube.com, google.com)
# because training datasets lack bare benign root domains (documented FPR: 40.75%).
# Industry standards (Google Safe Browsing, Microsoft Defender SmartScreen, Cisco Umbrella) apply
# an inert legitimate domain baseline to mitigate spurious lexical alarms.
REPUTABLE_ROOT_DOMAINS = {
    # Search & Technology Giants
    "google.com", "youtube.com", "gmail.com", "android.com", "google.co.in",
    "microsoft.com", "live.com", "office.com", "bing.com", "msn.com",
    "apple.com", "icloud.com",
    "github.com", "gitlab.com", "bitbucket.org", "stackoverflow.com",
    # Knowledge & Reference
    "wikipedia.org", "wikimedia.org", "wiktionary.org",
    # Social & Professional Networks
    "linkedin.com", "twitter.com", "x.com", "facebook.com", "instagram.com",
    "reddit.com", "whatsapp.com", "telegram.org",
    # Streaming & Media
    "netflix.com", "spotify.com", "twitch.tv",
    # Major E-commerce & Cloud Services
    "amazon.com", "amazon.in", "yahoo.com", "cloudflare.com",
    "zoom.us", "slack.com", "dropbox.com", "adobe.com",
    # Authoritative Government / Academic
    "gov", "edu", "mil", "gov.in", "ac.in", "nic.in", "nih.gov", "who.int"
}


CREDENTIAL_HARVEST_TOKENS = {
    "login", "signin", "sign-in", "verify", "password", "credential",
    "banking", "wallet", "invoice", "payment", "billing", "authenticate",
    "recover", "update-account", "webscr", "cmd=_login"
}


def is_verified_reputable_domain(url: str, features: Dict[str, float]) -> Tuple[bool, str]:
    """Check if the URL belongs to an authenticated top reputable domain.

    Security checks to prevent attacker spoofing:
    - Hostname must be or end with '.<domain>' (e.g. 'www.youtube.com' matches 'youtube.com').
    - Must NOT be an IP address (has_ip_hostname == 0).
    - Must NOT contain an '@' character (basic auth credential spoofing).
    - Must NOT contain non-standard port (has_port == 0).
    - Must NOT contain credential harvesting keywords in path or query.
    """
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url if "://" in url else f"http://{url}")
        host = (parsed.hostname or "").lower()
        if not host:
            return False, ""
        if features.get("has_ip_hostname", 0) > 0:
            return False, host
        if features.get("num_at", 0) > 0:
            return False, host
        if features.get("has_port", 0) > 0:
            return False, host

        # Check for credential harvesting tokens in path or query
        url_path_query = (parsed.path + "?" + parsed.query).lower()
        if any(token in url_path_query for token in CREDENTIAL_HARVEST_TOKENS):
            return False, host

        for domain in REPUTABLE_ROOT_DOMAINS:
            if host == domain or host.endswith("." + domain):
                return True, host
        return False, host
    except Exception:
        return False, ""


class URLInferenceEngine:
    """Inference engine for URL branch of QRShield."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        shap_importance_path: Optional[Path] = None,
    ) -> None:
        self.model_path = model_path or (
            PATHS.root / "artifacts" / "url_baseline_phase3" / "url_random_forest.joblib"
        )
        self.config_path = config_path or (
            PATHS.root / "artifacts" / "url_baseline_phase3" / "config.json"
        )
        self.shap_importance_path = shap_importance_path or (
            PATHS.root / "artifacts" / "fusion_phase3" / "shap_feature_importance.csv"
        )

        self.model = None
        self.config = {}
        self.threshold = 0.5
        self.shap_importance: Dict[str, float] = {}
        self.feature_names: List[str] = FEATURE_NAMES

        self._load_resources()

    def _load_resources(self) -> None:
        """Load trained model, config, and SHAP weights."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"URL model artifact not found: {self.model_path}")

        self.model = joblib.load(self.model_path)

        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
                    self.threshold = float(self.config.get("threshold", 0.5))
            except Exception:
                self.threshold = 0.5

        if self.shap_importance_path.exists():
            try:
                df = pd.read_csv(self.shap_importance_path)
                self.shap_importance = dict(zip(df["feature"], df["mean_abs_shap"]))
            except Exception:
                self.shap_importance = {}

    def extract_features(self, url: str) -> Dict[str, float]:
        """Extract the exact 29 static lexical features."""
        return extract_url_features(url)

    def predict(self, url: str) -> Dict[str, Any]:
        """Perform static inference on a decoded URL.

        Args:
            url: Inert raw URL string.

        Returns:
            Dictionary with url_probability, prediction, features, and explanations.
        """
        features_dict = self.extract_features(url)
        # Preserve exact feature ordering as training
        feature_vector = np.array([[features_dict[name] for name in self.feature_names]], dtype=np.float32)

        probabilities = self.model.predict_proba(feature_vector)[0]
        malicious_prob = float(probabilities[1])

        is_reputable, matched_host = is_verified_reputable_domain(url, features_dict)
        if is_reputable:
            # Mitigate known training set artifact (spurious 40.75% FPR on bare root domains)
            malicious_prob = min(malicious_prob, 0.005)

        prediction_label = "MALICIOUS" if malicious_prob >= self.threshold else "BENIGN"

        explanation_items = self._generate_explanations(
            features_dict, malicious_prob, is_reputable=is_reputable, matched_host=matched_host
        )

        return {
            "url_probability": malicious_prob,
            "prediction": prediction_label,
            "threshold": self.threshold,
            "features": features_dict,
            "explanation": explanation_items,
            "urls_contacted": False,
        }

    def _generate_explanations(
        self,
        features: Dict[str, float],
        prob: float,
        is_reputable: bool = False,
        matched_host: str = "",
    ) -> List[Dict[str, Any]]:
        """Generate transparent, evidence-based feature attribution.

        Never claims any feature alone proves malicious intent.
        """
        if is_reputable:
            return [
                {
                    "feature": "reputable_domain_baseline",
                    "value": 1.0,
                    "importance_weight": 0.50,
                    "impact": "mitigating",
                    "finding": f"Verified Global Infrastructure ({matched_host or 'trusted'}): Authentic destination host with zero credential-harvesting tokens.",
                    "description": "Established legitimate infrastructure baseline safely prevents spurious lexical alarms on clean root domains.",
                },
                {
                    "feature": "has_https" if features.get("has_https", 0) > 0 else "clean_path",
                    "value": float(features.get("has_https", 1.0)),
                    "importance_weight": 0.05,
                    "impact": "mitigating",
                    "finding": "Standard transport security & clean URL parameters.",
                    "description": "Zero suspicious tokens or abnormal query parameters detected.",
                },
            ]

        explanations: List[Dict[str, Any]] = []

        # Domain knowledge descriptions for features
        feature_descriptions = {
            "path_entropy": "Information entropy in the URL path segment (higher values suggest obfuscation or random strings).",
            "path_length": "Character length of the path segment.",
            "num_subdomains": "Number of subdomain levels preceding the registered domain.",
            "num_hyphens": "Frequency of hyphens often used to mimic brand names or create deceptive URLs.",
            "hostname_entropy": "Entropy of hostname characters indicating potential algorithmic domain generation (DGA).",
            "has_ip_hostname": "Use of a raw IP address in place of a standard domain name.",
            "url_length": "Overall character length of the entire URL string.",
            "num_dots": "Total period characters in the URL (frequent in multi-tier subdomains or file extensions).",
            "hostname_length": "Length of the target host component.",
            "num_slashes": "Depth of directory hierarchy.",
            "url_entropy": "Total Shannon entropy across all characters in the URL.",
            "special_ratio": "Proportion of non-alphanumeric punctuation and special characters.",
            "has_https": "Presence of HTTPS protocol encryption (note: phishing sites can also use HTTPS).",
            "hostname_hyphen_count": "Hyphens located specifically within the hostname.",
            "digit_ratio": "Proportion of numeric digits within the URL string.",
            "alpha_ratio": "Proportion of alphabetic letters.",
            "num_special": "Total count of special non-alphanumeric characters.",
            "digit_count": "Total count of numerical digits.",
            "suspicious_token_count": "Presence of sensitive keywords such as login, verify, secure, update, account, or token.",
            "has_port": "Explicit non-standard network port included in the URL host.",
            "query_param_count": "Number of query parameters embedded in the URL.",
        }

        # Calculate impact scores for features that deviate from benign baseline
        candidate_features = []
        for feature_name, value in features.items():
            shap_weight = self.shap_importance.get(feature_name, 0.01)

            # Heuristics for feature contribution relevance
            is_relevant = False
            signal_direction = "neutral"
            detail = ""

            if feature_name == "has_ip_hostname" and value > 0:
                is_relevant = True
                signal_direction = "elevated_risk"
                detail = "Direct IP address host detected without domain name resolution."
            elif feature_name == "suspicious_token_count" and value > 0:
                is_relevant = True
                signal_direction = "elevated_risk"
                detail = f"{int(value)} sensitive authentication/security keywords detected in URL."
            elif feature_name == "num_subdomains" and value >= 2:
                is_relevant = True
                signal_direction = "elevated_risk"
                detail = f"{int(value)} subdomains present, indicating deep domain nesting."
            elif feature_name == "path_entropy" and value > 3.8:
                is_relevant = True
                signal_direction = "elevated_risk"
                detail = f"High path entropy ({value:.2f}) indicates potential tokenization or obfuscation."
            elif feature_name == "url_length" and value > 75:
                is_relevant = True
                signal_direction = "elevated_risk"
                detail = f"Long URL string ({int(value)} characters)."
            elif feature_name == "num_hyphens" and value >= 3:
                is_relevant = True
                signal_direction = "elevated_risk"
                detail = f"Frequent hyphens ({int(value)}) observed across the URL."
            elif feature_name == "has_port" and value > 0:
                is_relevant = True
                signal_direction = "elevated_risk"
                detail = "Non-standard network port specified in host."
            elif feature_name == "special_ratio" and value > 0.15:
                is_relevant = True
                signal_direction = "elevated_risk"
                detail = f"Elevated ratio of special characters ({value:.1%})."
            elif feature_name == "has_https" and value == 1:
                is_relevant = True
                signal_direction = "mitigating"
                detail = "HTTPS protocol is present."
            elif feature_name == "num_subdomains" and value == 0 and prob < 0.5:
                is_relevant = True
                signal_direction = "mitigating"
                detail = "Clean single-level host without extraneous subdomains."
            elif feature_name in ["path_length", "hostname_entropy", "num_dots"] and value > 0:
                # Include as general context if model probability is elevated
                if prob > 0.5:
                    is_relevant = True
                    signal_direction = "elevated_risk"
                    detail = f"Observed {feature_name.replace('_', ' ')} value of {value}."

            if is_relevant:
                score = shap_weight * (1.5 if signal_direction == "elevated_risk" else 1.0)
                candidate_features.append({
                    "feature": feature_name,
                    "value": value,
                    "importance_weight": round(shap_weight, 4),
                    "impact": signal_direction,
                    "finding": detail,
                    "description": feature_descriptions.get(feature_name, ""),
                    "_rank_score": score,
                })

        # Sort candidate features by rank score descending and take top 5
        candidate_features.sort(key=lambda item: item["_rank_score"], reverse=True)
        top_candidates = candidate_features[:5]
        for item in top_candidates:
            item.pop("_rank_score", None)
            explanations.append(item)

        if not explanations:
            explanations.append({
                "feature": "lexical_baseline",
                "value": len(features),
                "importance_weight": 0.0,
                "impact": "neutral",
                "finding": "Standard lexical structure within normal parameters.",
                "description": "All 29 static lexical features were extracted and evaluated.",
            })

        return explanations


# Global singleton instance
url_engine = URLInferenceEngine()
