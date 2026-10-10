"""
QRShield — Global Configuration

Centralized configuration for paths, random seeds, model parameters,
and evaluation settings. All configurable values live here to ensure
reproducibility and avoid scattered magic numbers.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


# ── Project root ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PathConfig:
    """All project paths derived from PROJECT_ROOT."""

    root: Path = PROJECT_ROOT
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    models: Path = PROJECT_ROOT / "models"
    artifacts: Path = PROJECT_ROOT / "artifacts"
    data_audit: Path = PROJECT_ROOT / "artifacts" / "data_audit"
    url_baseline: Path = PROJECT_ROOT / "artifacts" / "url_baseline"
    image_baseline: Path = PROJECT_ROOT / "artifacts" / "image_baseline"
    soft_voting: Path = PROJECT_ROOT / "artifacts" / "soft_voting"
    qrshield_fusion: Path = PROJECT_ROOT / "artifacts" / "qrshield_fusion"
    external_test: Path = PROJECT_ROOT / "artifacts" / "external_test"
    failure_analysis: Path = PROJECT_ROOT / "artifacts" / "failure_analysis"

    def ensure_dirs(self) -> None:
        """Create all artifact directories if they do not exist."""
        for attr_name in self.__dataclass_fields__:
            path = getattr(self, attr_name)
            if isinstance(path, Path) and attr_name != "root":
                path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SeedConfig:
    """Reproducibility seeds."""

    global_seed: int = 42
    split_seed: int = 42
    model_seed: int = 42


@dataclass(frozen=True)
class DataConfig:
    """Dataset configuration."""

    # Expected column names — will be verified at runtime
    url_column: str = "url"
    label_column: str = "label"
    # Canonical label mapping
    label_benign: int = 0
    label_malicious: int = 1
    # Split ratios (approximate, subject to group constraints)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    # Maximum subset size (0 = use all)
    max_samples: int = 0
    # Minimum URL length to keep
    min_url_length: int = 10


@dataclass(frozen=True)
class URLModelConfig:
    """URL-only Random Forest baseline configuration."""

    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    class_weight: str = "balanced"
    n_jobs: int = -1
    random_state: int = 42


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation settings."""

    # Threshold selection metric on validation set
    threshold_metric: str = "f1"
    # Default classification threshold
    default_threshold: float = 0.5
    # Number of latency measurement runs
    latency_runs: int = 100


# ── Suspicious tokens for URL feature extraction ──────────────
SUSPICIOUS_TOKENS: List[str] = [
    "login", "signin", "sign-in", "verify", "account", "update",
    "secure", "banking", "confirm", "password", "credential",
    "wallet", "paypal", "ebay", "amazon", "microsoft", "apple",
    "google", "facebook", "netflix", "support", "helpdesk",
    "admin", "webscr", "cmd", "suspend", "unlock", "alert",
    "urgent", "expire", "invoice", "payment", "billing",
    "security", "authenticate", "recover", "restore",
]


# ── Singleton instances ───────────────────────────────────────
PATHS = PathConfig()
SEEDS = SeedConfig()
DATA_CFG = DataConfig()
URL_MODEL_CFG = URLModelConfig()
EVAL_CFG = EvalConfig()
