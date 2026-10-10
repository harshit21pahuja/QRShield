"""
QRShield — Leakage-Safe Dataset Splitting

Implements domain-grouped, stratified splitting to prevent data leakage.

Key invariants:
  - TRAIN domains ∩ VALIDATION domains = ∅
  - TRAIN domains ∩ TEST domains = ∅
  - VALIDATION domains ∩ TEST domains = ∅
  - All URL variants from the same domain stay in the same split

Uses StratifiedGroupKFold from scikit-learn as the core mechanism.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import DATA_CFG, SEEDS, PATHS

logger = logging.getLogger(__name__)


def _verify_no_overlap(train_domains: set, val_domains: set,
                       test_domains: set) -> Dict[str, int]:
    """Verify zero domain overlap between splits.

    Raises:
        AssertionError: If any overlap is found.
    """
    train_val = train_domains & val_domains
    train_test = train_domains & test_domains
    val_test = val_domains & test_domains

    overlap_report = {
        "train_val_overlap": len(train_val),
        "train_test_overlap": len(train_test),
        "val_test_overlap": len(val_test),
    }

    if train_val:
        raise AssertionError(
            f"LEAKAGE DETECTED: {len(train_val)} domains in BOTH train and val: "
            f"{list(train_val)[:5]}..."
        )
    if train_test:
        raise AssertionError(
            f"LEAKAGE DETECTED: {len(train_test)} domains in BOTH train and test: "
            f"{list(train_test)[:5]}..."
        )
    if val_test:
        raise AssertionError(
            f"LEAKAGE DETECTED: {len(val_test)} domains in BOTH val and test: "
            f"{list(val_test)[:5]}..."
        )

    logger.info("✓ Zero domain overlap verified across all splits")
    return overlap_report


def _split_report(name: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a summary report for a single split."""
    total = len(df)
    benign = int((df["label"] == 0).sum())
    malicious = int((df["label"] == 1).sum())
    return {
        "name": name,
        "sample_count": total,
        "benign_count": benign,
        "malicious_count": malicious,
        "malicious_pct": round(100 * malicious / max(total, 1), 2),
        "unique_domains": int(df["domain"].nunique()) if "domain" in df.columns else -1,
    }


def domain_grouped_split(
    df: pd.DataFrame,
    train_ratio: float = DATA_CFG.train_ratio,
    val_ratio: float = DATA_CFG.val_ratio,
    test_ratio: float = DATA_CFG.test_ratio,
    seed: int = SEEDS.split_seed,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset with domain-grouped stratification.

    Uses a two-stage StratifiedGroupKFold approach:
      1. Split into train+val vs test
      2. Split train+val into train vs val

    Args:
        df: DataFrame with 'url', 'label', and 'domain' columns.
        train_ratio: Approximate train proportion.
        val_ratio: Approximate validation proportion.
        test_ratio: Approximate test proportion.
        seed: Random seed.
        output_dir: Where to save split info. Defaults to PATHS.data_audit.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If 'domain' column is missing.
        AssertionError: If domain overlap is detected.
    """
    if "domain" not in df.columns:
        raise ValueError("DataFrame must have a 'domain' column. Run add_domain_column first.")

    output_dir = output_dir or PATHS.data_audit
    output_dir.mkdir(parents=True, exist_ok=True)

    X = df.index.values
    y = df["label"].values
    groups = df["domain"].values

    # ── Stage 1: Split off TEST ───────────────────────────────
    # We want ~test_ratio for test, so use n_splits ≈ 1/test_ratio
    n_splits_test = max(2, round(1.0 / test_ratio))
    sgkf_test = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True,
                                      random_state=seed)

    train_val_idx = None
    test_idx = None
    for tv_idx, t_idx in sgkf_test.split(X, y, groups):
        train_val_idx = tv_idx
        test_idx = t_idx
        break  # Take the first fold

    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # ── Stage 2: Split TRAIN_VAL into TRAIN and VAL ───────────
    relative_val = val_ratio / (train_ratio + val_ratio)
    n_splits_val = max(2, round(1.0 / relative_val))

    X2 = df_train_val.index.values
    y2 = df_train_val["label"].values
    groups2 = df_train_val["domain"].values

    sgkf_val = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True,
                                     random_state=seed)

    train_idx = None
    val_idx = None
    for tr_idx, v_idx in sgkf_val.split(X2, y2, groups2):
        train_idx = tr_idx
        val_idx = v_idx
        break

    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

    # ── Verify zero overlap ──────────────────────────────────
    train_domains = set(df_train["domain"].unique())
    val_domains = set(df_val["domain"].unique())
    test_domains = set(df_test["domain"].unique())

    overlap_report = _verify_no_overlap(train_domains, val_domains, test_domains)

    # ── Generate split report ─────────────────────────────────
    split_info = {
        "seed": seed,
        "train": _split_report("train", df_train),
        "validation": _split_report("validation", df_val),
        "test": _split_report("test", df_test),
        "domain_overlap": overlap_report,
        "total_unique_domains": int(df["domain"].nunique()),
        "train_domains": int(len(train_domains)),
        "val_domains": int(len(val_domains)),
        "test_domains": int(len(test_domains)),
    }

    # Log summary
    for split_name in ["train", "validation", "test"]:
        info = split_info[split_name]
        logger.info(
            "  %s: %d samples, %d benign, %d malicious (%.1f%% mal), %d domains",
            split_name.upper(), info["sample_count"], info["benign_count"],
            info["malicious_count"], info["malicious_pct"], info["unique_domains"],
        )

    # Save split report
    report_path = output_dir / "split_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    logger.info("Split report saved to %s", report_path)

    return df_train, df_val, df_test
