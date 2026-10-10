"""
QRShield — Dataset Audit

Generates comprehensive audit reports for the dataset.
Reports both machine-readable (JSON) and human-readable (TXT) formats.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from src.config import PATHS

logger = logging.getLogger(__name__)


def _url_scheme(url: str) -> str:
    """Extract scheme from URL safely."""
    try:
        return urlparse(url).scheme.lower()
    except Exception:
        return "invalid"


def audit_dataset(df: pd.DataFrame,
                  output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Run a full audit on the dataset and save reports.

    Args:
        df: DataFrame with at least 'url' and 'label' columns.
             Optionally 'domain', 'source', 'timestamp'.
        output_dir: Where to write reports. Defaults to PATHS.data_audit.

    Returns:
        Audit report as a dictionary.
    """
    output_dir = output_dir or PATHS.data_audit
    output_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {}

    # ── Basic counts ──────────────────────────────────────────
    report["total_rows"] = len(df)
    report["num_columns"] = len(df.columns)
    report["columns"] = list(df.columns)

    # ── Label distribution ────────────────────────────────────
    label_counts = df["label"].value_counts().to_dict()
    # Convert numpy int keys to Python int
    label_counts = {int(k): int(v) for k, v in label_counts.items()}
    report["label_distribution"] = label_counts
    total = len(df)
    report["benign_count"] = int(label_counts.get(0, 0))
    report["malicious_count"] = int(label_counts.get(1, 0))
    report["benign_pct"] = round(100 * report["benign_count"] / max(total, 1), 2)
    report["malicious_pct"] = round(100 * report["malicious_count"] / max(total, 1), 2)

    # ── Missing values ────────────────────────────────────────
    missing = df.isnull().sum().to_dict()
    report["missing_values"] = {k: int(v) for k, v in missing.items() if v > 0}

    # ── URL statistics ────────────────────────────────────────
    url_lengths = df["url"].str.len()
    report["url_length_stats"] = {
        "min": int(url_lengths.min()) if len(url_lengths) > 0 else 0,
        "max": int(url_lengths.max()) if len(url_lengths) > 0 else 0,
        "mean": round(float(url_lengths.mean()), 2) if len(url_lengths) > 0 else 0,
        "median": round(float(url_lengths.median()), 2) if len(url_lengths) > 0 else 0,
        "std": round(float(url_lengths.std()), 2) if len(url_lengths) > 0 else 0,
    }

    # ── Duplicates ────────────────────────────────────────────
    report["duplicate_urls"] = int(df["url"].duplicated().sum())
    report["duplicate_rows"] = int(df.duplicated().sum())
    report["unique_urls"] = int(df["url"].nunique())

    # ── Domain stats ──────────────────────────────────────────
    if "domain" in df.columns:
        report["unique_domains"] = int(df["domain"].nunique())
        report["unknown_domains"] = int((df["domain"] == "unknown").sum())
        # Top 10 most frequent domains
        top_domains = df["domain"].value_counts().head(10).to_dict()
        report["top_10_domains"] = {str(k): int(v) for k, v in top_domains.items()}

    # ── URL scheme analysis ───────────────────────────────────
    schemes = df["url"].apply(_url_scheme)
    scheme_counts = schemes.value_counts().to_dict()
    report["url_schemes"] = {str(k): int(v) for k, v in scheme_counts.items()}

    supported_schemes = {"http", "https"}
    unsupported = schemes[~schemes.isin(supported_schemes | {"invalid"})]
    report["unsupported_schemes_count"] = int(len(unsupported))

    # Malformed URLs (no scheme or can't parse)
    malformed = schemes[schemes == "invalid"]
    no_scheme = schemes[schemes == ""]
    report["malformed_urls"] = int(len(malformed) + len(no_scheme))

    # Empty URLs
    report["empty_urls"] = int((df["url"].str.strip() == "").sum())

    # ── Source distribution ───────────────────────────────────
    if "source" in df.columns:
        source_counts = df["source"].value_counts().to_dict()
        report["source_distribution"] = {str(k): int(v) for k, v in source_counts.items()}

    # ── Timestamp distribution ────────────────────────────────
    if "timestamp" in df.columns:
        try:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            valid_ts = ts.dropna()
            report["timestamp_stats"] = {
                "earliest": str(valid_ts.min()) if len(valid_ts) > 0 else "N/A",
                "latest": str(valid_ts.max()) if len(valid_ts) > 0 else "N/A",
                "count_valid": int(len(valid_ts)),
                "count_invalid": int(ts.isna().sum()),
            }
        except Exception:
            report["timestamp_stats"] = {"error": "Could not parse timestamps"}

    # ── Save reports ──────────────────────────────────────────
    json_path = output_dir / "report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Audit report (JSON) saved to %s", json_path)

    txt_path = output_dir / "report.txt"
    _write_txt_report(report, txt_path)
    logger.info("Audit report (TXT) saved to %s", txt_path)

    return report


def _write_txt_report(report: Dict[str, Any], path: Path) -> None:
    """Write a human-readable text report."""
    lines = []
    lines.append("=" * 60)
    lines.append("QRShield — Dataset Audit Report")
    lines.append("=" * 60)
    lines.append("")

    lines.append(f"Total rows:            {report['total_rows']}")
    lines.append(f"Columns:               {report['num_columns']}")
    lines.append(f"Column names:          {report['columns']}")
    lines.append("")

    lines.append("── Label Distribution ──")
    lines.append(f"  Benign (0):          {report['benign_count']}  ({report['benign_pct']}%)")
    lines.append(f"  Malicious (1):       {report['malicious_count']}  ({report['malicious_pct']}%)")
    lines.append("")

    lines.append("── URL Length Statistics ──")
    stats = report.get("url_length_stats", {})
    for k, v in stats.items():
        lines.append(f"  {k:20s}: {v}")
    lines.append("")

    lines.append("── Duplicates ──")
    lines.append(f"  Duplicate URLs:      {report['duplicate_urls']}")
    lines.append(f"  Duplicate rows:      {report['duplicate_rows']}")
    lines.append(f"  Unique URLs:         {report['unique_urls']}")
    lines.append("")

    if "unique_domains" in report:
        lines.append("── Domain Statistics ──")
        lines.append(f"  Unique domains:      {report['unique_domains']}")
        lines.append(f"  Unknown domains:     {report['unknown_domains']}")
        if "top_10_domains" in report:
            lines.append("  Top 10 domains:")
            for dom, cnt in report["top_10_domains"].items():
                lines.append(f"    {dom:40s}  {cnt}")
        lines.append("")

    lines.append("── URL Schemes ──")
    for scheme, cnt in report.get("url_schemes", {}).items():
        lines.append(f"  {scheme:20s}: {cnt}")
    lines.append(f"  Malformed URLs:      {report['malformed_urls']}")
    lines.append(f"  Empty URLs:          {report['empty_urls']}")
    lines.append(f"  Unsupported schemes: {report['unsupported_schemes_count']}")
    lines.append("")

    if "missing_values" in report and report["missing_values"]:
        lines.append("── Missing Values ──")
        for col, cnt in report["missing_values"].items():
            lines.append(f"  {col:20s}: {cnt}")
        lines.append("")

    if "source_distribution" in report:
        lines.append("── Source Distribution ──")
        for src, cnt in report["source_distribution"].items():
            lines.append(f"  {src:30s}: {cnt}")
        lines.append("")

    if "timestamp_stats" in report:
        lines.append("── Timestamp Statistics ──")
        for k, v in report["timestamp_stats"].items():
            lines.append(f"  {k:20s}: {v}")
        lines.append("")

    lines.append("=" * 60)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
