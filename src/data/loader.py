"""
QRShield — Dataset Loading and Preprocessing

Handles:
  - Loading CSV datasets
  - Schema detection and validation
  - Label normalization to canonical {0: BENIGN, 1: MALICIOUS}
  - Registrable domain extraction for leakage-safe splitting
  - Duplicate detection and removal

SECURITY: Raw URLs are loaded as inert strings. No URL is ever visited.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd

try:
    import tldextract
    _HAS_TLDEXTRACT = True
except ImportError:
    _HAS_TLDEXTRACT = False

from src.config import DATA_CFG

logger = logging.getLogger(__name__)


# ── Known label mappings ──────────────────────────────────────
_LABEL_MAP_MALICIOUS = {
    "phishing", "malicious", "phish", "1", "bad", "spam",
    "malware", "defacement", "suspicious",
}
_LABEL_MAP_BENIGN = {
    "benign", "legitimate", "safe", "0", "good", "clean", "legal",
}


def load_csv(path: Path, url_col: Optional[str] = None,
             label_col: Optional[str] = None) -> pd.DataFrame:
    """Load a CSV and validate the URL + label columns exist.

    Args:
        path: Path to CSV file.
        url_col: Name of URL column. If None, auto-detected.
        label_col: Name of label column. If None, auto-detected.

    Returns:
        DataFrame with at least 'url' and 'label' columns.

    Raises:
        FileNotFoundError: If CSV does not exist.
        ValueError: If required columns cannot be found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # ── Auto-detect columns ───────────────────────────────────
    cols_lower = {c.lower().strip(): c for c in df.columns}

    if url_col is None:
        for candidate in ["url", "urls", "uri", "link", "website", "webpage"]:
            if candidate in cols_lower:
                url_col = cols_lower[candidate]
                break
    if url_col is None:
        raise ValueError(
            f"Cannot detect URL column. Columns: {list(df.columns)}"
        )

    if label_col is None:
        for candidate in ["label", "labels", "target", "class", "status", "type"]:
            if candidate in cols_lower:
                label_col = cols_lower[candidate]
                break
    if label_col is None:
        raise ValueError(
            f"Cannot detect label column. Columns: {list(df.columns)}"
        )

    logger.info("Detected URL column: '%s', Label column: '%s'", url_col, label_col)

    # ── Rename to canonical names ─────────────────────────────
    df = df.rename(columns={url_col: "url", label_col: "label"})

    return df


def inspect_labels(df: pd.DataFrame) -> Dict[str, int]:
    """Return unique label values and their counts."""
    return df["label"].value_counts().to_dict()


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Convert labels to canonical binary {0, 1}.

    Raises:
        ValueError: If unknown label values are found.
    """
    df = df.copy()

    # If already numeric 0/1
    unique_vals = set(df["label"].dropna().unique())
    if unique_vals <= {0, 1} or unique_vals <= {0.0, 1.0}:
        df["label"] = df["label"].astype(int)
        logger.info("Labels already in {0, 1} format")
        return df

    # String labels
    str_vals = {str(v).lower().strip() for v in unique_vals}

    unknown = str_vals - _LABEL_MAP_MALICIOUS - _LABEL_MAP_BENIGN
    if unknown:
        raise ValueError(
            f"Unknown label values detected: {unknown}. "
            f"Expected values from {_LABEL_MAP_BENIGN | _LABEL_MAP_MALICIOUS}. "
            f"STOPPING — manual review required."
        )

    def _map_label(val):
        v = str(val).lower().strip()
        if v in _LABEL_MAP_MALICIOUS:
            return 1
        if v in _LABEL_MAP_BENIGN:
            return 0
        return None  # Should not reach here

    df["label"] = df["label"].apply(_map_label)
    dropped = df["label"].isna().sum()
    if dropped > 0:
        logger.warning("Dropped %d rows with unmappable labels", dropped)
        df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    logger.info("Labels normalized to canonical {0=BENIGN, 1=MALICIOUS}")
    return df


def extract_domain(url: str) -> str:
    """Extract registrable domain from a URL using tldextract.

    Falls back to basic hostname parsing if tldextract is unavailable.

    Args:
        url: Raw URL string (treated as inert data).

    Returns:
        Registrable domain string, or "unknown" if extraction fails.
    """
    if not url or not isinstance(url, str):
        return "unknown"

    try:
        if _HAS_TLDEXTRACT:
            ext = tldextract.extract(url)
            # Use new API name (registered_domain is deprecated)
            domain = getattr(ext, 'top_domain_under_public_suffix',
                             None) or ext.registered_domain
            if domain:
                return domain.lower()
            # If no registered domain, use the full domain
            full = ext.fqdn
            return full.lower() if full else "unknown"
        else:
            parsed = urlparse(url)
            hostname = parsed.hostname or ""
            return hostname.lower() if hostname else "unknown"
    except Exception:
        return "unknown"


def add_domain_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'domain' column for leakage-safe grouped splitting."""
    df = df.copy()
    df["domain"] = df["url"].apply(extract_domain)
    n_unknown = (df["domain"] == "unknown").sum()
    if n_unknown > 0:
        logger.warning("%d URLs have 'unknown' domain", n_unknown)
    logger.info("Extracted %d unique domains", df["domain"].nunique())
    return df


def clean_dataset(df: pd.DataFrame,
                  min_url_length: int = 10,
                  remove_duplicates: bool = True) -> pd.DataFrame:
    """Clean the dataset: remove missing, short, and duplicate URLs.

    Args:
        df: DataFrame with 'url' and 'label' columns.
        min_url_length: Minimum URL length to keep.
        remove_duplicates: Whether to remove duplicate URLs.

    Returns:
        Cleaned DataFrame.
    """
    initial_count = len(df)
    df = df.copy()

    # Remove rows with missing URL or label
    df = df.dropna(subset=["url", "label"])
    after_na = len(df)
    if after_na < initial_count:
        logger.info("Dropped %d rows with missing url/label", initial_count - after_na)

    # Remove empty URLs
    df = df[df["url"].str.strip().str.len() > 0]

    # Remove very short URLs
    df = df[df["url"].str.len() >= min_url_length]
    after_short = len(df)
    if after_short < after_na:
        logger.info("Dropped %d short URLs (< %d chars)", after_na - after_short, min_url_length)

    # Remove duplicates
    if remove_duplicates:
        before_dedup = len(df)
        df = df.drop_duplicates(subset=["url"], keep="first")
        after_dedup = len(df)
        if after_dedup < before_dedup:
            logger.info("Removed %d duplicate URLs", before_dedup - after_dedup)

    logger.info("Dataset cleaned: %d → %d rows", initial_count, len(df))
    return df.reset_index(drop=True)
