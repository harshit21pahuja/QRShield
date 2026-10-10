from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.data.qr_canonical import (
    build_split_rows,
    normalize_qr_path,
    reconcile_metadata,
    safe_domain,
    safe_url,
)


def write_png(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1, 1), color="white")
    image.save(path, format="PNG")


def test_normalize_qr_path_handles_variants() -> None:
    assert normalize_qr_path("Output\\data\\x.png") == "data/x.png"
    assert normalize_qr_path("./nested/qr.png") == "nested/qr.png"
    assert normalize_qr_path("") == ""


def test_safe_url_and_domain_extraction() -> None:
    ok, reason = safe_url("https://example.com/path")
    assert ok and reason is None
    assert safe_domain("https://sub.example.com/path") == "example.com"
    ok2, _ = safe_url("ftp://example.com")
    assert not ok2


def test_metadata_reconciliation_handles_valid_and_invalid_rows(tmp_path: Path) -> None:
    dataset_root = tmp_path / "DATASET" / "QR_EXTRACTED"
    benign_dir = dataset_root / "benign" / "sample"
    good_img = benign_dir / "good.png"
    write_png(good_img)

    bad_img = benign_dir / "bad.png"
    write_png(bad_img)

    csv_path = dataset_root / "benign" / "all_generated_urls.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["url", "qr_path", "label"])
        writer.writerow(["https://example.com/test", "benign/sample/good.png", "0"])
        writer.writerow(["bad-url", "benign/sample/bad.png", "0"])
        writer.writerow(["https://example.com/other", "missing.png", "0"])

    valid, invalid, conflicts, _ = reconcile_metadata(dataset_root)

    assert len(valid) == 1
    assert len(invalid) == 2
    assert any(row["reason"] == "unsupported_scheme" or row["reason"] == "missing_image_path" for row in invalid)
    assert isinstance(conflicts, list)


def test_grouped_splits_have_zero_overlap() -> None:
    rows = [
        {"sample_id": "qr_001", "image_path": "/tmp/a.png", "url": "https://a.com/u1", "label": 0, "domain_group": "a.com", "sha256": "aaa"},
        {"sample_id": "qr_002", "image_path": "/tmp/b.png", "url": "https://a.com/u2", "label": 1, "domain_group": "a.com", "sha256": "bbb"},
        {"sample_id": "qr_003", "image_path": "/tmp/c.png", "url": "https://b.com/u1", "label": 0, "domain_group": "b.com", "sha256": "ccc"},
        {"sample_id": "qr_004", "image_path": "/tmp/d.png", "url": "https://c.com/u1", "label": 1, "domain_group": "c.com", "sha256": "ddd"},
        {"sample_id": "qr_005", "image_path": "/tmp/e.png", "url": "https://d.com/u1", "label": 0, "domain_group": "d.com", "sha256": "eee"},
        {"sample_id": "qr_006", "image_path": "/tmp/f.png", "url": "https://e.com/u1", "label": 1, "domain_group": "e.com", "sha256": "fff"},
        {"sample_id": "qr_007", "image_path": "/tmp/g.png", "url": "https://f.com/u1", "label": 0, "domain_group": "f.com", "sha256": "ggg"},
    ]
    train_df, val_df, test_df = build_split_rows(rows)
    assert set(train_df["domain_group"]).isdisjoint(set(val_df["domain_group"]))
    assert set(train_df["domain_group"]).isdisjoint(set(test_df["domain_group"]))
    assert set(val_df["domain_group"]).isdisjoint(set(test_df["domain_group"]))
    assert not (set(train_df["url"]) & set(val_df["url"]))
    assert not (set(train_df["url"]) & set(test_df["url"]))
    assert not (set(train_df["sha256"]) & set(val_df["sha256"]))


def test_build_split_rows_keeps_domain_url_and_hash_unique_across_splits() -> None:
    rows = [
        {"sample_id": "qr_001", "image_path": "/tmp/a.png", "url": "https://dom1.com/u1", "label": 0, "domain_group": "dom1.com", "sha256": "aaa"},
        {"sample_id": "qr_002", "image_path": "/tmp/b.png", "url": "https://dom2.com/u1", "label": 1, "domain_group": "dom2.com", "sha256": "bbb"},
        {"sample_id": "qr_003", "image_path": "/tmp/c.png", "url": "https://dom3.com/u1", "label": 1, "domain_group": "dom3.com", "sha256": "ccc"},
        {"sample_id": "qr_004", "image_path": "/tmp/d.png", "url": "https://dom4.com/u1", "label": 0, "domain_group": "dom4.com", "sha256": "ddd"},
    ]
    train_df, val_df, test_df = build_split_rows(rows)
    assert not (set(train_df["url"]) & set(val_df["url"]))
    assert not (set(train_df["sha256"]) & set(val_df["sha256"]))
    assert not (set(train_df["domain_group"]) & set(val_df["domain_group"]))
