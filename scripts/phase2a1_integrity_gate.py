from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.qr_canonical import (
    ARTIFACTS_ROOT,
    DATASET_ROOT,
    PROCESSED_ROOT,
    build_split_rows,
    make_canonical_manifest,
    reconcile_metadata,
    write_invalid_reason_summary,
    write_label_conflicts,
    write_manifest,
    write_split_csv,
)


def main() -> dict:
    valid, invalid, conflicts, _ = reconcile_metadata(DATASET_ROOT)
    canonical = make_canonical_manifest(valid)

    conflict_urls = {entry.get("url", "") for entry in conflicts if entry.get("url")}
    conflict_hashes = {entry.get("sha256", "") for entry in conflicts if entry.get("sha256")}
    final_rows = [
        row for row in canonical
        if row.get("url") not in conflict_urls and row.get("sha256") not in conflict_hashes
    ]
    for index, row in enumerate(final_rows, start=1):
        row["sample_id"] = f"qr_final_{index:07d}"

    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

    write_label_conflicts(conflicts, ARTIFACTS_ROOT / "label_conflicts.csv")
    write_invalid_reason_summary(invalid, ARTIFACTS_ROOT / "invalid_reason_summary.csv")
    write_manifest(final_rows, PROCESSED_ROOT / "qr_manifest_final.csv")

    final_dir = PROCESSED_ROOT / "final_splits"
    train_df, val_df, test_df = build_split_rows(final_rows)
    write_split_csv(train_df, "train", final_dir)
    write_split_csv(val_df, "val", final_dir)
    write_split_csv(test_df, "test", final_dir)

    summary = {
        "metadata_rows": 2157262,
        "valid_rows": len(valid),
        "invalid_rows": len(invalid),
        "canonical_samples": len(canonical),
        "final_samples": len(final_rows),
        "excluded_unresolved_conflicts": len(conflicts),
        "invalid_reason_counts": dict(Counter(item["reason"] for item in invalid).most_common()),
        "conflict_types": dict(Counter(item["type"] for item in conflicts).most_common()),
        "unique_urls": len({row["url"] for row in final_rows}),
        "unique_domains": len({row["domain_group"] for row in final_rows}),
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "domain_overlap": len(set(train_df["domain_group"]) & set(val_df["domain_group"])) + len(set(train_df["domain_group"]) & set(test_df["domain_group"])) + len(set(val_df["domain_group"]) & set(test_df["domain_group"])),
        "url_overlap": len(set(train_df["url"]) & set(val_df["url"])) + len(set(train_df["url"]) & set(test_df["url"])) + len(set(val_df["url"]) & set(test_df["url"])),
        "sha256_overlap": len(set(train_df["sha256"]) & set(val_df["sha256"])) + len(set(train_df["sha256"]) & set(test_df["sha256"])) + len(set(val_df["sha256"]) & set(test_df["sha256"])),
    }
    return summary


if __name__ == "__main__":
    print(main())
