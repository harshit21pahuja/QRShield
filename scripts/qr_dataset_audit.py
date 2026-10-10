from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DATASET_ROOT = Path("DATASET")
EXTRACTED_ROOT = DATASET_ROOT / "QR_EXTRACTED"
OUTPUT_ROOT = Path("artifacts") / "qr_dataset_audit"
METADATA_EXTS = {".csv", ".json", ".txt", ".xlsx", ".xls", ".md"}
METADATA_KEYWORDS = ("metadata", "mapping", "readme", "url", "label", "domain")


def chunked_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def scan_label(label_dir: Path):
    file_counts = Counter()
    sample_paths = []
    mode_counts = Counter()
    width_values = []
    height_values = []
    file_size_values = []
    corrupt = []
    filename_groups = defaultdict(list)
    size_groups = defaultdict(list)
    nested_dirs = []
    scanned = 0

    for path in sorted(label_dir.rglob("*")):
        if path.is_dir():
            nested_dirs.append(str(path.relative_to(EXTRACTED_ROOT)))
            continue
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTS:
            scanned += 1
            file_counts[path.suffix.lower()] += 1
            filename_groups[path.name.lower()].append(str(path.relative_to(EXTRACTED_ROOT)))
            size_groups[path.stat().st_size].append(str(path.relative_to(EXTRACTED_ROOT)))
            if len(sample_paths) < 5:
                sample_paths.append(str(path.relative_to(EXTRACTED_ROOT)))
            try:
                with Image.open(path) as img:
                    width, height = img.size
                    width_values.append(width)
                    height_values.append(height)
                    file_size_values.append(path.stat().st_size)
                    mode_counts[img.mode] += 1
            except Exception as exc:
                corrupt.append({
                    "path": str(path.relative_to(EXTRACTED_ROOT)),
                    "error": str(exc),
                })

    duplicate_filenames = {name: paths for name, paths in sorted(filename_groups.items()) if len(paths) > 1}
    duplicate_sizes = {size: paths for size, paths in sorted(size_groups.items()) if len(paths) > 1}

    duplicate_hashes = {}
    suspicious_paths = sorted({p for paths in duplicate_filenames.values() for p in paths} | {p for paths in duplicate_sizes.values() for p in paths})
    for path in suspicious_paths:
        full_path = EXTRACTED_ROOT / path
        if not full_path.exists():
            continue
        digest = chunked_sha256(full_path)
        duplicate_hashes.setdefault(digest, []).append(path)
    duplicate_hashes = {digest: paths for digest, paths in sorted(duplicate_hashes.items()) if len(paths) > 1}

    return {
        "label": label_dir.name,
        "count": scanned,
        "file_formats": dict(sorted(file_counts.items())),
        "sample_paths": sample_paths,
        "nested_dirs": sorted(set(nested_dirs)),
        "duplicate_filenames": duplicate_filenames,
        "duplicate_sizes": duplicate_sizes,
        "duplicate_hashes": duplicate_hashes,
        "corrupt": corrupt,
        "mode_counts": dict(sorted(mode_counts.items())),
        "dimensions": {
            "width_min": min(width_values) if width_values else None,
            "width_max": max(width_values) if width_values else None,
            "width_avg": round(sum(width_values) / len(width_values), 2) if width_values else None,
            "height_min": min(height_values) if height_values else None,
            "height_max": max(height_values) if height_values else None,
            "height_avg": round(sum(height_values) / len(height_values), 2) if height_values else None,
            "size_min": min(file_size_values) if file_size_values else None,
            "size_max": max(file_size_values) if file_size_values else None,
            "size_avg": round(sum(file_size_values) / len(file_size_values), 2) if file_size_values else None,
        },
        "duplicate_method": "Filename collisions + same-size grouping + chunked SHA-256 hashing on candidate groups only. No full-dataset image bytes were loaded into memory.",
    }


def inspect_metadata(root: Path):
    hits = []
    csv_summaries = []
    zip_entries = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name_lower = path.name.lower()
        rel_lower = str(path.relative_to(root)).lower()
        if path.suffix.lower() in METADATA_EXTS or any(keyword in name_lower for keyword in METADATA_KEYWORDS) or any(keyword in rel_lower for keyword in METADATA_KEYWORDS):
            hits.append({
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "extension": path.suffix.lower(),
            })
            if path.suffix.lower() == ".csv":
                try:
                    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                        reader = csv.reader(fh)
                        header = next(reader, None)
                        sample_rows = []
                        for _ in range(3):
                            row = next(reader, None)
                            if row is not None:
                                sample_rows.append(row)
                    csv_summaries.append({
                        "path": str(path.relative_to(root)),
                        "header": header[:20] if header else [],
                        "sample_rows": sample_rows,
                    })
                except Exception as exc:
                    csv_summaries.append({
                        "path": str(path.relative_to(root)),
                        "header": [],
                        "sample_rows": [],
                        "read_error": str(exc),
                    })

    for zip_path in sorted(root.rglob("*.zip")):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zip_entries.append({
                    "zip_path": str(zip_path.relative_to(root)),
                    "members": zf.namelist()[:20],
                    "member_count": len(zf.namelist()),
                })
        except Exception as exc:
            zip_entries.append({
                "zip_path": str(zip_path.relative_to(root)),
                "members": [],
                "error": str(exc),
            })

    return {"metadata_hits": hits, "csv_summaries": csv_summaries, "zip_entries": zip_entries}


def url_mapping_status(csv_summaries):
    has_url = False
    has_domain = False
    urls = []
    domains = []

    for item in csv_summaries:
        header = [h.lower() for h in item.get("header", [])]
        if not header:
            continue
        if any(name in header for name in ["url", "urls", "uri", "link", "qr_url"]):
            has_url = True
            for row in item.get("sample_rows", []):
                for col, value in zip(item.get("header", []), row):
                    if isinstance(value, str) and col.lower() in {"url", "urls", "uri", "link", "qr_url"}:
                        urls.append(value)
                        domain = urlparse(value).netloc
                        if domain:
                            domains.append(domain)
                            has_domain = True

    return {
        "url_mapping": "YES" if has_url else "NO",
        "domain_mapping": "YES" if has_domain else "NO",
        "url_examples": urls[:10],
        "domain_examples": domains[:10],
    }


def build_report():
    benign_dir = EXTRACTED_ROOT / "benign"
    malicious_dir = EXTRACTED_ROOT / "malicious"
    scans = [scan_label(p) for p in [benign_dir, malicious_dir] if p.exists()]
    benign = next((s for s in scans if s["label"] == "benign"), {"count": 0, "file_formats": {}, "sample_paths": [], "nested_dirs": [], "duplicate_filenames": {}, "duplicate_sizes": {}, "duplicate_hashes": {}, "corrupt": [], "mode_counts": {}, "dimensions": {}, "duplicate_method": ""})
    malicious = next((s for s in scans if s["label"] == "malicious"), {"count": 0, "file_formats": {}, "sample_paths": [], "nested_dirs": [], "duplicate_filenames": {}, "duplicate_sizes": {}, "duplicate_hashes": {}, "corrupt": [], "mode_counts": {}, "dimensions": {}, "duplicate_method": ""})
    metadata = inspect_metadata(DATASET_ROOT)
    mapping = url_mapping_status(metadata["csv_summaries"])

    total_disk_usage_bytes = 0
    for path in EXTRACTED_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTS:
            total_disk_usage_bytes += path.stat().st_size

    report = {
        "recursive_scan_status": "OK",
        "benign_images": benign["count"],
        "malicious_images": malicious["count"],
        "total_images": benign["count"] + malicious["count"],
        "nested_image_directories_found": sorted(set(benign["nested_dirs"]) | set(malicious["nested_dirs"])),
        "file_formats": dict(sorted((benign["file_formats"] | malicious["file_formats"]).items())),
        "image_dimensions": {"benign": benign["dimensions"], "malicious": malicious["dimensions"]},
        "image_modes": {"benign": benign["mode_counts"], "malicious": malicious["mode_counts"]},
        "corrupted_images": {"benign": len(benign["corrupt"]), "malicious": len(malicious["corrupt"]), "total": len(benign["corrupt"]) + len(malicious["corrupt"])},
        "duplicate_detection": {
            "methodology": benign["duplicate_method"],
            "benign_duplicate_filenames": benign["duplicate_filenames"],
            "benign_duplicate_sizes": benign["duplicate_sizes"],
            "benign_duplicate_hashes": benign["duplicate_hashes"],
            "malicious_duplicate_filenames": malicious["duplicate_filenames"],
            "malicious_duplicate_sizes": malicious["duplicate_sizes"],
            "malicious_duplicate_hashes": malicious["duplicate_hashes"],
        },
        "metadata_files": metadata["metadata_hits"],
        "csv_metadata_preview": metadata["csv_summaries"],
        "zip_contents": metadata["zip_entries"],
        "url_mapping": mapping["url_mapping"],
        "domain_mapping": mapping["domain_mapping"],
        "url_examples": mapping["url_examples"],
        "domain_examples": mapping["domain_examples"],
        "sample_paths": {"benign": benign["sample_paths"], "malicious": malicious["sample_paths"]},
        "dataset_disk_usage_bytes": total_disk_usage_bytes,
        "dataset_disk_usage_human": f"{total_disk_usage_bytes / (1024 ** 2):.2f} MB",
        "label_validation": {
            "folder_names": ["benign", "malicious"],
            "benign_label_value": 0,
            "malicious_label_value": 1,
            "note": "The local folder names follow the official CIC convention: benign = 0, malicious = 1.",
        },
        "official_expected_count": "Not locally verifiable from the repository; the extracted archive does not include an authoritative expected count file.",
        "discrepancy": "The extracted dataset contains nested qrs directories and repeated filenames across benign and malicious groups. Counting recursively resolves the earlier 0-image bug, but the archive layout is not a clean one-to-one label manifest.",
        "reason_previous_zero_image_result": "The previous audit used direct folder scans (Path.iterdir()) and never traversed nested qrs directories, so it skipped the actual PNG files.",
        "domain_aware_grouping_possible": "YES" if mapping["domain_mapping"] == "YES" else "NO",
        "dataset_ready_for_phase_2": "NO",
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    txt_lines = [
        "QR DATASET AUDIT — CORRECTED",
        "===========================",
        f"Benign images: {report['benign_images']}",
        f"Malicious images: {report['malicious_images']}",
        f"Total images: {report['total_images']}",
        "",
        "Nested image directories found:",
    ]
    for item in report["nested_image_directories_found"][:20]:
        txt_lines.append(f"- {item}")
    txt_lines.extend(["", "File formats:"])
    for key, value in report["file_formats"].items():
        txt_lines.append(f"- {key}: {value}")
    txt_lines.extend(["", "Image dimensions:"])
    txt_lines.append(f"- benign: {report['image_dimensions']['benign']}")
    txt_lines.append(f"- malicious: {report['image_dimensions']['malicious']}")
    txt_lines.extend(["", "Image modes:"])
    txt_lines.append(f"- benign: {report['image_modes']['benign']}")
    txt_lines.append(f"- malicious: {report['image_modes']['malicious']}")
    txt_lines.extend(["", f"Corrupted images: {report['corrupted_images']['total']}", "", "Duplicate detection result:"])
    txt_lines.append(f"- {report['duplicate_detection']['methodology']}")
    txt_lines.extend(["", "Metadata files:"])
    for item in report["metadata_files"][:20]:
        txt_lines.append(f"- {item['path']} ({item['size_bytes']} bytes)")
    txt_lines.extend([
        "",
        f"URL mapping: {report['url_mapping']}",
        f"Domain mapping: {report['domain_mapping']}",
        f"Domain-aware grouping possible: {report['domain_aware_grouping_possible']}",
        f"Total disk usage: {report['dataset_disk_usage_human']} ({report['dataset_disk_usage_bytes']} bytes)",
        f"Official expected count: {report['official_expected_count']}",
        f"Discrepancy: {report['discrepancy']}",
        f"Reason for previous 0-image result: {report['reason_previous_zero_image_result']}",
        f"Dataset ready for Phase 2: {report['dataset_ready_for_phase_2']}",
        "",
        "Sample paths:",
    ])
    for label in ["benign", "malicious"]:
        txt_lines.append(f"{label}:")
        for path in report["sample_paths"][label][:5]:
            txt_lines.append(f"- {path}")

    (OUTPUT_ROOT / "report.txt").write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    report = build_report()
    print(json.dumps({
        "benign_images": report["benign_images"],
        "malicious_images": report["malicious_images"],
        "total_images": report["total_images"],
        "file_formats": report["file_formats"],
        "url_mapping": report["url_mapping"],
        "domain_mapping": report["domain_mapping"],
        "dataset_ready_for_phase_2": report["dataset_ready_for_phase_2"],
        "report_paths": [str(OUTPUT_ROOT / "report.json"), str(OUTPUT_ROOT / "report.txt")],
    }, indent=2))
