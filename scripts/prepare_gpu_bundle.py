"""
Prepare QRShield GPU Training Bundle for Google Colab.

Copies the verified 50K/10K/10K development subset (70,000 images total),
rewrites split CSVs with relative paths, packages core code, notebooks,
and configuration, verifies image integrity, and creates QRShield_GPU_bundle.zip.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = PROJECT_ROOT / "gpu_bundle"
ZIP_PATH = PROJECT_ROOT / "QRShield_GPU_bundle.zip"

IMAGE_DEV_DIR = PROJECT_ROOT / "data" / "processed" / "image_dev"


def setup_bundle_structure() -> None:
    if BUNDLE_DIR.exists():
        print(f"Cleaning existing {BUNDLE_DIR}...")
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    (BUNDLE_DIR / "src").mkdir(parents=True, exist_ok=True)
    (BUNDLE_DIR / "scripts").mkdir(parents=True, exist_ok=True)
    (BUNDLE_DIR / "notebooks").mkdir(parents=True, exist_ok=True)
    (BUNDLE_DIR / "data" / "processed" / "image_dev").mkdir(parents=True, exist_ok=True)
    (BUNDLE_DIR / "data" / "images" / "train").mkdir(parents=True, exist_ok=True)
    (BUNDLE_DIR / "data" / "images" / "val").mkdir(parents=True, exist_ok=True)
    (BUNDLE_DIR / "data" / "images" / "test").mkdir(parents=True, exist_ok=True)
    (BUNDLE_DIR / "artifacts" / "image_baseline_gpu").mkdir(parents=True, exist_ok=True)


def copy_project_code() -> None:
    print("Copying src/...")
    for root, dirs, files in os.walk(PROJECT_ROOT / "src"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        rel_root = Path(root).relative_to(PROJECT_ROOT / "src")
        target_dir = BUNDLE_DIR / "src" / rel_root
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            if f.endswith(".py"):
                shutil.copy2(Path(root) / f, target_dir / f)

    print("Copying scripts/...")
    for f in (PROJECT_ROOT / "scripts").glob("*.py"):
        if f.name != "__pycache__":
            shutil.copy2(f, BUNDLE_DIR / "scripts" / f.name)

    print("Copying notebooks/...")
    gpu_nb = PROJECT_ROOT / "notebooks" / "05_image_baseline_gpu.ipynb"
    if gpu_nb.exists():
        shutil.copy2(gpu_nb, BUNDLE_DIR / "notebooks" / gpu_nb.name)

    print("Copying requirements.txt...")
    req_file = PROJECT_ROOT / "requirements.txt"
    if req_file.exists():
        shutil.copy2(req_file, BUNDLE_DIR / "requirements.txt")


def create_readme() -> None:
    readme_content = """# QRShield — GPU Training Bundle for Google Colab

**Project:** QRShield — Explainable Multimodal Quishing Detection  
**Experiment:** Phase 2B MobileNetV2 Image-Only Baseline Development Training  
**Dataset:** 70,000 Stratified Development Images (50,000 Train / 10,000 Val / 10,000 Test)  
**Strict Policy:** QR IMAGE ONLY. No URL, domain, or metadata features passed to the MobileNetV2 model.

---

## Google Colab Instructions

### 1. Upload & Unzip
Upload `QRShield_GPU_bundle.zip` to your Colab session (`/content/`), then run:

```bash
!unzip -q /content/QRShield_GPU_bundle.zip -d /content/QRShield
%cd /content/QRShield
```

### 2. Install Requirements
```bash
!pip install -q -r requirements.txt
```

### 3. Verify GPU Runtime
Ensure your Colab runtime is set to **GPU** (`Runtime > Change runtime type > T4 GPU`).
```python
import torch
print("CUDA Available:", torch.cuda.is_available())
print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```

### 4. Run Training
You can run the notebook:
- Open `notebooks/05_image_baseline_gpu.ipynb` and run all cells.

Or run the CLI script directly:
```bash
python src/models/train_image_baseline.py \\
    --train-csv data/processed/image_dev/train.csv \\
    --val-csv data/processed/image_dev/val.csv \\
    --test-csv data/processed/image_dev/test.csv \\
    --artifact-dir artifacts/image_baseline_gpu \\
    --train-samples 50000 \\
    --val-samples 10000 \\
    --test-samples 10000 \\
    --batch-size 32 \\
    --epochs 4 \\
    --stage-a-epochs 2 \\
    --stage-b-epochs 2
```

### 5. Download Results
Download the generated `artifacts/image_baseline_gpu/` directory back to the local repository:
- `best_model.pt`
- `last_model.pt`
- `metrics.json`
- `config.json`
- `latency.json`
- `training_history.csv`
- `training_history.png`
- `confusion_matrix.png`
- `roc_curve.png`
- `pr_curve.png`
- `classification_report.txt`
- `gradcam/` visualizations
"""
    with open(BUNDLE_DIR / "README_GPU.md", "w", encoding="utf-8") as f:
        f.write(readme_content)


def process_split(
    split_name: str,
    source_csv: Path,
) -> Tuple[pd.DataFrame, int, int, List[str], Dict[str, int]]:
    print(f"\nProcessing {split_name} split from {source_csv}...")
    df = pd.read_csv(source_csv)
    total_rows = len(df)
    print(f"Total rows in {split_name}: {total_rows}")

    target_img_dir = BUNDLE_DIR / "data" / "images" / split_name
    target_img_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    missing_paths: List[str] = []
    formats: Dict[str, int] = {}
    new_image_paths: List[str] = []

    start_time = time.time()
    for idx, row in df.iterrows():
        sample_id = str(row["sample_id"])
        raw_path = str(row["image_path"])
        p = Path(raw_path)

        if not p.exists():
            # Try relative to PROJECT_ROOT
            if (PROJECT_ROOT / raw_path).exists():
                p = PROJECT_ROOT / raw_path
            else:
                missing_paths.append(raw_path)
                new_image_paths.append(raw_path)
                continue

        # Format detection
        ext = p.suffix.lower() or ".png"
        formats[ext] = formats.get(ext, 0) + 1

        # Deterministic collision-free filename
        dest_filename = f"{sample_id}_{p.name}"
        dest_path = target_img_dir / dest_filename

        shutil.copy2(p, dest_path)
        copied_count += 1

        # Relative path for Colab bundle
        new_rel_path = f"data/images/{split_name}/{dest_filename}"
        new_image_paths.append(new_rel_path)

        if copied_count % 10000 == 0:
            elapsed = time.time() - start_time
            print(f"  [{split_name}] Copied {copied_count}/{total_rows} images ({elapsed:.1f}s)...")

    df_rewritten = df.copy()
    df_rewritten["image_path"] = new_image_paths
    dest_csv = BUNDLE_DIR / "data" / "processed" / "image_dev" / f"{split_name}.csv"
    df_rewritten.to_csv(dest_csv, index=False)
    print(f"Saved rewritten {split_name}.csv to {dest_csv}")

    return df_rewritten, copied_count, len(missing_paths), missing_paths, formats


def verify_sample_readability(bundle_dir: Path, samples_per_split: int = 50) -> Tuple[int, int]:
    print("\nVerifying sample image readability across splits...")
    readable = 0
    unreadable = 0

    for split in ["train", "val", "test"]:
        csv_path = bundle_dir / "data" / "processed" / "image_dev" / f"{split}.csv"
        df = pd.read_csv(csv_path)
        sample_rows = df.head(samples_per_split)

        for _, row in sample_rows.iterrows():
            img_rel = row["image_path"]
            img_abs = bundle_dir / img_rel
            if not img_abs.exists():
                unreadable += 1
                continue
            try:
                with Image.open(img_abs) as img:
                    img.verify()
                with Image.open(img_abs) as img:
                    _ = img.convert("RGB").size
                readable += 1
            except Exception as exc:
                print(f"Unreadable image {img_abs}: {exc}")
                unreadable += 1

    print(f"Sample verification complete: {readable} readable, {unreadable} unreadable.")
    return readable, unreadable


def create_bundle_zip() -> Tuple[Path, int]:
    print(f"\nCreating ZIP archive: {ZIP_PATH}...")
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    total_files = 0
    start_time = time.time()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, files in os.walk(BUNDLE_DIR):
            dirs[:] = [d for d in dirs if d not in {"__pycache__", ".pytest_cache", ".venv"}]
            for f in files:
                abs_f = Path(root) / f
                rel_f = abs_f.relative_to(BUNDLE_DIR)
                zf.write(abs_f, arcname=str(rel_f))
                total_files += 1
                if total_files % 10000 == 0:
                    print(f"  Zipped {total_files} files...")

    elapsed = time.time() - start_time
    zip_size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"ZIP created in {elapsed:.1f}s: {zip_size_mb:.2f} MB ({total_files} files)")
    return ZIP_PATH, total_files


def main() -> None:
    print("=" * 60)
    print("QRShield — GPU Bundle Preparation for Google Colab")
    print("=" * 60)

    setup_bundle_structure()
    copy_project_code()
    create_readme()

    splits = ["train", "val", "test"]
    stats: Dict[str, Dict[str, Any]] = {}
    total_copied = 0
    total_missing = 0
    all_formats: Dict[str, int] = {}

    for split in splits:
        source_csv = IMAGE_DEV_DIR / f"{split}.csv"
        _, copied, missing, _, fmts = process_split(split, source_csv)
        stats[split] = {"copied": copied, "missing": missing, "formats": fmts}
        total_copied += copied
        total_missing += missing
        for k, v in fmts.items():
            all_formats[k] = all_formats.get(k, 0) + v

    readable, unreadable = verify_sample_readability(BUNDLE_DIR)

    bundle_bytes = sum(f.stat().st_size for f in BUNDLE_DIR.rglob("*") if f.is_file())
    bundle_size_mb = bundle_bytes / (1024 * 1024)

    zip_path, zip_file_count = create_bundle_zip()

    print("\n" + "=" * 60)
    print("GPU BUNDLE PREPARATION REPORT")
    print("=" * 60)
    print(f"Train images copied:      {stats['train']['copied']}")
    print(f"Validation images copied: {stats['val']['copied']}")
    print(f"Test images copied:       {stats['test']['copied']}")
    print(f"Total images copied:      {total_copied}")
    print(f"Missing images:           {total_missing}")
    print(f"Unreadable sample images: {unreadable}")
    print(f"Formats detected:         {all_formats}")
    print(f"Total uncompressed size:  {bundle_size_mb:.2f} MB")
    print(f"ZIP file path:            {zip_path}")
    print(f"ZIP file size:            {zip_path.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"ZIP file count:           {zip_file_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
