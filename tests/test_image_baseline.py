from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.models.image_baseline import (
    QRImageDataset,
    build_mobilenetv2_model,
    load_split_frame,
    preprocess_image,
    verify_split_images,
)


@pytest.fixture
def temp_qr_dataset(tmp_path: Path) -> Path:
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    for d in [train_dir, val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for label, name in [(0, "benign.png"), (1, "malicious.png")]:
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[:, :, 0] = 255
        if label == 1:
            img[:, :, 1] = 255
        path = train_dir / name
        Image.fromarray(img).save(path)

    df = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "image_path": [str(train_dir / "benign.png"), str(train_dir / "malicious.png")],
            "url": ["https://a.com", "https://b.com"],
            "label": [0, 1],
            "domain_group": ["a.com", "b.com"],
            "sha256": ["aaa", "bbb"],
        }
    )
    csv_path = train_dir / "dummy.csv"
    df.to_csv(csv_path, index=False)

    return tmp_path


def test_load_split_frame(temp_qr_dataset: Path) -> None:
    df = load_split_frame(temp_qr_dataset / "train" / "dummy.csv")
    assert list(df.columns) == [
        "sample_id",
        "image_path",
        "url",
        "label",
        "domain_group",
        "sha256",
    ]


def test_preprocess_image_returns_rgb_tensor() -> None:
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    tensor = preprocess_image(arr, size=224)
    assert tensor.shape == (3, 224, 224)
    assert np.isfinite(tensor.numpy()).all()
    assert tensor.min() >= -3.0 and tensor.max() <= 5.0


def test_dataset_returns_image_and_label() -> None:
    path = Path("data/processed/final_splits/train.csv")
    if not path.exists():
        pytest.skip("Final split dataset not available in this environment.")

    df = pd.read_csv(path).head(4)
    dataset = QRImageDataset(df)
    sample = dataset[0]
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["label"] in {0, 1}
    assert isinstance(sample["sample_id"], str)


def test_mobilenetv2_forward_pass() -> None:
    model = build_mobilenetv2_model(num_classes=1)
    x = torch.rand(2, 3, 224, 224)
    logits = model(x)
    probs = torch.sigmoid(logits)
    assert logits.shape == (2, 1)
    assert probs.shape == (2, 1)
    assert torch.all((probs >= 0.0) & (probs <= 1.0)).item() is True


def test_verify_split_images_handles_missing_file() -> None:
    df = pd.DataFrame({
        "sample_id": ["missing"],
        "image_path": ["/definitely/missing.png"],
        "label": [0],
    })
    result = verify_split_images(df, sample_limit=1)
    assert result["status"] in {"ok", "error"}
    assert result["issues"]
