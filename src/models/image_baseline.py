from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

from src.config import PATHS

REQUIRED_COLUMNS = [
    "sample_id",
    "image_path",
    "label",
]

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass
class ImageBaselineConfig:
    output_dir: Path = PATHS.image_baseline
    model_name: str = "MobileNetV2"
    pretrained: bool = True
    image_size: int = 224
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    patience: int = 3
    threshold: float = 0.5
    seed: int = 42
    class_weighting: str = "balanced"

    def to_dict(self) -> Dict[str, Any]:
        config = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split_frame(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Split CSV not found: {path}")

    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Split CSV missing required columns: {missing}")

    if "label" in df.columns:
        df["label"] = df["label"].astype(int)
    if "sample_id" not in df.columns:
        df["sample_id"] = df.index.astype(str)

    return df.reset_index(drop=True)


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def preprocess_image(image: np.ndarray | Image.Image, size: int = 224) -> torch.Tensor:
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    else:
        pil_image = ensure_rgb(image)

    if not isinstance(pil_image, Image.Image):
        raise TypeError("Unsupported image type for preprocessing.")

    rgb_image = ensure_rgb(pil_image).resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(rgb_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).float()
    mean = torch.tensor(MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(STD, dtype=tensor.dtype).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor


def resolve_image_path(image_path: str | Path, base_dir: Optional[Path] = None) -> Path:
    p = Path(image_path)
    if p.exists():
        return p
    candidate_rel = (base_dir or PATHS.root) / p
    if candidate_rel.exists():
        return candidate_rel
    candidate_cwd = Path.cwd() / p
    if candidate_cwd.exists():
        return candidate_cwd
    path_str = str(image_path).replace("\\", "/")
    marker = "DATASET/QR_EXTRACTED/"
    if marker in path_str:
        sub = path_str.split(marker, 1)[1]
        candidates = [
            (base_dir or PATHS.root) / "DATASET" / "QR_EXTRACTED" / sub,
            Path.cwd() / "DATASET" / "QR_EXTRACTED" / sub,
            Path("/content/QRShield/DATASET/QR_EXTRACTED") / sub,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return p


def verify_split_images(split_df: pd.DataFrame, sample_limit: int = 20) -> Dict[str, Any]:
    split_df = split_df.copy()
    limit = min(len(split_df), sample_limit)
    issues: List[str] = []
    dimension_counts: Counter = Counter()
    mode_counts: Counter = Counter()
    valid_count = 0

    for row in split_df.head(limit).itertuples(index=False):
        raw_path = str(getattr(row, "image_path", ""))
        image_path = resolve_image_path(raw_path)
        if not image_path.exists():
            issues.append(f"missing:{raw_path}")
            continue
        try:
            with Image.open(image_path) as im:
                im = ensure_rgb(im)
                dimension_counts[im.size] += 1
                mode_counts[im.mode] += 1
                valid_count += 1
        except Exception as exc:  # pragma: no cover - validation path guard
            issues.append(f"corrupt:{raw_path}:{exc}")

    status = "ok" if not issues else "error"
    dimension_summary = {
        f"{width}x{height}": count
        for (width, height), count in dimension_counts.most_common()
    }
    return {
        "status": status,
        "sample_limit": limit,
        "valid_samples": valid_count,
        "issues": issues,
        "dimension_summary": dimension_summary,
        "mode_summary": dict(mode_counts.most_common()),
    }


class QRImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 224,
        train: bool = False,
        augmentations: Optional[Sequence[Any]] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.train = train
        self.augmentations = list(augmentations or [])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        image_path = resolve_image_path(str(row["image_path"]))
        label = int(row["label"])
        image = Image.open(image_path)
        image = ensure_rgb(image)
        image_tensor = preprocess_image(image, size=self.image_size)
        sample_id = str(row.get("sample_id", idx))
        return {"image": image_tensor, "label": label, "sample_id": sample_id}


def build_mobilenetv2_model(num_classes: int = 1, pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = _BinaryClassifierWrapper(model)
    return model


class _BinaryClassifierWrapper(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if logits.dim() == 2 and logits.shape[1] == 1:
            return logits
        return logits


def compute_class_weights(labels: Sequence[int]) -> torch.Tensor:
    counts = Counter(int(v) for v in labels)
    total = sum(counts.values())
    inverse = {cls: total / max(count, 1) for cls, count in counts.items()}
    weights = torch.tensor([
        inverse.get(0, 1.0),
        inverse.get(1, 1.0),
    ], dtype=torch.float32)
    return weights / weights.mean()


def select_threshold_from_scores(y_true: Sequence[int], y_proba: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    if y_true.size == 0:
        return 0.5

    thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        preds = (y_proba >= threshold).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        tn = np.sum((preds == 0) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if f1 > best_score:
            best_score = float(f1)
            best_threshold = float(threshold)
    return best_threshold


def encode_model_response(probs: Sequence[float], threshold: float = 0.5) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for prob in probs:
        prob_f = float(prob)
        pred_int = 1 if prob_f >= threshold else 0
        label_name = "MALICIOUS" if pred_int == 1 else "BENIGN"
        results.append({
            "probability_malicious": prob_f,
            "prediction": pred_int,
            "label": label_name,
        })
    return results


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=lambda item: str(item))]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def safe_json_dump(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2)


def make_development_subset(
    df: pd.DataFrame,
    target_size: int = 50000,
    random_state: int = 42,
) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    if len(df) <= target_size:
        return df

    rng = np.random.default_rng(random_state)
    labels = df["label"].astype(int).tolist()
    pos_idx = df.index[df["label"].astype(int) == 1].tolist()
    neg_idx = df.index[df["label"].astype(int) == 0].tolist()

    pos_target = min(len(pos_idx), max(1, int(round(target_size * 0.5))))
    neg_target = min(len(neg_idx), max(1, target_size - pos_target))

    pos_sample = rng.choice(pos_idx, size=pos_target, replace=False)
    neg_sample = rng.choice(neg_idx, size=neg_target, replace=False)
    chosen = sorted(set(pos_sample.tolist()) | set(neg_sample.tolist()))
    return df.iloc[chosen].reset_index(drop=True)


def save_development_subset(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
    output_dir: str | Path,
    train_target: int = 50000,
    val_target: int = 10000,
    test_target: int = 10000,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subsets: Dict[str, pd.DataFrame] = {}
    for split_name, source_csv, target_size in [
        ("train", train_csv, train_target),
        ("val", val_csv, val_target),
        ("test", test_csv, test_target),
    ]:
        df = load_split_frame(source_csv)
        if len(df) <= target_size:
            subset = df.copy().reset_index(drop=True)
        else:
            rng = np.random.default_rng(random_state + (0 if split_name == "train" else 1 if split_name == "val" else 2))
            pos_df = df[df["label"] == 1].copy()
            neg_df = df[df["label"] == 0].copy()
            pos_goal = min(len(pos_df), max(1, int(round(target_size * (len(pos_df) / len(df))))))
            neg_goal = target_size - pos_goal
            if neg_goal > len(neg_df):
                neg_goal = len(neg_df)
                pos_goal = target_size - neg_goal
            if pos_goal > len(pos_df):
                pos_goal = len(pos_df)
                neg_goal = target_size - pos_goal
            pos_sample = pos_df.sample(n=pos_goal, random_state=random_state + 10, replace=False)
            neg_sample = neg_df.sample(n=neg_goal, random_state=random_state + 20, replace=False)
            subset = pd.concat([pos_sample, neg_sample], ignore_index=True).sample(frac=1, random_state=random_state + 30).reset_index(drop=True)
        subset = subset.reset_index(drop=True)
        subset.to_csv(output_dir / f"{split_name}.csv", index=False)
        subsets[split_name] = subset
    return subsets


def _read_rgb_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ensure_rgb(image)
        return np.asarray(image, dtype=np.uint8)


def gradcam_heatmap(model: nn.Module, image_tensor: torch.Tensor) -> np.ndarray:
    if not hasattr(model, "backbone"):
        raise ValueError("Grad-CAM requires a model with a backbone attribute.")

    model.eval()
    tensor = image_tensor.clone().detach().unsqueeze(0).requires_grad_(True)
    logits = model(tensor)
    logits[:, 0].sum().backward()
    gradients = tensor.grad.detach().abs().mean(dim=1).cpu().numpy()[0]
    heatmap = gradients / (gradients.max() + 1e-8)
    return heatmap.astype(np.float32)
