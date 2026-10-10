from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from src.config import PATHS, SEEDS
from src.evaluation.plots import plot_confusion_matrix, plot_pr_curve, plot_roc_curve
from src.models.image_baseline import (
    ImageBaselineConfig,
    QRImageDataset,
    build_mobilenetv2_model,
    compute_class_weights,
    gradcam_heatmap,
    load_split_frame,
    make_development_subset,
    preprocess_image,
    resolve_image_path,
    safe_json_dump,
    save_development_subset,
    select_threshold_from_scores,
    set_seed,
    verify_split_images,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MobileNetV2 QR image-only baseline.")
    parser.add_argument("--train-csv", type=str, default=str(PATHS.data_processed / "final_splits" / "train.csv"))
    parser.add_argument("--val-csv", type=str, default=str(PATHS.data_processed / "final_splits" / "val.csv"))
    parser.add_argument("--test-csv", type=str, default=str(PATHS.data_processed / "final_splits" / "test.csv"))
    parser.add_argument("--artifact-dir", type=str, default=str(PATHS.image_baseline))
    parser.add_argument("--train-samples", type=int, default=50000)
    parser.add_argument("--val-samples", type=int, default=10000)
    parser.add_argument("--test-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--stage-a-epochs", type=int, default=2)
    parser.add_argument("--stage-b-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=SEEDS.model_seed)
    return parser.parse_args()


def print_class_distribution(label_name: str, df: pd.DataFrame) -> None:
    counts = df["label"].value_counts().sort_index().to_dict()
    benign = int(counts.get(0, 0))
    malignant = int(counts.get(1, 0))
    total = max(1, len(df))
    print(f"{label_name}:")
    print(f"  benign = {benign}")
    print(f"  malicious = {malignant}")
    print(f"  benign_pct = {benign / total * 100:.2f}%")
    print(f"  malicious_pct = {malignant / total * 100:.2f}%")


def freeze_backbone(model: nn.Module, freeze: bool) -> None:
    if hasattr(model, "backbone"):
        for name, param in model.backbone.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = not freeze
    elif hasattr(model, "features"):
        for param in model.features.parameters():
            param.requires_grad = not freeze


def create_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = QRImageDataset(train_df, image_size=224, train=True)
    val_ds = QRImageDataset(val_df, image_size=224, train=False)
    test_ds = QRImageDataset(test_df, image_size=224, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader, test_loader


def gather_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: List[int] = []
    y_prob: List[float] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy().astype(int)
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy().astype(float)
            y_true.extend(labels.tolist())
            y_prob.extend(probs.tolist())
    return np.asarray(y_true, dtype=int), np.asarray(y_prob, dtype=float)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def save_training_plots(history: List[Dict[str, float]], artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_csv(artifact_dir / "training_history.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    ax.plot(history_df["epoch"], history_df["val_f1"], label="val_f1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(artifact_dir / "training_history.png", dpi=150)
    plt.close(fig)


def benchmark_latency(model: nn.Module, loader: DataLoader, device: torch.device, num_runs: int = 50) -> Dict[str, float]:
    model.eval()
    latencies: List[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            for batch in loader:
                images = batch["image"].to(device)
                t0 = time.perf_counter()
                _ = model(images)
                latencies.append((time.perf_counter() - t0) * 1000)
                break
    return {"average_ms": float(np.mean(latencies)), "p95_ms": float(np.percentile(latencies, 95)), "runs": len(latencies)}


def save_gradcam_examples(model: nn.Module, df: pd.DataFrame, artifact_dir: Path, device: torch.device, max_examples: int = 4) -> None:
    gradcam_dir = artifact_dir / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    selected = df.iloc[:max_examples].copy()
    for idx, row in selected.iterrows():
        image_path = resolve_image_path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        tensor = preprocess_image(image, size=224).to(device)
        heatmap = gradcam_heatmap(model, tensor.cpu())
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(heatmap, cmap="jet", alpha=0.6)
        ax.imshow(np.asarray(image.resize((224, 224))), alpha=0.7)
        ax.set_axis_off()
        fig.savefig(gradcam_dir / f"gradcam_{row['sample_id']}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    set_seed(args.seed)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if (artifact_dir / "model.pt").exists():
        sanity_path = artifact_dir / "sanity_model.pt"
        if not sanity_path.exists():
            import shutil
            shutil.copy2(artifact_dir / "model.pt", sanity_path)

    dev_dir = Path("data/processed/image_dev")
    dev_dir.mkdir(parents=True, exist_ok=True)
    subset_map = save_development_subset(args.train_csv, args.val_csv, args.test_csv, dev_dir, train_target=args.train_samples, val_target=args.val_samples, test_target=args.test_samples, random_state=args.seed)
    train_df = subset_map["train"]
    val_df = subset_map["val"]
    test_df = subset_map["test"]

    print_class_distribution("TRAIN", train_df)
    print_class_distribution("VALIDATION", val_df)
    print_class_distribution("TEST", test_df)

    class_distribution = {
        "train": train_df["label"].value_counts().sort_index().to_dict(),
        "val": val_df["label"].value_counts().sort_index().to_dict(),
        "test": test_df["label"].value_counts().sort_index().to_dict(),
    }
    safe_json_dump(class_distribution, artifact_dir / "class_distribution.json")
    safe_json_dump({
        "dataset_root": str(dev_dir),
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "train_label_counts": train_df["label"].value_counts().sort_index().to_dict(),
        "val_label_counts": val_df["label"].value_counts().sort_index().to_dict(),
        "test_label_counts": test_df["label"].value_counts().sort_index().to_dict(),
    }, artifact_dir / "development_dataset_summary.json")

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        safe_json_dump(verify_split_images(df, sample_limit=10), artifact_dir / f"{name}_image_validation.json")

    train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    model = build_mobilenetv2_model(num_classes=1, pretrained=True).to(device)
    print("Pretrained weights loaded successfully for MobileNetV2.")

    pos_weight = compute_class_weights(train_df["label"].tolist())[1:2].to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history: List[Dict[str, float]] = []
    best_val_f1 = -1.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_threshold = args.threshold
    patience = 0
    current_lr = args.lr

    stage_a_epochs = args.stage_a_epochs
    stage_b_epochs = args.stage_b_epochs
    stage_a_done = False

    for stage_name, epochs, lr, unfreeze in [
        ("stage_a", stage_a_epochs, args.lr, False),
        ("stage_b", stage_b_epochs, args.fine_tune_lr, True),
    ]:
        if epochs <= 0:
            continue
        if stage_name == "stage_a":
            freeze_backbone(model, freeze=True)
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=args.weight_decay)
        else:
            freeze_backbone(model, freeze=False)
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

        for epoch_offset in range(1, epochs + 1):
            epoch = len(history) + 1
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device).float().view(-1, 1)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
            train_loss = running_loss / len(train_df)

            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device).float().view(-1, 1)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * images.size(0)
            val_loss /= len(val_df)

            train_true, train_prob = gather_predictions(model, train_loader, device)
            train_metrics = compute_metrics(train_true, train_prob, args.threshold)
            val_true, val_prob = gather_predictions(model, val_loader, device)
            val_threshold = select_threshold_from_scores(val_true, val_prob)
            val_metrics = compute_metrics(val_true, val_prob, val_threshold)

            record = {
                "epoch": epoch,
                "stage": stage_name,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_f1": float(train_metrics["f1"]),
                "val_f1": float(val_metrics["f1"]),
                "learning_rate": float(lr),
                "val_threshold": float(val_threshold),
            }
            history.append(record)
            message = (
                f"Epoch {epoch}/{args.stage_a_epochs + args.stage_b_epochs} "
                f"stage={stage_name} train_loss={train_loss:.4f} train_f1={train_metrics['f1']:.4f} "
                f"val_loss={val_loss:.4f} val_f1={val_metrics['f1']:.4f} lr={lr:.5f}"
            )
            print(message)
            logger.info(message)

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = float(val_metrics["f1"])
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_threshold = float(val_threshold)
                patience = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "stage": stage_name,
                        "state_dict": model.state_dict(),
                        "threshold": best_threshold,
                        "val_f1": best_val_f1,
                    },
                    artifact_dir / "best_model.pt",
                )
                print(f"  --> Saved new best checkpoint to {artifact_dir / 'best_model.pt'} (val_f1={best_val_f1:.4f})")
            else:
                patience += 1

            scheduler.step()
            if patience >= 2 and stage_name == "stage_b":
                break

        if stage_name == "stage_a" and unfreeze:
            stage_a_done = True

    # Save LAST checkpoint immediately
    torch.save(
        {
            "epoch": len(history),
            "stage": stage_name,
            "state_dict": model.state_dict(),
            "threshold": best_threshold,
        },
        artifact_dir / "last_model.pt",
    )
    print(f"Saved last model checkpoint to {artifact_dir / 'last_model.pt'}")

    # Restore best model weights
    if (artifact_dir / "best_model.pt").exists():
        best_ckpt = torch.load(artifact_dir / "best_model.pt", map_location=device)
        model.load_state_dict(best_ckpt["state_dict"])
    else:
        model.load_state_dict(best_state)

    # Compute final metrics
    val_true, val_prob = gather_predictions(model, val_loader, device)
    val_metrics = compute_metrics(val_true, val_prob, best_threshold)
    test_true, test_prob = gather_predictions(model, test_loader, device)
    test_metrics = compute_metrics(test_true, test_prob, best_threshold)

    # Save metrics.json immediately
    safe_json_dump({"validation": val_metrics, "test": test_metrics}, artifact_dir / "metrics.json")
    print(f"Saved metrics.json to {artifact_dir / 'metrics.json'}")

    # Save config.json immediately (strictly JSON-safe)
    config = {
        "output_dir": str(artifact_dir),
        "model_name": "MobileNetV2",
        "pretrained": True,
        "image_size": 224,
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.lr),
        "fine_tuning_learning_rate": float(args.fine_tune_lr),
        "weight_decay": float(args.weight_decay),
        "epochs": int(len(history)),
        "stage_a_epochs": int(stage_a_epochs),
        "stage_b_epochs": int(stage_b_epochs),
        "threshold": float(best_threshold),
        "threshold_selection": "validation_only",
        "seed": int(args.seed),
        "class_weighting": "balanced",
        "train_csv": str(dev_dir / "train.csv"),
        "val_csv": str(dev_dir / "val.csv"),
        "test_csv": str(dev_dir / "test.csv"),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "train_label_counts": {str(k): int(v) for k, v in train_df["label"].value_counts().sort_index().items()},
        "val_label_counts": {str(k): int(v) for k, v in val_df["label"].value_counts().sort_index().items()},
        "test_label_counts": {str(k): int(v) for k, v in test_df["label"].value_counts().sort_index().items()},
    }
    safe_json_dump(config, artifact_dir / "config.json")
    print(f"Saved config.json to {artifact_dir / 'config.json'}")

    # Save training_history.csv immediately
    history_df = pd.DataFrame(history)
    history_df.to_csv(artifact_dir / "training_history.csv", index=False)
    print(f"Saved training_history.csv to {artifact_dir / 'training_history.csv'}")

    # Optional visualizations and diagnostics guarded in try/except
    try:
        save_training_plots(history, artifact_dir)
    except Exception as exc:
        logger.warning("Failed to save training plots: %s", exc)

    try:
        plot_confusion_matrix(test_true, (test_prob >= best_threshold).astype(int), artifact_dir / "confusion_matrix.png", title="QR Image Baseline Confusion Matrix")
        plot_roc_curve(test_true, test_prob, artifact_dir / "roc_curve.png", title="QR Image Baseline ROC")
        plot_pr_curve(test_true, test_prob, artifact_dir / "pr_curve.png", title="QR Image Baseline PR")
    except Exception as exc:
        logger.warning("Failed to save evaluation plots: %s", exc)

    try:
        latency = benchmark_latency(model, test_loader, device, num_runs=50)
        safe_json_dump({"average": latency["average_ms"], "p95": latency["p95_ms"], "runs": latency["runs"]}, artifact_dir / "latency.json")
        safe_json_dump({"validation": val_metrics, "test": test_metrics, "latency_ms": latency}, artifact_dir / "metrics.json")
    except Exception as exc:
        logger.warning("Failed to benchmark latency: %s", exc)

    try:
        with open(artifact_dir / "classification_report.txt", "w", encoding="utf-8") as handle:
            handle.write(classification_report(test_true, (test_prob >= best_threshold).astype(int), target_names=["BENIGN", "MALICIOUS"]))
    except Exception as exc:
        logger.warning("Failed to save classification report: %s", exc)

    try:
        save_gradcam_examples(model, test_df.head(4), artifact_dir, device)
    except Exception as exc:
        logger.warning("Failed to save Grad-CAM examples: %s", exc)

    logger.info("Completed QR image-only development training. Artifacts saved under %s", artifact_dir)

    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Locked threshold: {best_threshold:.4f}")
    print(f"Final test F1: {test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
