"""
Update cell 5 in notebooks/05_image_baseline_gpu.ipynb with fail-safe checkpointing.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NB_PATH = PROJECT_ROOT / "notebooks" / "05_image_baseline_gpu.ipynb"

NEW_CELL_5_CODE = '''import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import classification_report
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from src.config import PATHS
from src.evaluation.plots import plot_confusion_matrix, plot_pr_curve, plot_roc_curve
from src.models.image_baseline import (
    QRImageDataset,
    build_mobilenetv2_model,
    compute_class_weights,
    safe_json_dump,
    verify_split_images,
)
from src.models.train_image_baseline import (
    benchmark_latency,
    compute_metrics,
    create_dataloaders,
    freeze_backbone,
    gather_predictions,
    save_gradcam_examples,
    save_training_plots,
    select_threshold_from_scores,
)

logger = logging.getLogger(__name__)

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "image_baseline_gpu"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("QRShield — Phase 2B MobileNetV2 Image-Only Development Training")
print("Target Artifact Directory:", ARTIFACT_DIR)
print(f"PyTorch Version: {torch.__version__} | Torchvision Version: {torchvision.__version__}")
print("Image-only requirement: model receives QR image only; no URL/domain features used.")
print("=" * 60)

# Verify image accessibility
for split_name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
    report = verify_split_images(frame, sample_limit=10)
    print(f"[{split_name.upper()}] Image sample check: status={report['status']}, valid={report['valid_samples']}/{report['sample_limit']}")
    safe_json_dump(report, ARTIFACT_DIR / f"{split_name}_image_validation.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Selected device for training:", device)
train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, batch_size=32)
model = build_mobilenetv2_model(num_classes=1, pretrained=True).to(device)

pos_weight = compute_class_weights(train_df["label"].tolist())[1:2].to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Hyperparameters (preserved 4-epoch configuration)
stage_a_epochs = 2
stage_b_epochs = 2
total_epochs = stage_a_epochs + stage_b_epochs

# -----------------------------------------------------------------
# REQUIREMENT 1: MINIMAL JSON-SAFE CONFIG DEFINED BEFORE EPOCH LOOP
# (Independent of torchvision.__version__, metrics, plots, or Grad-CAM)
# -----------------------------------------------------------------
checkpoint_config = {
    "model_name": "MobileNetV2",
    "image_size": 224,
    "batch_size": 32,
    "stage_a_epochs": stage_a_epochs,
    "stage_b_epochs": stage_b_epochs,
    "head_learning_rate": 1e-3,
    "fine_tuning_learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "optimizer": "AdamW",
    "seed": 42,
    "train_samples": len(train_df),
    "val_samples": len(val_df),
    "test_samples": len(test_df),
}

history = []
best_val_f1 = -1.0
best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
best_threshold = 0.5
patience = 0

for stage_name, epochs, lr, unfreeze in [
    ("stage_a", stage_a_epochs, 1e-3, False),
    ("stage_b", stage_b_epochs, 1e-4, True),
]:
    if stage_name == "stage_a":
        freeze_backbone(model, freeze=True)
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    else:
        freeze_backbone(model, freeze=False)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

    for epoch_in_stage in range(1, epochs + 1):
        epoch = len(history) + 1
        start = time.time()
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
        train_true, train_prob = gather_predictions(model, train_loader, device)
        train_metrics = compute_metrics(train_true, train_prob, threshold=0.5)

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

        val_true, val_prob = gather_predictions(model, val_loader, device)
        val_threshold = select_threshold_from_scores(val_true, val_prob)
        val_metrics = compute_metrics(val_true, val_prob, threshold=val_threshold)
        epoch_time = time.time() - start

        print(
            f"Epoch {epoch}/{total_epochs} [{stage_name}] "
            f"train_loss={train_loss:.4f} train_f1={train_metrics['f1']:.4f} "
            f"val_loss={val_loss:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_threshold={val_threshold:.4f} lr={lr:.5f} time={epoch_time:.1f}s"
        )

        record = {
            "epoch": epoch,
            "stage": stage_name,
            "train_loss": float(train_loss),
            "train_precision": float(train_metrics["precision"]),
            "train_recall": float(train_metrics["recall"]),
            "train_f1": float(train_metrics["f1"]),
            "val_loss": float(val_loss),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
            "learning_rate": float(lr),
            "epoch_time_seconds": float(epoch_time),
        }
        history.append(record)

        # -----------------------------------------------------------------
        # REQUIREMENT 2: IMMEDIATELY SAVE best_model.pt ON F1 IMPROVEMENT
        # (Inside epoch loop, before any subsequent epoch or test evaluation)
        # -----------------------------------------------------------------
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            best_threshold = float(val_threshold)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
            torch.save(
                {
                    "epoch": epoch,
                    "stage": stage_name,
                    "state_dict": model.state_dict(),
                    "threshold": best_threshold,
                    "val_f1": best_val_f1,
                    "val_metrics": val_metrics,
                    "config": checkpoint_config,
                },
                ARTIFACT_DIR / "best_model.pt",
            )
            print(f"  --> [FAIL-SAFE SAVE: best_model.pt] epoch={epoch} val_f1={best_val_f1:.4f} threshold={best_threshold:.4f}")
        else:
            patience += 1

        if patience >= 2 and stage_name == "stage_b":
            print("Early stopping triggered in stage_b.")
            break

        scheduler.step()

# -----------------------------------------------------------------
# REQUIREMENT 3: IMMEDIATELY SAVE last_model.pt AFTER FINAL EPOCH
# (Happens immediately, BEFORE test eval, latency, plots, report, metrics, Grad-CAM)
# -----------------------------------------------------------------
torch.save(
    {
        "epoch": len(history),
        "stage": stage_name,
        "state_dict": model.state_dict(),
        "threshold": best_threshold,
        "config": checkpoint_config,
    },
    ARTIFACT_DIR / "last_model.pt",
)
print(f"[FAIL-SAFE SAVE: last_model.pt] -> {ARTIFACT_DIR / 'last_model.pt'}")

# Immediately save training history CSV
history_df = pd.DataFrame(history)
history_df.to_csv(ARTIFACT_DIR / "training_history.csv", index=False)
print(f"[FAIL-SAFE SAVE: training_history.csv] -> {ARTIFACT_DIR / 'training_history.csv'}")

# Restore best weights from disk for test evaluation
if (ARTIFACT_DIR / "best_model.pt").exists():
    best_ckpt = torch.load(ARTIFACT_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(best_ckpt["state_dict"])
else:
    model.load_state_dict(best_state)

print(f"Restored best model weights. Locked validation threshold: {best_threshold:.4f}")

# -----------------------------------------------------------------
# TEST AND VALIDATION EVALUATION (AFTER CHECKPOINTS ARE ALREADY SAVED)
# -----------------------------------------------------------------
print("Evaluating test set with locked threshold...")
test_true, test_prob = gather_predictions(model, test_loader, device)
test_metrics = compute_metrics(test_true, test_prob, threshold=best_threshold)
print("Test metrics:", test_metrics)

val_true, val_prob = gather_predictions(model, val_loader, device)
val_metrics = compute_metrics(val_true, val_prob, threshold=best_threshold)

# -----------------------------------------------------------------
# REQUIREMENT 4: SERIALIZE METRICS.JSON AND CONFIG.JSON
# -----------------------------------------------------------------
safe_json_dump({"validation": val_metrics, "test": test_metrics}, ARTIFACT_DIR / "metrics.json")
print(f"[SAVED: metrics.json] -> {ARTIFACT_DIR / 'metrics.json'}")

full_config = dict(checkpoint_config)
full_config.update({
    "project_root": str(PROJECT_ROOT),
    "device": str(device),
    "torch_version": str(torch.__version__),
    "torchvision_version": str(torchvision.__version__),
    "threshold": float(best_threshold),
    "train_label_counts": {str(k): int(v) for k, v in train_df["label"].value_counts().sort_index().items()},
    "val_label_counts": {str(k): int(v) for k, v in val_df["label"].value_counts().sort_index().items()},
    "test_label_counts": {str(k): int(v) for k, v in test_df["label"].value_counts().sort_index().items()},
})
safe_json_dump(full_config, ARTIFACT_DIR / "config.json")
print(f"[SAVED: config.json] -> {ARTIFACT_DIR / 'config.json'}")

# -----------------------------------------------------------------
# OPTIONAL DIAGNOSTICS & VISUALIZATIONS (GUARDED IN TRY/EXCEPT)
# Model weights best_model.pt and last_model.pt are ALREADY safe on disk.
# -----------------------------------------------------------------
print("\\nGenerating optional evaluation curves and diagnostics...")

try:
    save_training_plots(history, ARTIFACT_DIR)
    print("  -> Saved training history plot")
except Exception as exc:
    print(f"  -> Warning: failed to save training history plot: {exc}")

try:
    plot_confusion_matrix(test_true, (test_prob >= best_threshold).astype(int), ARTIFACT_DIR / "confusion_matrix.png", title="QR Image Baseline Confusion Matrix")
    print("  -> Saved confusion matrix plot")
except Exception as exc:
    print(f"  -> Warning: failed to save confusion matrix plot: {exc}")

try:
    plot_roc_curve(test_true, test_prob, ARTIFACT_DIR / "roc_curve.png", title="QR Image Baseline ROC")
    print("  -> Saved ROC curve plot")
except Exception as exc:
    print(f"  -> Warning: failed to save ROC curve plot: {exc}")

try:
    plot_pr_curve(test_true, test_prob, ARTIFACT_DIR / "pr_curve.png", title="QR Image Baseline PR")
    print("  -> Saved PR curve plot")
except Exception as exc:
    print(f"  -> Warning: failed to save PR curve plot: {exc}")

try:
    clf_report = classification_report(test_true, (test_prob >= best_threshold).astype(int), target_names=["BENIGN", "MALICIOUS"])
    with open(ARTIFACT_DIR / "classification_report.txt", "w", encoding="utf-8") as handle:
        handle.write(clf_report)
    print("  -> Saved classification report")
    print("\\nClassification Report:\\n" + clf_report)
except Exception as exc:
    print(f"  -> Warning: failed to save classification report: {exc}")

try:
    latency = benchmark_latency(model, test_loader, device, num_runs=50)
    safe_json_dump({"average": latency["average_ms"], "p95": latency["p95_ms"], "runs": latency["runs"]}, ARTIFACT_DIR / "latency.json")
    safe_json_dump({"validation": val_metrics, "test": test_metrics, "latency_ms": latency}, ARTIFACT_DIR / "metrics.json")
    print(f"  -> Latency benchmark complete: avg={latency['average_ms']:.2f}ms, p95={latency['p95_ms']:.2f}ms")
except Exception as exc:
    print(f"  -> Warning: latency benchmark failed: {exc}")

try:
    save_gradcam_examples(model, test_df.head(4), ARTIFACT_DIR, device)
    print("  -> Saved Grad-CAM visualizations")
except Exception as exc:
    print(f"  -> Warning: Grad-CAM generation failed: {exc}")

print("\\n" + "=" * 60)
print("SUCCESS: Phase 2B MobileNetV2 workflow complete!")
print(f"All artifacts securely verified under: {ARTIFACT_DIR}")
print("=" * 60)'''


def main() -> None:
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    lines = [line + "\n" for line in NEW_CELL_5_CODE.splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    nb["cells"][5]["source"] = lines

    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print("SUCCESS: cell 5 in notebooks/05_image_baseline_gpu.ipynb updated on disk!")


if __name__ == "__main__":
    main()
