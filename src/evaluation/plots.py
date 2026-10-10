"""
QRShield — Evaluation Visualization

Generates publication-quality plots:
  - Confusion matrix
  - ROC curve
  - Precision-Recall curve
  - Feature importance
  - Calibration curve
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_curve,
    precision_recall_curve,
    auc,
)
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["BENIGN", "MALICIOUS"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", output_path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: Path,
    title: str = "ROC Curve",
) -> None:
    """Plot and save ROC curve with AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2563eb", linewidth=2,
            label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1,
            label="Random baseline")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curve saved to %s", output_path)


def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: Path,
    title: str = "Precision-Recall Curve",
) -> None:
    """Plot and save Precision-Recall curve with AUC."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#dc2626", linewidth=2,
            label=f"PR (AUC = {pr_auc:.4f})")
    # Baseline = positive class prevalence
    baseline = y_true.mean()
    ax.axhline(y=baseline, color="#94a3b8", linestyle="--", linewidth=1,
               label=f"Baseline (prevalence = {baseline:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("PR curve saved to %s", output_path)


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    output_path: Path,
    top_n: int = 20,
    title: str = "Feature Importance (Random Forest)",
) -> None:
    """Plot and save horizontal bar chart of feature importances."""
    # Sort and take top N
    indices = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in indices]
    values = importances[indices]

    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.35)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color="#2563eb", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance plot saved to %s", output_path)


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: Path,
    n_bins: int = 10,
    title: str = "Calibration Curve",
) -> None:
    """Plot and save a reliability (calibration) diagram."""
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2, color="#2563eb",
            label="Model calibration")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8",
            label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Calibration curve saved to %s", output_path)
