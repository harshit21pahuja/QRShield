"""
QRShield — Phase 1 Pipeline: Data Audit + URL Baseline

End-to-end script that:
  1. Loads dataset (real or synthetic smoke test)
  2. Runs dataset audit
  3. Normalizes labels
  4. Extracts registrable domains
  5. Performs leakage-safe domain-grouped split
  6. Verifies zero domain overlap
  7. Extracts URL features
  8. Trains Random Forest baseline
  9. Selects threshold on validation data
  10. Evaluates on test data with locked threshold
  11. Generates all metrics, plots, and artifacts

Usage:
    python -m scripts.run_phase1                          # smoke test (no real data)
    python -m scripts.run_phase1 --data data/raw/dataset.csv   # real dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, SEEDS, DATA_CFG, URL_MODEL_CFG, EVAL_CFG
from src.data.loader import (
    load_csv, inspect_labels, normalize_labels,
    add_domain_column, clean_dataset,
)
from src.data.audit import audit_dataset
from src.data.splitter import domain_grouped_split
from src.data.synthetic import generate_smoke_dataset
from src.features.url_features import FEATURE_NAMES, extract_url_features
from src.models.url_model import (
    train_url_model, select_threshold, evaluate_with_threshold,
    measure_latency, get_feature_importance, _features_to_array,
)
from src.evaluation.plots import (
    plot_confusion_matrix, plot_roc_curve, plot_pr_curve,
    plot_feature_importance, plot_calibration_curve,
)


def setup_logging() -> None:
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_phase1(data_path: str | None = None, max_samples: int = 0) -> None:
    """Execute the complete Phase 1 pipeline."""

    setup_logging()
    logger = logging.getLogger("QRShield.Phase1")

    logger.info("=" * 60)
    logger.info("QRShield — Phase 1: Data Audit + URL Baseline")
    logger.info("=" * 60)

    # Ensure directories exist
    PATHS.ensure_dirs()

    # ── Step 1: Load dataset ──────────────────────────────────
    if data_path and Path(data_path).exists():
        logger.info("Loading REAL dataset from: %s", data_path)
        df = load_csv(Path(data_path))
    else:
        if data_path:
            logger.warning("Dataset file not found: %s", data_path)
        logger.info("Using SYNTHETIC smoke-test dataset")
        df = generate_smoke_dataset(seed=SEEDS.global_seed)

    logger.info("Initial dataset shape: %s", df.shape)

    # ── Step 2: Inspect labels ────────────────────────────────
    label_info = inspect_labels(df)
    logger.info("Raw label distribution: %s", label_info)

    # ── Step 3: Normalize labels ──────────────────────────────
    df = normalize_labels(df)
    logger.info("Label distribution after normalization: %s", inspect_labels(df))

    # ── Step 4: Clean dataset ─────────────────────────────────
    df = clean_dataset(df, min_url_length=DATA_CFG.min_url_length)

    # ── Step 5: Apply max_samples if configured ───────────────
    if max_samples > 0 and len(df) > max_samples:
        logger.info("Subsampling to %d samples (from %d)", max_samples, len(df))
        df = df.sample(n=max_samples, random_state=SEEDS.global_seed).reset_index(drop=True)

    # ── Step 6: Extract domains ───────────────────────────────
    df = add_domain_column(df)

    # ── Step 7: Run dataset audit ─────────────────────────────
    logger.info("Running dataset audit...")
    audit_report = audit_dataset(df, output_dir=PATHS.data_audit)
    logger.info("Audit complete. Report saved to %s", PATHS.data_audit)

    # ── Step 8: Domain-grouped split ──────────────────────────
    logger.info("Performing domain-grouped stratified split...")
    df_train, df_val, df_test = domain_grouped_split(
        df, seed=SEEDS.split_seed, output_dir=PATHS.data_audit,
    )

    logger.info("Split sizes — Train: %d, Val: %d, Test: %d",
                len(df_train), len(df_val), len(df_test))

    # Save split CSVs
    df_train.to_csv(PATHS.data_processed / "train.csv", index=False)
    df_val.to_csv(PATHS.data_processed / "val.csv", index=False)
    df_test.to_csv(PATHS.data_processed / "test.csv", index=False)
    logger.info("Split CSVs saved to %s", PATHS.data_processed)

    # ── Step 9: Train URL model ───────────────────────────────
    logger.info("Training URL-only Random Forest baseline...")
    model, X_train = train_url_model(df_train)

    # ── Step 10: Select threshold on validation ───────────────
    logger.info("Selecting threshold on validation set...")
    threshold, val_metrics = select_threshold(model, df_val)
    logger.info("Validation metrics at threshold %.4f:", threshold)
    for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]:
        logger.info("  %s: %.4f", k, val_metrics[k])

    # ── Step 11: Evaluate on test with LOCKED threshold ──────
    logger.info("Evaluating on test set with LOCKED threshold %.4f...", threshold)
    X_test = _features_to_array(df_test)
    y_test = df_test["label"].values
    test_probs = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_with_threshold(y_test, test_probs, threshold)

    logger.info("TEST RESULTS (threshold=%.4f):", threshold)
    for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc",
              "pr_auc", "false_positive_rate", "brier_score"]:
        logger.info("  %s: %.4f", k, test_metrics[k])

    # Also evaluate at default threshold for comparison
    default_metrics = evaluate_with_threshold(y_test, test_probs, EVAL_CFG.default_threshold)

    # ── Step 12: Measure latency ──────────────────────────────
    logger.info("Measuring inference latency...")
    latency = measure_latency(model)
    logger.info("Latency — total mean: %.3f ms, p95: %.3f ms",
                latency["total_mean_ms"], latency["total_p95_ms"])

    # ── Step 13: Feature importance ───────────────────────────
    fi_df = get_feature_importance(model)
    fi_df.to_csv(PATHS.url_baseline / "feature_importance.csv", index=False)
    logger.info("Feature importance saved")

    # ── Step 14: Generate plots ───────────────────────────────
    y_test_pred = (test_probs >= threshold).astype(int)

    plot_confusion_matrix(
        y_test, y_test_pred,
        PATHS.url_baseline / "confusion_matrix.png",
        title="URL Baseline — Confusion Matrix (Test)",
    )
    plot_roc_curve(
        y_test, test_probs,
        PATHS.url_baseline / "roc_curve.png",
        title="URL Baseline — ROC Curve (Test)",
    )
    plot_pr_curve(
        y_test, test_probs,
        PATHS.url_baseline / "pr_curve.png",
        title="URL Baseline — Precision-Recall Curve (Test)",
    )
    plot_feature_importance(
        FEATURE_NAMES, model.feature_importances_,
        PATHS.url_baseline / "feature_importance.png",
        title="URL Baseline — Feature Importance",
    )
    plot_calibration_curve(
        y_test, test_probs,
        PATHS.url_baseline / "calibration_curve.png",
        title="URL Baseline — Calibration Curve (Test)",
    )

    # ── Step 15: Save model ───────────────────────────────────
    model_path = PATHS.models / "url_rf_baseline.joblib"
    PATHS.models.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Model saved to %s", model_path)

    # ── Step 16: Save experiment config ───────────────────────
    experiment = {
        "experiment": "url_baseline_phase1",
        "model": "RandomForest",
        "config": {
            "n_estimators": URL_MODEL_CFG.n_estimators,
            "max_depth": URL_MODEL_CFG.max_depth,
            "min_samples_split": URL_MODEL_CFG.min_samples_split,
            "min_samples_leaf": URL_MODEL_CFG.min_samples_leaf,
            "class_weight": URL_MODEL_CFG.class_weight,
            "random_state": URL_MODEL_CFG.random_state,
        },
        "seeds": {
            "global": SEEDS.global_seed,
            "split": SEEDS.split_seed,
            "model": SEEDS.model_seed,
        },
        "features": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "dataset_source": data_path or "synthetic_smoke_test",
        "train_size": len(df_train),
        "val_size": len(df_val),
        "test_size": len(df_test),
        "selected_threshold": threshold,
        "default_threshold": EVAL_CFG.default_threshold,
        "validation_metrics": val_metrics,
        "test_metrics_selected_threshold": test_metrics,
        "test_metrics_default_threshold": default_metrics,
        "latency": latency,
    }

    exp_path = PATHS.url_baseline / "experiment.json"
    with open(exp_path, "w", encoding="utf-8") as f:
        json.dump(experiment, f, indent=2, default=str)
    logger.info("Experiment config saved to %s", exp_path)

    # ── Final summary ─────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 1 COMPLETE")
    logger.info("=" * 60)
    logger.info("Dataset: %s", data_path or "SYNTHETIC SMOKE TEST")
    logger.info("Train: %d | Val: %d | Test: %d", len(df_train), len(df_val), len(df_test))
    logger.info("Selected threshold: %.4f", threshold)
    logger.info("Test F1: %.4f | Test AUC-ROC: %.4f | Test PR-AUC: %.4f",
                test_metrics["f1_score"], test_metrics["roc_auc"], test_metrics["pr_auc"])
    logger.info("Test FPR: %.4f | Brier: %.4f",
                test_metrics["false_positive_rate"], test_metrics["brier_score"])
    logger.info("Inference latency: %.3f ms (mean) | %.3f ms (p95)",
                latency["total_mean_ms"], latency["total_p95_ms"])
    logger.info("")
    logger.info("Artifacts saved to: %s", PATHS.url_baseline)
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QRShield Phase 1 Pipeline")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to CSV dataset")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Maximum samples to use (0 = all)")
    args = parser.parse_args()

    run_phase1(data_path=args.data, max_samples=args.max_samples)
