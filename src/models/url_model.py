"""
QRShield — URL-Only Random Forest Baseline

Trains, evaluates, and persists a Random Forest classifier
on URL lexical/structural features.

Threshold selection is done on VALIDATION data only.
The test threshold is LOCKED before test evaluation.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
)

from src.config import URL_MODEL_CFG, EVAL_CFG, SEEDS, PATHS
from src.features.url_features import extract_url_features, FEATURE_NAMES

logger = logging.getLogger(__name__)


def _features_to_array(df: pd.DataFrame) -> np.ndarray:
    """Extract URL features from DataFrame and return as numpy array.

    Args:
        df: DataFrame with 'url' column.

    Returns:
        2D numpy array of shape (n_samples, n_features).
    """
    feature_dicts = [extract_url_features(url) for url in df["url"]]
    feature_df = pd.DataFrame(feature_dicts, columns=FEATURE_NAMES)
    return feature_df.values.astype(np.float64)


def train_url_model(
    df_train: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[RandomForestClassifier, np.ndarray]:
    """Train a Random Forest on URL features.

    Args:
        df_train: Training DataFrame with 'url' and 'label' columns.
        config: Optional override for model hyperparameters.

    Returns:
        Tuple of (trained model, training feature array).
    """
    logger.info("Extracting features for %d training samples...", len(df_train))
    X_train = _features_to_array(df_train)
    y_train = df_train["label"].values

    cfg = config or {}
    model = RandomForestClassifier(
        n_estimators=cfg.get("n_estimators", URL_MODEL_CFG.n_estimators),
        max_depth=cfg.get("max_depth", URL_MODEL_CFG.max_depth),
        min_samples_split=cfg.get("min_samples_split", URL_MODEL_CFG.min_samples_split),
        min_samples_leaf=cfg.get("min_samples_leaf", URL_MODEL_CFG.min_samples_leaf),
        class_weight=cfg.get("class_weight", URL_MODEL_CFG.class_weight),
        n_jobs=cfg.get("n_jobs", URL_MODEL_CFG.n_jobs),
        random_state=cfg.get("random_state", URL_MODEL_CFG.random_state),
    )

    logger.info("Training Random Forest (%d trees)...", model.n_estimators)
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    logger.info("Training completed in %.2f seconds", train_time)

    return model, X_train


def select_threshold(
    model: RandomForestClassifier,
    df_val: pd.DataFrame,
    metric: str = EVAL_CFG.threshold_metric,
) -> Tuple[float, Dict[str, Any]]:
    """Select optimal classification threshold on VALIDATION data.

    Args:
        model: Trained Random Forest.
        df_val: Validation DataFrame.
        metric: Metric to optimize ('f1' or 'precision' or 'recall').

    Returns:
        Tuple of (selected_threshold, validation_metrics_dict).
    """
    logger.info("Selecting threshold on %d validation samples...", len(df_val))
    X_val = _features_to_array(df_val)
    y_val = df_val["label"].values
    probs = model.predict_proba(X_val)[:, 1]

    # Search thresholds
    precisions, recalls, thresholds_pr = precision_recall_curve(y_val, probs)
    f1_scores = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls),
        0.0,
    )

    if metric == "f1":
        best_idx = np.argmax(f1_scores)
        best_threshold = float(thresholds_pr[best_idx]) if best_idx < len(thresholds_pr) else 0.5
    else:
        best_threshold = EVAL_CFG.default_threshold

    logger.info("Selected threshold: %.4f (optimizing %s)", best_threshold, metric)

    # Evaluate at selected threshold
    val_metrics = evaluate_with_threshold(y_val, probs, best_threshold)
    val_metrics["threshold"] = best_threshold
    val_metrics["metric_optimized"] = metric

    return best_threshold, val_metrics


def evaluate_with_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Evaluate predictions at a specific threshold.

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities for class 1.
        threshold: Classification threshold.

    Returns:
        Dictionary of metrics.
    """
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr_val = fp / max(fp + tn, 1)

    metrics: Dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 6),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 6),
        "false_positive_rate": round(float(fpr_val), 6),
        "brier_score": round(float(brier_score_loss(y_true, y_proba)), 6),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "classification_report": classification_report(
            y_true, y_pred, target_names=["BENIGN", "MALICIOUS"], output_dict=True,
        ),
        "threshold": threshold,
    }

    return metrics


def measure_latency(
    model: RandomForestClassifier,
    sample_url: str = "https://www.example.com/test?param=value",
    n_runs: int = EVAL_CFG.latency_runs,
) -> Dict[str, float]:
    """Measure real inference latency.

    Args:
        model: Trained model.
        sample_url: A URL to use for latency measurement.
        n_runs: Number of measurement iterations.

    Returns:
        Dictionary with mean and p95 latency in milliseconds.
    """
    latencies_feature: List[float] = []
    latencies_inference: List[float] = []
    latencies_total: List[float] = []

    for _ in range(n_runs):
        t0 = time.perf_counter()
        features = extract_url_features(sample_url)
        t1 = time.perf_counter()

        feature_array = np.array([[features[f] for f in FEATURE_NAMES]])
        _ = model.predict_proba(feature_array)
        t2 = time.perf_counter()

        latencies_feature.append((t1 - t0) * 1000)
        latencies_inference.append((t2 - t1) * 1000)
        latencies_total.append((t2 - t0) * 1000)

    return {
        "feature_extraction_mean_ms": round(float(np.mean(latencies_feature)), 3),
        "model_inference_mean_ms": round(float(np.mean(latencies_inference)), 3),
        "total_mean_ms": round(float(np.mean(latencies_total)), 3),
        "feature_extraction_p95_ms": round(float(np.percentile(latencies_feature, 95)), 3),
        "model_inference_p95_ms": round(float(np.percentile(latencies_inference, 95)), 3),
        "total_p95_ms": round(float(np.percentile(latencies_total, 95)), 3),
        "n_runs": n_runs,
    }


def get_feature_importance(
    model: RandomForestClassifier,
) -> pd.DataFrame:
    """Extract and sort feature importances.

    Returns:
        DataFrame with columns ['feature', 'importance'], sorted descending.
    """
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return fi_df
