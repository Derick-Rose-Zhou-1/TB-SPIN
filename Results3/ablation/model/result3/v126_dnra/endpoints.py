"""Training-only endpoint heads and participant-level metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler


def apply_endpoint_gene_weighting(
    expression: np.ndarray,
    development_mean: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Recenter around a development-only mean, then apply endpoint weights."""
    x = np.asarray(expression, dtype=np.float32)
    mean = np.asarray(development_mean, dtype=np.float32)
    feature_weights = np.asarray(weights, dtype=np.float32)
    if x.ndim != 2 or mean.shape != (x.shape[1],) or feature_weights.shape != (x.shape[1],):
        raise ValueError("ENDPOINT_GENE_WEIGHT_INPUT_SHAPE_INVALID")
    return mean + (x - mean) * feature_weights


def binary_metrics(y: np.ndarray, score: np.ndarray, sample_weight: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    weight = np.asarray(sample_weight, dtype=np.float64)
    if len(y) != len(score) or len(y) != len(weight):
        raise ValueError("BINARY_METRIC_INPUT_SHAPE_INVALID")
    if len(np.unique(y)) < 2:
        return {"AUROC": np.nan, "AUPRC": np.nan, "Brier": np.nan, "status": "STRUCTURAL_NE_SINGLE_CLASS"}
    predicted = score >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, predicted, sample_weight=weight, labels=[0, 1]).ravel()
    return {
        "AUROC": float(roc_auc_score(y, score, sample_weight=weight)),
        "AUPRC": float(average_precision_score(y, score, sample_weight=weight)),
        "Brier": float(brier_score_loss(y, score, sample_weight=weight)),
        "sensitivity": float(tp / max(tp + fn, 1e-12)),
        "specificity": float(tn / max(tn + fp, 1e-12)),
        "PPV": float(tp / max(tp + fp, 1e-12)),
        "NPV": float(tn / max(tn + fn, 1e-12)),
        "status": "ESTIMABLE",
    }


def fit_binary_head(train_x: np.ndarray, train_y: np.ndarray, heldout_x: np.ndarray, seed: int) -> dict:
    if len(train_x) != len(train_y) or train_x.ndim != 2 or heldout_x.ndim != 2 or train_x.shape[1] != heldout_x.shape[1]:
        raise ValueError("BINARY_HEAD_INPUT_SHAPE_INVALID")
    if len(np.unique(train_y)) != 2:
        raise ValueError("BINARY_HEAD_REQUIRES_TWO_TRAINING_CLASSES")
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=seed, class_weight="balanced")
    model.fit(scaler.transform(train_x), train_y)
    return {
        "scores": model.predict_proba(scaler.transform(heldout_x))[:, 1].astype(np.float64),
        "scaler_mean": scaler.mean_.astype(np.float64),
        "scaler_scale": scaler.scale_.astype(np.float64),
        "coef": model.coef_.reshape(-1).astype(np.float64),
        "intercept": float(model.intercept_[0]),
    }


def fit_cox_head(
    train_x: np.ndarray,
    train_duration: np.ndarray,
    train_event: np.ndarray,
    heldout_x: np.ndarray,
    penalizer: float = 0.1,
) -> dict:
    """Fit a development-only regularized Cox head and score held-out rows."""
    if (
        train_x.ndim != 2
        or heldout_x.ndim != 2
        or train_x.shape[1] != heldout_x.shape[1]
        or len(train_x) != len(train_duration)
        or len(train_x) != len(train_event)
    ):
        raise ValueError("COX_HEAD_INPUT_SHAPE_INVALID")
    duration = np.asarray(train_duration, dtype=np.float64)
    event = np.asarray(train_event, dtype=np.int64)
    if not np.isfinite(duration).all() or (duration <= 0).any() or not np.isin(event, [0, 1]).all():
        raise ValueError("COX_HEAD_OUTCOME_INVALID")
    if int(event.sum()) < 2:
        raise ValueError("COX_HEAD_REQUIRES_AT_LEAST_TWO_EVENTS")

    scaler = StandardScaler().fit(train_x)
    columns = [f"z_{i:02d}" for i in range(train_x.shape[1])]
    train_frame = pd.DataFrame(scaler.transform(train_x), columns=columns)
    train_frame["duration"] = duration
    train_frame["event"] = event
    model = CoxPHFitter(penalizer=penalizer)
    model.fit(train_frame, duration_col="duration", event_col="event")

    heldout_frame = pd.DataFrame(scaler.transform(heldout_x), columns=columns)
    scores = model.predict_log_partial_hazard(heldout_frame).to_numpy(dtype=np.float64)
    return {
        "scores": scores,
        "scaler_mean": scaler.mean_.astype(np.float64),
        "scaler_scale": scaler.scale_.astype(np.float64),
        "coef": model.params_.loc[columns].to_numpy(dtype=np.float64),
        "intercept": 0.0,
        "penalizer": float(penalizer),
    }
