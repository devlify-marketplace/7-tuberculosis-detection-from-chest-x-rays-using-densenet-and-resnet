"""Screening metrics; positive class 1 is an abnormal/TB-associated reading."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, roc_auc_score)


def _point_metrics(y_true: np.ndarray, pred: np.ndarray, probabilities: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else 0.0
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    try:
        auc = float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None
    except ValueError:
        auc = None
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": auc,
        "mcc": float(matthews_corrcoef(y_true, pred)) if len(np.unique(y_true)) == 2 else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _stratified_bootstrap_ci(y_true: np.ndarray, probabilities: np.ndarray,
                             threshold: float, n_bootstrap: int, seed: int) -> dict:
    """Class-stratified case bootstrap, returning percentile 95% intervals."""
    metrics = ("accuracy", "sensitivity", "specificity", "roc_auc")
    classes = [np.flatnonzero(y_true == label) for label in (0, 1)]
    if not n_bootstrap or any(len(indices) == 0 for indices in classes):
        return {name: None for name in metrics}
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in metrics}
    for _ in range(n_bootstrap):
        indices = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in classes])
        y = y_true[indices]
        p = probabilities[indices]
        measured = _point_metrics(y, (p >= threshold).astype(np.int64), p)
        for name in metrics:
            value = measured[name]
            if value is not None and np.isfinite(value):
                samples[name].append(value)
    return {name: ([float(np.quantile(values, .025)), float(np.quantile(values, .975))]
                   if values else None) for name, values in samples.items()}


def calculate_metrics(y_true, probabilities, threshold: float = 0.5,
                      n_bootstrap: int = 1000, seed: int = 42) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(y_true) != len(probabilities) or len(y_true) == 0:
        raise ValueError("y_true and probabilities must have the same nonzero length")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    measured = _point_metrics(y_true, (probabilities >= threshold).astype(np.int64), probabilities)
    measured.update({
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "confidence_intervals_95": _stratified_bootstrap_ci(
            y_true, probabilities, threshold, n_bootstrap, seed),
        "bootstrap": {"method": "class-stratified percentile", "replicates": int(n_bootstrap), "seed": int(seed)},
    })
    return measured
