"""Validation metrics and threshold selection for binary defect scores."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_curve(labels: Sequence[int], scores: Sequence[float]) -> list[dict[str, float | int]]:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    if labels_array.ndim != 1 or scores_array.ndim != 1 or labels_array.shape != scores_array.shape:
        raise ValueError("Labels and scores must be one-dimensional arrays with matching shapes")
    if set(np.unique(labels_array)) != {0, 1}:
        raise ValueError("Threshold analysis requires both binary classes")
    if not np.isfinite(scores_array).all():
        raise ValueError("Scores must all be finite")

    positive_count = int(labels_array.sum())
    negative_count = int(len(labels_array) - positive_count)
    thresholds = [math.inf, *sorted(np.unique(scores_array).tolist(), reverse=True)]
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        predicted = scores_array >= threshold
        true_positive = int(np.sum(predicted & (labels_array == 1)))
        false_positive = int(np.sum(predicted & (labels_array == 0)))
        false_negative = positive_count - true_positive
        true_negative = negative_count - false_positive
        recall = true_positive / positive_count
        fpr = false_positive / negative_count
        precision = true_positive / (true_positive + false_positive) if predicted.any() else math.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if predicted.any() and precision + recall > 0
            else math.nan
        )
        rows.append(
            {
                "threshold": float(threshold),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
                "recall": recall,
                "false_positive_rate": fpr,
                "precision": precision,
                "f1": f1,
                "accuracy": (true_positive + true_negative) / len(labels_array),
            }
        )
    return rows


def select_fpr_constrained_threshold(
    curve: Sequence[dict[str, float | int]], maximum_fpr: float = 0.10
) -> dict[str, float | int]:
    eligible = [row for row in curve if float(row["false_positive_rate"]) <= maximum_fpr]
    if not eligible:
        raise ValueError(f"No threshold has FPR <= {maximum_fpr}")
    return min(
        eligible,
        key=lambda row: (
            -float(row["recall"]),
            float(row["false_positive_rate"]),
            -float(row["threshold"]),
        ),
    )


def operating_point_at_recall(
    curve: Sequence[dict[str, float | int]], target_recall: float
) -> dict[str, float | int]:
    eligible = [row for row in curve if float(row["recall"]) >= target_recall]
    if not eligible:
        raise ValueError(f"No threshold reaches recall {target_recall}")
    return min(
        eligible,
        key=lambda row: (
            float(row["false_positive_rate"]),
            -float(row["threshold"]),
        ),
    )


def ranking_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    return {
        "auroc": float(roc_auc_score(labels_array, scores_array)),
        "average_precision": float(average_precision_score(labels_array, scores_array)),
    }
