from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)


Threshold = float | dict[str, float] | Sequence[float] | np.ndarray


def extract_logits(prediction_output: Any) -> np.ndarray:
    """Extract a logits matrix from Trainer predictions or a raw value."""
    logits = (
        prediction_output.predictions
        if hasattr(prediction_output, "predictions")
        else prediction_output
    )
    if isinstance(logits, tuple):
        logits = logits[0]
    logits_array = np.asarray(logits)
    if logits_array.ndim != 2:
        raise ValueError(f"Expected a two-dimensional logits array, got {logits_array.shape}.")
    return logits_array


def sigmoid_probabilities(logits: np.ndarray) -> np.ndarray:
    """Convert logits to numerically stable probabilities."""
    return expit(np.asarray(logits, dtype=float))


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Compatibility name for the notebook's sigmoid helper."""
    return sigmoid_probabilities(logits)


def threshold_vector(
    threshold: Threshold,
    labels: Sequence[str] | int,
) -> np.ndarray:
    """Expand scalar, named, or vector thresholds to one value per label."""
    label_names = (
        [str(index) for index in range(labels)]
        if isinstance(labels, int)
        else list(labels)
    )
    label_count = len(label_names)
    if isinstance(threshold, dict):
        fallback = float(threshold.get("global", 0.5))
        values = np.asarray(
            [threshold.get(label, fallback) for label in label_names],
            dtype=float,
        )
    elif np.isscalar(threshold):
        values = np.full(label_count, float(threshold), dtype=float)
    else:
        values = np.asarray(threshold, dtype=float)
        if values.shape != (label_count,):
            raise ValueError(
                f"Expected {label_count} thresholds, got shape {values.shape}."
            )
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Decision thresholds must be finite values between 0 and 1.")
    return values


def threshold_predictions(
    probabilities: np.ndarray,
    threshold: Threshold,
    labels: Sequence[str] | None = None,
) -> np.ndarray:
    """Convert probabilities into binary multi-label predictions."""
    probability_array = np.asarray(probabilities, dtype=float)
    if probability_array.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional probability array, got {probability_array.shape}."
        )
    threshold_labels = labels if labels is not None else probability_array.shape[1]
    return (probability_array >= threshold_vector(threshold, threshold_labels)).astype(int)


def multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: Threshold,
    labels: Sequence[str] | None = None,
) -> dict[str, float]:
    """Compute threshold-dependent aggregate metrics for multi-label outputs."""
    true_values = np.asarray(y_true).astype(int)
    predicted_values = threshold_predictions(y_prob, threshold, labels)
    return {
        "macro_f1": float(f1_score(true_values, predicted_values, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(true_values, predicted_values, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(true_values, predicted_values, average="weighted", zero_division=0)),
        "macro_precision": float(
            precision_score(true_values, predicted_values, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(true_values, predicted_values, average="macro", zero_division=0)
        ),
        "micro_precision": float(
            precision_score(true_values, predicted_values, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(true_values, predicted_values, average="micro", zero_division=0)
        ),
        "subset_accuracy": float(accuracy_score(true_values, predicted_values)),
        "hamming_loss": float(hamming_loss(true_values, predicted_values)),
    }


def threshold_independent_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Compute metrics that do not depend on a decision threshold."""
    true_values = np.asarray(y_true).astype(int)
    probabilities = np.asarray(y_prob, dtype=float)
    average_precision = average_precision_score(
        true_values,
        probabilities,
        average="macro",
    )
    return {
        "macro_average_precision": float(np.nan_to_num(average_precision)),
        "mean_brier_score": float(np.mean((probabilities - true_values) ** 2)),
    }


def _best_candidate(
    scores: Sequence[float],
    candidates: np.ndarray,
) -> tuple[float, float]:
    best_index = int(np.argmax(np.asarray(scores, dtype=float)))
    return float(candidates[best_index]), float(scores[best_index])


def calibrate_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    candidates: Sequence[float],
    labels: Sequence[str],
    min_positives: int,
    shrinkage: float,
) -> dict[str, Any]:
    """Calibrate global and regularized per-label thresholds on validation data."""
    candidate_values = np.sort(np.unique(np.asarray(candidates, dtype=float)))
    if len(candidate_values) == 0 or np.any((candidate_values < 0) | (candidate_values > 1)):
        raise ValueError("Threshold candidates must be non-empty values between 0 and 1.")
    if min_positives < 1:
        raise ValueError("min_positives must be at least 1.")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be between 0 and 1.")

    true_values = np.asarray(y_true).astype(int)
    probabilities = np.asarray(y_prob, dtype=float)
    if true_values.shape != probabilities.shape:
        raise ValueError("y_true and y_prob must have the same shape.")
    if true_values.ndim != 2 or true_values.shape[1] != len(labels):
        raise ValueError("The target matrix width must match labels.")

    sweep_results = []
    global_scores = []
    for candidate in candidate_values:
        metrics = multilabel_metrics(true_values, probabilities, float(candidate), labels)
        global_scores.append(metrics["macro_f1"])
        sweep_results.append({
            "threshold": float(candidate),
            "macro_f1": metrics["macro_f1"],
            "micro_f1": metrics["micro_f1"],
            "subset_accuracy": metrics["subset_accuracy"],
        })
    global_threshold, global_macro_f1 = _best_candidate(global_scores, candidate_values)

    per_label_thresholds: dict[str, float] = {"global": global_threshold}
    calibration_rows = []
    for label_index, label in enumerate(labels):
        label_scores = [
            f1_score(
                true_values[:, label_index],
                probabilities[:, label_index] >= candidate,
                zero_division=0,
            )
            for candidate in candidate_values
        ]
        raw_threshold, raw_f1 = _best_candidate(label_scores, candidate_values)
        positives = int(true_values[:, label_index].sum())
        if positives >= min_positives:
            selected_threshold = (
                shrinkage * global_threshold + (1.0 - shrinkage) * raw_threshold
            )
            source = "regularized label optimum"
        else:
            selected_threshold = global_threshold
            source = "global fallback (low support)"
        selected_threshold = float(round(selected_threshold, 4))
        per_label_thresholds[label] = selected_threshold
        calibration_rows.append({
            "label": label,
            "validation_positives": positives,
            "raw_best_threshold": raw_threshold,
            "raw_best_f1": raw_f1,
            "applied_threshold": selected_threshold,
            "threshold_source": source,
        })

    return {
        "global_threshold": global_threshold,
        "global_macro_f1": global_macro_f1,
        "per_label_thresholds": per_label_thresholds,
        "per_label_calibration": calibration_rows,
        "sweep_results": sweep_results,
        "validation_metrics": multilabel_metrics(
            true_values,
            probabilities,
            per_label_thresholds,
            labels,
        ),
    }


def canonical_airo_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Map held-out evaluation metrics to the four AIRO aggregate names."""
    aliases = {
        "eval_accuracy": "subset_accuracy",
        "eval_precision": "macro_precision",
        "eval_recall": "macro_recall",
        "eval_f1": "macro_f1",
    }
    result = {}
    for canonical_name, source_name in aliases.items():
        if source_name not in metrics:
            raise KeyError(f"Held-out metrics are missing {source_name}.")
        result[canonical_name] = float(metrics[source_name])
    return result