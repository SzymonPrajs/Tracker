"""Centroid-specific localization and confidence-ranked detection metrics."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .matching import confidence_greedy_match, localization_match


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    rank = max(1, int(np.ceil(percentile * len(ordered))))
    return float(ordered[rank - 1])


def _error_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": float(np.mean(values)) if values else None,
        "p50": _nearest_rank(values, 0.50),
        "p90": _nearest_rank(values, 0.90),
        "p95": _nearest_rank(values, 0.95),
        "p99": _nearest_rank(values, 0.99),
        "max": max(values) if values else None,
    }


def _average_precision(labels: np.ndarray, scores: np.ndarray, total_truth: int) -> float | None:
    if total_truth == 0:
        return None
    if len(labels) == 0:
        return 0.0
    order = sorted(range(len(labels)), key=lambda index: (-float(scores[index]), index))
    true_positive = np.cumsum(labels[order], dtype=np.float64)
    false_positive = np.cumsum(~labels[order], dtype=np.float64)
    recall = true_positive / total_truth
    precision = true_positive / (true_positive + false_positive)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    changes = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))


def evaluate_centroids(
    predictions_by_image: Sequence[np.ndarray],
    ground_truth_by_image: Sequence[np.ndarray],
    *,
    distance_thresholds: Sequence[float] = (0.05, 0.10, 0.20),
    localization_gate: float = 0.5,
) -> dict[str, Any]:
    """Evaluate multi-face predictions without hiding misses behind conditional error."""
    if len(predictions_by_image) != len(ground_truth_by_image):
        raise ValueError("prediction and ground-truth image counts differ")
    thresholds = tuple(float(value) for value in distance_thresholds)
    if not thresholds or any(value <= 0 or not np.isfinite(value) for value in thresholds):
        raise ValueError("distance thresholds must be positive and finite")

    total_truth = sum(len(np.asarray(values)) for values in ground_truth_by_image)
    pixel_errors: list[float] = []
    normalized_errors: list[float] = []
    localization_matches = 0
    for predictions, truth in zip(predictions_by_image, ground_truth_by_image, strict=True):
        for match in localization_match(predictions, truth, max_normalized_distance=localization_gate):
            localization_matches += 1
            pixel_errors.append(match.distance_px)
            normalized_errors.append(match.normalized_distance)

    threshold_results: dict[str, Any] = {}
    for threshold in thresholds:
        records: list[tuple[float, int, int, bool]] = []
        for image_index, (predictions, truth) in enumerate(
            zip(predictions_by_image, ground_truth_by_image, strict=True)
        ):
            pred = np.asarray(predictions, dtype=np.float64)
            labels, _ = confidence_greedy_match(
                pred, truth, max_normalized_distance=threshold
            )
            for prediction_index in range(len(pred)):
                records.append(
                    (float(pred[prediction_index, 2]), image_index, prediction_index, bool(labels[prediction_index]))
                )
        records.sort(key=lambda item: (-item[0], item[1], item[2]))
        labels = np.asarray([item[3] for item in records], dtype=bool)
        scores = np.asarray([item[0] for item in records], dtype=np.float64)
        true_positive = int(np.sum(labels))
        false_positive = len(labels) - true_positive
        false_negative = total_truth - true_positive
        precision = true_positive / len(labels) if len(labels) else None
        recall = true_positive / total_truth if total_truth else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else None
        )
        threshold_results[f"{threshold:.3f}"] = {
            "ap": _average_precision(labels, scores, total_truth),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "images": len(predictions_by_image),
        "ground_truth": total_truth,
        "predictions": sum(len(np.asarray(values)) for values in predictions_by_image),
        "localization_matches": localization_matches,
        "pixel_error": _error_summary(pixel_errors),
        "normalized_error": _error_summary(normalized_errors),
        "thresholds": threshold_results,
    }
