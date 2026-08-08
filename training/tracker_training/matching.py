"""Deterministic one-to-one matching for centroid localization and PR evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CentroidMatch:
    prediction_index: int
    ground_truth_index: int
    distance_px: float
    normalized_distance: float


def _prediction_array(predictions: np.ndarray) -> np.ndarray:
    values = np.asarray(predictions, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 3 or not np.all(np.isfinite(values)):
        raise ValueError("predictions must be a finite Nx3+ array of x, y, score")
    return values


def _ground_truth_array(ground_truth: np.ndarray) -> np.ndarray:
    values = np.asarray(ground_truth, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 4 or not np.all(np.isfinite(values)):
        raise ValueError("ground_truth must be a finite Nx4+ array of x, y, width, height")
    if np.any(values[:, 2:4] <= 0):
        raise ValueError("ground-truth width and height must be positive")
    return values


def normalized_distance_matrix(predictions: np.ndarray, ground_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = _prediction_array(predictions)
    truth = _ground_truth_array(ground_truth)
    if len(pred) == 0 or len(truth) == 0:
        shape = (len(pred), len(truth))
        return np.empty(shape), np.empty(shape)
    deltas = pred[:, None, :2] - truth[None, :, :2]
    pixels = np.linalg.norm(deltas, axis=2)
    diagonals = np.linalg.norm(truth[:, 2:4], axis=1)
    return pixels, pixels / diagonals[None, :]


def _hungarian_square(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost square assignment with deterministic smallest-column ties."""
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.all(np.isfinite(matrix)):
        raise ValueError("cost matrix must be finite and square")
    n = matrix.shape[0]
    if n == 0:
        return []
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=np.int64)
    way = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        j0 = 0
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                current = matrix[i0 - 1, j - 1] - u[i0] - v[j]
                if current < minv[j]:
                    minv[j] = current
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return sorted((int(p[j]) - 1, j - 1) for j in range(1, n + 1))


def localization_match(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    *,
    max_normalized_distance: float = 0.5,
) -> list[CentroidMatch]:
    """Globally minimize normalized centroid distance with explicit unmatched nodes."""
    pred = _prediction_array(predictions)
    truth = _ground_truth_array(ground_truth)
    if max_normalized_distance <= 0 or not np.isfinite(max_normalized_distance):
        raise ValueError("max_normalized_distance must be positive and finite")
    if len(pred) == 0 or len(truth) == 0:
        return []
    pixels, normalized = normalized_distance_matrix(pred, truth)
    count = len(pred) + len(truth)
    forbidden = float(10 * count)
    cost = np.full((count, count), forbidden)
    real = normalized.copy()
    real[real > max_normalized_distance] = forbidden
    cost[: len(pred), : len(truth)] = real
    cost[: len(pred), len(truth) :] = 1.0
    cost[len(pred) :, : len(truth)] = 1.0
    cost[len(pred) :, len(truth) :] = 0.0

    matches = []
    for prediction_index, truth_index in _hungarian_square(cost):
        if prediction_index < len(pred) and truth_index < len(truth):
            distance = float(normalized[prediction_index, truth_index])
            if distance <= max_normalized_distance:
                matches.append(
                    CentroidMatch(
                        prediction_index,
                        truth_index,
                        float(pixels[prediction_index, truth_index]),
                        distance,
                    )
                )
    return matches


def confidence_greedy_match(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    *,
    max_normalized_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Match score-ranked predictions for precision-recall and calibration."""
    pred = _prediction_array(predictions)
    truth = _ground_truth_array(ground_truth)
    if max_normalized_distance <= 0 or not np.isfinite(max_normalized_distance):
        raise ValueError("max_normalized_distance must be positive and finite")
    labels = np.zeros(len(pred), dtype=bool)
    matched_truth = np.full(len(pred), -1, dtype=np.int64)
    if len(pred) == 0 or len(truth) == 0:
        return labels, matched_truth
    _, normalized = normalized_distance_matrix(pred, truth)
    used: set[int] = set()
    order = sorted(range(len(pred)), key=lambda index: (-pred[index, 2], index))
    for prediction_index in order:
        candidates = [
            truth_index
            for truth_index in range(len(truth))
            if truth_index not in used
            and normalized[prediction_index, truth_index] <= max_normalized_distance
        ]
        if not candidates:
            continue
        truth_index = min(candidates, key=lambda index: (normalized[prediction_index, index], index))
        labels[prediction_index] = True
        matched_truth[prediction_index] = truth_index
        used.add(truth_index)
    return labels, matched_truth
