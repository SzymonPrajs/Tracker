"""Deterministic heatmap decoding shared by training evaluation and export checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecodedPeak:
    score: float
    x: float
    y: float
    cell_x: int
    cell_y: int
    flat_index: int
    width: float | None = None
    height: float | None = None
    x_q16: int | None = None
    y_q16: int | None = None


def _as_hwc2(values: np.ndarray | None, height: int, width: int, name: str) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values)
    if array.shape == (height, width, 2):
        result = array
    elif array.shape == (2, height, width):
        result = np.moveaxis(array, 0, -1)
    else:
        raise ValueError(f"{name} must have shape HxWx2 or 2xHxW")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains a non-finite value")
    return result


def local_maxima(
    heatmap: np.ndarray,
    *,
    score_threshold: float = 0.0,
    nms_kernel: int = 3,
) -> list[tuple[int, int]]:
    """Return local maxima with deterministic row-major plateau suppression."""
    scores = np.asarray(heatmap)
    if scores.ndim != 2 or scores.size == 0:
        raise ValueError("heatmap must be a non-empty 2D array")
    if not np.all(np.isfinite(scores)):
        raise ValueError("heatmap contains a non-finite value")
    if nms_kernel <= 0 or nms_kernel % 2 == 0:
        raise ValueError("nms_kernel must be a positive odd integer")
    if not np.isfinite(score_threshold):
        raise ValueError("score_threshold must be finite")

    radius = nms_kernel // 2
    height, width = scores.shape
    peaks: list[tuple[int, int]] = []
    for y in range(height):
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        for x in range(width):
            score = scores[y, x]
            if score < score_threshold:
                continue
            x0, x1 = max(0, x - radius), min(width, x + radius + 1)
            window = scores[y0:y1, x0:x1]
            maximum = np.max(window)
            if score != maximum:
                continue
            first = int(np.flatnonzero(window == maximum)[0])
            first_y, first_x = divmod(first, window.shape[1])
            if y == y0 + first_y and x == x0 + first_x:
                peaks.append((y, x))
    return peaks


def _local_centroid(scores: np.ndarray, y: int, x: int, window: int) -> tuple[float, float]:
    radius = window // 2
    y0, y1 = max(0, y - radius), min(scores.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(scores.shape[1], x + radius + 1)
    local = scores[y0:y1, x0:x1].astype(np.float64)
    weights = local - np.min(local)
    grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
    total = float(np.sum(weights))
    if total == 0.0:
        return float(x), float(y)
    return float(np.sum(weights * grid_x) / total), float(np.sum(weights * grid_y) / total)


def decode_heatmap(
    heatmap: np.ndarray,
    *,
    offsets: np.ndarray | None = None,
    sizes: np.ndarray | None = None,
    output_stride: int = 4,
    top_k: int = 20,
    score_threshold: float = 0.0,
    nms_kernel: int = 3,
    centroid_window: int = 3,
) -> list[DecodedPeak]:
    """Decode score, center-relative offsets, and sizes into input-image pixels."""
    scores = np.asarray(heatmap)
    if scores.ndim != 2 or scores.size == 0:
        raise ValueError("heatmap must be a non-empty 2D array")
    if output_stride <= 0 or top_k < 0:
        raise ValueError("output_stride must be positive and top_k non-negative")
    if centroid_window <= 0 or centroid_window % 2 == 0:
        raise ValueError("centroid_window must be a positive odd integer")
    height, width = scores.shape
    offset_values = _as_hwc2(offsets, height, width, "offsets")
    size_values = _as_hwc2(sizes, height, width, "sizes")

    candidates = local_maxima(scores, score_threshold=score_threshold, nms_kernel=nms_kernel)
    candidates.sort(key=lambda point: (-float(scores[point]), point[0] * width + point[1]))
    decoded: list[DecodedPeak] = []
    for y, x in candidates[:top_k]:
        local_x, local_y = _local_centroid(scores, y, x, centroid_window)
        offset_x = float(offset_values[y, x, 0]) if offset_values is not None else 0.0
        offset_y = float(offset_values[y, x, 1]) if offset_values is not None else 0.0
        center_x = (local_x + 0.5 + offset_x) * output_stride
        center_y = (local_y + 0.5 + offset_y) * output_stride
        decoded.append(
            DecodedPeak(
                score=float(scores[y, x]),
                x=center_x,
                y=center_y,
                cell_x=x,
                cell_y=y,
                flat_index=y * width + x,
                width=float(size_values[y, x, 0]) if size_values is not None else None,
                height=float(size_values[y, x, 1]) if size_values is not None else None,
            )
        )
    return decoded


def decode_hwc16_c_parity(
    head: np.ndarray,
    *,
    output_stride: int = 4,
    offset_fraction_bits: int = 7,
    top_k: int = 20,
    score_threshold: int = -128,
    nms_kernel: int = 3,
) -> list[DecodedPeak]:
    """Mirror the repository C decoder's fixed-point 3x3 centroid arithmetic."""
    tensor = np.asarray(head)
    if tensor.ndim != 3 or tensor.shape[2] != 16 or tensor.size == 0:
        raise ValueError("head must have shape HxWx16")
    if tensor.dtype != np.int8:
        raise ValueError("head must use int8 elements")
    if output_stride <= 0 or not 0 <= offset_fraction_bits <= 15 or top_k < 0:
        raise ValueError("invalid stride, fractional-bit count, or top_k")

    scores = tensor[:, :, 0]
    height, width = scores.shape
    candidates = local_maxima(scores, score_threshold=score_threshold, nms_kernel=nms_kernel)
    candidates.sort(key=lambda point: (-int(scores[point]), point[0] * width + point[1]))
    decoded: list[DecodedPeak] = []
    offset_scale_q16 = 65536 // (1 << offset_fraction_bits)
    for y, x in candidates[:top_k]:
        y0, y1 = max(0, y - 1), min(height, y + 2)
        x0, x1 = max(0, x - 1), min(width, x + 2)
        local = scores[y0:y1, x0:x1].astype(np.int16)
        weights = local - int(np.min(local))
        grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
        weight_sum = int(np.sum(weights, dtype=np.int64))
        weighted_x = int(np.sum(weights * grid_x, dtype=np.int64))
        weighted_y = int(np.sum(weights * grid_y, dtype=np.int64))
        if weight_sum == 0:
            weighted_x = x
            weighted_y = y
            weight_sum = 1
        x_q16 = ((weighted_x << 16) + weight_sum // 2) // weight_sum + 32768
        y_q16 = ((weighted_y << 16) + weight_sum // 2) // weight_sum + 32768
        x_q16 += int(tensor[y, x, 1]) * offset_scale_q16
        y_q16 += int(tensor[y, x, 2]) * offset_scale_q16
        x_q16 = int(np.clip(x_q16 * output_stride, -(1 << 31), (1 << 31) - 1))
        y_q16 = int(np.clip(y_q16 * output_stride, -(1 << 31), (1 << 31) - 1))
        decoded.append(
            DecodedPeak(
                score=float(scores[y, x]),
                x=x_q16 / 65536.0,
                y=y_q16 / 65536.0,
                cell_x=x,
                cell_y=y,
                flat_index=y * width + x,
                width=float(tensor[y, x, 3]),
                height=float(tensor[y, x, 4]),
                x_q16=x_q16,
                y_q16=y_q16,
            )
        )
    return decoded
