#!/usr/bin/env python3
"""Compare float and ESP-PPQ-simulated tracker heads on the identical input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from tracker_training.decode import decode_heatmap  # noqa: E402


def sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _head(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.shape == (1, 16, 40, 72):
        values = values[0]
    elif values.shape == (1, 40, 72, 16):
        values = np.moveaxis(values[0], -1, 0)
    if values.shape != (16, 40, 72):
        raise ValueError(f"expected [1,16,40,72] or [1,40,72,16], found {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("head contains non-finite values")
    return values.astype(np.float64, copy=False)


def compare_heads(
    float_output: np.ndarray,
    quantized_output: np.ndarray,
    *,
    top_k: int = 10,
    score_threshold: float = 0.05,
) -> dict[str, Any]:
    float_head = _head(float_output)
    quantized_head = _head(quantized_output)
    absolute = np.abs(float_head - quantized_head)
    channel_error = [float(np.mean(absolute[channel])) for channel in range(16)]

    def peaks(head: np.ndarray) -> list[Any]:
        return decode_heatmap(
            sigmoid(head[0]),
            offsets=head[1:3],
            sizes=head[3:5],
            output_stride=4,
            top_k=top_k,
            score_threshold=score_threshold,
        )

    float_peaks = peaks(float_head)
    quantized_peaks = peaks(quantized_head)
    unmatched = set(range(len(quantized_peaks)))
    deltas: list[float] = []
    for reference in float_peaks:
        if not unmatched:
            break
        candidate = min(
            unmatched,
            key=lambda index: (
                (quantized_peaks[index].x - reference.x) ** 2
                + (quantized_peaks[index].y - reference.y) ** 2,
                index,
            ),
        )
        distance = float(
            np.hypot(
                quantized_peaks[candidate].x - reference.x,
                quantized_peaks[candidate].y - reference.y,
            )
        )
        if distance <= 16.0:
            deltas.append(distance)
            unmatched.remove(candidate)
    sorted_deltas = sorted(deltas)
    p95_index = max(0, int(np.ceil(0.95 * len(sorted_deltas))) - 1)
    return {
        "tensor": {
            "max_absolute_error": float(np.max(absolute)),
            "mean_absolute_error": float(np.mean(absolute)),
            "mean_absolute_error_by_channel": channel_error,
        },
        "decoded": {
            "float_peaks": len(float_peaks),
            "quantized_peaks": len(quantized_peaks),
            "matched_peaks": len(deltas),
            "unmatched_float_peaks": len(float_peaks) - len(deltas),
            "unmatched_quantized_peaks": len(quantized_peaks) - len(deltas),
            "mean_centroid_delta_pixels": float(np.mean(deltas)) if deltas else None,
            "p95_centroid_delta_pixels": sorted_deltas[p95_index] if deltas else None,
            "max_centroid_delta_pixels": max(deltas) if deltas else None,
        },
        "warning": "A synthetic conversion comparison is not representative-data INT8 accuracy.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("float_output", type=Path)
    parser.add_argument("quantized_output", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    args = parser.parse_args()
    result = compare_heads(
        np.load(args.float_output, allow_pickle=False),
        np.load(args.quantized_output, allow_pickle=False),
        top_k=args.top_k,
        score_threshold=args.score_threshold,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
