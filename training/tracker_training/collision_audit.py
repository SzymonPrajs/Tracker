"""Audit the representational ceiling of fixed-stride, fixed-slot heatmaps."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def audit_collisions(
    centers_by_image: Sequence[np.ndarray],
    *,
    output_stride: int,
    slots_per_cell: int = 1,
) -> dict[str, Any]:
    if output_stride <= 0 or slots_per_cell <= 0:
        raise ValueError("output_stride and slots_per_cell must be positive")
    total_targets = 0
    overflow_targets = 0
    collision_pairs = 0
    colliding_images = 0
    max_occupancy = 0
    per_image = []
    for image_index, raw_centers in enumerate(centers_by_image):
        centers = np.asarray(raw_centers, dtype=np.float64)
        if centers.size == 0:
            centers = np.empty((0, 2), dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] < 2 or not np.all(np.isfinite(centers)):
            raise ValueError("each center array must be finite and Nx2+")
        if np.any(centers[:, :2] < 0):
            raise ValueError("centers must be non-negative image coordinates")
        cells = np.floor(centers[:, :2] / output_stride).astype(np.int64)
        occupancy: dict[tuple[int, int], int] = {}
        for x, y in cells:
            key = (int(x), int(y))
            occupancy[key] = occupancy.get(key, 0) + 1
        image_overflow = sum(max(0, count - slots_per_cell) for count in occupancy.values())
        image_pairs = sum(count * (count - 1) // 2 for count in occupancy.values())
        image_max = max(occupancy.values(), default=0)
        image_collides = image_overflow > 0
        total_targets += len(centers)
        overflow_targets += image_overflow
        collision_pairs += image_pairs
        colliding_images += int(image_collides)
        max_occupancy = max(max_occupancy, image_max)
        per_image.append(
            {
                "image_index": image_index,
                "targets": len(centers),
                "occupied_cells": len(occupancy),
                "overflow_targets": image_overflow,
                "collision_pairs": image_pairs,
                "max_occupancy": image_max,
            }
        )
    representable = total_targets - overflow_targets
    return {
        "images": len(centers_by_image),
        "output_stride": output_stride,
        "slots_per_cell": slots_per_cell,
        "total_targets": total_targets,
        "representable_targets": representable,
        "overflow_targets": overflow_targets,
        "collision_pairs": collision_pairs,
        "colliding_images": colliding_images,
        "max_occupancy": max_occupancy,
        "theoretical_max_recall": representable / total_targets if total_targets else None,
        "per_image": per_image,
    }


def audit_strides(
    centers_by_image: Sequence[np.ndarray],
    *,
    output_strides: Sequence[int] = (2, 4, 8),
    slots_per_cell: int = 1,
) -> dict[int, dict[str, Any]]:
    strides = tuple(int(value) for value in output_strides)
    if not strides or len(set(strides)) != len(strides):
        raise ValueError("output_strides must be non-empty and unique")
    return {
        stride: audit_collisions(
            centers_by_image, output_stride=stride, slots_per_cell=slots_per_cell
        )
        for stride in strides
    }
