from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_outputs.py"
SPEC = importlib.util.spec_from_file_location("compare_outputs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_identical_heads_have_zero_tensor_and_centroid_delta() -> None:
    head = np.full((1, 16, 40, 72), -8.0, dtype=np.float32)
    head[0, 0, 12, 20] = 8.0
    result = MODULE.compare_heads(head, head.copy(), top_k=1)
    assert result["tensor"]["max_absolute_error"] == 0.0
    assert result["decoded"]["matched_peaks"] == 1
    assert result["decoded"]["max_centroid_delta_pixels"] == 0.0


def test_hwc_layout_is_accepted() -> None:
    head = np.zeros((1, 16, 40, 72), dtype=np.float32)
    hwc = np.moveaxis(head, 1, -1)
    result = MODULE.compare_heads(head, hwc, top_k=1)
    assert result["tensor"]["mean_absolute_error"] == 0.0
