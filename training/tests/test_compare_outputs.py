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
    assert result["evidence_class"] == "converter_smoke_calibration_sample"


def test_hwc_layout_is_accepted() -> None:
    head = np.zeros((1, 16, 40, 72), dtype=np.float32)
    hwc = np.moveaxis(head, 1, -1)
    result = MODULE.compare_heads(head, hwc, top_k=1)
    assert result["tensor"]["mean_absolute_error"] == 0.0


def test_global_match_reports_misses_outside_gate() -> None:
    reference = np.full((1, 16, 40, 72), -8.0, dtype=np.float32)
    shifted = reference.copy()
    reference[0, 0, 10, 10] = 8.0
    shifted[0, 0, 10, 20] = 8.0
    result = MODULE.compare_heads(reference, shifted, top_k=1, match_gate_pixels=4.0)
    assert result["decoded"]["matched_peaks"] == 0
    assert result["decoded"]["unmatched_float_peaks"] == 1
    assert result["decoded"]["unmatched_quantized_peaks"] == 1


def test_encoded_outputs_are_compared_in_semantic_units() -> None:
    reference = np.zeros((1, 16, 40, 72), dtype=np.float32)
    quantized = reference.copy()
    quantized[0, 0] = 2.0
    quantized[0, 1:5] = 16.0
    quantized[0, 5] = 1.0

    result = MODULE.compare_heads(
        reference,
        quantized,
        top_k=0,
        output_encoding="encoded",
    )

    assert result["output_encoding"] == {
        "name": "encoded",
        "comparison_space": "semantic",
        "encoded_channel_divisors": [2.0, 16.0, 16.0, 16.0, 16.0],
    }
    channel_error = result["tensor"]["mean_absolute_error_by_channel"]
    assert channel_error[:6] == [1.0] * 6


def test_encoded_decode_matches_semantic_decode() -> None:
    semantic = np.full((1, 16, 40, 72), -8.0, dtype=np.float32)
    semantic[0, 0, 12, 20] = 8.0
    semantic[0, 1, 12, 20] = 0.25
    semantic[0, 2, 12, 20] = -0.125
    encoded = semantic.copy()
    encoded[:, 0] *= 2.0
    encoded[:, 1:5] *= 16.0

    semantic_result = MODULE.compare_heads(semantic, semantic, top_k=1)
    encoded_result = MODULE.compare_heads(
        encoded,
        encoded,
        top_k=1,
        output_encoding="encoded",
    )

    assert encoded_result["decoded"] == semantic_result["decoded"]
    assert encoded_result["tensor"]["max_absolute_error"] == 0.0


def test_unknown_output_encoding_is_rejected() -> None:
    head = np.zeros((1, 16, 40, 72), dtype=np.float32)
    try:
        MODULE.compare_heads(head, head, output_encoding="mystery")
    except ValueError as error:
        assert "output_encoding" in str(error)
    else:
        raise AssertionError("unknown output encoding was accepted")
