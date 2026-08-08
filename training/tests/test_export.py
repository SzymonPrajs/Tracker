from __future__ import annotations

import sys
import pickle
import random
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT / "scripts"))

from check_onnx import check_parity, inspect_model  # noqa: E402
from export_onnx import export_static_onnx, load_checkpoint, make_smoke_model  # noqa: E402
from quantize_espdl import save_quantized_output  # noqa: E402


def test_static_opset18_export_and_parity(tmp_path: Path) -> None:
    output = tmp_path / "smoke.onnx"
    artifacts = export_static_onnx(
        make_smoke_model(),
        output,
        (1, 3, 160, 288),
        write_reference=True,
    )

    summary = inspect_model(output)
    parity = check_parity(output, artifacts["input"], artifacts["output"])

    assert summary["opset"] == 18
    assert summary["input"]["shape"] == (1, 3, 160, 288)
    assert summary["output"]["shape"] == (1, 16, 40, 72)
    assert set(summary["operators"]) <= {"Conv", "Relu"}
    assert parity["max_abs_error"] < 1e-5


def test_checker_rejects_arbitrary_grouped_convolution(tmp_path: Path) -> None:
    import torch

    class GroupedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 16, kernel_size=3, padding=1, groups=2)

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return self.conv(image)

    output = tmp_path / "grouped.onnx"
    export_static_onnx(GroupedModel(), output, (1, 4, 16, 16))
    with pytest.raises(ValueError, match="groups must be 1 or input channels"):
        inspect_model(output)


def test_tracker_checkpoint_v1_loads_with_safe_numpy_rng_allowlist(tmp_path: Path) -> None:
    import torch

    source = make_smoke_model(seed=13)
    checkpoint = tmp_path / "tracker.pt"
    torch.save(
        {
            "schema_version": "tracker-checkpoint-v1",
            "model": source.state_dict(),
            "optimizer": {},
            "config": {"model": {"name": "smoke"}},
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
        },
        checkpoint,
    )
    destination = make_smoke_model(seed=99)

    load_checkpoint(destination, checkpoint)

    for expected, actual in zip(
        source.state_dict().values(), destination.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(actual, expected)


def test_checkpoint_rejects_unknown_schema(tmp_path: Path) -> None:
    import torch

    checkpoint = tmp_path / "future.pt"
    torch.save(
        {
            "schema_version": "tracker-checkpoint-v2",
            "model": make_smoke_model().state_dict(),
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="unsupported checkpoint schema"):
        load_checkpoint(make_smoke_model(), checkpoint)


def test_checkpoint_does_not_retry_corrupt_pickle_unsafely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    calls: list[bool] = []

    def rejected_load(*args: object, **kwargs: object) -> object:
        calls.append(bool(kwargs.get("weights_only")))
        raise pickle.UnpicklingError("corrupt checkpoint")

    monkeypatch.setattr(torch, "load", rejected_load)
    with pytest.raises(pickle.UnpicklingError, match="corrupt checkpoint"):
        load_checkpoint(make_smoke_model(), tmp_path / "corrupt.pt")
    assert calls == [True]


def test_quantized_output_capture_saves_the_only_executor_tensor(
    tmp_path: Path,
) -> None:
    import torch

    class FakeExecutor:
        def __init__(self, *, graph: object, device: str) -> None:
            assert graph == "quantized-graph"
            assert device == "cpu"

        def __call__(self, input_tensor: torch.Tensor) -> list[torch.Tensor]:
            return [input_tensor.mul(0.5).add(1.0)]

    sample = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    output = tmp_path / "quantized.npy"
    saved = save_quantized_output(
        "quantized-graph", sample, output, executor_type=FakeExecutor
    )

    assert saved == output.resolve()
    np.testing.assert_array_equal(np.load(saved, allow_pickle=False), sample.numpy() * 0.5 + 1.0)


def test_quantized_output_capture_rejects_multiple_outputs(tmp_path: Path) -> None:
    import torch

    class MultipleOutputExecutor:
        def __init__(self, *, graph: object, device: str) -> None:
            pass

        def __call__(self, input_tensor: torch.Tensor) -> list[torch.Tensor]:
            return [input_tensor, input_tensor]

    with pytest.raises(ValueError, match="requires one graph output"):
        save_quantized_output(
            object(),
            torch.zeros(1),
            tmp_path / "quantized.npy",
            executor_type=MultipleOutputExecutor,
        )
