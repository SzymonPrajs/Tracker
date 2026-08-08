"""Strict layout conversion and optional PyTorch-to-MLX checkpoint bridge."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn


def torch_conv_to_mlx(weight: torch.Tensor) -> np.ndarray:
    if weight.ndim != 4:
        raise ValueError("convolution weight must be four-dimensional")
    return weight.detach().cpu().permute(0, 2, 3, 1).contiguous().numpy()


def mlx_conv_to_torch(weight: np.ndarray) -> torch.Tensor:
    array = np.asarray(weight)
    if array.ndim != 4:
        raise ValueError("convolution weight must be four-dimensional")
    return torch.from_numpy(array.transpose(0, 3, 1, 2).copy())


def torch_state_for_mlx(model: nn.Module) -> dict[str, np.ndarray]:
    """Return an explicit NumPy checkpoint with MLX convolution layouts."""
    converted: dict[str, np.ndarray] = {}
    conv_names = {name for name, module in model.named_modules()
                  if isinstance(module, nn.Conv2d)}
    for name, value in model.state_dict().items():
        owner, _, field = name.rpartition(".")
        if owner in conv_names and field == "weight":
            converted[name] = torch_conv_to_mlx(value)
        else:
            converted[name] = value.detach().cpu().numpy().copy()
    return converted


def load_mlx_layout_state(
    model: nn.Module, state: Mapping[str, np.ndarray], *, strict: bool = True
) -> None:
    reference = model.state_dict()
    conv_names = {name for name, module in model.named_modules()
                  if isinstance(module, nn.Conv2d)}
    if strict and set(state) != set(reference):
        missing = sorted(set(reference) - set(state))
        unexpected = sorted(set(state) - set(reference))
        raise KeyError(f"checkpoint keys differ: missing={missing}, unexpected={unexpected}")
    restored: dict[str, torch.Tensor] = {}
    for name, target in reference.items():
        if name not in state:
            restored[name] = target
            continue
        owner, _, field = name.rpartition(".")
        value = mlx_conv_to_torch(state[name]) if owner in conv_names and field == "weight" \
            else torch.from_numpy(np.asarray(state[name]).copy())
        if value.shape != target.shape:
            raise ValueError(f"shape mismatch for {name}: {value.shape} != {target.shape}")
        restored[name] = value.to(dtype=target.dtype)
    model.load_state_dict(restored, strict=strict)


def copy_pytorch_to_mlx(torch_model: nn.Module, mlx_model: object) -> None:
    """Copy an isomorphic checkpoint into MLX by matching nested parameter names."""
    try:
        import mlx.core as mx
        from mlx.utils import tree_unflatten
    except ImportError as error:
        raise ImportError("MLX is required for the live bridge") from error

    flat = torch_state_for_mlx(torch_model)
    parameters = {
        name: mx.array(value)
        for name, value in flat.items()
        if not name.endswith("num_batches_tracked")
    }
    # MLX uses nested dictionaries for dotted module paths.
    mlx_model.update(tree_unflatten(list(parameters.items())))
    mx.eval(mlx_model.state)


def copy_mlx_to_pytorch(mlx_model: object, torch_model: nn.Module) -> None:
    """Copy an MLX-trained checkpoint into the isomorphic PyTorch exporter."""
    try:
        from mlx.utils import tree_flatten
    except ImportError as error:
        raise ImportError("MLX is required for the live bridge") from error

    reference = torch_model.state_dict()
    expected = {name for name in reference if not name.endswith("num_batches_tracked")}
    # MLX ``state`` also exposes immutable operator configuration such as
    # ``conv.stride.0``. Only named arrays corresponding to PyTorch state are
    # checkpoint values; structural entries are deliberately ignored.
    flat_mlx = {
        name: np.asarray(value)
        for name, value in tree_flatten(mlx_model.state)
        if name in expected
    }
    state: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for name, value in reference.items():
        if name.endswith("num_batches_tracked"):
            state[name] = value.detach().cpu().numpy().copy()
        elif name in flat_mlx:
            state[name] = flat_mlx[name]
        else:
            missing.append(name)
    if missing:
        raise KeyError(f"MLX checkpoint is missing parameters/state: {missing}")
    load_mlx_layout_state(torch_model, state, strict=True)
