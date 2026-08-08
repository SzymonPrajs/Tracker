import numpy as np
import pytest
import torch

from tracker_training.mlx_bridge import (
    copy_mlx_to_pytorch,
    copy_pytorch_to_mlx,
    load_mlx_layout_state,
    mlx_conv_to_torch,
    torch_conv_to_mlx,
    torch_state_for_mlx,
)
from tracker_training.model import HCDS31


def test_convolution_layout_round_trip_including_depthwise():
    for shape in ((16, 3, 3, 3), (32, 1, 3, 3), (16, 32, 1, 1)):
        source = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
        assert torch.equal(mlx_conv_to_torch(torch_conv_to_mlx(source)), source)


def test_full_checkpoint_layout_round_trip():
    source = HCDS31()
    destination = HCDS31()
    state = torch_state_for_mlx(source)
    load_mlx_layout_state(destination, state)
    for expected, actual in zip(source.state_dict().values(),
                                destination.state_dict().values(), strict=True):
        assert torch.equal(expected, actual)


def test_bridge_rejects_non_convolution_rank():
    with pytest.raises(ValueError):
        torch_conv_to_mlx(torch.zeros(3, 3))


def test_live_mlx_pytorch_numerical_parity():
    mx = pytest.importorskip("mlx.core")
    from tracker_training.mlx_model import HCDS31MLX

    torch.manual_seed(20260808)
    torch_model = HCDS31().eval()
    mlx_model = HCDS31MLX()
    mlx_model.eval()
    copy_pytorch_to_mlx(torch_model, mlx_model)

    source = torch.randn(1, 3, 160, 288)
    with torch.no_grad():
        torch_output = torch_model(source).permute(0, 2, 3, 1).numpy()
    mlx_output = mlx_model(mx.array(source.permute(0, 2, 3, 1).numpy()))
    mx.eval(mlx_output)
    np.testing.assert_allclose(np.asarray(mlx_output), torch_output,
                               rtol=2e-5, atol=2e-6)

    restored = HCDS31().eval()
    copy_mlx_to_pytorch(mlx_model, restored)
    for expected, actual in zip(torch_model.state_dict().values(),
                                restored.state_dict().values(), strict=True):
        assert torch.equal(expected, actual)
