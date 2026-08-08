import torch
from torch import nn

from tracker_training.model import HCDS31


def test_exact_hcds31_envelope_and_shape():
    model = HCDS31().eval()
    macs = 0

    def count_macs(module, _inputs, output):
        nonlocal macs
        macs += output.shape[-2] * output.shape[-1] * module.weight.numel()

    hooks = [module.register_forward_hook(count_macs) for module in model.modules()
             if isinstance(module, nn.Conv2d)]
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 160, 288))
    for hook in hooks:
        hook.remove()
    convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    assert output.shape == (1, 16, 40, 72)
    assert len(convolutions) == 36
    assert model.convolution_weight_count() == 66_224
    assert macs == 31_256_640
    assert all(conv.groups in (1, conv.in_channels) for conv in convolutions)


def test_feature_pyramid_shapes():
    model = HCDS31().eval()
    with torch.no_grad():
        features = model.forward_features(torch.zeros(1, 3, 160, 288))
    assert {name: tuple(value.shape[1:]) for name, value in features.items()} == {
        "stem": (16, 80, 144), "s1": (32, 40, 72), "s2": (48, 20, 36),
        "s3": (64, 10, 18), "s4": (96, 5, 9), "p3": (32, 10, 18),
        "p2": (32, 20, 36), "p1": (32, 40, 72),
    }
