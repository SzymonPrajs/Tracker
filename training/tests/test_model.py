import torch
from torch import nn

from tracker_training.model import (
    HCDS31,
    OUTPUT_ENCODING_GAINS,
    OUTPUT_ENCODING_ID,
    OUTPUT_ENCODED_MINIMUM,
    OUTPUT_ENCODED_SATURATION_LIMIT,
    OUTPUT_EXPECTED_EXPONENT,
    make_encoded_export_model,
)


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


def test_encoded_export_clone_scales_head_without_mutating_semantic_model():
    torch.manual_seed(17)
    semantic = HCDS31().eval()
    original_weight = semantic.head.weight.detach().clone()
    original_bias = semantic.head.bias.detach().clone()
    encoded = make_encoded_export_model(semantic)
    gains = torch.tensor(OUTPUT_ENCODING_GAINS)

    assert OUTPUT_ENCODING_ID == "hcds31-output-q4-q7-v1"
    assert OUTPUT_EXPECTED_EXPONENT == -3
    assert torch.equal(semantic.head.weight, original_weight)
    assert torch.equal(semantic.head.bias, original_bias)
    assert semantic.encoded_output_clamp is False
    assert encoded.encoded_output_clamp is True
    assert torch.equal(encoded.head.weight, original_weight * gains[:, None, None, None])
    assert torch.equal(encoded.head.bias, original_bias * gains)

    source = torch.randn(1, 3, 160, 288)
    with torch.no_grad():
        semantic_output = semantic(source)
        encoded_output = encoded(source)
    torch.testing.assert_close(
        encoded_output,
        torch.clamp(
            semantic_output * gains[None, :, None, None],
            min=OUTPUT_ENCODED_MINIMUM,
            max=OUTPUT_ENCODED_SATURATION_LIMIT,
        ),
        rtol=1e-5,
        atol=1e-6,
    )
    assert encoded_output.min() >= OUTPUT_ENCODED_MINIMUM
    assert encoded_output.max() <= OUTPUT_ENCODED_SATURATION_LIMIT
    assert torch.count_nonzero(encoded_output[:, 5:]) == 0


def test_encoded_export_clamp_is_graph_visible_and_semantic_model_is_unbounded():
    semantic = HCDS31().eval()
    with torch.no_grad():
        semantic.head.weight.zero_()
        semantic.head.bias.copy_(torch.tensor([20.0, -2.0] + [1.0] * 14))
    encoded = make_encoded_export_model(semantic)
    source = torch.zeros(1, 3, 160, 288)
    with torch.no_grad():
        semantic_output = semantic(source)
        encoded_output = encoded(source)
    assert semantic_output[0, 0, 0, 0].item() == 20.0
    assert encoded_output[0, 0, 0, 0].item() == OUTPUT_ENCODED_SATURATION_LIMIT
    assert encoded_output[0, 1, 0, 0].item() == OUTPUT_ENCODED_MINIMUM
    assert torch.count_nonzero(encoded_output[:, 5:]) == 0


def test_encoded_export_rejects_double_encoding():
    encoded = make_encoded_export_model(HCDS31())
    try:
        make_encoded_export_model(encoded)
    except ValueError as error:
        assert "already encoded" in str(error)
    else:
        raise AssertionError("double output encoding was accepted")
