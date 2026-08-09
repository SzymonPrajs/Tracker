import torch

from tracker_training.model import HCDS31, deployment_model
from train import loss, scene, targets


def test_model_and_training_step():
    model = HCDS31()
    image, heads = scene(5)
    output = model(image[None])
    assert output.shape == (1, 16, 40, 72)
    value = loss(output, targets([heads]))
    value.backward()
    assert torch.isfinite(value)


def test_multiple_heads_make_multiple_peaks():
    _, heads = scene(5)
    heat, _, _, _ = targets([heads])
    assert int(heat.eq(1).sum()) == len(heads)


def test_deployment_output_is_bounded():
    output = deployment_model(HCDS31())(torch.zeros(1, 3, 160, 288))
    assert output.min() >= -16
    assert output.max() <= 15.5
    assert torch.count_nonzero(output[:, 5:]) == 0
