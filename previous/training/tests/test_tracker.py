import torch
from types import SimpleNamespace

from tracker_training.augmentation import (
    _warp_sparse_targets,
    forward_lens_points,
    inverse_lens_points,
    photometric_augmentation,
)
from tracker_training.model import HCDS31, deployment_model
from train import augment, loss, scene, targets


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


def test_horizontal_flip_preserves_size_and_negates_x_offset():
    image = torch.arange(12).reshape(1, 3, 2, 2).float()
    heat = torch.tensor([[[[0.0, 1.0]]]])
    offset = torch.tensor([[[[0.0, 0.25]], [[0.0, -0.1]]]])
    size = torch.tensor([[[[0.0, 0.3]], [[0.0, 0.4]]]])
    mask = torch.tensor([[[[False, True]]]])
    args = SimpleNamespace(horizontal_flip=1.0, brightness=0.0, contrast=0.0, channel_jitter=0.0)
    _, (flipped_heat, flipped_offset, flipped_size, flipped_mask) = augment(
        image, (heat, offset, size, mask), args
    )
    assert flipped_heat[0, 0, 0, 0] == 1
    assert flipped_mask[0, 0, 0, 0]
    assert torch.allclose(flipped_offset[0, :, 0, 0], torch.tensor([-0.25, -0.1]))
    assert torch.allclose(flipped_size[0, :, 0, 0], torch.tensor([0.3, 0.4]))


def test_fisheye_point_mapping_round_trips():
    source = torch.tensor([
        [[-0.8, -0.4], [0.0, 0.0], [0.65, 0.3]],
        [[-0.2, 0.2], [0.2, -0.5], [0.4, 0.1]],
    ])
    strength = torch.tensor([0.18, 0.43])[:, None, None]
    scale = torch.tensor([1.0, 0.4])[:, None, None]
    center = torch.tensor([[0.0, 0.0], [0.1, -0.15]])[:, None]
    destination = forward_lens_points(source, strength, scale, center, aspect=1.8)
    reconstructed = inverse_lens_points(destination, strength, scale, center, aspect=1.8)
    assert torch.allclose(reconstructed, source, atol=2e-5)


def test_telephoto_warp_moves_dense_and_sparse_targets_together():
    heat = torch.zeros(1, 1, 40, 72)
    heat[0, 0, 20, 36] = 1
    offset = torch.zeros(1, 2, 40, 72)
    size = torch.zeros(1, 2, 40, 72)
    size[0, :, 20, 36] = torch.tensor([0.20, 0.25])
    mask = torch.zeros(1, 1, 40, 72, dtype=torch.bool)
    mask[0, 0, 20, 36] = True
    warped = _warp_sparse_targets(
        (heat, offset, size, mask),
        strength=torch.zeros(1, 1, 1, 1),
        crop_scale=torch.full((1, 1, 1, 1), 0.5),
        crop_center=torch.zeros(1, 1, 1, 2),
        width=288,
        height=160,
    )
    new_heat, _, new_size, new_mask = warped
    locations = new_mask[0, 0].nonzero()
    assert len(locations) == 1
    iy, ix = locations[0]
    assert new_heat[0, 0, iy, ix] == 1
    assert torch.allclose(new_size[0, :, iy, ix], torch.tensor([0.40, 0.50]), atol=1e-5)


def test_low_light_augmentation_is_bounded_finite_and_darker():
    args = SimpleNamespace(
        exposure_probability=1.0, low_light_probability=1.0,
        exposure_min_ev=0.0, exposure_max_ev=0.0,
        low_light_min_ev=-3.0, low_light_max_ev=-3.0,
        white_balance_probability=0.0, illumination_gradient_probability=0.0,
        shadow_probability=0.0, vignette_probability=0.0, noise_probability=0.0,
        gamma_probability=0.0, saturation_probability=0.0, blur_probability=0.0,
        brightness=0.0, contrast=0.0, channel_jitter=0.0,
    )
    image = torch.full((2, 3, 16, 16), 0.5)
    augmented = photometric_augmentation(image, args)
    assert torch.isfinite(augmented).all()
    assert augmented.min() >= -1 and augmented.max() <= 1
    assert augmented.mean() < image.mean() - 0.5


def test_clean_branch_bypasses_every_augmentation():
    image = torch.rand(2, 3, 8, 8) * 2 - 1
    target = (
        torch.rand(2, 1, 2, 2), torch.rand(2, 2, 2, 2),
        torch.rand(2, 2, 2, 2), torch.ones(2, 1, 2, 2, dtype=torch.bool),
    )
    args = SimpleNamespace(
        augmentation_clean_probability=1.0, horizontal_flip=1.0,
        fisheye_130_probability=1.0, fisheye_180_probability=0.0,
        telephoto_probability=0.0, fisheye_130_min=0.2, fisheye_130_max=0.2,
        exposure_probability=1.0, low_light_probability=1.0,
        low_light_min_ev=-4.0, low_light_max_ev=-4.0,
        white_balance_probability=1.0, illumination_gradient_probability=1.0,
        shadow_probability=1.0, vignette_probability=1.0, noise_probability=1.0,
        gamma_probability=1.0, saturation_probability=1.0, blur_probability=1.0,
        brightness=0.2, contrast=0.2, channel_jitter=0.2,
    )
    augmented_image, augmented_target = augment(image, target, args)
    assert torch.equal(augmented_image, image)
    assert all(torch.equal(actual, expected) for actual, expected in zip(augmented_target, target))
