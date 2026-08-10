"""Fast, label-aware camera augmentation for the fixed-size head detector."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


_BASE_GRIDS: dict[tuple[int, int, str, torch.dtype], torch.Tensor] = {}


def option(args, name: str, default):
    return getattr(args, name, default)


def _pixel_grid(
    height: int,
    width: int,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (height, width, str(device), dtype)
    grid = _BASE_GRIDS.get(key)
    if grid is None:
        y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2 / height) - 1
        x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2 / width) - 1
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((xx, yy), dim=-1)[None]
        _BASE_GRIDS[key] = grid
    return grid.expand(batch, -1, -1, -1)


def inverse_lens_points(
    destination: torch.Tensor,
    strength: torch.Tensor,
    crop_scale: torch.Tensor,
    crop_center: torch.Tensor,
    aspect: float,
) -> torch.Tensor:
    """Map output coordinates to source coordinates for sampling.

    The monotonic radial polynomial keeps the rectangular boundary fixed while
    compressing off-axis content, which approximates the dominant barrel effect
    of an uncalibrated wide/fisheye lens without inventing unseen scene content.
    """
    corner_radius = math.sqrt(aspect * aspect + 1)
    isotropic = torch.stack(
        (destination[..., 0] * aspect / corner_radius,
         destination[..., 1] / corner_radius),
        dim=-1,
    )
    radius_squared = isotropic.square().sum(dim=-1, keepdim=True).clamp_max(1)
    factor = 1 + strength * (1 - radius_squared)
    radial = destination * factor
    return crop_center + crop_scale * radial


def forward_lens_points(
    source: torch.Tensor,
    strength: torch.Tensor,
    crop_scale: torch.Tensor,
    crop_center: torch.Tensor,
    aspect: float,
) -> torch.Tensor:
    """Map source coordinates to their output positions.

    This numerically inverts the monotonic radial sampling polynomial. Eight
    Newton steps are ample for the configured strength range (at most 0.45).
    """
    undistorted = (source - crop_center) / crop_scale
    corner_radius = math.sqrt(aspect * aspect + 1)
    isotropic = torch.stack(
        (undistorted[..., 0] * aspect / corner_radius,
         undistorted[..., 1] / corner_radius),
        dim=-1,
    )
    source_radius = isotropic.square().sum(dim=-1, keepdim=True).sqrt()
    destination_radius = source_radius.clamp(0, 1)
    for _ in range(8):
        value = destination_radius * (1 + strength * (1 - destination_radius.square()))
        derivative = 1 + strength - 3 * strength * destination_radius.square()
        destination_radius = (
            destination_radius - (value - source_radius) / derivative.clamp_min(0.05)
        ).clamp(0, 1)
    ratio = torch.where(source_radius > 1e-7, destination_radius / source_radius, torch.ones_like(source_radius))
    return undistorted * ratio


def _head_aware_crop_centers(
    mask: torch.Tensor,
    offset: torch.Tensor,
    crop_scale: torch.Tensor,
    telephoto: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    batch = mask.shape[0]
    flat_mask = mask[:, 0].flatten(1)
    scores = torch.rand(flat_mask.shape, device=mask.device)
    scores.masked_fill_(~flat_mask, -1)
    selected = scores.argmax(dim=1)
    iy = selected // mask.shape[-1]
    ix = selected % mask.shape[-1]
    batch_index = torch.arange(batch, device=mask.device)
    selected_offset = offset[batch_index, :, iy, ix]
    x = (ix.float() + 0.5 + selected_offset[:, 0]) * (2 / mask.shape[-1]) - 1
    y = (iy.float() + 0.5 + selected_offset[:, 1]) * (2 / mask.shape[-2]) - 1
    centers = torch.stack((x, y), dim=-1)

    # Most telephoto crops retain a randomly chosen head; a minority are random
    # crops so empty and partially visible scenes remain represented.
    has_head = flat_mask.any(dim=1)
    track_head = telephoto & has_head & (torch.rand(batch, device=mask.device) < 0.75)
    random_center = torch.rand((batch, 2), device=mask.device) * 2 - 1
    centers = torch.where(track_head[:, None], centers, random_center)
    jitter = (
        (torch.rand((batch, 2), device=mask.device) * 2 - 1)
        * crop_scale[:, 0, None]
        * 0.30
    )
    centers = centers + torch.where(track_head[:, None], jitter, torch.zeros_like(jitter))
    limit = 1 - crop_scale[:, 0]
    centers = torch.maximum(torch.minimum(centers, limit[:, None]), -limit[:, None])
    return torch.where(telephoto[:, None], centers, torch.zeros_like(centers))[:, None, None, :]


def sample_lens_parameters(images: torch.Tensor, target, args):
    batch = images.shape[0]
    device, dtype = images.device, images.dtype
    p130 = option(args, "fisheye_130_probability", 0.0)
    p180 = option(args, "fisheye_180_probability", 0.0)
    ptele = option(args, "telephoto_probability", 0.0)
    draw = torch.rand(batch, device=device)
    fish130 = draw < p130
    fish180 = (draw >= p130) & (draw < p130 + p180)
    telephoto = (draw >= p130 + p180) & (draw < p130 + p180 + ptele)

    strength = torch.zeros(batch, device=device, dtype=dtype)
    value130 = torch.empty(batch, device=device, dtype=dtype).uniform_(
        option(args, "fisheye_130_min", 0.10), option(args, "fisheye_130_max", 0.25)
    )
    value180 = torch.empty(batch, device=device, dtype=dtype).uniform_(
        option(args, "fisheye_180_min", 0.25), option(args, "fisheye_180_max", 0.45)
    )
    strength = torch.where(fish130, value130, strength)
    strength = torch.where(fish180, value180, strength)
    strength = strength[:, None, None, None]

    zoom_min = option(args, "telephoto_zoom_min", 2.0)
    zoom_max = option(args, "telephoto_zoom_max", 6.0)
    log_zoom = torch.empty(batch, device=device, dtype=dtype).uniform_(
        math.log(zoom_min), math.log(zoom_max)
    )
    crop_scale = torch.where(telephoto, torch.exp(-log_zoom), torch.ones_like(log_zoom))
    crop_scale = crop_scale[:, None, None, None]
    crop_center = _head_aware_crop_centers(
        target[3], target[1], crop_scale.flatten(1), telephoto,
        images.shape[-1], images.shape[-2],
    )
    modes = torch.zeros(batch, device=device, dtype=torch.int8)
    modes[fish130] = 1
    modes[fish180] = 2
    modes[telephoto] = 3
    return strength, crop_scale, crop_center, modes


def _warp_sparse_targets(target, strength, crop_scale, crop_center, width: int, height: int):
    heat, offset, size, mask = target
    batch, _, output_height, output_width = mask.shape
    aspect = width / height
    heat_grid = _pixel_grid(output_height, output_width, batch, heat.device, heat.dtype)
    source_grid = inverse_lens_points(
        heat_grid,
        strength,
        crop_scale,
        crop_center,
        aspect,
    )
    new_heat = F.grid_sample(
        heat, source_grid, mode="bilinear", padding_mode="zeros", align_corners=False
    ).clamp_(0, 1)
    new_offset = torch.zeros_like(offset)
    new_size = torch.zeros_like(size)
    new_mask = torch.zeros_like(mask)

    locations = mask[:, 0].nonzero(as_tuple=False)
    if locations.numel() == 0:
        return new_heat, new_offset, new_size, new_mask
    b, iy, ix = locations.unbind(dim=1)
    old_offset = offset[b, :, iy, ix]
    old_size = size[b, :, iy, ix]
    center = torch.stack(
        (
            (ix.float() + 0.5 + old_offset[:, 0]) * (2 / output_width) - 1,
            (iy.float() + 0.5 + old_offset[:, 1]) * (2 / output_height) - 1,
        ),
        dim=-1,
    )

    # Transform a 3x3 sampling of every source box. Corners alone are not
    # sufficient under radial distortion because an edge midpoint can become an
    # axis-aligned extremum.
    fractions = torch.tensor((-1.0, 0.0, 1.0), device=heat.device, dtype=heat.dtype)
    yy, xx = torch.meshgrid(fractions, fractions, indexing="ij")
    box_points = center[:, None, :] + torch.stack(
        (xx.flatten()[None] * old_size[:, 0:1], yy.flatten()[None] * old_size[:, 1:2]),
        dim=-1,
    )
    item_strength = strength[b].reshape(-1, 1, 1)
    item_scale = crop_scale[b].reshape(-1, 1, 1)
    item_center = crop_center[b].reshape(-1, 1, 2)
    transformed_center = forward_lens_points(
        center[:, None, :], item_strength, item_scale, item_center, aspect
    )[:, 0]
    transformed_box = forward_lens_points(
        box_points, item_strength, item_scale, item_center, aspect
    )
    minimum = transformed_box.amin(dim=1).clamp(-1, 1)
    maximum = transformed_box.amax(dim=1).clamp(-1, 1)
    transformed_size = (maximum - minimum) / 2
    crop_relative_center = (center - item_center[:, 0]) / item_scale[:, 0]
    valid = (
        (crop_relative_center.abs() < 1).all(dim=1)
        & (transformed_center.abs() < 1).all(dim=1)
        & (transformed_size[:, 0] >= 2 / width)
        & (transformed_size[:, 1] >= 2 / height)
    )
    if not valid.any():
        return new_heat, new_offset, new_size, new_mask

    b = b[valid]
    transformed_center = transformed_center[valid]
    transformed_size = transformed_size[valid]
    u = (transformed_center[:, 0] + 1) * output_width / 2
    v = (transformed_center[:, 1] + 1) * output_height / 2
    new_ix = u.floor().long().clamp(0, output_width - 1)
    new_iy = v.floor().long().clamp(0, output_height - 1)
    new_offset[b, 0, new_iy, new_ix] = u - new_ix - 0.5
    new_offset[b, 1, new_iy, new_ix] = v - new_iy - 0.5
    new_size[b, :, new_iy, new_ix] = transformed_size
    new_mask[b, 0, new_iy, new_ix] = True
    new_heat[b, 0, new_iy, new_ix] = 1
    return new_heat, new_offset, new_size, new_mask


def lens_augmentation(images: torch.Tensor, target, args):
    strength, crop_scale, crop_center, modes = sample_lens_parameters(images, target, args)
    if not modes.any():
        return images, target, modes
    batch, _, height, width = images.shape
    destination = _pixel_grid(height, width, batch, images.device, images.dtype)
    source = inverse_lens_points(
        destination, strength, crop_scale, crop_center, width / height
    )
    images = F.grid_sample(
        (images + 1) / 2, source, mode="bilinear", padding_mode="zeros", align_corners=False
    ) * 2 - 1
    target = _warp_sparse_targets(
        target, strength, crop_scale, crop_center, width, height
    )
    return images, target, modes


def _gate(batch: int, probability: float, device: torch.device, dimensions: int = 4):
    shape = (batch,) + (1,) * (dimensions - 1)
    return (torch.rand(shape, device=device) < probability)


def photometric_augmentation(images: torch.Tensor, args, lens_modes=None) -> torch.Tensor:
    """Approximate camera exposure, illumination, optics, and sensor failure modes."""
    batch, _, height, width = images.shape
    device, dtype = images.device, images.dtype
    rgb = ((images + 1) / 2).clamp(0, 1)
    # An inexpensive power-law approximation is sufficient here; applying
    # exposure and noise in this space is substantially more physical than
    # adding offsets directly to gamma-encoded RGB.
    linear = rgb.clamp_min(1e-6).pow(2.2)

    exposure_probability = option(args, "exposure_probability", 0.0)
    low_probability = option(args, "low_light_probability", 0.0)
    regular_ev = torch.empty((batch, 1, 1, 1), device=device, dtype=dtype).uniform_(
        option(args, "exposure_min_ev", -1.0), option(args, "exposure_max_ev", 0.75)
    )
    low_ev = torch.empty((batch, 1, 1, 1), device=device, dtype=dtype).uniform_(
        option(args, "low_light_min_ev", -4.0), option(args, "low_light_max_ev", -1.5)
    )
    low = _gate(batch, low_probability, device)
    exposure = torch.where(low, low_ev, regular_ev)
    exposure = torch.where(_gate(batch, exposure_probability, device), exposure, torch.zeros_like(exposure))
    linear = linear * torch.exp2(exposure)

    wb_probability = option(args, "white_balance_probability", 0.0)
    wb_magnitude = option(args, "white_balance_magnitude", 0.18)
    temperature = (torch.rand((batch, 1, 1, 1), device=device, dtype=dtype) * 2 - 1) * wb_magnitude
    tint = (torch.rand((batch, 1, 1, 1), device=device, dtype=dtype) * 2 - 1) * wb_magnitude * 0.6
    gains = torch.cat((1 + temperature, 1 + tint, 1 - temperature), dim=1).clamp(0.55, 1.55)
    gains = torch.where(_gate(batch, wb_probability, device), gains, torch.ones_like(gains))
    linear = linear * gains

    illumination_grid = _pixel_grid(height, width, 1, device, dtype)
    xx = illumination_grid[0, ..., 0]
    yy = illumination_grid[0, ..., 1]
    gradient_probability = option(args, "illumination_gradient_probability", 0.0)
    gradient_strength = option(args, "illumination_gradient_strength", 0.45)
    gx = (torch.rand((batch, 1, 1, 1), device=device, dtype=dtype) * 2 - 1) * gradient_strength
    gy = (torch.rand((batch, 1, 1, 1), device=device, dtype=dtype) * 2 - 1) * gradient_strength
    gradient = (1 + gx * xx + gy * yy).clamp(0.25, 1.75)
    gradient = torch.where(_gate(batch, gradient_probability, device), gradient, torch.ones_like(gradient))
    linear = linear * gradient

    shadow_probability = option(args, "shadow_probability", 0.0)
    shadow_strength = option(args, "shadow_strength", 0.65)
    shadow_x = torch.rand((batch, 1, 1, 1), device=device, dtype=dtype) * 2 - 1
    shadow_y = torch.rand((batch, 1, 1, 1), device=device, dtype=dtype) * 2 - 1
    shadow_sigma = torch.empty((batch, 1, 1, 1), device=device, dtype=dtype).uniform_(0.25, 0.85)
    shadow_depth = torch.rand((batch, 1, 1, 1), device=device, dtype=dtype) * shadow_strength
    shadow = 1 - shadow_depth * torch.exp(
        -((xx - shadow_x).square() + (yy - shadow_y).square()) / (2 * shadow_sigma.square())
    )
    shadow = torch.where(_gate(batch, shadow_probability, device), shadow, torch.ones_like(shadow))
    linear = linear * shadow

    vignette_probability = option(args, "vignette_probability", 0.0)
    vignette_strength = option(args, "vignette_strength", 0.45)
    vignette_amount = torch.rand((batch, 1, 1, 1), device=device, dtype=dtype) * vignette_strength
    if lens_modes is not None:
        fisheye = ((lens_modes == 1) | (lens_modes == 2))[:, None, None, None]
        vignette_amount = torch.where(fisheye, vignette_amount.clamp_min(0.18), vignette_amount)
        vignette_gate = _gate(batch, vignette_probability, device) | fisheye
    else:
        vignette_gate = _gate(batch, vignette_probability, device)
    radius_squared = (xx.square() + yy.square()).clamp_max(2)
    vignette = (1 - vignette_amount * radius_squared).clamp_min(0.15)
    linear = linear * torch.where(vignette_gate, vignette, torch.ones_like(vignette))

    noise_probability = option(args, "noise_probability", 0.0)
    shot = option(args, "shot_noise", 0.035)
    read = option(args, "read_noise", 0.012)
    noise_std = read + shot * linear.clamp_min(0).sqrt()
    noisy = linear + torch.randn_like(linear) * noise_std
    linear = torch.where(_gate(batch, noise_probability, device), noisy, linear)
    rgb = linear.clamp(0, 1).pow(1 / 2.2)

    gamma_probability = option(args, "gamma_probability", 0.0)
    gamma_magnitude = option(args, "gamma_magnitude", 0.25)
    gamma = torch.exp(
        (torch.rand((batch, 1, 1, 1), device=device, dtype=dtype) * 2 - 1) * gamma_magnitude
    )
    gamma = torch.where(_gate(batch, gamma_probability, device), gamma, torch.ones_like(gamma))
    rgb = rgb.clamp_min(1e-6).pow(gamma)

    saturation_probability = option(args, "saturation_probability", 0.0)
    saturation_magnitude = option(args, "saturation_magnitude", 0.35)
    saturation = 1 + (
        torch.rand((batch, 1, 1, 1), device=device, dtype=dtype) * 2 - 1
    ) * saturation_magnitude
    saturation = torch.where(
        _gate(batch, saturation_probability, device), saturation, torch.ones_like(saturation)
    )
    luminance = (rgb * rgb.new_tensor((0.2126, 0.7152, 0.0722))[None, :, None, None]).sum(1, keepdim=True)
    rgb = luminance + saturation * (rgb - luminance)

    brightness = option(args, "brightness", 0.0)
    contrast = option(args, "contrast", 0.0)
    channel_jitter = option(args, "channel_jitter", 0.0)
    additive = (torch.rand((batch, 1, 1, 1), device=device, dtype=dtype) * 2 - 1) * brightness
    contrast_value = 1 + (torch.rand((batch, 1, 1, 1), device=device, dtype=dtype) * 2 - 1) * contrast
    channels = 1 + (
        torch.rand((batch, 3, 1, 1), device=device, dtype=dtype) * 2 - 1
    ) * channel_jitter
    rgb = (rgb - 0.5) * contrast_value * channels + 0.5 + additive

    blur_probability = option(args, "blur_probability", 0.0)
    blurred = F.avg_pool2d(rgb, kernel_size=3, stride=1, padding=1)
    rgb = torch.where(_gate(batch, blur_probability, device), blurred, rgb)
    return (rgb.clamp(0, 1) * 2 - 1).contiguous()


def augment_batch(images: torch.Tensor, target, args):
    original_images = images
    original_target = target
    clean_probability = option(args, "augmentation_clean_probability", 0.0)
    clean = (
        torch.rand(images.shape[0], device=images.device) < clean_probability
    )[:, None, None, None]
    heat, offset, size, mask = target
    horizontal_flip = option(args, "horizontal_flip", 0.0)
    if horizontal_flip:
        selected = (torch.rand(images.shape[0], device=images.device) < horizontal_flip)[:, None, None, None]
        images = torch.where(selected, images.flip(-1), images)
        heat = torch.where(selected, heat.flip(-1), heat)
        flipped_offset = offset.flip(-1)
        flipped_offset[:, 0].neg_()
        offset = torch.where(selected, flipped_offset, offset)
        size = torch.where(selected, size.flip(-1), size)
        mask = torch.where(selected, mask.flip(-1), mask)
    images, target, modes = lens_augmentation(images, (heat, offset, size, mask), args)
    images = photometric_augmentation(images, args, modes)
    if clean_probability:
        images = torch.where(clean, original_images, images)
        target = tuple(
            torch.where(clean, original, augmented)
            for original, augmented in zip(original_target, target, strict=True)
        )
    return images, target
