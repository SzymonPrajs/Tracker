"""Losses for the five logical HC-DS31 output channels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import fields
import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import OUTPUT_ENCODED_SATURATION_LIMIT, OUTPUT_ENCODING_GAINS


def centernet_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_mask: torch.Tensor | None = None,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("focal alpha must be finite and non-negative")
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("focal beta must be finite and non-negative")
    probabilities = logits.sigmoid().clamp(1e-6, 1.0 - 1e-6)
    positives = target.eq(1.0)
    valid = torch.ones_like(positives) if ignore_mask is None else ~ignore_mask.bool()
    negatives = target.lt(1.0) & valid
    positive_loss = -torch.log(probabilities) * (1.0 - probabilities).pow(alpha)
    negative_loss = (
        -torch.log(1.0 - probabilities)
        * probabilities.pow(alpha)
        * (1.0 - target).pow(beta)
    )
    positive_loss = positive_loss.masked_select(positives & valid).sum()
    negative_loss = negative_loss.masked_select(negatives).sum()
    positive_count = (positives & valid).sum().to(logits.dtype)
    return (positive_loss + negative_loss) / positive_count.clamp_min(1.0)


def masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float = 1.0 / 9.0,
) -> torch.Tensor:
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("Smooth-L1 beta must be finite and positive")
    expanded_mask = mask.bool().expand_as(prediction)
    if not bool(expanded_mask.any()):
        return prediction.sum() * 0.0
    loss = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta)
    return loss.masked_select(expanded_mask).mean()


@dataclass(frozen=True)
class LossWeights:
    heatmap: float = 1.0
    offset: float = 1.0
    size: float = 0.15
    padding: float = 0.10
    background: float = 0.01
    saturation: float = 0.01

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"loss weight {field.name} must be finite and non-negative")


class HCDS31Loss(nn.Module):
    def __init__(
        self,
        weights: LossWeights = LossWeights(),
        *,
        focal_alpha: float = 2.0,
        focal_beta: float = 4.0,
        smooth_l1_beta: float = 1.0 / 9.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(focal_alpha) or focal_alpha < 0.0:
            raise ValueError("focal_alpha must be finite and non-negative")
        if not math.isfinite(focal_beta) or focal_beta < 0.0:
            raise ValueError("focal_beta must be finite and non-negative")
        if not math.isfinite(smooth_l1_beta) or smooth_l1_beta <= 0.0:
            raise ValueError("smooth_l1_beta must be finite and positive")
        self.weights = weights
        self.focal_alpha = focal_alpha
        self.focal_beta = focal_beta
        self.smooth_l1_beta = smooth_l1_beta

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "HCDS31Loss":
        decode_consistency = float(config.get("decode_consistency", 0.0))
        if not math.isfinite(decode_consistency) or decode_consistency != 0.0:
            raise ValueError("decode_consistency must remain zero until implemented")
        weights = LossWeights(
            heatmap=float(config["heatmap"]),
            offset=float(config["offset"]),
            size=float(config["size"]),
            padding=float(config["padding"]),
            background=float(config["background_regression"]),
            saturation=float(config["encoded_saturation"]),
        )
        return cls(
            weights,
            focal_alpha=float(config["focal_alpha"]),
            focal_beta=float(config["focal_beta"]),
            smooth_l1_beta=float(config["smooth_l1_beta"]),
        )

    @staticmethod
    def _batched(value: torch.Tensor, dimensions: int) -> torch.Tensor:
        return value.unsqueeze(0) if value.ndim == dimensions - 1 else value

    def forward(
        self, prediction: torch.Tensor, targets: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if prediction.ndim != 4 or prediction.shape[1] != 16:
            raise ValueError("prediction must have shape [N,16,H,W]")
        heatmap = self._batched(targets["heatmap"], 4).to(prediction.device)
        offset_target = self._batched(targets["offset"], 4).to(prediction.device)
        size_target = self._batched(targets["size"], 4).to(prediction.device)
        reg_mask = self._batched(targets["reg_mask"], 4).to(prediction.device)
        ignore = self._batched(targets["ignore_mask"], 4).to(prediction.device)

        heatmap_loss = centernet_focal_loss(
            prediction[:, 0:1],
            heatmap,
            ignore,
            alpha=self.focal_alpha,
            beta=self.focal_beta,
        )
        offset_loss = masked_smooth_l1(
            prediction[:, 1:3], offset_target, reg_mask, beta=self.smooth_l1_beta
        )
        size_loss = masked_smooth_l1(
            prediction[:, 3:5], size_target, reg_mask, beta=self.smooth_l1_beta
        )
        padding_loss = prediction[:, 5:16].square().mean()
        regression = prediction[:, 1:5]
        background_mask = (~reg_mask.bool()).expand_as(regression)
        if bool(background_mask.any()):
            background_loss = regression.square().masked_select(background_mask).mean()
        else:
            background_loss = regression.sum() * 0.0
        gains = prediction.new_tensor(OUTPUT_ENCODING_GAINS[:5]).view(1, 5, 1, 1)
        encoded = prediction[:, :5] * gains
        saturation_loss = F.relu(
            encoded.abs() - OUTPUT_ENCODED_SATURATION_LIMIT
        ).square().mean()
        total = (
            self.weights.heatmap * heatmap_loss
            + self.weights.offset * offset_loss
            + self.weights.size * size_loss
            + self.weights.padding * padding_loss
            + self.weights.background * background_loss
            + self.weights.saturation * saturation_loss
        )
        return {
            "loss": total,
            "heatmap": heatmap_loss,
            "offset": offset_loss,
            "size": size_loss,
            "padding": padding_loss,
            "background": background_loss,
            "saturation": saturation_loss,
        }
