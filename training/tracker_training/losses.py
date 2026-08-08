"""Losses for the five logical HC-DS31 output channels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def centernet_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_mask: torch.Tensor | None = None,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
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
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    expanded_mask = mask.bool().expand_as(prediction)
    if not bool(expanded_mask.any()):
        return prediction.sum() * 0.0
    loss = F.smooth_l1_loss(prediction, target, reduction="none")
    return loss.masked_select(expanded_mask).mean()


@dataclass(frozen=True)
class LossWeights:
    offset: float = 1.0
    size: float = 0.15
    padding: float = 0.10


class HCDS31Loss(nn.Module):
    def __init__(self, weights: LossWeights = LossWeights()) -> None:
        super().__init__()
        self.weights = weights

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

        heatmap_loss = centernet_focal_loss(prediction[:, 0:1], heatmap, ignore)
        offset_loss = masked_smooth_l1(prediction[:, 1:3], offset_target, reg_mask)
        size_loss = masked_smooth_l1(prediction[:, 3:5], size_target, reg_mask)
        padding_loss = prediction[:, 5:16].square().mean()
        total = (
            heatmap_loss
            + self.weights.offset * offset_loss
            + self.weights.size * size_loss
            + self.weights.padding * padding_loss
        )
        return {
            "loss": total,
            "heatmap": heatmap_loss,
            "offset": offset_loss,
            "size": size_loss,
            "padding": padding_loss,
        }
