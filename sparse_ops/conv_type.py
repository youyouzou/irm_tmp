import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_keep_count(numel: int, prune_rate: float) -> int:
    keep_ratio = max(0.0, min(1.0, 1.0 - float(prune_rate)))
    keep_count = int(round(numel * keep_ratio))
    return min(numel, max(1, keep_count))


class SparseMixin:
    prune_rate: float
    score: nn.Parameter

    def _mask_from_score(self) -> torch.Tensor:
        probs = torch.sigmoid(self.score)
        flat = probs.reshape(-1)
        keep_count = _compute_keep_count(flat.numel(), self.prune_rate)
        if keep_count >= flat.numel():
            hard_mask = torch.ones_like(probs)
        else:
            topk_vals = torch.topk(flat, keep_count, sorted=False).values
            threshold = topk_vals.min()
            hard_mask = (probs >= threshold).to(probs.dtype)
        return hard_mask + probs - probs.detach()

    def hard_mask(self) -> torch.Tensor:
        probs = torch.sigmoid(self.score)
        flat = probs.reshape(-1)
        keep_count = _compute_keep_count(flat.numel(), self.prune_rate)
        if keep_count >= flat.numel():
            return torch.ones_like(probs)
        topk_vals = torch.topk(flat, keep_count, sorted=False).values
        threshold = topk_vals.min()
        return (probs >= threshold).to(probs.dtype)

    def mask_density(self) -> float:
        return float(self.hard_mask().float().mean().item())

    def effective_parameter_count(self) -> float:
        return float(self.hard_mask().float().sum().item())


class SparseConv2d(nn.Conv2d, SparseMixin):
    def __init__(self, *args, prune_rate: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.prune_rate = float(prune_rate)
        self.score = nn.Parameter(torch.empty_like(self.weight))
        nn.init.kaiming_uniform_(self.score, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked_weight = self.weight * self._mask_from_score()
        return F.conv2d(
            x,
            masked_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class SparseLinear(nn.Linear, SparseMixin):
    def __init__(self, *args, prune_rate: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.prune_rate = float(prune_rate)
        self.score = nn.Parameter(torch.empty_like(self.weight))
        nn.init.kaiming_uniform_(self.score, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked_weight = self.weight * self._mask_from_score()
        return F.linear(x, masked_weight, self.bias)


def mask_density(module: nn.Module) -> Optional[float]:
    if hasattr(module, "mask_density"):
        return float(module.mask_density())
    if hasattr(module, "weight") and module.weight is not None:
        return 1.0
    return None
