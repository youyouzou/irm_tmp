from typing import Type

import torch.nn as nn

from sparse_ops.bn_type import batch_norm
from sparse_ops.conv_type import SparseConv2d


class LayerFactory:
    def __init__(self, backbone_type: str, prune_rate: float):
        if backbone_type not in {"dense", "sparse"}:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")
        self.backbone_type = backbone_type
        self.prune_rate = float(prune_rate)

    def conv2d(self, *args, **kwargs) -> nn.Module:
        if self.backbone_type == "dense":
            return nn.Conv2d(*args, **kwargs)
        return SparseConv2d(*args, prune_rate=self.prune_rate, **kwargs)

    def linear(self, *args, **kwargs) -> nn.Module:
        return nn.Linear(*args, **kwargs)

    def batch_norm(self, num_features: int) -> nn.Module:
        return batch_norm(num_features)
