import torch.nn as nn


def batch_norm(num_features: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(num_features)
