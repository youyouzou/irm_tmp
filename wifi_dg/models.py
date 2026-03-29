from __future__ import annotations

import torch
import torch.nn as nn

from sparse_ops.builder import LayerFactory


class CSIConvBlock(nn.Module):
    def __init__(self, factory: LayerFactory, in_channels: int, out_channels: int, pool_kernel=None):
        super().__init__()
        self.block = nn.Sequential(
            factory.conv2d(in_channels, out_channels, kernel_size=(5, 3), padding=(2, 1), bias=False),
            factory.batch_norm(out_channels),
            nn.ReLU(inplace=True),
            factory.conv2d(out_channels, out_channels, kernel_size=(3, 5), padding=(1, 2), bias=False),
            factory.batch_norm(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_kernel) if pool_kernel else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.block(x))


class WiFiDGBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        factory = LayerFactory(config["backbone_type"], config["prune_rate"])
        dropout = float(config["dropout"])
        input_shape = config.get("input_shape")
        if input_shape is None:
            raise ValueError("config['input_shape'] is required before model construction.")
        in_channels = int(input_shape[0])
        stem_channels = int(config.get("stem_channels", 32))
        stage_channels = [int(v) for v in config.get("stage_channels", [64, 128, 128])]
        stage_pool_kernels = config.get("stage_pool_kernels", [[2, 1], [2, 2], [2, 2]])
        pool1 = tuple(stage_pool_kernels[0]) if stage_pool_kernels[0] else None
        pool2 = tuple(stage_pool_kernels[1]) if stage_pool_kernels[1] else None
        pool3 = tuple(stage_pool_kernels[2]) if stage_pool_kernels[2] else None

        self.stem = nn.Sequential(
            factory.conv2d(in_channels, stem_channels, kernel_size=(7, 3), padding=(3, 1), bias=False),
            factory.batch_norm(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.block1 = CSIConvBlock(factory, stem_channels, stage_channels[0], pool_kernel=pool1)
        self.block2 = CSIConvBlock(factory, stage_channels[0], stage_channels[1], pool_kernel=pool2)
        self.block3 = CSIConvBlock(factory, stage_channels[1], stage_channels[2], pool_kernel=pool3)
        self.refine = nn.Sequential(
            factory.conv2d(stage_channels[2], stage_channels[2], kernel_size=(3, 3), padding=(1, 1), bias=False),
            factory.batch_norm(stage_channels[2]),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(stage_channels[2], int(config["num_classes"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.refine(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)


class MMFiBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride=(1, 1), dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.shortcut = nn.Identity()
        if stride != (1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = torch.relu(out)
        return out


class MMFiWiFiBaseline(nn.Module):
    def __init__(self, in_channels: int = 30, num_classes: int = 7, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=(2, 2), padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1),
        )
        self.layer1 = self._make_layer(32, 32, blocks=2, stride=(1, 1), dropout=dropout)
        self.layer2 = self._make_layer(32, 64, blocks=2, stride=(2, 2), dropout=dropout)
        self.layer3 = self._make_layer(64, 128, blocks=2, stride=(2, 2), dropout=dropout)
        self.layer4 = self._make_layer(128, 256, blocks=2, stride=(2, 2), dropout=dropout)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, blocks: int, stride, dropout: float) -> nn.Sequential:
        layers = [MMFiBasicBlock(in_channels, out_channels, stride=stride, dropout=dropout)]
        for _ in range(1, blocks):
            layers.append(MMFiBasicBlock(out_channels, out_channels, stride=(1, 1), dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


def build_model(config) -> nn.Module:
    model_family = str(config.get("model_family", "wifi_csi_default")).lower()
    if model_family == "wifi_csi_default":
        return WiFiDGBackbone(config)
    if model_family == "mmfi_resnet":
        if str(config.get("backbone_type", "dense")).lower() != "dense":
            raise ValueError("mmfi_resnet currently supports backbone_type=dense only.")
        input_shape = config.get("input_shape")
        if input_shape is None:
            raise ValueError("config['input_shape'] is required before model construction.")
        return MMFiWiFiBaseline(
            in_channels=int(input_shape[0]),
            num_classes=int(config["num_classes"]),
            dropout=float(config.get("dropout", 0.2)),
        )
    raise ValueError(f"Unsupported model_family: {model_family}")
