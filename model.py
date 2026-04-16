from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


WINDOW_SIZE = 10
DILATIONS = (1, 2, 4)
KERNEL_SIZE = 4
HIDDEN_CHANNELS = 128


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        use_weight_norm: bool = False,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.conv = weight_norm(conv) if use_weight_norm else conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.left_padding, 0))
        return self.conv(x)


class PlainTCNLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            CausalConv1d(
                in_channels,
                out_channels,
                kernel_size,
                dilation,
                use_weight_norm=True,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation,
            use_weight_norm=True,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(
            out_channels,
            out_channels,
            kernel_size,
            dilation,
            use_weight_norm=True,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.skip = None
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.skip is None else self.skip(x)

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        return out + residual


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, reduced_channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(reduced_channels, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x).squeeze(-1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale).unsqueeze(-1)
        return x * scale


class SETCNModel(nn.Module):
    def __init__(self, model_name: str, input_channels: int = 1, dropout: float = 0.0):
        super().__init__()
        if model_name not in {"tcn", "tcn_r1", "tcn_r2", "tcn_r3", "setcn"}:
            raise ValueError(f"Unsupported model: {model_name}")

        self.model_name = model_name
        self.features = nn.ModuleList()

        for index, dilation in enumerate(DILATIONS):
            in_channels = input_channels if index == 0 else HIDDEN_CHANNELS
            out_channels = HIDDEN_CHANNELS

            use_residual = (
                (model_name == "tcn_r1" and index < 1)
                or (model_name == "tcn_r2" and index < 2)
                or (model_name in {"tcn_r3", "setcn"})
            )

            if model_name == "tcn":
                layer = PlainTCNLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=KERNEL_SIZE,
                    dilation=dilation,
                    dropout=dropout,
                )
            elif use_residual:
                layer = ResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=KERNEL_SIZE,
                    dilation=dilation,
                    dropout=dropout,
                )
            else:
                layer = PlainTCNLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=KERNEL_SIZE,
                    dilation=dilation,
                    dropout=dropout,
                )

            self.features.append(layer)

        self.se_block = SEBlock(HIDDEN_CHANNELS) if model_name == "setcn" else None
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Linear(HIDDEN_CHANNELS, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.features:
            x = layer(x)

        if self.se_block is not None:
            x = self.se_block(x)

        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.output(x)
        return x.squeeze(-1)


def build_model(model_name: str, input_channels: int = 1, dropout: float = 0.0) -> nn.Module:
    return SETCNModel(model_name=model_name, input_channels=input_channels, dropout=dropout)
