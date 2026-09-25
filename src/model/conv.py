"""Causal convolutions, transposed convolutions, and residual blocks."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CausalConv1d(nn.Module):
    """Mimi/Encodec causal padding; keep the inner name `conv` for checkpoint keys."""

    def __init__(
        self,
        config: SimpleNamespace,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        pad_mode: str | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.causal = config.use_causal_conv
        self.pad_mode = config.pad_mode if pad_mode is None else pad_mode

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # Effective kernel size after dilation.
        self.effective_kernel_size = (kernel_size - 1) * dilation + 1
        self.stride_value = stride
        self.padding_total = self.effective_kernel_size - stride
        self.padding_right = self.padding_total // 2
        self.padding_left = self.padding_total - self.padding_right

    @staticmethod
    def _pad1d(
        x: Tensor, padding: tuple[int, int], mode: str, value: float = 0.0
    ) -> Tensor:
        """F.pad wrapper matching the Encodec/Mimi reflect-padding edge case."""
        left, right = padding
        if mode != "reflect":
            return F.pad(x, (left, right), mode=mode, value=value)

        length = x.shape[-1]
        max_pad = max(left, right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))

        x = F.pad(x, (left, right), mode="reflect")
        return x[..., : x.shape[-1] - extra_pad] if extra_pad else x

    def _extra_padding(self, input_length: int) -> int:
        # Right-pad to the next stride boundary.
        return (-input_length) % self.stride_value

    def forward(self, x: Tensor) -> Tensor:
        extra = self._extra_padding(x.shape[-1])
        if self.causal:
            x = self._pad1d(x, (self.padding_total, extra), self.pad_mode)
        else:
            x = self._pad1d(
                x,
                (self.padding_left, self.padding_right + extra),
                self.pad_mode,
            )
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    """ConvTranspose1d with Mimi's asymmetric output trimming."""

    def __init__(
        self,
        config: SimpleNamespace,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.causal = config.use_causal_conv
        self.trim_right_ratio = config.trim_right_ratio

        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            groups=groups,
            bias=bias,
        )

        padding_total = kernel_size - stride
        if self.causal:
            self.padding_right = math.ceil(padding_total * self.trim_right_ratio)
        else:
            self.padding_right = padding_total // 2
        self.padding_left = padding_total - self.padding_right

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        end = x.shape[-1] - self.padding_right
        if self.padding_right == 0:
            return x[..., self.padding_left :]
        return x[..., self.padding_left : end]


class SEANetResnetBlock(nn.Module):
    """SEANet residual block used by Mimi."""

    def __init__(
        self, config: SimpleNamespace, dim: int, dilations: Sequence[int]
    ) -> None:
        super().__init__()
        kernel_sizes = (config.residual_kernel_size, 1)

        hidden = dim // config.compress
        block: list[nn.Module] = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_ch = dim if i == 0 else hidden
            out_ch = dim if i == len(kernel_sizes) - 1 else hidden
            block.append(nn.ELU())
            block.append(
                CausalConv1d(
                    config,
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
            )

        self.block = nn.Sequential(*block)
        self.shortcut = (
            CausalConv1d(config, dim, dim, kernel_size=1)
            if config.use_conv_shortcut
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.shortcut(x) + self.block(x)


class DecoderCausalConv1d(CausalConv1d):
    """Zero-padded causal convolution for the waveform decoder."""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1
    ):
        config = SimpleNamespace(use_causal_conv=True, pad_mode="constant")
        super().__init__(
            config,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )


class DecoderCausalConvTranspose1d(CausalConvTranspose1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        config = SimpleNamespace(use_causal_conv=True, trim_right_ratio=1.0)
        super().__init__(config, in_channels, out_channels, kernel_size, stride=stride)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = DecoderCausalConv1d(dim, dim, 7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), 1e-6))

    def forward(self, x):
        y = self.dwconv(x).transpose(1, 2)
        y = self.pwconv2(F.gelu(self.pwconv1(self.norm(y))))
        return x + (self.gamma * y).transpose(1, 2)


class SnakeBeta(nn.Module):
    """Checkpoint alpha and beta are stored in log space."""

    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        alpha = self.alpha.exp()[None, :, None]
        beta = self.beta.exp()[None, :, None]
        return x + (1.0 / (beta + 1e-9)) * torch.sin(x * alpha).square()


class DecoderResidualUnit(nn.Module):
    def __init__(self, dim, dilation):
        super().__init__()
        self.act1 = SnakeBeta(dim)
        self.conv1 = DecoderCausalConv1d(dim, dim, 7, dilation=dilation)
        self.act2 = SnakeBeta(dim)
        self.conv2 = DecoderCausalConv1d(dim, dim, 1)

    def forward(self, x):
        return x + self.conv2(self.act2(self.conv1(self.act1(x))))


class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, rate):
        super().__init__()
        self.block = nn.Sequential(
            SnakeBeta(in_dim),
            DecoderCausalConvTranspose1d(in_dim, out_dim, 2 * rate, rate),
            *(DecoderResidualUnit(out_dim, d) for d in (1, 3, 9)),
        )

    def forward(self, x):
        return self.block(x)
