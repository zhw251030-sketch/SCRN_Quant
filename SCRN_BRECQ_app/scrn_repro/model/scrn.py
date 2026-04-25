"""SCRN 网络结构的独立复现。

该实现参考论文中“CNN 局部特征 + Swin Transformer 全局/窗口特征融合”的设计，
以及公开源码展示出的默认超参数。模块运行时不导入 SCRN-main 中的任何文件。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SCRNConfig:
    """SCRN 默认结构参数。

    stage_depths 对应 5 个特征融合阶段；公开实现默认每个阶段 1 个融合块。
    """

    in_channels: int = 1
    dim: int = 64
    stage_depths: tuple[int, int, int, int, int] = (1, 1, 1, 1, 1)
    head_dim: int = 32
    window_size: int = 8
    drop_path_rate: float = 0.0
    input_resolution: int = 128


class DropPath(nn.Module):
    """随机深度层。

    训练阶段按样本随机丢弃残差分支；推理阶段保持恒等映射。
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def _pad_to_window_multiple(x: Tensor, window_size: int) -> tuple[Tensor, tuple[int, int]]:
    """把特征图补齐到窗口大小的整数倍，避免非 128 尺寸输入在窗口划分时失败。"""
    height, width = x.shape[-2:]
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (pad_h, pad_w)


def _window_partition(x: Tensor, window_size: int) -> Tensor:
    """将 NHWC 特征划分为窗口序列。

    输入:  [B, H, W, C]
    输出:  [B, num_windows, window_size * window_size, C]
    """
    batch, height, width, channels = x.shape
    x = x.view(batch, height // window_size, window_size, width // window_size, window_size, channels)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch, -1, window_size * window_size, channels)


def _window_reverse(windows: Tensor, window_size: int, height: int, width: int) -> Tensor:
    """把窗口序列还原为 NHWC 特征图。"""
    batch, _, _, channels = windows.shape
    x = windows.view(batch, height // window_size, width // window_size, window_size, window_size, channels)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch, height, width, channels)


class WindowMultiHeadSelfAttention(nn.Module):
    """窗口多头自注意力，支持普通窗口 W 和移位窗口 SW。

    SCRN 使用窗口注意力共享局部窗口内的时间/空间信息；SW 模式通过半窗口平移
    让相邻窗口发生信息交换，从而改善事件连续性。
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        window_size: int,
        window_type: str = "W",
    ) -> None:
        super().__init__()
        if window_type not in {"W", "SW"}:
            raise ValueError(f"Unsupported window type: {window_type}")
        if dim % head_dim != 0:
            raise ValueError(f"dim={dim} must be divisible by head_dim={head_dim}")

        self.dim = dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.window_type = window_type
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        # 相对位置偏置表大小为 (2M-1)*(2M-1)，M 是窗口边长。
        table_size = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, self.num_heads))
        self.register_buffer("relative_position_index", self._build_relative_position_index(), persistent=False)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        batch, height, width, channels = x.shape
        if self.window_type == "SW":
            shift = self.window_size // 2
            x = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))

        windows = _window_partition(x, self.window_size)
        qkv = self.qkv(windows)
        qkv = qkv.view(batch, -1, self.window_size * self.window_size, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv.unbind(dim=0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self._relative_position_bias().unsqueeze(0).unsqueeze(2)
        if self.window_type == "SW":
            # 移位窗口会把原本不相邻的边界 patch 卷到同一窗口，因此需要 mask 掉跨边界注意力。
            attn = attn + self._shift_attention_mask(height, width, x.device, x.dtype)

        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch, -1, self.window_size * self.window_size, channels)
        out = self.proj(out)
        out = _window_reverse(out, self.window_size, height, width)

        if self.window_type == "SW":
            out = torch.roll(out, shifts=(shift, shift), dims=(1, 2))
        return out

    def _build_relative_position_index(self) -> Tensor:
        coords = torch.stack(torch.meshgrid(
            torch.arange(self.window_size),
            torch.arange(self.window_size),
            indexing="ij",
        ))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        return relative_coords.sum(-1)

    def _relative_position_bias(self) -> Tensor:
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(self.window_size**2, self.window_size**2, self.num_heads)
        return bias.permute(2, 0, 1).contiguous()

    def _shift_attention_mask(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        shift = self.window_size // 2
        mask = torch.zeros((1, height, width, 1), device=device)
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -shift), slice(-shift, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -shift), slice(-shift, None))

        label = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                mask[:, h_slice, w_slice, :] = label
                label += 1

        mask_windows = _window_partition(mask, self.window_size).view(1, -1, self.window_size**2)
        attn_mask = mask_windows.unsqueeze(-1) - mask_windows.unsqueeze(-2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask.to(dtype=dtype).unsqueeze(1)


class SwinBlock(nn.Module):
    """一个 SCRN 使用的 Swin Transformer 块。"""

    def __init__(
        self,
        dim: int,
        head_dim: int,
        window_size: int,
        drop_path: float,
        window_type: str,
        input_resolution: int,
    ) -> None:
        super().__init__()
        if input_resolution <= window_size:
            window_type = "W"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowMultiHeadSelfAttention(dim, head_dim, window_size, window_type)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class FeatureFusionBlock(nn.Module):
    """SCRN 特征融合块。

    输入通道先被 1x1 卷积分成 CNN 分支和 Transformer 分支：
    CNN 分支强调局部相似性与弱信号保留；Transformer 分支通过窗口注意力增强事件连续性。
    """

    def __init__(
        self,
        conv_dim: int,
        trans_dim: int,
        head_dim: int,
        window_size: int,
        drop_path: float,
        window_type: str,
        input_resolution: int,
    ) -> None:
        super().__init__()
        total_dim = conv_dim + trans_dim
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.window_size = window_size

        self.split_proj = nn.Conv2d(total_dim, total_dim, kernel_size=1, bias=True)
        self.merge_proj = nn.Conv2d(total_dim, total_dim, kernel_size=1, bias=True)
        self.conv_branch = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(inplace=True),
        )
        self.trans_branch = SwinBlock(
            dim=trans_dim,
            head_dim=head_dim,
            window_size=window_size,
            drop_path=drop_path,
            window_type=window_type,
            input_resolution=input_resolution,
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x, _ = _pad_to_window_multiple(x, self.window_size)

        conv_x, trans_x = torch.split(self.split_proj(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_branch(conv_x) + conv_x

        # Transformer 分支使用 NHWC，以便 LayerNorm 直接作用于通道维。
        trans_x = trans_x.permute(0, 2, 3, 1).contiguous()
        trans_x = self.trans_branch(trans_x)
        trans_x = trans_x.permute(0, 3, 1, 2).contiguous()

        x = self.merge_proj(torch.cat((conv_x, trans_x), dim=1))
        x = x[..., : residual.shape[-2], : residual.shape[-1]]
        return residual + x


class SCRN(nn.Module):
    """Swin Transformer Convolutional Residual Network。

    模型输入和输出均为单通道地震 patch，形状约定为 [B, 1, H, W]。
    """

    def __init__(
        self,
        in_channels: int = 1,
        stage_depths: Sequence[int] = (1, 1, 1, 1, 1),
        dim: int = 64,
        drop_path_rate: float = 0.0,
        input_resolution: int = 128,
        head_dim: int = 32,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        if len(stage_depths) != 5:
            raise ValueError("SCRN expects exactly five feature fusion stages.")
        if dim % 2 != 0:
            raise ValueError("dim must be even because SCRN splits channels into two branches.")

        self.config = SCRNConfig(
            in_channels=in_channels,
            dim=dim,
            stage_depths=tuple(int(x) for x in stage_depths),
            head_dim=head_dim,
            window_size=window_size,
            drop_path_rate=drop_path_rate,
            input_resolution=input_resolution,
        )

        total_blocks = sum(stage_depths)
        drop_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        self.head = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=False)

        block_index = 0
        stages = []
        for depth in stage_depths:
            blocks = []
            for i in range(depth):
                blocks.append(FeatureFusionBlock(
                    conv_dim=dim // 2,
                    trans_dim=dim // 2,
                    head_dim=head_dim,
                    window_size=window_size,
                    drop_path=drop_rates[block_index],
                    window_type="W" if i % 2 == 0 else "SW",
                    input_resolution=input_resolution,
                ))
                block_index += 1
            blocks.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False))
            stages.append(nn.Sequential(*blocks))

        self.stage1, self.stage2, self.stage3, self.stage4, self.stage5 = stages
        self.tail = nn.Conv2d(dim, in_channels, kernel_size=3, padding=1, bias=False)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        x1 = self.head(x)
        x2 = self.stage1(x1)
        x3 = self.stage2(x2)
        x4 = self.stage3(x3)
        x5 = self.stage4(x4 + x3)
        x6 = self.stage5(x5 + x2)
        out = self.tail(x6 + x1)
        return out[..., :height, :width]

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)


def build_scrn_from_config(config: SCRNConfig | None = None) -> SCRN:
    """根据配置构建 SCRN，供脚本和加载器复用。"""
    config = config or SCRNConfig()
    return SCRN(
        in_channels=config.in_channels,
        stage_depths=config.stage_depths,
        dim=config.dim,
        drop_path_rate=config.drop_path_rate,
        input_resolution=config.input_resolution,
        head_dim=config.head_dim,
        window_size=config.window_size,
    )
