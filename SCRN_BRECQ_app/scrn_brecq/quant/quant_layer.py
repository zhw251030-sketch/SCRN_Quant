"""BRECQ 基础量化层。

本文件参考 `BRECQ-main/quant/quant_layer.py` 的算法结构，重新实现 SCRN-BRECQ
需要的基础组件。这里不导入 BRECQ 原源码，后续 QuantModel、AdaRound 和重构流程
都会复用本文件中的接口。
"""

from __future__ import annotations

from typing import Union
import warnings

import torch
from torch import nn
import torch.nn.functional as F


class StraightThrough(nn.Module):
    """恒等映射层。

    BRECQ 会把某些 ReLU 合并到前一个量化层中；原来 ReLU 所在的位置需要一个占位
    模块保持网络结构可遍历，因此使用恒等映射。
    """

    def __init__(self, channel_num: int = 1) -> None:
        super().__init__()
        self.channel_num = int(channel_num)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """直接返回输入，不改变数值和梯度。"""
        return input_tensor


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """带直通估计器的 round 操作。

    量化前向需要 `round` 得到整数网格，但普通 `round` 的梯度几乎处处为 0。
    STE 写法让前向等价于 `round(x)`，反向近似为恒等函数梯度，便于后续优化
    activation scale 或 AdaRound 参数。
    """
    return (x.round() - x).detach() + x


def lp_loss(pred: torch.Tensor, tgt: torch.Tensor, p: float = 2.0, reduction: str = "none") -> torch.Tensor:
    """计算 Lp reconstruction loss。

    BRECQ 的 scale 初始化和重构阶段会比较量化前后张量误差。`reduction="none"`
    保留原实现风格：先沿通道维求和再求均值；其他取值则退化为全元素均值。
    """
    if reduction == "none":
        return (pred - tgt).abs().pow(p).sum(1).mean()
    return (pred - tgt).abs().pow(p).mean()


class UniformAffineQuantizer(nn.Module):
    """均匀仿射量化器。

    量化公式为：

    1. 用 `delta` 表示 scale，用 `zero_point` 表示浮点 0 对应的整数位置。
    2. 将浮点值映射到整数网格：`round(x / delta) + zero_point`。
    3. clamp 到 `[0, 2**n_bits - 1]`。
    4. 反量化回浮点：`(x_int - zero_point) * delta`。

    参数:
        n_bits: 量化 bit 数，保持 BRECQ 原始约束，支持 2 到 8 bit。
        symmetric: 是否使用对称范围；对称时零点会围绕绝对最大值确定。
        channel_wise: 是否逐输出通道计算权重量化 scale。
        scale_method: scale 初始化方式，支持 `max`、`max_scale` 和 `mse`。
        leaf_param: True 时把 `delta` 注册为可学习参数，供激活量化 LSQ 阶段优化。
    """

    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        channel_wise: bool = False,
        scale_method: str = "max",
        leaf_param: bool = False,
    ) -> None:
        super().__init__()
        if not 2 <= int(n_bits) <= 8:
            raise ValueError(f"bitwidth not supported: {n_bits}")
        self.sym = bool(symmetric)
        self.n_bits = int(n_bits)
        self.n_levels = 2 ** self.n_bits
        self.delta: torch.Tensor | nn.Parameter | None = None
        # zero_point 需要进入 state_dict，否则 W+A checkpoint 重新加载时会用
        # eval 输入重新初始化激活量化零点，导致 run 内指标和重载评估不一致。
        self.register_buffer("zero_point", None)
        self.inited = False
        self.leaf_param = bool(leaf_param)
        self.channel_wise = bool(channel_wise)
        self.scale_method = str(scale_method)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行量化和反量化，输出仍是浮点张量。

        第一次前向时根据输入张量初始化 `delta` 和 `zero_point`；后续前向复用同一组
        参数，保证校准阶段行为稳定。
        """
        if not self.inited:
            delta, zero_point = self.init_quantization_scale(x, self.channel_wise)
            if self.leaf_param:
                self.delta = nn.Parameter(delta)
            else:
                self.delta = delta
            self.zero_point = zero_point
            self.inited = True

        if self.delta is None or self.zero_point is None:
            raise RuntimeError("Quantizer scale is not initialized.")

        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        return (x_quant - self.zero_point) * self.delta

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """初始化 scale 与 zero point。

        channel-wise 模式通常用于 Conv2d/Linear 权重，按输出通道分别估计范围；
        tensor-wise 模式通常用于激活，整张量共享一组量化参数。
        """
        if channel_wise:
            return self._init_channel_wise_scale(x)
        return self._init_tensor_wise_scale(x)

    def quantize(self, x: torch.Tensor, max_value: torch.Tensor, min_value: torch.Tensor) -> torch.Tensor:
        """用给定上下界量化张量，供 MSE 搜索 scale 时反复评估。"""
        delta = (max_value - min_value) / (self.n_levels - 1)
        delta = torch.clamp(delta, min=torch.finfo(x.dtype).eps)
        zero_point = (-min_value / delta).round()
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        return (x_quant - zero_point) * delta

    def bitwidth_refactor(self, refactored_bit: int) -> None:
        """修改 bitwidth。

        BRECQ 会把首层/尾层设成 8bit；该函数保留这个接口，供后续 QuantModel 复用。
        """
        if not 2 <= int(refactored_bit) <= 8:
            raise ValueError(f"bitwidth not supported: {refactored_bit}")
        self.n_bits = int(refactored_bit)
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self) -> str:
        """让 `print(module)` 时显示关键量化配置。"""
        return (
            f"bit={self.n_bits}, scale_method={self.scale_method}, symmetric={self.sym}, "
            f"channel_wise={self.channel_wise}, leaf_param={self.leaf_param}"
        )

    def _init_channel_wise_scale(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """逐输出通道初始化量化参数。

        对 Conv2d 权重，输出通道是第 0 维，返回的 `delta` 形状为
        `[out_channels, 1, 1, 1]`，可以直接广播到权重张量。
        """
        if x.ndim not in {2, 4}:
            raise ValueError(f"Channel-wise quantization expects 2D or 4D tensor, got shape {tuple(x.shape)}")

        x_clone = x.detach().clone()
        n_channels = x_clone.shape[0]
        deltas = []
        zero_points = []
        for channel_index in range(n_channels):
            delta, zero_point = self._init_tensor_wise_scale(x_clone[channel_index])
            deltas.append(delta)
            zero_points.append(zero_point)

        delta_tensor = torch.stack(deltas).to(dtype=x.dtype, device=x.device)
        zero_point_tensor = torch.stack(zero_points).to(dtype=x.dtype, device=x.device)
        if x.ndim == 4:
            return delta_tensor.view(-1, 1, 1, 1), zero_point_tensor.view(-1, 1, 1, 1)
        return delta_tensor.view(-1, 1), zero_point_tensor.view(-1, 1)

    def _init_tensor_wise_scale(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """整张量初始化量化参数。"""
        if "max" in self.scale_method:
            return self._init_max_scale(x)
        if self.scale_method == "mse":
            return self._init_mse_scale(x)
        raise NotImplementedError(f"Unsupported scale_method: {self.scale_method}")

    def _init_max_scale(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """按最小值/最大值直接确定量化范围。"""
        x_detached = x.detach()
        x_min = min(float(x_detached.min().item()), 0.0)
        x_max = max(float(x_detached.max().item()), 0.0)
        if "scale" in self.scale_method:
            # 原 BRECQ 的 max_scale 会按 bit 数轻微收缩范围。
            scale_factor = (self.n_bits + 2) / 8
            x_min *= scale_factor
            x_max *= scale_factor

        x_absmax = max(abs(x_min), x_max)
        if self.sym:
            x_min = -x_absmax if x_min < 0 else 0.0
            x_max = x_absmax

        delta_value = (x_max - x_min) / (self.n_levels - 1)
        if delta_value < 1e-8:
            warnings.warn(f"Quantization range close to zero: [{x_min}, {x_max}]", stacklevel=2)
            delta_value = 1e-8

        delta = torch.tensor(delta_value, dtype=x.dtype, device=x.device)
        zero_point = torch.tensor(round(-x_min / delta_value), dtype=x.dtype, device=x.device)
        return delta, zero_point

    def _init_mse_scale(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """通过 MSE/Lp 搜索更合适的量化范围。

        该方法从原始 min/max 开始逐步收缩范围，选择量化误差最小的一组 scale 和
        zero point。BRECQ 默认用它初始化权重量化参数。
        """
        x_detached = x.detach()
        x_max = x_detached.max()
        x_min = x_detached.min()
        best_score = torch.tensor(float("inf"), dtype=x.dtype, device=x.device)
        best_delta: torch.Tensor | None = None
        best_zero_point: torch.Tensor | None = None

        for shrink_step in range(80):
            shrink = 1.0 - shrink_step * 0.01
            new_max = x_max * shrink
            new_min = x_min * shrink
            x_quant = self.quantize(x_detached, new_max, new_min)
            score = lp_loss(x_detached, x_quant, p=2.4, reduction="all")
            if score < best_score:
                best_score = score
                delta = (new_max - new_min) / (self.n_levels - 1)
                delta = torch.clamp(delta, min=torch.finfo(x.dtype).eps)
                best_delta = delta
                best_zero_point = (-new_min / delta).round()

        if best_delta is None or best_zero_point is None:
            raise RuntimeError("Failed to initialize MSE quantization scale.")
        return best_delta.to(dtype=x.dtype, device=x.device), best_zero_point.to(dtype=x.dtype, device=x.device)


class QuantModule(nn.Module):
    """Conv2d/Linear 的量化包装模块。

    该模块保存一份原始 FP32 权重副本，并通过 `set_quant_state` 控制前向时是否使用
    权重量化、激活量化。后续 QuantModel 会递归把 SCRN 中的 Conv2d 和 Linear 替换
    为本模块。
    """

    def __init__(
        self,
        org_module: Union[nn.Conv2d, nn.Linear],
        weight_quant_params: dict | None = None,
        act_quant_params: dict | None = None,
        disable_act_quant: bool = False,
        se_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(org_module, (nn.Conv2d, nn.Linear)):
            raise TypeError(f"QuantModule only supports Conv2d and Linear, got {type(org_module)!r}")

        weight_quant_params = weight_quant_params or {}
        act_quant_params = act_quant_params or {}
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = {}
            self.fwd_func = F.linear

        # 保留原 module 的 Parameter 引用，使优化器和 state_dict 仍能看到可训练权重。
        self.weight = org_module.weight
        # 原始 FP32 权重作为 buffer 保存。这样 QuantModel 后续 `.to(device)` 时，
        # FP32 路径和量化路径使用的张量会一起迁移到目标设备。
        self.register_buffer("org_weight", org_module.weight.detach().clone())
        if org_module.bias is None:
            self.bias = None
            self.register_buffer("org_bias", None)
        else:
            self.bias = org_module.bias
            self.register_buffer("org_bias", org_module.bias.detach().clone())

        # 默认不启用量化，保证刚替换模块时输出仍是 FP32 路径。
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = bool(disable_act_quant)
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

        self.activation_function: nn.Module = StraightThrough()
        self.ignore_reconstruction = False
        self.se_module = se_module
        self.extra_repr = org_module.extra_repr

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """执行 FP32 或量化前向。

        - `use_weight_quant=False` 时使用初始化时保存的 FP32 权重副本。
        - `use_weight_quant=True` 时先对当前权重做量化/反量化，再执行 Conv/Linear。
        - 激活函数位于 Conv/Linear 后；若启用激活量化，则再量化激活输出。
        """
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias

        out = self.fwd_func(input_tensor, weight, bias, **self.fwd_kwargs)
        if self.se_module is not None:
            out = self.se_module(out)
        out = self.activation_function(out)

        if self.disable_act_quant:
            return out
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False) -> None:
        """设置当前前向是否启用权重量化和激活量化。"""
        self.use_weight_quant = bool(weight_quant)
        self.use_act_quant = bool(act_quant)
