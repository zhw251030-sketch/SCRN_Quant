"""AdaRound 权重量化器。

AdaRound 的目标是在固定量化 scale/zero point 后，学习每个权重到底向上取整还是
向下取整。该模块只实现量化器本身，后续 layer/block reconstruction 会负责把
`QuantModule.weight_quantizer` 替换成这里的 `AdaRoundQuantizer` 并优化 `alpha`。
"""

from __future__ import annotations

import torch
from torch import nn

from SCRN_BRECQ_app.scrn_brecq.quant.quant_layer import UniformAffineQuantizer, round_ste


class AdaRoundQuantizer(nn.Module):
    """BRECQ 使用的自适应 rounding 量化器。

    参数:
        uaq: 已经初始化过的 `UniformAffineQuantizer`。AdaRound 复用它的 bitwidth、
            scale(`delta`) 和 zero point。
        weight_tensor: 用于初始化 `alpha` 的 FP32 权重。
        round_mode: rounding 模式。支持 `nearest`、`nearest_ste`、`stochastic`、
            `learned_hard_sigmoid`；兼容原 BRECQ 中的 `learned_round_sigmoid` 名称。
    """

    def __init__(
        self,
        uaq: UniformAffineQuantizer,
        weight_tensor: torch.Tensor,
        round_mode: str = "learned_round_sigmoid",
    ) -> None:
        super().__init__()
        if uaq.delta is None or uaq.zero_point is None:
            raise RuntimeError("AdaRoundQuantizer requires an initialized UniformAffineQuantizer.")

        self.n_bits = int(uaq.n_bits)
        self.sym = bool(uaq.sym)
        self.n_levels = int(uaq.n_levels)
        self.round_mode = self._normalize_round_mode(round_mode)
        self.soft_targets = False

        # delta/zero_point 是固定量化参数，不在 AdaRound 中优化；注册为 buffer 便于
        # 随模型迁移设备和保存 state_dict。
        self.register_buffer("delta", uaq.delta.detach().clone())
        self.register_buffer("zero_point", uaq.zero_point.detach().clone())

        # hard-sigmoid 参数来自 AdaRound 论文和 BRECQ 原实现。
        self.gamma = -0.1
        self.zeta = 1.1
        self.beta = 2.0 / 3.0
        self.alpha: nn.Parameter | None = None

        if self.round_mode == "learned_hard_sigmoid":
            self.init_alpha(weight_tensor.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 AdaRound 量化和反量化。"""
        scaled = x / self.delta
        if self.round_mode == "nearest":
            x_int = torch.round(scaled)
        elif self.round_mode == "nearest_ste":
            x_int = round_ste(scaled)
        elif self.round_mode == "stochastic":
            x_floor = torch.floor(scaled)
            rest = torch.clamp(scaled - x_floor, 0.0, 1.0)
            x_int = x_floor + torch.bernoulli(rest)
        elif self.round_mode == "learned_hard_sigmoid":
            x_floor = torch.floor(scaled)
            if self.soft_targets:
                x_int = x_floor + self.get_soft_targets()
            else:
                if self.alpha is None:
                    raise RuntimeError("AdaRound alpha is not initialized.")
                x_int = x_floor + (self.alpha >= 0).to(dtype=x.dtype)
        else:
            raise ValueError(f"Unsupported round_mode: {self.round_mode}")

        x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
        return (x_quant - self.zero_point) * self.delta

    def get_soft_targets(self) -> torch.Tensor:
        """返回连续 relaxed rounding 目标，范围严格限制在 `[0, 1]`。"""
        if self.alpha is None:
            raise RuntimeError("AdaRound alpha is not initialized.")
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0.0, 1.0)

    def init_alpha(self, x: torch.Tensor) -> None:
        """根据 FP32 权重的小数部分初始化 `alpha`。

        hard-sigmoid 的反函数在边界值附近可能出现 NaN/Inf，因此对 remainder 做极小
        clamp，只影响数值稳定性，不改变 AdaRound 学习向上/向下取整的核心思想。
        """
        x_floor = torch.floor(x / self.delta)
        rest = (x / self.delta) - x_floor
        eps = torch.finfo(x.dtype).eps
        rest = torch.clamp(rest, min=eps, max=1.0 - eps)
        alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1.0)
        self.alpha = nn.Parameter(alpha)

    @staticmethod
    def _normalize_round_mode(round_mode: str) -> str:
        """兼容 BRECQ 原实现中不一致的 round mode 名称。"""
        if round_mode == "learned_round_sigmoid":
            return "learned_hard_sigmoid"
        valid_modes = {"nearest", "nearest_ste", "stochastic", "learned_hard_sigmoid"}
        if round_mode not in valid_modes:
            raise ValueError(f"Unsupported round_mode: {round_mode}")
        return round_mode

