"""SCRN 量化模型包装。

本文件负责把已经加载好的 FP32 SCRN 递归改写成可量化模型：先折叠 Conv-BN，
再把 `Conv2d` 和 `Linear` 替换为 `QuantModule`。这里暂不实现 SCRN 专用
`QuantBlock`，后续会单独适配 `FeatureFusionBlock`。
"""

from __future__ import annotations

from collections.abc import Iterator

from torch import nn

from SCRN_BRECQ_app.scrn_brecq.quant.fold_bn import search_fold_and_remove_bn
from SCRN_BRECQ_app.scrn_brecq.quant.quant_layer import QuantModule, StraightThrough


class QuantModel(nn.Module):
    """包装 SCRN 的量化模型。

    参数:
        model: 已加载好权重的 FP32 SCRN 模型。
        weight_quant_params: 传给权重量化器的参数，如 `n_bits`、`channel_wise`。
        act_quant_params: 传给激活量化器的参数，如 `n_bits`、`leaf_param`。
        fold_bn: 是否在替换量化层之前执行 Conv-BN 折叠。
    """

    def __init__(
        self,
        model: nn.Module,
        weight_quant_params: dict | None = None,
        act_quant_params: dict | None = None,
        *,
        fold_bn: bool = True,
    ) -> None:
        super().__init__()
        weight_quant_params = weight_quant_params or {}
        act_quant_params = act_quant_params or {}

        if fold_bn:
            search_fold_and_remove_bn(model)
        self.model = model
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(
        self,
        module: nn.Module,
        weight_quant_params: dict | None = None,
        act_quant_params: dict | None = None,
    ) -> None:
        """递归替换 Conv2d/Linear，并把相邻 ReLU 合并到前一个 QuantModule。

        BRECQ 的重构单位是量化层或量化块；本阶段先保证 SCRN 内部所有基础可量化
        算子都被替换。`FeatureFusionBlock` 的整体 block 包装留到下一部分。
        """
        weight_quant_params = weight_quant_params or {}
        act_quant_params = act_quant_params or {}
        previous_quant_module: QuantModule | None = None

        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Linear)):
                quant_module = QuantModule(child_module, weight_quant_params, act_quant_params)
                setattr(module, name, quant_module)
                previous_quant_module = quant_module
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if previous_quant_module is not None:
                    previous_quant_module.activation_function = child_module
                    setattr(module, name, StraightThrough())
            elif isinstance(child_module, StraightThrough):
                # BN folding 后留下的占位层不应打断 Conv -> ReLU 的激活合并关系。
                continue
            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)
                previous_quant_module = None

    def forward(self, input_tensor):
        """转发到量化改写后的 SCRN 模型。"""
        return self.model(input_tensor)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False) -> None:
        """统一设置所有 QuantModule 的权重量化和激活量化开关。"""
        for module in self.quant_modules():
            module.set_quant_state(weight_quant, act_quant)

    def quant_modules(self) -> Iterator[QuantModule]:
        """遍历模型中的所有 QuantModule，供状态设置和调试统计复用。"""
        for module in self.model.modules():
            if isinstance(module, QuantModule):
                yield module

    def set_first_last_layer_to_8bit(self) -> None:
        """将首层和尾层相关量化器设为 8bit。

        BRECQ 常用策略是降低中间层 bitwidth，同时保持首层和输出附近更高精度，
        以减少输入/输出端误差放大。
        """
        module_list = list(self.quant_modules())
        if len(module_list) < 2:
            raise RuntimeError("set_first_last_layer_to_8bit requires at least two QuantModule layers.")
        module_list[0].weight_quantizer.bitwidth_refactor(8)
        module_list[0].act_quantizer.bitwidth_refactor(8)
        module_list[-1].weight_quantizer.bitwidth_refactor(8)
        module_list[-2].act_quantizer.bitwidth_refactor(8)
        module_list[0].ignore_reconstruction = True

    def disable_network_output_quantization(self) -> None:
        """关闭最后一个 QuantModule 的激活量化。

        网络最终输出不会继续作为后续层输入，通常不需要额外做 activation quant。
        """
        module_list = list(self.quant_modules())
        if not module_list:
            raise RuntimeError("No QuantModule found in model.")
        module_list[-1].disable_act_quant = True

