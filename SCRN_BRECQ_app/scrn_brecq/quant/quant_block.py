"""SCRN 专用量化 block 适配。

BRECQ 原始 `quant_block.py` 面向 ResNet/MobileNet/RegNet 的残差块。SCRN 的核心
结构是 `FeatureFusionBlock`，包含 CNN 分支、Swin Transformer 分支和残差连接，
因此这里实现 SCRN 自己的 block 包装，供后续 block reconstruction 识别和控制。
"""

from __future__ import annotations

from torch import nn

from SCRN_BRECQ_app.scrn_brecq.quant.quant_layer import QuantModule, StraightThrough, UniformAffineQuantizer
from SCRN_BRECQ_app.scrn_repro.model.scrn import FeatureFusionBlock


class BaseQuantBlock(nn.Module):
    """BRECQ block reconstruction 使用的基础 block 接口。

    该类只定义量化状态控制和 reconstruction 需要的通用字段。具体 block 的前向逻辑
    由子类实现或委托给原始模块。
    """

    def __init__(self, act_quant_params: dict | None = None) -> None:
        super().__init__()
        act_quant_params = act_quant_params or {}
        self.use_weight_quant = False
        self.use_act_quant = False
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        self.activation_function: nn.Module = StraightThrough()
        self.ignore_reconstruction = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False) -> None:
        """设置 block 内所有 QuantModule 的量化状态。

        BRECQ 的 block reconstruction 会以 block 为单位打开或关闭量化。这里同时
        记录 block 自身状态，并把状态传递给内部已经替换好的 `QuantModule`。
        """
        self.use_weight_quant = bool(weight_quant)
        self.use_act_quant = bool(act_quant)
        for module in self.modules():
            if isinstance(module, QuantModule):
                module.set_quant_state(weight_quant, act_quant)


class QuantFeatureFusionBlock(BaseQuantBlock):
    """SCRN `FeatureFusionBlock` 的量化 block 包装。

    当前版本保持原始 `FeatureFusionBlock.forward()` 行为不变，只通过 wrapper 暴露
    `BaseQuantBlock` 接口。这样后续 `block_reconstruction.py` 可以通过
    `isinstance(module, BaseQuantBlock)` 找到 SCRN 的块级重构单元。
    """

    def __init__(
        self,
        feature_block: FeatureFusionBlock,
        weight_quant_params: dict | None = None,
        act_quant_params: dict | None = None,
    ) -> None:
        super().__init__(act_quant_params)
        # weight_quant_params 当前由 QuantModel 在包装前递归替换内部层时使用。
        # 这里保留参数是为了和 BRECQ 原 quant_block 构造接口保持一致。
        self.weight_quant_params = weight_quant_params or {}
        self.block = feature_block
        self.conv_dim = feature_block.conv_dim
        self.trans_dim = feature_block.trans_dim
        self.window_size = feature_block.window_size

    def forward(self, input_tensor):
        """委托给原始 FeatureFusionBlock，避免第五部分改变模型数值行为。"""
        return self.block(input_tensor)


specials = {
    FeatureFusionBlock: QuantFeatureFusionBlock,
}

