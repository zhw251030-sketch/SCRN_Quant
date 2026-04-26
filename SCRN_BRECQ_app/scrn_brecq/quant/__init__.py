"""BRECQ 量化算法子包。

后续会在这里逐步迁移并重写 BRECQ 的核心组件：
量化层、AdaRound、自适应重构、量化模型包装、BN 折叠和校准数据缓存。
"""

from .fold_bn import fold_bn_into_conv, search_fold_and_remove_bn, search_fold_and_reset_bn
from .quant_block import BaseQuantBlock, QuantFeatureFusionBlock, specials
from .quant_layer import QuantModule, StraightThrough, UniformAffineQuantizer, lp_loss, round_ste
from .quant_model import QuantModel

__all__ = [
    "BaseQuantBlock",
    "QuantModel",
    "QuantModule",
    "QuantFeatureFusionBlock",
    "StraightThrough",
    "UniformAffineQuantizer",
    "fold_bn_into_conv",
    "lp_loss",
    "round_ste",
    "search_fold_and_remove_bn",
    "search_fold_and_reset_bn",
    "specials",
]
