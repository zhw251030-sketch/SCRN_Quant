"""BRECQ 量化算法子包。

后续会在这里逐步迁移并重写 BRECQ 的核心组件：
量化层、AdaRound、自适应重构、量化模型包装、BN 折叠和校准数据缓存。
"""

from .adaptive_rounding import AdaRoundQuantizer
from .activation_diagnostics import (
    build_activation_diagnostics_report,
    collect_activation_stats,
    collect_quantizer_rows,
    summarize_activation_quantizers,
)
from .activation_precision import (
    apply_activation_bitwidth_overrides,
    normalize_activation_bitwidth_overrides,
    summarize_activation_bitwidths,
)
from .block_recon import BlockLossFunction, LinearTempDecay, block_reconstruction, reconstruction_loss
from .data_utils import (
    DataSaverHook,
    GetLayerGrad,
    GetLayerInpOut,
    GradSaverHook,
    StopForwardException,
    quantize_model_till,
    save_grad_data,
    save_inp_oup_data,
)
from .fold_bn import fold_bn_into_conv, search_fold_and_remove_bn, search_fold_and_reset_bn
from .layer_recon import LayerLossFunction, layer_reconstruction
from .quant_block import BaseQuantBlock, QuantFeatureFusionBlock, specials
from .quant_layer import QuantModule, StraightThrough, UniformAffineQuantizer, lp_loss, round_ste
from .quant_model import QuantModel

__all__ = [
    "AdaRoundQuantizer",
    "apply_activation_bitwidth_overrides",
    "BaseQuantBlock",
    "BlockLossFunction",
    "build_activation_diagnostics_report",
    "collect_activation_stats",
    "collect_quantizer_rows",
    "DataSaverHook",
    "GetLayerGrad",
    "GetLayerInpOut",
    "GradSaverHook",
    "LayerLossFunction",
    "LinearTempDecay",
    "normalize_activation_bitwidth_overrides",
    "QuantModel",
    "QuantModule",
    "QuantFeatureFusionBlock",
    "StraightThrough",
    "StopForwardException",
    "UniformAffineQuantizer",
    "block_reconstruction",
    "fold_bn_into_conv",
    "layer_reconstruction",
    "lp_loss",
    "quantize_model_till",
    "reconstruction_loss",
    "round_ste",
    "search_fold_and_remove_bn",
    "search_fold_and_reset_bn",
    "save_grad_data",
    "save_inp_oup_data",
    "specials",
    "summarize_activation_bitwidths",
    "summarize_activation_quantizers",
]
