"""Conv-BN 折叠工具。

SCRN 的 `FeatureFusionBlock.conv_branch` 中存在多组 `Conv2d + BatchNorm2d + ReLU`。
后训练量化通常先把 BatchNorm 的均值、方差和仿射参数吸收到前一个 Conv 权重中，
这样后续量化时只需要处理 Conv/Linear，不需要单独量化 BN。
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.init as init

from SCRN_BRECQ_app.scrn_brecq.quant.quant_layer import StraightThrough


def fold_bn_into_conv(conv_module: nn.Conv2d, bn_module: nn.BatchNorm2d) -> None:
    """把一个 BatchNorm2d 的效果折叠进前一个 Conv2d。

    折叠后的 Conv 在 eval 模式下等价于原来的 `Conv + BN`。如果 Conv 原本没有 bias，
    这里会创建一个 bias 参数来承接 BN 的平移项。
    """
    weight, bias = _fold_bn_parameters(conv_module, bn_module)
    conv_module.weight.data.copy_(weight)
    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(bias)
    else:
        conv_module.bias.data.copy_(bias)


def reset_bn(module: nn.modules.batchnorm._BatchNorm) -> None:
    """重置 BN 参数。

    当前默认流程会直接移除 BN；保留该函数是为了后续如果要对比 reset-BN 策略，
    可以复用同一工具。
    """
    if module.track_running_stats:
        module.running_mean.zero_()
        module.running_var.fill_(1 - module.eps)
    if module.affine:
        init.ones_(module.weight)
        init.zeros_(module.bias)


def is_bn(module: nn.Module | None) -> bool:
    """判断模块是否是 1D/2D BatchNorm。"""
    return isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))


def is_absorbing(module: nn.Module | None) -> bool:
    """判断模块是否能吸收 BN 参数。

    这里保持 BRECQ 原始接口风格，支持 Conv2d 和 Linear；SCRN 中实际需要的是
    Conv2d，因为 `conv_branch` 使用的是 `Conv2d + BatchNorm2d`。
    """
    return isinstance(module, (nn.Conv2d, nn.Linear))


def search_fold_and_remove_bn(model: nn.Module) -> nn.Module | None:
    """递归查找相邻的 Conv/Linear + BN，并把 BN 替换为 StraightThrough。

    返回当前层级最后一个可吸收模块，供父级递归继续判断相邻关系。SCRN 的 BN 都在
    `nn.Sequential` 内，递归遍历可以覆盖这些情况。
    """
    model.eval()
    previous: nn.Module | None = None
    for name, child in model.named_children():
        if is_bn(child) and is_absorbing(previous):
            if not isinstance(previous, nn.Conv2d):
                raise TypeError("Only Conv2d + BatchNorm2d folding is supported in SCRN-BRECQ.")
            if not isinstance(child, nn.BatchNorm2d):
                raise TypeError("Only BatchNorm2d folding is supported for SCRN Conv branches.")
            fold_bn_into_conv(previous, child)
            setattr(model, name, StraightThrough())
        elif is_absorbing(child):
            previous = child
        else:
            previous = search_fold_and_remove_bn(child)
    return previous


def search_fold_and_reset_bn(model: nn.Module) -> None:
    """递归折叠 BN 后重置 BN 参数，但不替换模块。

    本函数不是默认量化路径，只作为调试/对照工具保留。
    """
    model.eval()
    previous: nn.Module | None = None
    for _, child in model.named_children():
        if is_bn(child) and isinstance(previous, nn.Conv2d) and isinstance(child, nn.BatchNorm2d):
            fold_bn_into_conv(previous, child)
            reset_bn(child)
        else:
            search_fold_and_reset_bn(child)
        previous = child


def _fold_bn_parameters(conv_module: nn.Conv2d, bn_module: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """计算折叠后的 Conv 权重和 bias。"""
    weight = conv_module.weight.data
    running_mean = bn_module.running_mean
    running_var = bn_module.running_var
    safe_std = torch.sqrt(running_var + bn_module.eps)
    weight_view = (conv_module.out_channels, 1, 1, 1)

    if bn_module.affine:
        folded_weight = weight * (bn_module.weight / safe_std).view(weight_view)
        folded_bias = bn_module.bias - bn_module.weight * running_mean / safe_std
        if conv_module.bias is not None:
            folded_bias = bn_module.weight * conv_module.bias / safe_std + folded_bias
    else:
        folded_weight = weight / safe_std.view(weight_view)
        folded_bias = -running_mean / safe_std
        if conv_module.bias is not None:
            folded_bias = conv_module.bias / safe_std + folded_bias

    return folded_weight, folded_bias

