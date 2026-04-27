"""SCRN-BRECQ 模型大小与理论压缩率统计工具。

当前 `quantized_scrn_brecq.pth` 仍是 PyTorch checkpoint，里面会保存 FP32 权重副本、
AdaRound alpha、scale/zero point 等调试与恢复信息。因此 checkpoint 文件大小不等于
真实部署时 bit-packed 4bit 模型大小。本文件同时报告两类指标：

1. 实际 checkpoint 文件大小。
2. 按权重 bitwidth 估算的理论部署大小。
"""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

import torch

from SCRN_BRECQ_app.scrn_brecq.quant import AdaRoundQuantizer, QuantModule


def build_model_size_report(
    model: torch.nn.Module,
    *,
    source_checkpoint_path: str | Path | None = None,
    quant_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """统计当前量化模型的大小、bit 分布和理论压缩率。

    该报告只估算权重打包后的部署大小：QuantModule 的权重按其量化 bitwidth 计入，
    非权重参数保留 FP32，权重量化 scale/zero point 按当前 tensor dtype 计入开销。
    AdaRound alpha 和 `org_weight` 只属于当前 checkpoint 恢复格式，不计入理论部署大小。
    """
    quant_modules = [(name, module) for name, module in model.named_modules() if isinstance(module, QuantModule)]
    quant_aux_param_ids = _collect_quant_aux_parameter_ids(model)

    base_parameter_count = _count_base_parameters(model, exclude_ids=quant_aux_param_ids)
    quantized_weight_parameter_count = 0
    quantized_weight_bits = 0
    quantizer_overhead_bits = 0
    weight_layer_counts_by_bit: dict[int, int] = {}
    weight_param_counts_by_bit: dict[int, int] = {}
    quantizer_overhead_bytes_by_bit: dict[int, int] = {}
    adaround_alpha_parameter_count = 0

    for _, module in quant_modules:
        quantizer = module.weight_quantizer
        bit = int(getattr(quantizer, "n_bits", 32))
        weight_params = int(module.weight.numel())
        overhead_bits = _weight_quantizer_overhead_bits(quantizer)

        quantized_weight_parameter_count += weight_params
        quantized_weight_bits += weight_params * bit
        quantizer_overhead_bits += overhead_bits
        weight_layer_counts_by_bit[bit] = weight_layer_counts_by_bit.get(bit, 0) + 1
        weight_param_counts_by_bit[bit] = weight_param_counts_by_bit.get(bit, 0) + weight_params
        quantizer_overhead_bytes_by_bit[bit] = quantizer_overhead_bytes_by_bit.get(bit, 0) + ceil(overhead_bits / 8)

        if isinstance(quantizer, AdaRoundQuantizer) and quantizer.alpha is not None:
            adaround_alpha_parameter_count += int(quantizer.alpha.numel())

    non_quantized_parameter_count = max(0, base_parameter_count - quantized_weight_parameter_count)
    fp32_model_bits = base_parameter_count * 32
    fp32_quantized_weight_bits = quantized_weight_parameter_count * 32
    estimated_packed_model_bits = quantized_weight_bits + non_quantized_parameter_count * 32 + quantizer_overhead_bits

    report: dict[str, Any] = {
        "checkpoint_files": build_checkpoint_file_size_report(
            source_checkpoint_path=source_checkpoint_path,
            quant_checkpoint_path=quant_checkpoint_path,
        ),
        "parameters": {
            "base_parameter_count_excluding_quant_aux": int(base_parameter_count),
            "quantized_weight_parameter_count": int(quantized_weight_parameter_count),
            "non_quantized_parameter_count": int(non_quantized_parameter_count),
            "quantized_weight_parameter_ratio": _safe_div(
                quantized_weight_parameter_count,
                base_parameter_count,
            ),
            "adaround_alpha_parameter_count_checkpoint_only": int(adaround_alpha_parameter_count),
            "weight_layer_counts_by_bit": _stringify_int_keys(weight_layer_counts_by_bit),
            "weight_param_counts_by_bit": _stringify_int_keys(weight_param_counts_by_bit),
        },
        "estimated_storage": {
            "fp32_model_size_bytes": _bits_to_bytes(fp32_model_bits),
            "fp32_model_size_mib": _bytes_to_mib(_bits_to_bytes(fp32_model_bits)),
            "fp32_quantized_weight_size_bytes": _bits_to_bytes(fp32_quantized_weight_bits),
            "estimated_quantized_weight_size_bytes": _bits_to_bytes(quantized_weight_bits),
            "estimated_quantized_weight_size_mib": _bytes_to_mib(_bits_to_bytes(quantized_weight_bits)),
            "estimated_weight_quantizer_overhead_bytes": _bits_to_bytes(quantizer_overhead_bits),
            "estimated_weight_quantizer_overhead_bytes_by_bit": _stringify_int_keys(quantizer_overhead_bytes_by_bit),
            "estimated_packed_model_size_bytes": _bits_to_bytes(estimated_packed_model_bits),
            "estimated_packed_model_size_mib": _bytes_to_mib(_bits_to_bytes(estimated_packed_model_bits)),
            "estimated_weight_compression_ratio": _safe_div(fp32_quantized_weight_bits, quantized_weight_bits),
            "estimated_model_compression_ratio": _safe_div(fp32_model_bits, estimated_packed_model_bits),
        },
        "notes": {
            "checkpoint_size_is_not_packed_int4_size": True,
            "adaround_alpha_and_org_weight_are_checkpoint_recovery_overhead": True,
        },
    }
    return report


def build_checkpoint_file_size_report(
    *,
    source_checkpoint_path: str | Path | None = None,
    quant_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """读取 FP32/量化 checkpoint 的实际文件大小。"""
    source_size = _file_size_bytes(source_checkpoint_path)
    quant_size = _file_size_bytes(quant_checkpoint_path)
    return {
        "source_checkpoint_file_size_bytes": source_size,
        "source_checkpoint_file_size_mib": _bytes_to_mib(source_size),
        "quant_checkpoint_file_size_bytes": quant_size,
        "quant_checkpoint_file_size_mib": _bytes_to_mib(quant_size),
        "quant_to_source_checkpoint_file_size_ratio": _safe_div(quant_size, source_size),
    }


def refresh_checkpoint_file_sizes(
    report: dict[str, Any],
    *,
    source_checkpoint_path: str | Path | None = None,
    quant_checkpoint_path: str | Path | None = None,
) -> None:
    """在 checkpoint 写出后刷新报告中的实际文件大小。"""
    report["checkpoint_files"] = build_checkpoint_file_size_report(
        source_checkpoint_path=source_checkpoint_path,
        quant_checkpoint_path=quant_checkpoint_path,
    )


def _count_base_parameters(model: torch.nn.Module, *, exclude_ids: set[int]) -> int:
    """统计原始模型参数数量，排除 AdaRound/量化 scale 等辅助参数。"""
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        parameter_id = id(parameter)
        if parameter_id in seen or parameter_id in exclude_ids:
            continue
        seen.add(parameter_id)
        total += int(parameter.numel())
    return total


def _collect_quant_aux_parameter_ids(model: torch.nn.Module) -> set[int]:
    """收集 checkpoint 恢复用的量化辅助参数，理论部署大小不计入它们。"""
    aux_ids: set[int] = set()
    for name, parameter in model.named_parameters():
        if ".weight_quantizer.alpha" in name or ".act_quantizer.delta" in name:
            aux_ids.add(id(parameter))
    return aux_ids


def _weight_quantizer_overhead_bits(quantizer: torch.nn.Module) -> int:
    """估算权重量化 scale/zero point 存储开销。"""
    total = 0
    for attr_name in ("delta", "zero_point"):
        tensor = getattr(quantizer, attr_name, None)
        if isinstance(tensor, torch.Tensor):
            total += int(tensor.numel()) * int(tensor.element_size()) * 8
    return total


def _file_size_bytes(path: str | Path | None) -> int | None:
    if path is None:
        return None
    target = Path(path)
    if not target.exists() or not target.is_file():
        return None
    return int(target.stat().st_size)


def _bits_to_bytes(bits: int | float | None) -> int | None:
    if bits is None:
        return None
    return int(ceil(float(bits) / 8.0))


def _bytes_to_mib(size_bytes: int | None) -> float | None:
    if size_bytes is None:
        return None
    return float(size_bytes) / 1024.0 / 1024.0


def _safe_div(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    denominator_value = float(denominator)
    if denominator_value == 0.0:
        return None
    return float(numerator) / denominator_value


def _stringify_int_keys(payload: dict[int, int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(payload.items())}


__all__ = [
    "build_checkpoint_file_size_report",
    "build_model_size_report",
    "refresh_checkpoint_file_sizes",
]
