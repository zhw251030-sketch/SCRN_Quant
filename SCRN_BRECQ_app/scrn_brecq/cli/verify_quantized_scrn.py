"""验证已保存 SCRN-BRECQ checkpoint 的量化真实性。

本脚本不重新训练或重构模型，只读取 `quantized_scrn_brecq.pth`，恢复 QuantModel，
并检查权重量化器是否真的落在目标 bitwidth 的整数网格上。
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Mapping

import numpy as np
import torch

from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn import (
    build_quant_model_from_checkpoint,
    load_eval_arrays,
    load_quant_checkpoint,
    normalize_quant_config,
    predict_array,
    require_file,
    resolve_eval_paths,
    restore_quantizer_state_shapes,
    select_device,
)
from SCRN_BRECQ_app.scrn_brecq.quant import AdaRoundQuantizer, QuantModule
from SCRN_BRECQ_app.scrn_brecq.quant.activation_precision import apply_activation_bitwidth_overrides, summarize_activation_bitwidths
from SCRN_BRECQ_app.scrn_brecq.utils import build_model_size_report
from SCRN_BRECQ_app.scrn_repro.training import write_json
from SCRN_BRECQ_app.scrn_repro.utils import snr_db, ssim_score


def build_parser() -> argparse.ArgumentParser:
    """构建量化真实性验证参数解析器。"""
    parser = argparse.ArgumentParser(description="Verify a saved SCRN-BRECQ quantized checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to quantized_scrn_brecq.pth")
    parser.add_argument("--eval-clean-path", default=None, help="Clean reference .npy; defaults to checkpoint quant_config")
    parser.add_argument("--eval-input-path", default=None, help="Degraded input .npy; defaults to checkpoint quant_config")
    parser.add_argument("--output-json", default=None, help="Optional path to save the verification JSON report")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--max-offenders", type=int, default=20, help="Maximum level offender rows to include")
    return parser


def main() -> None:
    """命令行主流程。"""
    args = build_parser().parse_args()
    checkpoint_path = require_file(args.checkpoint, "quantized checkpoint")
    checkpoint = load_quant_checkpoint(checkpoint_path)
    quant_config = normalize_quant_config(checkpoint.get("quant_config", {}))
    clean_path, input_path = resolve_eval_paths(args, quant_config)
    device = select_device(args.device)

    quant_model = build_quant_model_from_checkpoint(checkpoint)
    state_dict = checkpoint["quant_model_state_dict"]
    restore_quantizer_state_shapes(quant_model, state_dict)
    quant_model.load_state_dict(state_dict, strict=True)
    quant_model.to(device)
    quant_model.eval()

    final_state = checkpoint.get("final_quant_state", {})
    weight_quant = bool(final_state.get("weight_quant", True))
    act_quant = bool(final_state.get("act_quant", quant_config.get("act_quant", False)))
    if act_quant:
        quant_model.disable_network_output_quantization()

    layer_report = inspect_weight_quantization(quant_model, max_offenders=int(args.max_offenders))
    activation_report = inspect_activation_quantization(quant_model)
    activation_report.update(
        apply_activation_bitwidth_overrides(
            quant_model,
            quant_config["activation_bitwidth_overrides"],
        )
    )
    clean, degraded = load_eval_arrays(clean_path, input_path)
    output_report = compare_fp32_and_quant_outputs(
        quant_model,
        clean=clean,
        degraded=degraded,
        device=device,
        weight_quant=weight_quant,
        act_quant=act_quant,
    )

    checks = {
        "has_quant_modules": layer_report["quant_modules"] > 0,
        "no_level_offenders": layer_report["level_offender_count"] == 0,
        "output_changed_after_quantization": output_report["fp32_quant_max_abs_diff"] > 0.0,
        "activation_quantizers_restored": (not act_quant) or activation_report["missing_activation_state_count"] == 0,
    }
    report = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "final_quant_state": {"weight_quant": weight_quant, "act_quant": act_quant},
        "quant_config": {
            "n_bits_w": quant_config["n_bits_w"],
            "n_bits_a": quant_config["n_bits_a"],
            "channel_wise": quant_config["channel_wise"],
            "scale_method": quant_config["scale_method"],
            "act_quant": quant_config["act_quant"],
            "disable_8bit_head_stem": quant_config["disable_8bit_head_stem"],
            "activation_bitwidth_overrides": quant_config["activation_bitwidth_overrides"],
        },
        "eval_paths": {"clean": str(clean_path), "input": str(input_path)},
        "layer_quantization": layer_report,
        "activation_quantization": activation_report,
        "model_size": build_model_size_report(
            quant_model,
            source_checkpoint_path=checkpoint.get("source_checkpoint"),
            quant_checkpoint_path=checkpoint_path,
        ),
        "output_comparison": output_report,
        "checks": checks,
        "passed": all(checks.values()),
    }

    if args.output_json is not None:
        write_json(args.output_json, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def inspect_activation_quantization(quant_model: torch.nn.Module) -> dict[str, Any]:
    """检查 activation quantizer 的 delta/zero_point 是否从 checkpoint 恢复。"""
    rows: list[dict[str, Any]] = []
    initialized_count = 0
    delta_count = 0
    zero_point_count = 0
    learnable_delta_count = 0
    quant_modules = [(name, module) for name, module in quant_model.named_modules() if isinstance(module, QuantModule)]
    for name, module in quant_modules:
        quantizer = module.act_quantizer
        delta = getattr(quantizer, "delta", None)
        zero_point = getattr(quantizer, "zero_point", None)
        has_delta = delta is not None
        has_zero_point = zero_point is not None
        is_inited = bool(getattr(quantizer, "inited", False))
        if is_inited:
            initialized_count += 1
        if has_delta:
            delta_count += 1
        if has_zero_point:
            zero_point_count += 1
        if isinstance(delta, torch.nn.Parameter):
            learnable_delta_count += 1
        if has_delta and not has_zero_point:
            rows.append({"name": name, "reason": "activation delta exists but zero_point is missing"})
        elif has_delta and not is_inited:
            rows.append({"name": name, "reason": "activation quantizer is not marked initialized"})

    report = {
        "quant_modules": len(quant_modules),
        "initialized_activation_quantizers": initialized_count,
        "activation_delta_count": delta_count,
        "activation_zero_point_count": zero_point_count,
        "learnable_activation_delta_count": learnable_delta_count,
        "missing_activation_state_count": len(rows),
        "missing_activation_state": rows,
    }
    report.update(summarize_activation_bitwidths(quant_model))
    return report


def inspect_weight_quantization(quant_model: torch.nn.Module, *, max_offenders: int) -> dict[str, Any]:
    """检查每个 QuantModule 权重量化后的整数等级数量。"""
    quant_modules = [(name, module) for name, module in quant_model.named_modules() if isinstance(module, QuantModule)]
    bit_counts: dict[int, int] = {}
    max_unique_by_bit: dict[int, int] = {}
    min_int_by_bit: dict[int, int] = {}
    max_int_by_bit: dict[int, int] = {}
    offenders: list[dict[str, Any]] = []
    initialized = 0
    adaround_count = 0

    for index, (name, module) in enumerate(quant_modules):
        quantizer = module.weight_quantizer
        bit = int(getattr(quantizer, "n_bits", -1))
        n_levels = int(getattr(quantizer, "n_levels", 2**bit))
        bit_counts[bit] = bit_counts.get(bit, 0) + 1
        if isinstance(quantizer, AdaRoundQuantizer):
            adaround_count += 1

        delta = getattr(quantizer, "delta", None)
        zero_point = getattr(quantizer, "zero_point", None)
        if delta is None or zero_point is None:
            with torch.no_grad():
                _ = quantizer(module.weight)
            delta = getattr(quantizer, "delta", None)
            zero_point = getattr(quantizer, "zero_point", None)
            if delta is None or zero_point is None:
                add_offender(
                    offenders,
                    max_offenders=max_offenders,
                    row={
                        "index": index,
                        "name": name,
                        "bit": bit,
                        "reason": "weight quantizer is not initialized",
                    },
                )
                continue
        initialized += 1

        with torch.no_grad():
            quant_weight = quantizer(module.weight)
            quant_int = torch.round(quant_weight / delta + zero_point).detach().cpu()
        rows = quant_int.reshape(quant_int.shape[0], -1) if quant_int.ndim >= 2 else quant_int.reshape(1, -1)
        unique_levels = [int(torch.unique(row).numel()) for row in rows]
        max_unique = max(unique_levels)
        min_int = int(quant_int.min().item())
        max_int = int(quant_int.max().item())
        max_unique_by_bit[bit] = max(max_unique_by_bit.get(bit, 0), max_unique)
        min_int_by_bit[bit] = min(min_int_by_bit.get(bit, min_int), min_int)
        max_int_by_bit[bit] = max(max_int_by_bit.get(bit, max_int), max_int)

        if max_unique > n_levels or min_int < 0 or max_int > n_levels - 1:
            add_offender(
                offenders,
                max_offenders=max_offenders,
                row={
                    "index": index,
                    "name": name,
                    "bit": bit,
                    "n_levels": n_levels,
                    "max_unique_levels_per_output_channel": max_unique,
                    "min_int": min_int,
                    "max_int": max_int,
                },
            )

    return {
        "quant_modules": len(quant_modules),
        "adaround_modules": adaround_count,
        "initialized_weight_quantizers": initialized,
        "weight_bit_counts": stringify_int_keys(bit_counts),
        "max_unique_int_levels_per_channel_by_bit": stringify_int_keys(max_unique_by_bit),
        "min_int_by_bit": stringify_int_keys(min_int_by_bit),
        "max_int_by_bit": stringify_int_keys(max_int_by_bit),
        "level_offender_count": len(offenders),
        "level_offenders": offenders,
    }


def compare_fp32_and_quant_outputs(
    quant_model: torch.nn.Module,
    *,
    clean: np.ndarray,
    degraded: np.ndarray,
    device: torch.device,
    weight_quant: bool,
    act_quant: bool,
) -> dict[str, float]:
    """比较同一 checkpoint 下 FP32 路径和量化路径输出。"""
    quant_model.set_quant_state(False, False)
    fp32_prediction, fp32_seconds = predict_array(quant_model, degraded, device)
    quant_model.set_quant_state(weight_quant, act_quant)
    quant_prediction, quant_seconds = predict_array(quant_model, degraded, device)
    diff = fp32_prediction - quant_prediction
    return {
        "input_snr_db": snr_db(degraded, clean),
        "input_ssim": ssim_score(degraded, clean),
        "fp32_snr_db": snr_db(fp32_prediction, clean),
        "fp32_ssim": ssim_score(fp32_prediction, clean),
        "quant_snr_db": snr_db(quant_prediction, clean),
        "quant_ssim": ssim_score(quant_prediction, clean),
        "fp32_inference_seconds": float(fp32_seconds),
        "quant_inference_seconds": float(quant_seconds),
        "fp32_quant_max_abs_diff": float(np.max(np.abs(diff))),
        "fp32_quant_mean_abs_diff": float(np.mean(np.abs(diff))),
        "fp32_quant_mse": float(np.mean(diff**2)),
    }


def add_offender(offenders: list[dict[str, Any]], *, max_offenders: int, row: dict[str, Any]) -> None:
    """限制报告中保存的异常层数量，避免 JSON 过大。"""
    if len(offenders) < max(0, max_offenders):
        offenders.append(row)


def stringify_int_keys(payload: Mapping[int, int]) -> dict[str, int]:
    """JSON 对象键统一转成字符串，避免不同解析器处理整数键不一致。"""
    return {str(key): int(value) for key, value in sorted(payload.items())}


if __name__ == "__main__":
    main()
