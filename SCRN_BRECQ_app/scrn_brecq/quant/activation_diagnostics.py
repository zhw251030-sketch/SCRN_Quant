"""Activation quantization diagnostics for SCRN-BRECQ.

The utilities in this module inspect already-built ``QuantModel`` instances.
They do not change quantization algorithms or reconstruction behavior; they only
summarize quantizer state and collect forward-hook statistics.
"""

from __future__ import annotations

from contextlib import contextmanager
import re
from typing import Any, Iterable

import torch
from torch import nn

from SCRN_BRECQ_app.scrn_brecq.quant.quant_layer import QuantModule


EPS = 1e-12
TORCH_QUANTILE_MAX_EXACT_VALUES = 500_000


def collect_quantizer_rows(model: nn.Module) -> list[dict[str, Any]]:
    """Return one JSON/CSV-friendly row per ``QuantModule`` activation quantizer."""
    rows: list[dict[str, Any]] = []
    for index, (name, module) in enumerate(_named_quant_modules(model)):
        quantizer = module.act_quantizer
        delta = getattr(quantizer, "delta", None)
        zero_point = getattr(quantizer, "zero_point", None)
        delta_stats = _tensor_stats(delta)
        zero_point_stats = _tensor_stats(zero_point)
        structure = infer_quantizer_structure(name)
        rows.append(
            {
                "index": index,
                "name": name,
                "stage": structure["stage"],
                "branch": structure["branch"],
                "role": structure["role"],
                "module_type": infer_module_type(module),
                "weight_bit": int(getattr(module.weight_quantizer, "n_bits", -1)),
                "act_bit": int(getattr(quantizer, "n_bits", -1)),
                "act_disabled": bool(getattr(module, "disable_act_quant", False)),
                "act_inited": bool(getattr(quantizer, "inited", False)),
                "act_quantizer_leaf_param": bool(getattr(quantizer, "leaf_param", False)),
                "act_delta_exists": delta is not None,
                "act_delta_learnable": isinstance(delta, nn.Parameter),
                "act_delta_shape": delta_stats["shape"],
                "act_delta_min": delta_stats["min"],
                "act_delta_max": delta_stats["max"],
                "act_delta_mean": delta_stats["mean"],
                "act_delta_non_positive_elements": _non_positive_count(delta),
                "act_zero_point_exists": zero_point is not None,
                "act_zero_point_shape": zero_point_stats["shape"],
                "act_zero_point_min": zero_point_stats["min"],
                "act_zero_point_max": zero_point_stats["max"],
                "act_zero_point_mean": zero_point_stats["mean"],
            }
        )
    return rows


def summarize_activation_quantizers(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize activation quantizer state rows."""
    materialized = list(rows)
    delta_values = [value for row in materialized for value in _row_min_max(row, "act_delta")]
    zero_point_values = [value for row in materialized for value in _row_min_max(row, "act_zero_point")]
    offender_layers = [
        {
            "index": row["index"],
            "name": row["name"],
            "stage": row["stage"],
            "branch": row["branch"],
            "role": row["role"],
            "act_delta_min": row["act_delta_min"],
            "act_delta_max": row["act_delta_max"],
            "act_delta_non_positive_elements": row["act_delta_non_positive_elements"],
        }
        for row in materialized
        if int(row["act_delta_non_positive_elements"]) > 0
    ]
    return {
        "quant_modules": len(materialized),
        "activation_quantizers": len(materialized),
        "activation_delta_count": sum(1 for row in materialized if row["act_delta_exists"]),
        "activation_zero_point_count": sum(1 for row in materialized if row["act_zero_point_exists"]),
        "initialized_activation_quantizers": sum(1 for row in materialized if row["act_inited"]),
        "learnable_activation_delta_count": sum(1 for row in materialized if row["act_delta_learnable"]),
        "disabled_activation_quantizers": sum(1 for row in materialized if row["act_disabled"]),
        "delta_min": min(delta_values) if delta_values else None,
        "delta_max": max(delta_values) if delta_values else None,
        "zero_point_min": min(zero_point_values) if zero_point_values else None,
        "zero_point_max": max(zero_point_values) if zero_point_values else None,
        "non_positive_delta_count": len(offender_layers),
        "non_positive_delta_elements": sum(int(row["act_delta_non_positive_elements"]) for row in materialized),
        "offender_layers": offender_layers,
    }


def collect_activation_stats(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    weight_quant: bool = True,
) -> list[dict[str, Any]]:
    """Collect pre-activation-quantization output stats for each ``QuantModule``.

    The model is run with weight quantization enabled by default and activation
    quantization disabled. Each module output is then manually fake-quantized
    with that module's existing activation quantizer state, when available.
    """
    rows: list[dict[str, Any]] = []
    handles: list[Any] = []
    named_modules = list(_named_quant_modules(model))

    def make_hook(index: int, name: str, module: QuantModule):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            if not isinstance(output, torch.Tensor):
                return
            rows.append(_activation_row(index, name, module, output.detach()))

        return hook

    for index, (name, module) in enumerate(named_modules):
        handles.append(module.register_forward_hook(make_hook(index, name, module)))

    try:
        with _temporary_quant_state(model, weight_quant=weight_quant, act_quant=False):
            with torch.no_grad():
                _ = model(inputs)
    finally:
        for handle in handles:
            handle.remove()

    rows.sort(key=lambda row: int(row["index"]))
    return rows


def build_activation_diagnostics_report(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    weight_quant: bool = True,
) -> dict[str, Any]:
    """Build the complete E001 activation diagnostics payload."""
    quantizer_rows = collect_quantizer_rows(model)
    activation_stats = collect_activation_stats(model, inputs, weight_quant=weight_quant)
    summary = summarize_activation_quantizers(quantizer_rows)
    summary.update(summarize_activation_stats(activation_stats))
    return {
        "summary": summary,
        "quantizers": quantizer_rows,
        "activation_stats": activation_stats,
        "offender_layers": summary["offender_layers"],
    }


def summarize_activation_stats(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize activation distribution and fake-quant rows."""
    materialized = list(rows)
    mse_values = [float(row["fake_quant_mse"]) for row in materialized if row["fake_quant_mse"] is not None]
    level_values = [int(row["effective_int_levels"]) for row in materialized if row["effective_int_levels"] is not None]
    outlier_values = [float(row["absmax_over_p99"]) for row in materialized if row["absmax_over_p99"] is not None]
    return {
        "activation_stat_count": len(materialized),
        "fake_quant_mse_max": max(mse_values) if mse_values else None,
        "fake_quant_mse_mean": sum(mse_values) / len(mse_values) if mse_values else None,
        "effective_int_levels_min": min(level_values) if level_values else None,
        "effective_int_levels_max": max(level_values) if level_values else None,
        "absmax_over_p99_max": max(outlier_values) if outlier_values else None,
        "top_outlier_layers": _ranked_layer_rows(materialized, "absmax_over_p99", reverse=True),
        "lowest_effective_level_layers": _ranked_layer_rows(materialized, "effective_int_levels", reverse=False),
        "worst_fake_quant_mse_layers": _ranked_layer_rows(materialized, "fake_quant_mse", reverse=True),
        "worst_relative_mse_layers": _ranked_layer_rows(materialized, "fake_quant_relative_mse", reverse=True),
        "top_per_channel_imbalance_layers": _ranked_layer_rows(materialized, "per_channel_absmax_ratio", reverse=True),
        "branch_summary": _group_activation_stats(materialized, "branch"),
        "stage_summary": _group_activation_stats(materialized, "stage"),
        "role_summary": _group_activation_stats(materialized, "role"),
        "module_type_summary": _group_activation_stats(materialized, "module_type"),
        "conv2d_range_summary": _conv2d_range_summary(materialized),
    }


def infer_quantizer_structure(name: str) -> dict[str, str]:
    """Infer SCRN structural labels from a QuantModule path."""
    stage_match = re.search(r"(?:^|\.)stage(\d+)(?:\.|$)", name)
    is_stage_output_conv = re.search(r"(?:^|\.)stage\d+\.1$", name) is not None
    if stage_match:
        stage = f"stage{stage_match.group(1)}"
    elif "head" in name:
        stage = "head"
    elif "tail" in name or "output" in name:
        stage = "tail"
    else:
        stage = _top_level_stage(name)
    if is_stage_output_conv:
        branch = "stage_output"
    elif "trans_branch" in name:
        branch = "transformer"
    elif "cnn_branch" in name or "conv_branch" in name:
        branch = "cnn"
    elif "split_proj" in name or "merge_proj" in name:
        branch = "fusion"
    elif ".attn." in name or name.endswith(".attn"):
        branch = "attention"
    elif ".mlp." in name or name.endswith(".mlp"):
        branch = "mlp"
    elif "head" in name:
        branch = "head"
    elif "tail" in name or "output" in name:
        branch = "tail"
    else:
        branch = "unknown"

    if is_stage_output_conv:
        role = "stage_output_conv"
    elif ".attn.qkv" in name or name.endswith(".qkv"):
        role = "attention_qkv"
    elif ".attn.proj" in name or name.endswith(".proj"):
        role = "attention_proj"
    elif "split_proj" in name:
        role = "split_proj"
    elif "merge_proj" in name:
        role = "merge_proj"
    elif ".mlp." in name:
        role = "mlp"
    elif "conv" in name:
        role = "conv"
    elif "head" in name:
        role = "head"
    elif "tail" in name or "output" in name:
        role = "tail"
    else:
        role = "unknown"
    return {"stage": stage, "branch": branch, "role": role}


def infer_module_type(module: QuantModule) -> str:
    """Return the wrapped module family from the quantized weight shape."""
    if module.weight.ndim == 4:
        return "Conv2d"
    if module.weight.ndim == 2:
        return "Linear"
    return f"weight_{module.weight.ndim}d"


def _activation_row(index: int, name: str, module: QuantModule, output: torch.Tensor) -> dict[str, Any]:
    structure = infer_quantizer_structure(name)
    value_stats = _activation_value_stats(output)
    quant_stats = _fake_quant_stats(output, module.act_quantizer)
    return {
        "index": index,
        "name": name,
        "stage": structure["stage"],
        "branch": structure["branch"],
        "role": structure["role"],
        "module_type": infer_module_type(module),
        "shape": list(output.shape),
        **value_stats,
        **quant_stats,
    }


def _activation_value_stats(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().float().reshape(-1)
    abs_values = values.abs()
    absmax = float(abs_values.max().item()) if values.numel() else None
    quantile_levels = (0.99, 0.999, 0.9999, 0.99999)
    value_quantiles = _quantiles(values, quantile_levels)
    abs_quantiles = _quantiles(abs_values, quantile_levels)
    p99_abs, p999_abs, p9999_abs, p99999_abs = abs_quantiles
    return {
        "numel": int(values.numel()),
        "min": _float(values.min()) if values.numel() else None,
        "max": _float(values.max()) if values.numel() else None,
        "mean": _float(values.mean()) if values.numel() else None,
        "std": _float(values.std(unbiased=False)) if values.numel() else None,
        "p99": value_quantiles[0],
        "p99_9": value_quantiles[1],
        "p99_99": value_quantiles[2],
        "p99_999": value_quantiles[3],
        "abs_p99": p99_abs,
        "abs_p99_9": p999_abs,
        "abs_p99_99": p9999_abs,
        "abs_p99_999": p99999_abs,
        "absmax": absmax,
        "absmax_over_p99": _safe_ratio(absmax, p99_abs),
        "absmax_over_p99_9": _safe_ratio(absmax, p999_abs),
        "absmax_over_p99_99": _safe_ratio(absmax, p9999_abs),
        "absmax_over_p99_999": _safe_ratio(absmax, p99999_abs),
        **_per_channel_absmax_stats(tensor),
    }


def _fake_quant_stats(tensor: torch.Tensor, quantizer: nn.Module) -> dict[str, Any]:
    delta = getattr(quantizer, "delta", None)
    zero_point = getattr(quantizer, "zero_point", None)
    if delta is None or zero_point is None or not bool(getattr(quantizer, "inited", False)):
        return _skipped_fake_quant_stats("activation_quantizer_not_initialized")

    try:
        values = tensor.detach()
        delta_tensor = _detach_tensor(delta).to(device=values.device, dtype=values.dtype)
        zero_point_tensor = _detach_tensor(zero_point).to(device=values.device, dtype=values.dtype)
        if bool((delta_tensor == 0).any().item()):
            return _skipped_fake_quant_stats("zero_delta")
        sampled = False
        if (
            values.numel() > TORCH_QUANTILE_MAX_EXACT_VALUES
            and delta_tensor.numel() == 1
            and zero_point_tensor.numel() == 1
        ):
            values = _sample_large_tensor(values)
            sampled = True
        q_int = torch.round(values / delta_tensor) + zero_point_tensor
        q_int = torch.clamp(q_int, 0, int(getattr(quantizer, "n_levels", 2 ** getattr(quantizer, "n_bits", 8))) - 1)
        quantized = (q_int - zero_point_tensor) * delta_tensor
        diff = (quantized - values).float()
        values_float = values.float()
        mse = float(diff.pow(2).mean().item())
        mae = float(diff.abs().mean().item())
        max_abs = float(diff.abs().max().item())
        denom = float(values_float.pow(2).mean().item())
        return {
            "fake_quant_skipped": False,
            "fake_quant_skip_reason": None,
            "fake_quant_mse": mse,
            "fake_quant_mae": mae,
            "fake_quant_max_abs": max_abs,
            "fake_quant_relative_mse": _safe_ratio(mse, denom),
            "effective_int_levels": int(torch.unique(q_int.detach().cpu()).numel()),
            "int_min": int(q_int.min().item()),
            "int_max": int(q_int.max().item()),
            "fake_quant_sampled": sampled,
            "fake_quant_sample_count": int(values.numel()),
        }
    except Exception as exc:  # pragma: no cover - keeps CLI diagnostic robust.
        return _skipped_fake_quant_stats(f"{type(exc).__name__}: {exc}")


def _skipped_fake_quant_stats(reason: str) -> dict[str, Any]:
    return {
        "fake_quant_skipped": True,
        "fake_quant_skip_reason": reason,
        "fake_quant_mse": None,
        "fake_quant_mae": None,
        "fake_quant_max_abs": None,
        "fake_quant_relative_mse": None,
        "effective_int_levels": None,
        "int_min": None,
        "int_max": None,
        "fake_quant_sampled": None,
        "fake_quant_sample_count": None,
    }


def _per_channel_absmax_stats(tensor: torch.Tensor) -> dict[str, Any]:
    channel_dim = _activation_channel_dim(tensor.ndim)
    if channel_dim is None:
        return _skipped_per_channel_stats("unsupported_shape")
    if tensor.shape[channel_dim] <= 0:
        return _skipped_per_channel_stats("empty_channel_dim")

    values = tensor.detach().float().abs()
    per_channel = torch.movedim(values, channel_dim, 0).reshape(values.shape[channel_dim], -1).max(dim=1).values
    if per_channel.numel() == 0:
        return _skipped_per_channel_stats("empty_channel_values")
    max_value = float(per_channel.max().item())
    median_value = float(torch.median(per_channel).item())
    return {
        "per_channel_axis": int(channel_dim),
        "per_channel_count": int(per_channel.numel()),
        "per_channel_absmax_max": max_value,
        "per_channel_absmax_median": median_value,
        "per_channel_absmax_ratio": _safe_ratio(max_value, median_value),
        "per_channel_absmax_skip_reason": None,
    }


def _activation_channel_dim(ndim: int) -> int | None:
    if ndim == 4:
        return 1
    if ndim == 3:
        return 2
    if ndim == 2:
        return 1
    return None


def _skipped_per_channel_stats(reason: str) -> dict[str, Any]:
    return {
        "per_channel_axis": None,
        "per_channel_count": None,
        "per_channel_absmax_max": None,
        "per_channel_absmax_median": None,
        "per_channel_absmax_ratio": None,
        "per_channel_absmax_skip_reason": reason,
    }


def _ranked_layer_rows(
    rows: Iterable[dict[str, Any]],
    metric: str,
    *,
    reverse: bool,
    limit: int = 10,
) -> list[dict[str, Any]]:
    ranked = sorted(
        (row for row in rows if row.get(metric) is not None),
        key=lambda row: float(row[metric]),
        reverse=reverse,
    )[:limit]
    return [_layer_summary_row(row, metric) for row in ranked]


def _layer_summary_row(row: dict[str, Any], metric: str) -> dict[str, Any]:
    summary = {
        "index": row.get("index"),
        "name": row.get("name"),
        "stage": row.get("stage"),
        "branch": row.get("branch"),
        "role": row.get("role"),
        "module_type": row.get("module_type"),
    }
    summary[metric] = row.get(metric)
    return summary


def _group_activation_stats(rows: Iterable[dict[str, Any]], group_key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(group_key) or "unknown"), []).append(row)
    return {name: _summarize_activation_group(group_rows) for name, group_rows in sorted(groups.items())}


def _conv2d_range_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    conv2d_rows = [row for row in rows if row.get("module_type") == "Conv2d"]
    return {
        "overall": _summarize_activation_group(conv2d_rows),
        "by_stage": _group_activation_stats(conv2d_rows, "stage"),
        "by_branch": _group_activation_stats(conv2d_rows, "branch"),
        "by_role": _group_activation_stats(conv2d_rows, "role"),
        "by_module_type": _group_activation_stats(conv2d_rows, "module_type"),
    }


def _summarize_activation_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fake_quant_mse = _numeric_values(rows, "fake_quant_mse")
    relative_mse = _numeric_values(rows, "fake_quant_relative_mse")
    effective_levels = _numeric_values(rows, "effective_int_levels")
    outlier_ratio = _numeric_values(rows, "absmax_over_p99")
    outlier_ratio_p999 = _numeric_values(rows, "absmax_over_p99_9")
    outlier_ratio_p9999 = _numeric_values(rows, "absmax_over_p99_99")
    outlier_ratio_p99999 = _numeric_values(rows, "absmax_over_p99_999")
    channel_ratio = _numeric_values(rows, "per_channel_absmax_ratio")
    return {
        "count": len(rows),
        "fake_quant_mse_mean": _mean(fake_quant_mse),
        "fake_quant_mse_max": max(fake_quant_mse) if fake_quant_mse else None,
        "fake_quant_relative_mse_mean": _mean(relative_mse),
        "fake_quant_relative_mse_max": max(relative_mse) if relative_mse else None,
        "effective_int_levels_min": min(effective_levels) if effective_levels else None,
        "effective_int_levels_mean": _mean(effective_levels),
        "absmax_over_p99_mean": _mean(outlier_ratio),
        "absmax_over_p99_max": max(outlier_ratio) if outlier_ratio else None,
        "absmax_over_p99_9_mean": _mean(outlier_ratio_p999),
        "absmax_over_p99_9_max": max(outlier_ratio_p999) if outlier_ratio_p999 else None,
        "absmax_over_p99_99_mean": _mean(outlier_ratio_p9999),
        "absmax_over_p99_99_max": max(outlier_ratio_p9999) if outlier_ratio_p9999 else None,
        "absmax_over_p99_999_mean": _mean(outlier_ratio_p99999),
        "absmax_over_p99_999_max": max(outlier_ratio_p99999) if outlier_ratio_p99999 else None,
        "per_channel_absmax_ratio_mean": _mean(channel_ratio),
        "per_channel_absmax_ratio_max": max(channel_ratio) if channel_ratio else None,
    }


def _numeric_values(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


@contextmanager
def _temporary_quant_state(model: nn.Module, *, weight_quant: bool, act_quant: bool):
    quant_modules = [module for _, module in _named_quant_modules(model)]
    previous_states = [
        (module, bool(getattr(module, "use_weight_quant", False)), bool(getattr(module, "use_act_quant", False)))
        for module in quant_modules
    ]
    if hasattr(model, "set_quant_state"):
        model.set_quant_state(weight_quant, act_quant)
    else:
        for module in quant_modules:
            module.set_quant_state(weight_quant, act_quant)
    try:
        yield
    finally:
        for module, use_weight_quant, use_act_quant in previous_states:
            module.set_quant_state(use_weight_quant, use_act_quant)


def _named_quant_modules(model: nn.Module) -> list[tuple[str, QuantModule]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, QuantModule)]


def _tensor_stats(tensor: torch.Tensor | nn.Parameter | None) -> dict[str, Any]:
    if tensor is None:
        return {"shape": None, "min": None, "max": None, "mean": None}
    detached = _detach_tensor(tensor).float().reshape(-1)
    if detached.numel() == 0:
        return {"shape": list(tensor.shape), "min": None, "max": None, "mean": None}
    return {
        "shape": list(tensor.shape),
        "min": _float(detached.min()),
        "max": _float(detached.max()),
        "mean": _float(detached.mean()),
    }


def _detach_tensor(tensor: torch.Tensor | nn.Parameter) -> torch.Tensor:
    return tensor.detach() if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)


def _non_positive_count(tensor: torch.Tensor | nn.Parameter | None) -> int:
    if tensor is None:
        return 0
    return int((_detach_tensor(tensor) <= 0).sum().item())


def _row_min_max(row: dict[str, Any], prefix: str) -> list[float]:
    values = []
    for key in (f"{prefix}_min", f"{prefix}_max"):
        value = row.get(key)
        if value is not None:
            values.append(float(value))
    return values


def _quantile(values: torch.Tensor, q: float) -> float | None:
    return _quantiles(values, (q,))[0]


def _quantiles(values: torch.Tensor, qs: tuple[float, ...]) -> list[float | None]:
    if values.numel() == 0:
        return [None for _ in qs]
    flattened = values.reshape(-1)
    if values.numel() > TORCH_QUANTILE_MAX_EXACT_VALUES:
        flattened = _sample_large_tensor(flattened)
    q_tensor = torch.tensor(qs, device=flattened.device, dtype=flattened.dtype)
    quantiles = torch.quantile(flattened, q_tensor)
    return [float(value.item()) for value in quantiles.reshape(-1)]


def _sample_large_tensor(values: torch.Tensor) -> torch.Tensor:
    flattened = values.reshape(-1)
    stride = max(1, (flattened.numel() + TORCH_QUANTILE_MAX_EXACT_VALUES - 1) // TORCH_QUANTILE_MAX_EXACT_VALUES)
    return flattened[::stride]


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(float(denominator)) <= EPS:
        return None
    return float(numerator) / float(denominator)


def _float(value: torch.Tensor) -> float:
    return float(value.item())


def _top_level_stage(name: str) -> str:
    first = name.split(".", 1)[0]
    return first if first else "unknown"
