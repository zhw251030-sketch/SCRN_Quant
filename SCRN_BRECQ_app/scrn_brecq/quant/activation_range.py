"""Activation quantizer range calibration helpers for E005 experiments."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Iterator

import torch
from torch import nn

from SCRN_BRECQ_app.scrn_brecq.quant.activation_sensitivity import select_activation_quantizers
from SCRN_BRECQ_app.scrn_brecq.quant.quant_layer import QuantModule


DEFAULT_MAX_VALUES_PER_LAYER = 500_000
ACTIVATION_RANGE_EPS = 1e-8


def apply_percentile_activation_ranges(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    percentile: float,
    index: int | None = None,
    name_contains: str | None = None,
    stage: str | None = None,
    branch: str | None = None,
    role: str | None = None,
    module_type: str | None = None,
    include_output_quantizer: bool = False,
    max_values_per_layer: int = DEFAULT_MAX_VALUES_PER_LAYER,
    weight_quant: bool = True,
) -> dict[str, Any]:
    """Recalibrate selected activation quantizers with two-sided percentile ranges."""
    percentile = _validate_percentile(percentile)
    max_values_per_layer = _validate_max_values(max_values_per_layer)
    selected_rows = select_activation_quantizers(
        model,
        index=index,
        name_contains=name_contains,
        stage=stage,
        branch=branch,
        role=role,
        module_type=module_type,
        include_output_quantizer=include_output_quantizer,
    )
    if not selected_rows:
        raise ValueError("Percentile activation range calibration selected no activation quantizers.")

    named_modules = _named_quant_modules(model)
    module_by_index = {index_: module for index_, (_name, module) in enumerate(named_modules)}
    selected_by_index = {int(row["index"]): row for row in selected_rows}
    records: dict[int, dict[str, Any]] = {}
    hooks = []
    for index_, row in selected_by_index.items():
        hooks.append(module_by_index[index_].register_forward_hook(_make_percentile_hook(index_, row, records, percentile, max_values_per_layer)))

    try:
        with torch.no_grad(), _temporary_quant_state(model, weight_quant=weight_quant, act_quant=False):
            _ = model(inputs)
    finally:
        for hook in hooks:
            hook.remove()

    missing = sorted(set(selected_by_index) - set(records))
    if missing:
        raise RuntimeError(f"Selected activation quantizers were not observed during forward pass: {missing}")

    layers = [records[int(row["index"])] for row in selected_rows]
    return {
        "method": "percentile",
        "percentile": percentile,
        "lower_quantile": _lower_quantile(percentile),
        "upper_quantile": 1.0 - _lower_quantile(percentile),
        "selected_count": len(layers),
        "selected_names": [layer["name"] for layer in layers],
        "selector": {
            "index": index,
            "name_contains": name_contains,
            "stage": stage,
            "branch": branch,
            "role": role,
            "module_type": module_type,
            "include_output_quantizer": bool(include_output_quantizer),
        },
        "max_values_per_layer": max_values_per_layer,
        "layers": layers,
    }


def _make_percentile_hook(
    index: int,
    row: dict[str, Any],
    records: dict[int, dict[str, Any]],
    percentile: float,
    max_values_per_layer: int,
):
    def hook(module: QuantModule, _inputs, output: torch.Tensor) -> None:
        stats = _percentile_range_stats(output, percentile=percentile, max_values_per_layer=max_values_per_layer)
        _write_activation_quantizer_range(module.act_quantizer, output, stats["clipped_min"], stats["clipped_max"])
        records[index] = {
            "index": int(index),
            "name": row["name"],
            "stage": row["stage"],
            "branch": row["branch"],
            "role": row["role"],
            "module_type": row["module_type"],
            **stats,
            "delta": _float_value(module.act_quantizer.delta),
            "zero_point": _float_value(module.act_quantizer.zero_point),
        }

    return hook


def _percentile_range_stats(
    tensor: torch.Tensor,
    *,
    percentile: float,
    max_values_per_layer: int,
) -> dict[str, Any]:
    values = tensor.detach().float().reshape(-1)
    if values.numel() == 0:
        raise ValueError("Cannot calibrate activation range from an empty tensor.")
    original_min = float(values.min().item())
    original_max = float(values.max().item())
    sampled_values, sampled = _sample_values(values, max_values_per_layer=max_values_per_layer)
    lower_q = _lower_quantile(percentile)
    quantiles = torch.quantile(
        sampled_values,
        torch.tensor([lower_q, 1.0 - lower_q], dtype=sampled_values.dtype, device=sampled_values.device),
    )
    lower_value = float(quantiles[0].item())
    upper_value = float(quantiles[1].item())
    clipped_min = min(lower_value, 0.0)
    clipped_max = max(upper_value, 0.0)
    if clipped_max - clipped_min < ACTIVATION_RANGE_EPS:
        clipped_max = clipped_min + ACTIVATION_RANGE_EPS
    original_range = max(original_max - original_min, ACTIVATION_RANGE_EPS)
    clipped_range = clipped_max - clipped_min
    return {
        "original_min": original_min,
        "original_max": original_max,
        "lower_value": lower_value,
        "upper_value": upper_value,
        "clipped_min": clipped_min,
        "clipped_max": clipped_max,
        "original_range": original_range,
        "clipped_range": clipped_range,
        "range_shrink_ratio": clipped_range / original_range,
        "sampled": sampled,
        "sample_count": int(sampled_values.numel()),
        "numel": int(values.numel()),
    }


def _write_activation_quantizer_range(
    quantizer: nn.Module,
    reference: torch.Tensor,
    clipped_min: float,
    clipped_max: float,
) -> None:
    n_levels = int(getattr(quantizer, "n_levels", 2 ** int(getattr(quantizer, "n_bits", 8))))
    delta_value = max((float(clipped_max) - float(clipped_min)) / float(n_levels - 1), ACTIVATION_RANGE_EPS)
    delta = torch.tensor(delta_value, dtype=reference.dtype, device=reference.device)
    zero_point = torch.tensor(round(-float(clipped_min) / delta_value), dtype=reference.dtype, device=reference.device)
    current_delta = getattr(quantizer, "delta", None)
    if isinstance(current_delta, nn.Parameter):
        current_delta.data.copy_(delta)
    elif current_delta is not None:
        quantizer.delta = delta
    elif bool(getattr(quantizer, "leaf_param", False)):
        quantizer.delta = nn.Parameter(delta)
    else:
        quantizer.delta = delta
    quantizer.zero_point = zero_point
    quantizer.inited = True


def _sample_values(values: torch.Tensor, *, max_values_per_layer: int) -> tuple[torch.Tensor, bool]:
    if values.numel() <= max_values_per_layer:
        return values, False
    stride = max(1, (values.numel() + max_values_per_layer - 1) // max_values_per_layer)
    return values[::stride], True


def _validate_percentile(percentile: float) -> float:
    value = float(percentile)
    if value <= 0.0 or value >= 100.0:
        raise ValueError(f"activation percentile must be between 0 and 100, got {percentile}")
    return value


def _validate_max_values(max_values_per_layer: int) -> int:
    value = int(max_values_per_layer)
    if value <= 0:
        raise ValueError(f"range_max_values_per_layer must be positive, got {max_values_per_layer}")
    return value


def _lower_quantile(percentile: float) -> float:
    return (1.0 - float(percentile) / 100.0) / 2.0


def _float_value(value: torch.Tensor | nn.Parameter | None) -> float | None:
    if value is None:
        return None
    return float(value.detach().float().reshape(-1)[0].item())


@contextmanager
def _temporary_quant_state(model: nn.Module, *, weight_quant: bool, act_quant: bool) -> Iterator[None]:
    modules = _named_quant_modules(model)
    previous_states = [
        (module, bool(getattr(module, "use_weight_quant", False)), bool(getattr(module, "use_act_quant", False)))
        for _name, module in modules
    ]
    if hasattr(model, "set_quant_state"):
        model.set_quant_state(weight_quant, act_quant)
    else:
        for _name, module in modules:
            module.set_quant_state(weight_quant, act_quant)
    try:
        yield
    finally:
        for module, previous_weight, previous_act in previous_states:
            module.set_quant_state(previous_weight, previous_act)


def _named_quant_modules(model: nn.Module) -> list[tuple[str, QuantModule]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, QuantModule)]
