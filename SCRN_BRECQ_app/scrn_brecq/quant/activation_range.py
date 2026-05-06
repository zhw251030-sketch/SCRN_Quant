"""Activation quantizer range calibration helpers for E005 experiments."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Mapping

import torch
from torch import nn

from SCRN_BRECQ_app.scrn_brecq.quant.activation_sensitivity import select_activation_quantizers
from SCRN_BRECQ_app.scrn_brecq.quant.quant_layer import QuantModule


DEFAULT_MAX_VALUES_PER_LAYER = 500_000
ACTIVATION_RANGE_EPS = 1e-8
DEFAULT_MSE_SHRINK_RATIOS = [
    1.0,
    0.999,
    0.995,
    0.99,
    0.98,
    0.97,
    0.96,
    0.95,
    0.925,
    0.9,
    0.875,
    0.85,
    0.8,
    0.75,
    0.7,
    0.65,
    0.6,
    0.55,
    0.5,
]
RANGE_SELECTOR_KEYS = {"index", "name_contains", "stage", "branch", "role", "module_type"}


def apply_activation_ranges(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    method: str,
    percentile: float = 99.99,
    mse_shrink_ratios: str | Iterable[float] | None = None,
    loss_p: float = 2.4,
    index: int | None = None,
    name_contains: str | None = None,
    stage: str | None = None,
    branch: str | None = None,
    role: str | None = None,
    module_type: str | None = None,
    selector_groups: Iterable[Mapping[str, Any]] | None = None,
    exclude_selector_groups: Iterable[Mapping[str, Any]] | None = None,
    include_output_quantizer: bool = False,
    max_values_per_layer: int = DEFAULT_MAX_VALUES_PER_LAYER,
    weight_quant: bool = True,
) -> dict[str, Any]:
    """Recalibrate selected activation quantizers with a named range method."""
    method = str(method)
    if method not in {"percentile", "max", "mse_grid"}:
        raise ValueError(f"Unsupported activation range method: {method}")
    percentile = _validate_percentile(percentile)
    max_values_per_layer = _validate_max_values(max_values_per_layer)
    loss_p = _validate_loss_p(loss_p)
    mse_ratios = parse_mse_shrink_ratios(mse_shrink_ratios)
    selector_groups = normalize_selector_groups(selector_groups, "range_selector_groups")
    exclude_selector_groups = normalize_selector_groups(exclude_selector_groups, "range_exclude_selector_groups")
    selected_rows = _select_range_quantizers(
        model,
        index=index,
        name_contains=name_contains,
        stage=stage,
        branch=branch,
        role=role,
        module_type=module_type,
        selector_groups=selector_groups,
        exclude_selector_groups=exclude_selector_groups,
        include_output_quantizer=include_output_quantizer,
    )
    if not selected_rows:
        raise ValueError(f"{method} activation range calibration selected no activation quantizers.")

    named_modules = _named_quant_modules(model)
    module_by_index = {index_: module for index_, (_name, module) in enumerate(named_modules)}
    selected_by_index = {int(row["index"]): row for row in selected_rows}
    records: dict[int, dict[str, Any]] = {}
    hooks = []
    for index_, row in selected_by_index.items():
        hooks.append(
            module_by_index[index_].register_forward_hook(
                _make_range_hook(index_, row, records, method, percentile, mse_ratios, loss_p, max_values_per_layer)
            )
        )

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
    summary = {
        "method": method,
        "selected_count": len(layers),
        "selected_names": [layer["name"] for layer in layers],
        "selector": {
            "index": index,
            "name_contains": name_contains,
            "stage": stage,
            "branch": branch,
            "role": role,
            "module_type": module_type,
            "selector_groups": selector_groups,
            "exclude_selector_groups": exclude_selector_groups,
            "include_output_quantizer": bool(include_output_quantizer),
        },
        "max_values_per_layer": max_values_per_layer,
        "layers": layers,
    }
    if method == "percentile":
        summary.update(
            {
                "percentile": percentile,
                "lower_quantile": _lower_quantile(percentile),
                "upper_quantile": 1.0 - _lower_quantile(percentile),
            }
        )
    if method == "mse_grid":
        summary.update({"mse_shrink_ratios": mse_ratios, "loss_p": loss_p})
    return summary


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
    selector_groups: Iterable[Mapping[str, Any]] | None = None,
    exclude_selector_groups: Iterable[Mapping[str, Any]] | None = None,
    include_output_quantizer: bool = False,
    max_values_per_layer: int = DEFAULT_MAX_VALUES_PER_LAYER,
    weight_quant: bool = True,
) -> dict[str, Any]:
    """Recalibrate selected activation quantizers with two-sided percentile ranges."""
    return apply_activation_ranges(
        model,
        inputs,
        method="percentile",
        percentile=percentile,
        index=index,
        name_contains=name_contains,
        stage=stage,
        branch=branch,
        role=role,
        module_type=module_type,
        selector_groups=selector_groups,
        exclude_selector_groups=exclude_selector_groups,
        include_output_quantizer=include_output_quantizer,
        max_values_per_layer=max_values_per_layer,
        weight_quant=weight_quant,
    )


def normalize_selector_groups(
    value: Iterable[Mapping[str, Any]] | None,
    field_name: str,
) -> list[dict[str, Any]] | None:
    """Normalize selector group config for activation range calibration."""
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of selector objects.")
    groups: list[dict[str, Any]] = []
    for group_index, group in enumerate(value):
        if not isinstance(group, Mapping):
            raise ValueError(f"{field_name}[{group_index}] must be a selector object.")
        unsupported = sorted(set(group) - RANGE_SELECTOR_KEYS)
        if unsupported:
            raise ValueError(f"Unsupported selector key in {field_name}[{group_index}]: {unsupported[0]}")
        normalized: dict[str, Any] = {}
        for key in ["index", "name_contains", "stage", "branch", "role", "module_type"]:
            if key not in group or group[key] is None:
                continue
            normalized[key] = int(group[key]) if key == "index" else str(group[key])
        if not normalized:
            raise ValueError(f"{field_name}[{group_index}] must not be empty.")
        groups.append(normalized)
    return groups


def _select_range_quantizers(
    model: nn.Module,
    *,
    index: int | None,
    name_contains: str | None,
    stage: str | None,
    branch: str | None,
    role: str | None,
    module_type: str | None,
    selector_groups: list[dict[str, Any]] | None,
    exclude_selector_groups: list[dict[str, Any]] | None,
    include_output_quantizer: bool,
) -> list[dict[str, Any]]:
    legacy_selector = {
        "index": index,
        "name_contains": name_contains,
        "stage": stage,
        "branch": branch,
        "role": role,
        "module_type": module_type,
    }
    if selector_groups is not None and any(value is not None for value in legacy_selector.values()):
        raise ValueError("range_selector_groups cannot be combined with single range selector fields.")

    candidates = select_activation_quantizers(model, include_output_quantizer=include_output_quantizer)
    if selector_groups is None:
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
    else:
        selected_indices: set[int] = set()
        for group in selector_groups:
            for row in select_activation_quantizers(model, include_output_quantizer=include_output_quantizer, **group):
                selected_indices.add(int(row["index"]))
        selected_rows = [row for row in candidates if int(row["index"]) in selected_indices]

    if exclude_selector_groups:
        excluded_indices: set[int] = set()
        for group in exclude_selector_groups:
            for row in select_activation_quantizers(model, include_output_quantizer=include_output_quantizer, **group):
                excluded_indices.add(int(row["index"]))
        selected_rows = [row for row in selected_rows if int(row["index"]) not in excluded_indices]
    return selected_rows


def _make_range_hook(
    index: int,
    row: dict[str, Any],
    records: dict[int, dict[str, Any]],
    method: str,
    percentile: float,
    mse_shrink_ratios: list[float],
    loss_p: float,
    max_values_per_layer: int,
):
    def hook(module: QuantModule, _inputs, output: torch.Tensor) -> None:
        if method == "percentile":
            stats = _percentile_range_stats(output, percentile=percentile, max_values_per_layer=max_values_per_layer)
        elif method == "max":
            stats = _max_range_stats(output)
        elif method == "mse_grid":
            n_levels = int(getattr(module.act_quantizer, "n_levels", 2 ** int(getattr(module.act_quantizer, "n_bits", 8))))
            stats = _mse_grid_range_stats(
                output,
                mse_shrink_ratios=mse_shrink_ratios,
                loss_p=loss_p,
                max_values_per_layer=max_values_per_layer,
                n_levels=n_levels,
            )
        else:
            raise ValueError(f"Unsupported activation range method: {method}")
        _write_activation_quantizer_range(module.act_quantizer, output, stats["chosen_min"], stats["chosen_max"])
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
        "chosen_min": clipped_min,
        "chosen_max": clipped_max,
        "chosen_range": clipped_range,
        "original_range": original_range,
        "clipped_range": clipped_range,
        "range_shrink_ratio": clipped_range / original_range,
        "best_shrink_ratio": None,
        "best_score": None,
        "sampled": sampled,
        "sample_count": int(sampled_values.numel()),
        "numel": int(values.numel()),
    }


def _max_range_stats(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().float().reshape(-1)
    if values.numel() == 0:
        raise ValueError("Cannot calibrate activation range from an empty tensor.")
    original_min = float(values.min().item())
    original_max = float(values.max().item())
    chosen_min = original_min
    chosen_max = original_max
    if chosen_max - chosen_min < ACTIVATION_RANGE_EPS:
        chosen_max = chosen_min + ACTIVATION_RANGE_EPS
    original_range = max(original_max - original_min, ACTIVATION_RANGE_EPS)
    chosen_range = chosen_max - chosen_min
    return {
        "original_min": original_min,
        "original_max": original_max,
        "lower_value": original_min,
        "upper_value": original_max,
        "clipped_min": chosen_min,
        "clipped_max": chosen_max,
        "chosen_min": chosen_min,
        "chosen_max": chosen_max,
        "chosen_range": chosen_range,
        "original_range": original_range,
        "clipped_range": chosen_range,
        "range_shrink_ratio": chosen_range / original_range,
        "best_shrink_ratio": 1.0,
        "best_score": None,
        "sampled": False,
        "sample_count": int(values.numel()),
        "numel": int(values.numel()),
    }


def _mse_grid_range_stats(
    tensor: torch.Tensor,
    *,
    mse_shrink_ratios: Iterable[float],
    loss_p: float,
    max_values_per_layer: int,
    n_levels: int,
) -> dict[str, Any]:
    values = tensor.detach().float().reshape(-1)
    if values.numel() == 0:
        raise ValueError("Cannot calibrate activation range from an empty tensor.")
    original_min = float(values.min().item())
    original_max = float(values.max().item())
    original_range = max(original_max - original_min, ACTIVATION_RANGE_EPS)
    sampled_values, sampled = _sample_values(values, max_values_per_layer=max_values_per_layer)
    best_score = float("inf")
    best_ratio = None
    best_min = original_min
    best_max = original_max
    candidate_scores: dict[str, float] = {}

    for ratio in mse_shrink_ratios:
        candidate_min = original_min * float(ratio)
        candidate_max = original_max * float(ratio)
        if candidate_max - candidate_min < ACTIVATION_RANGE_EPS:
            candidate_max = candidate_min + ACTIVATION_RANGE_EPS
        quantized = _fake_quantize_with_range(sampled_values, candidate_min, candidate_max, n_levels=n_levels)
        score = float((sampled_values - quantized).abs().pow(loss_p).mean().item())
        candidate_scores[str(float(ratio))] = score
        if score < best_score:
            best_score = score
            best_ratio = float(ratio)
            best_min = candidate_min
            best_max = candidate_max

    if best_ratio is None:
        raise RuntimeError("Failed to choose MSE-grid activation range.")
    chosen_range = max(best_max - best_min, ACTIVATION_RANGE_EPS)
    return {
        "original_min": original_min,
        "original_max": original_max,
        "lower_value": best_min,
        "upper_value": best_max,
        "clipped_min": best_min,
        "clipped_max": best_max,
        "chosen_min": best_min,
        "chosen_max": best_max,
        "chosen_range": chosen_range,
        "original_range": original_range,
        "clipped_range": chosen_range,
        "range_shrink_ratio": chosen_range / original_range,
        "best_shrink_ratio": best_ratio,
        "best_score": best_score,
        "candidate_scores": candidate_scores,
        "sampled": sampled,
        "sample_count": int(sampled_values.numel()),
        "numel": int(values.numel()),
    }


def _fake_quantize_with_range(values: torch.Tensor, min_value: float, max_value: float, *, n_levels: int) -> torch.Tensor:
    delta_value = max((float(max_value) - float(min_value)) / float(n_levels - 1), ACTIVATION_RANGE_EPS)
    delta = torch.tensor(delta_value, dtype=values.dtype, device=values.device)
    zero_point = torch.tensor(round(-float(min_value) / delta_value), dtype=values.dtype, device=values.device)
    x_int = torch.round(values / delta)
    x_quant = torch.clamp(x_int + zero_point, 0, n_levels - 1)
    return (x_quant - zero_point) * delta


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


def _validate_loss_p(loss_p: float) -> float:
    value = float(loss_p)
    if value <= 0.0:
        raise ValueError(f"range_loss_p must be positive, got {loss_p}")
    return value


def parse_mse_shrink_ratios(value: str | Iterable[float] | None) -> list[float]:
    """Parse and validate MSE-grid shrink ratios."""
    if value is None:
        ratios = list(DEFAULT_MSE_SHRINK_RATIOS)
    elif isinstance(value, str):
        if not value.strip():
            raise ValueError("range_mse_shrink_ratios must not be empty.")
        try:
            ratios = [float(part.strip()) for part in value.split(",") if part.strip()]
        except ValueError as exc:
            raise ValueError(f"range_mse_shrink_ratios contains a non-numeric value: {value}") from exc
    else:
        ratios = [float(item) for item in value]
    if not ratios:
        raise ValueError("range_mse_shrink_ratios must not be empty.")
    for ratio in ratios:
        if ratio <= 0.0:
            raise ValueError(f"range_mse_shrink_ratios must be positive, got {ratio}")
    return ratios


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
