"""Activation mixed precision helpers for selective A8 experiments."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from torch import nn

from SCRN_BRECQ_app.scrn_brecq.quant.activation_diagnostics import collect_quantizer_rows
from SCRN_BRECQ_app.scrn_brecq.quant.activation_range import normalize_selector_groups
from SCRN_BRECQ_app.scrn_brecq.quant.activation_sensitivity import select_activation_quantizers
from SCRN_BRECQ_app.scrn_brecq.quant.quant_layer import QuantModule


OVERRIDE_KEYS = {"n_bits", "selector_groups", "exclude_selector_groups", "include_output_quantizer"}


def normalize_activation_bitwidth_overrides(
    value: Any,
    field_name: str = "activation_bitwidth_overrides",
) -> list[dict[str, Any]]:
    """Normalize selective activation bitwidth override config."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of override objects.")

    overrides: list[dict[str, Any]] = []
    for index, override in enumerate(value):
        if not isinstance(override, Mapping):
            raise ValueError(f"{field_name}[{index}] must be an override object.")
        unsupported = sorted(set(override) - OVERRIDE_KEYS)
        if unsupported:
            raise ValueError(f"Unsupported key in {field_name}[{index}]: {unsupported[0]}")
        if "n_bits" not in override:
            raise ValueError(f"{field_name}[{index}].n_bits is required.")
        n_bits = int(override["n_bits"])
        if n_bits < 2 or n_bits > 8:
            raise ValueError(f"{field_name}[{index}].n_bits must be between 2 and 8, got {n_bits}.")

        normalized: dict[str, Any] = {"n_bits": n_bits}
        if "selector_groups" in override:
            normalized["selector_groups"] = normalize_selector_groups(
                override.get("selector_groups"),
                f"{field_name}[{index}].selector_groups",
            )
        if "exclude_selector_groups" in override:
            normalized["exclude_selector_groups"] = normalize_selector_groups(
                override.get("exclude_selector_groups"),
                f"{field_name}[{index}].exclude_selector_groups",
            )
        if "include_output_quantizer" in override:
            normalized["include_output_quantizer"] = bool(override["include_output_quantizer"])
        overrides.append(normalized)
    return overrides


def apply_activation_bitwidth_overrides(model: nn.Module, overrides: Any) -> dict[str, Any]:
    """Apply selective activation bitwidth overrides and return an audit summary."""
    normalized = normalize_activation_bitwidth_overrides(overrides)
    named_modules = _named_quant_modules(model)
    module_by_index = {index: module for index, (_name, module) in enumerate(named_modules)}
    applied_overrides: list[dict[str, Any]] = []

    for override in normalized:
        selected_rows = _select_override_rows(model, override)
        n_bits = int(override["n_bits"])
        for row in selected_rows:
            module_by_index[int(row["index"])].act_quantizer.bitwidth_refactor(n_bits)
        applied_overrides.append(
            {
                "n_bits": n_bits,
                "selector_groups": override.get("selector_groups"),
                "exclude_selector_groups": override.get("exclude_selector_groups"),
                "include_output_quantizer": bool(override.get("include_output_quantizer", False)),
                "selected_count": len(selected_rows),
                "selected_indices": [int(row["index"]) for row in selected_rows],
                "selected_names": [str(row["name"]) for row in selected_rows],
            }
        )

    summary = summarize_activation_bitwidths(model)
    summary["override_count"] = len(normalized)
    summary["activation_bitwidth_overrides"] = normalized
    summary["applied_overrides"] = applied_overrides
    return summary


def summarize_activation_bitwidths(model: nn.Module) -> dict[str, Any]:
    """Summarize effective activation bitwidths for every QuantModule."""
    rows = collect_quantizer_rows(model)
    all_counts: Counter[int] = Counter()
    enabled_counts: Counter[int] = Counter()
    disabled_counts: Counter[int] = Counter()

    for row in rows:
        bit = int(row["act_bit"])
        all_counts[bit] += 1
        if bool(row["act_disabled"]):
            disabled_counts[bit] += 1
        else:
            enabled_counts[bit] += 1

    return {
        "activation_quantizers": len(rows),
        "disabled_activation_quantizers": sum(1 for row in rows if bool(row["act_disabled"])),
        "activation_bit_counts": _json_bit_counts(all_counts),
        "enabled_activation_bit_counts": _json_bit_counts(enabled_counts),
        "disabled_activation_bit_counts": _json_bit_counts(disabled_counts),
    }


def _select_override_rows(model: nn.Module, override: Mapping[str, Any]) -> list[dict[str, Any]]:
    include_output_quantizer = bool(override.get("include_output_quantizer", False))
    candidates = select_activation_quantizers(model, include_output_quantizer=include_output_quantizer)
    selector_groups = override.get("selector_groups")
    if selector_groups is None:
        selected_rows = list(candidates)
    else:
        selected_indices: set[int] = set()
        for group in selector_groups:
            for row in select_activation_quantizers(model, include_output_quantizer=include_output_quantizer, **group):
                selected_indices.add(int(row["index"]))
        selected_rows = [row for row in candidates if int(row["index"]) in selected_indices]

    exclude_selector_groups = override.get("exclude_selector_groups")
    if exclude_selector_groups:
        excluded_indices: set[int] = set()
        for group in exclude_selector_groups:
            for row in select_activation_quantizers(model, include_output_quantizer=include_output_quantizer, **group):
                excluded_indices.add(int(row["index"]))
        selected_rows = [row for row in selected_rows if int(row["index"]) not in excluded_indices]
    return selected_rows


def _named_quant_modules(model: nn.Module) -> list[tuple[str, QuantModule]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, QuantModule)]


def _json_bit_counts(counts: Counter[int]) -> dict[str, int]:
    return {str(bit): int(counts[bit]) for bit in sorted(counts)}
