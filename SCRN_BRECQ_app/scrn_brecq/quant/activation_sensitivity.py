"""Selective activation quantizer controls for E004 sensitivity experiments."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Iterator

from torch import nn

from SCRN_BRECQ_app.scrn_brecq.quant.activation_diagnostics import collect_quantizer_rows
from SCRN_BRECQ_app.scrn_brecq.quant.quant_layer import QuantModule


SENSITIVITY_MODES = {
    "all_on",
    "all_off",
    "disable_one",
    "enable_one",
    "disable_group",
    "enable_group",
}


def select_activation_quantizers(
    model: nn.Module,
    *,
    index: int | None = None,
    name_contains: str | None = None,
    stage: str | None = None,
    branch: str | None = None,
    role: str | None = None,
    module_type: str | None = None,
    include_output_quantizer: bool = False,
) -> list[dict[str, Any]]:
    """Return quantizer rows matching the provided selector fields."""
    rows = _candidate_rows(collect_quantizer_rows(model), include_output_quantizer=include_output_quantizer)
    return [
        row
        for row in rows
        if _row_matches(
            row,
            index=index,
            name_contains=name_contains,
            stage=stage,
            branch=branch,
            role=role,
            module_type=module_type,
        )
    ]


@contextmanager
def apply_activation_sensitivity_mode(
    model: nn.Module,
    *,
    mode: str,
    index: int | None = None,
    name_contains: str | None = None,
    stage: str | None = None,
    branch: str | None = None,
    role: str | None = None,
    module_type: str | None = None,
    include_output_quantizer: bool = False,
) -> Iterator[list[dict[str, Any]]]:
    """Temporarily apply an E004 activation quantizer sensitivity mode."""
    if mode not in SENSITIVITY_MODES:
        raise ValueError(f"Unsupported activation sensitivity mode: {mode}")

    named_modules = _named_quant_modules(model)
    module_by_index = {index_: module for index_, (_name, module) in enumerate(named_modules)}
    original_disabled = {index_: bool(module.disable_act_quant) for index_, (_name, module) in enumerate(named_modules)}

    rows = collect_quantizer_rows(model)
    candidates = _candidate_rows(rows, include_output_quantizer=include_output_quantizer)
    candidate_indices = {int(row["index"]) for row in candidates}
    selected = _selected_rows_for_mode(
        mode,
        candidates,
        index=index,
        name_contains=name_contains,
        stage=stage,
        branch=branch,
        role=role,
        module_type=module_type,
    )
    selected_indices = {int(row["index"]) for row in selected}
    _validate_selection(mode, selected)

    try:
        if mode == "all_on":
            _set_all_disabled(module_by_index, True)
            _set_indices_disabled(module_by_index, candidate_indices, False)
        elif mode in {"disable_one", "disable_group"}:
            _set_all_disabled(module_by_index, True)
            _set_indices_disabled(module_by_index, candidate_indices, False)
            _set_indices_disabled(module_by_index, selected_indices, True)
        elif mode == "all_off":
            _set_all_disabled(module_by_index, True)
        elif mode in {"enable_one", "enable_group"}:
            _set_all_disabled(module_by_index, True)
            _set_indices_disabled(module_by_index, selected_indices, False)
        yield _augment_rows_with_proxy(selected, named_modules)
    finally:
        for index_, disabled in original_disabled.items():
            module_by_index[index_].disable_act_quant = disabled


def _selected_rows_for_mode(
    mode: str,
    candidates: Iterable[dict[str, Any]],
    *,
    index: int | None,
    name_contains: str | None,
    stage: str | None,
    branch: str | None,
    role: str | None,
    module_type: str | None,
) -> list[dict[str, Any]]:
    if mode in {"all_on", "all_off"}:
        return list(candidates)
    return [
        row
        for row in candidates
        if _row_matches(
            row,
            index=index,
            name_contains=name_contains,
            stage=stage,
            branch=branch,
            role=role,
            module_type=module_type,
        )
    ]


def _validate_selection(mode: str, selected: list[dict[str, Any]]) -> None:
    if mode in {"disable_one", "enable_one"} and len(selected) != 1:
        raise ValueError(f"{mode} requires exactly one selected quantizer, got {len(selected)}")
    if mode in {"disable_group", "enable_group"} and not selected:
        raise ValueError(f"{mode} requires at least one selected quantizer.")


def _row_matches(
    row: dict[str, Any],
    *,
    index: int | None,
    name_contains: str | None,
    stage: str | None,
    branch: str | None,
    role: str | None,
    module_type: str | None,
) -> bool:
    if index is not None and int(row["index"]) != int(index):
        return False
    if name_contains and str(name_contains) not in str(row["name"]):
        return False
    if stage and str(row["stage"]) != str(stage):
        return False
    if branch and str(row["branch"]) != str(branch):
        return False
    if role and str(row["role"]) != str(role):
        return False
    if module_type and str(row["module_type"]) != str(module_type):
        return False
    return True


def _candidate_rows(rows: list[dict[str, Any]], *, include_output_quantizer: bool) -> list[dict[str, Any]]:
    if include_output_quantizer or not rows:
        return list(rows)
    output_index = max(int(row["index"]) for row in rows)
    return [row for row in rows if int(row["index"]) != output_index]


def _set_all_disabled(module_by_index: dict[int, QuantModule], disabled: bool) -> None:
    for module in module_by_index.values():
        module.disable_act_quant = bool(disabled)


def _set_indices_disabled(module_by_index: dict[int, QuantModule], indices: Iterable[int], disabled: bool) -> None:
    for index in indices:
        module_by_index[int(index)].disable_act_quant = bool(disabled)


def _augment_rows_with_proxy(
    rows: list[dict[str, Any]],
    named_modules: list[tuple[str, QuantModule]],
) -> list[dict[str, Any]]:
    result = []
    module_by_index = {index: module for index, (_name, module) in enumerate(named_modules)}
    for row in rows:
        module = module_by_index[int(row["index"])]
        augmented = dict(row)
        augmented["weight_shape"] = list(module.weight.shape)
        augmented["activation_numel"] = None
        result.append(augmented)
    return result


def _named_quant_modules(model: nn.Module) -> list[tuple[str, QuantModule]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, QuantModule)]
