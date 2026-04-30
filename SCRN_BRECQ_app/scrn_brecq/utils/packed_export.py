"""Packed deployment export helpers for SCRN-BRECQ checkpoints.

The regular ``quantized_scrn_brecq.pth`` file is a recovery checkpoint: it keeps
FP32 weights, AdaRound alpha, original weights, and quantizer state.  This module
exports a compact artifact that stores quantized weights as integers plus the
small FP32 metadata needed to dequantize them.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, BinaryIO

import torch

from SCRN_BRECQ_app.scrn_brecq.quant import AdaRoundQuantizer, QuantModule
from SCRN_BRECQ_app.scrn_brecq.utils.model_size import build_model_size_report


FORMAT_VERSION = 1


def pack_uint4(values: torch.Tensor) -> torch.Tensor:
    """Pack unsigned 4-bit integer values into bytes, low nibble first."""
    flat = _validate_unsigned_values(values, bits=4)
    if flat.numel() % 2:
        flat = torch.cat([flat, torch.zeros(1, dtype=flat.dtype)])
    low = flat[0::2]
    high = flat[1::2]
    return (low | (high << 4)).to(torch.uint8).contiguous()


def pack_unsigned_values(values: torch.Tensor, *, bits: int) -> tuple[torch.Tensor, str]:
    """Pack unsigned integer tensor values for a supported deployment bitwidth."""
    bitwidth = int(bits)
    if bitwidth == 4:
        return pack_uint4(values), "uint4_lownibble_first"
    if bitwidth == 8:
        return _validate_unsigned_values(values, bits=8).to(torch.uint8).contiguous(), "uint8"
    if bitwidth == 2:
        return _pack_uint2(values), "uint2_lownibble_first"
    raise ValueError(f"Packed export supports 2/4/8-bit weights, got {bits}")


def export_packed_deployment(
    model: torch.nn.Module,
    output_dir: str | Path,
    *,
    source_checkpoint_path: str | Path | None = None,
    quant_checkpoint_path: str | Path | None = None,
    final_quant_state: dict[str, bool] | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export packed quantized weights and FP32 aux tensors into ``output_dir``.

    The resulting directory contains:

    - ``weights.bin``: concatenated packed integer weights.
    - ``aux_fp32.bin``: concatenated FP32 metadata and non-quantized parameters.
    - ``manifest.json``: tensor offsets, shapes, bitwidths, and layer metadata.
    - ``summary.json``: size accounting and theoretical compression report.
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    weights_path = target_dir / "weights.bin"
    aux_path = target_dir / "aux_fp32.bin"
    manifest_path = target_dir / "manifest.json"
    summary_path = target_dir / "summary.json"

    quant_modules = [(name or "_root", module) for name, module in model.named_modules() if isinstance(module, QuantModule)]
    quant_weight_param_names = {f"{name}.weight" if name != "_root" else "weight" for name, _ in quant_modules}

    manifest: dict[str, Any] = {
        "format": "scrn_brecq_packed_deployment",
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_checkpoint": str(source_checkpoint_path) if source_checkpoint_path is not None else None,
        "quant_checkpoint": str(quant_checkpoint_path) if quant_checkpoint_path is not None else None,
        "final_quant_state": final_quant_state,
        "checkpoint_metadata": checkpoint_metadata or {},
        "weight_file": weights_path.name,
        "aux_file": aux_path.name,
        "layers": [],
        "aux_tensors": [],
    }

    payload_stats = {
        "quantized_layer_count": 0,
        "quantized_weight_values": 0,
        "packed_weight_bytes": 0,
        "aux_fp32_bytes": 0,
        "activation_quantizer_count": 0,
        "non_quantized_tensor_count": 0,
        "recomputed_weight_quantizer_count": 0,
    }

    with weights_path.open("wb") as weight_file, aux_path.open("wb") as aux_file:
        for name, module in quant_modules:
            layer_entry, packed_bytes = _export_quant_module(name, module, weight_file, aux_file, manifest)
            manifest["layers"].append(layer_entry)
            payload_stats["quantized_layer_count"] += 1
            payload_stats["quantized_weight_values"] += int(module.weight.numel())
            payload_stats["packed_weight_bytes"] += int(packed_bytes)
            if bool(layer_entry["weight_quantizer_recomputed"]):
                payload_stats["recomputed_weight_quantizer_count"] += 1
            payload_stats["aux_fp32_bytes"] += int(layer_entry["weight_delta"]["num_bytes"])
            payload_stats["aux_fp32_bytes"] += int(layer_entry["weight_zero_point"]["num_bytes"])
            if layer_entry.get("activation_delta") is not None:
                payload_stats["activation_quantizer_count"] += 1
                payload_stats["aux_fp32_bytes"] += int(layer_entry["activation_delta"]["num_bytes"])
            if layer_entry.get("activation_zero_point") is not None:
                payload_stats["aux_fp32_bytes"] += int(layer_entry["activation_zero_point"]["num_bytes"])

        for parameter_name, parameter in model.named_parameters():
            if _is_recovery_only_parameter(parameter_name) or parameter_name in quant_weight_param_names:
                continue
            entry = _write_aux_tensor(aux_file, parameter.detach(), name=parameter_name, role="non_quantized_parameter")
            manifest["aux_tensors"].append(entry)
            payload_stats["non_quantized_tensor_count"] += 1
            payload_stats["aux_fp32_bytes"] += int(entry["num_bytes"])

    _write_json(manifest_path, manifest)
    summary = _build_summary(
        model,
        target_dir,
        payload_stats,
        source_checkpoint_path=source_checkpoint_path,
        quant_checkpoint_path=quant_checkpoint_path,
    )
    _write_json(summary_path, summary)
    return summary


def quantized_weight_int(module: QuantModule) -> torch.Tensor:
    """Return the deployed integer weight tensor for a ``QuantModule``."""
    _ensure_weight_quantizer_ready(module)
    quantizer = module.weight_quantizer
    delta = getattr(quantizer, "delta", None)
    zero_point = getattr(quantizer, "zero_point", None)
    if delta is None or zero_point is None:
        raise RuntimeError("Weight quantizer must be initialized before packed export.")

    weight = module.weight.detach()
    scaled = weight / delta
    if isinstance(quantizer, AdaRoundQuantizer) and quantizer.alpha is not None:
        x_floor = torch.floor(scaled)
        x_int = x_floor + (quantizer.alpha.detach() >= 0).to(dtype=weight.dtype)
    else:
        x_int = torch.round(scaled)
    q_int = torch.clamp(x_int + zero_point, 0, int(getattr(quantizer, "n_levels", 2**int(quantizer.n_bits))) - 1)
    return torch.round(q_int).to(torch.int64).contiguous()


def _export_quant_module(
    name: str,
    module: QuantModule,
    weight_file: BinaryIO,
    aux_file: BinaryIO,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    quantizer = module.weight_quantizer
    recomputed_quantizer = _ensure_weight_quantizer_ready(module)
    bitwidth = int(getattr(quantizer, "n_bits", 32))
    q_int = quantized_weight_int(module)
    packed, pack_format = pack_unsigned_values(q_int, bits=bitwidth)
    offset = weight_file.tell()
    payload = packed.cpu().numpy().tobytes(order="C")
    weight_file.write(payload)

    layer_entry: dict[str, Any] = {
        "name": name,
        "module_type": _module_type(module),
        "weight_bits": bitwidth,
        "weight_quantizer_recomputed": recomputed_quantizer,
        "weight_shape": list(module.weight.shape),
        "weight_values": int(module.weight.numel()),
        "packed_weight": {
            "file": manifest["weight_file"],
            "offset_bytes": int(offset),
            "num_bytes": len(payload),
            "pack_format": pack_format,
            "storage_dtype": "uint8",
        },
        "weight_delta": _write_aux_tensor(aux_file, quantizer.delta, name=f"{name}.weight_quantizer.delta", role="weight_delta"),
        "weight_zero_point": _write_aux_tensor(
            aux_file,
            quantizer.zero_point,
            name=f"{name}.weight_quantizer.zero_point",
            role="weight_zero_point",
        ),
        "fwd_kwargs": _jsonify(getattr(module, "fwd_kwargs", {})),
    }

    act_quantizer = module.act_quantizer
    act_delta = getattr(act_quantizer, "delta", None)
    act_zero_point = getattr(act_quantizer, "zero_point", None)
    layer_entry["activation_bits"] = int(getattr(act_quantizer, "n_bits", 32))
    layer_entry["activation_disabled"] = bool(getattr(module, "disable_act_quant", False))
    layer_entry["activation_delta"] = (
        _write_aux_tensor(aux_file, act_delta, name=f"{name}.act_quantizer.delta", role="activation_delta")
        if isinstance(act_delta, torch.Tensor)
        else None
    )
    layer_entry["activation_zero_point"] = (
        _write_aux_tensor(aux_file, act_zero_point, name=f"{name}.act_quantizer.zero_point", role="activation_zero_point")
        if isinstance(act_zero_point, torch.Tensor)
        else None
    )
    return layer_entry, len(payload)


def _ensure_weight_quantizer_ready(module: QuantModule) -> bool:
    """Ensure weight delta/zero_point exist; recompute deterministic Uniform state if missing."""
    quantizer = module.weight_quantizer
    delta = getattr(quantizer, "delta", None)
    zero_point = getattr(quantizer, "zero_point", None)
    if delta is not None and zero_point is not None:
        return False
    if isinstance(quantizer, AdaRoundQuantizer):
        raise RuntimeError("AdaRound weight quantizer is missing delta/zero_point and cannot be recomputed safely.")
    recomputed_delta, recomputed_zero_point = quantizer.init_quantization_scale(
        module.weight.detach(),
        bool(getattr(quantizer, "channel_wise", False)),
    )
    quantizer.delta = recomputed_delta
    quantizer.zero_point = recomputed_zero_point
    quantizer.inited = True
    return True


def _write_aux_tensor(aux_file: BinaryIO, tensor: torch.Tensor, *, name: str, role: str) -> dict[str, Any]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected tensor for {name}, got {type(tensor)!r}")
    cpu_tensor = tensor.detach().cpu().to(torch.float32).contiguous()
    offset = aux_file.tell()
    payload = cpu_tensor.numpy().tobytes(order="C")
    aux_file.write(payload)
    return {
        "name": name,
        "role": role,
        "file": "aux_fp32.bin",
        "offset_bytes": int(offset),
        "num_bytes": len(payload),
        "shape": list(cpu_tensor.shape),
        "dtype": "float32",
    }


def _build_summary(
    model: torch.nn.Module,
    target_dir: Path,
    payload_stats: dict[str, int],
    *,
    source_checkpoint_path: str | Path | None,
    quant_checkpoint_path: str | Path | None,
) -> dict[str, Any]:
    files = {
        path.name: int(path.stat().st_size)
        for path in sorted(target_dir.iterdir())
        if path.is_file()
    }
    raw_payload_bytes = int(payload_stats["packed_weight_bytes"]) + int(payload_stats["aux_fp32_bytes"])
    report = build_model_size_report(
        model,
        source_checkpoint_path=source_checkpoint_path,
        quant_checkpoint_path=quant_checkpoint_path,
    )
    estimated = report["estimated_storage"]
    return {
        "format": "scrn_brecq_packed_deployment",
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "files": {
            "sizes_bytes": files,
            "total_export_file_size_bytes": sum(files.values()),
            "total_export_file_size_mib": _bytes_to_mib(sum(files.values())),
        },
        "payload": {
            **payload_stats,
            "raw_deployment_payload_bytes": raw_payload_bytes,
            "raw_deployment_payload_mib": _bytes_to_mib(raw_payload_bytes),
        },
        "reference_model_size": report,
        "comparison": {
            "estimated_packed_model_size_bytes": estimated["estimated_packed_model_size_bytes"],
            "estimated_packed_model_size_mib": estimated["estimated_packed_model_size_mib"],
            "estimated_model_compression_ratio": estimated["estimated_model_compression_ratio"],
            "raw_payload_to_estimated_packed_ratio": _safe_div(
                raw_payload_bytes,
                estimated["estimated_packed_model_size_bytes"],
            ),
            "export_files_to_estimated_packed_ratio": _safe_div(
                sum(files.values()),
                estimated["estimated_packed_model_size_bytes"],
            ),
        },
    }


def _pack_uint2(values: torch.Tensor) -> torch.Tensor:
    flat = _validate_unsigned_values(values, bits=2)
    padding = (-int(flat.numel())) % 4
    if padding:
        flat = torch.cat([flat, torch.zeros(padding, dtype=flat.dtype)])
    packed = flat[0::4] | (flat[1::4] << 2) | (flat[2::4] << 4) | (flat[3::4] << 6)
    return packed.to(torch.uint8).contiguous()


def _validate_unsigned_values(values: torch.Tensor, *, bits: int) -> torch.Tensor:
    flat = values.detach().cpu().to(torch.int64).flatten().contiguous()
    max_value = 2**int(bits) - 1
    if flat.numel() and (int(flat.min()) < 0 or int(flat.max()) > max_value):
        raise ValueError(f"Values for uint{bits} packing must be in [0, {max_value}].")
    return flat


def _is_recovery_only_parameter(name: str) -> bool:
    return ".weight_quantizer.alpha" in name or ".act_quantizer.delta" in name


def _module_type(module: QuantModule) -> str:
    if module.weight.ndim == 4:
        return "conv2d"
    if module.weight.ndim == 2:
        return "linear"
    return f"weight_ndim_{module.weight.ndim}"


def _jsonify(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bytes_to_mib(value: int | None) -> float | None:
    if value is None:
        return None
    return float(value) / 1024.0 / 1024.0


def _safe_div(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    denominator_value = float(denominator)
    if denominator_value == 0.0:
        return None
    return float(numerator) / denominator_value


__all__ = [
    "export_packed_deployment",
    "pack_uint4",
    "pack_unsigned_values",
    "quantized_weight_int",
]
