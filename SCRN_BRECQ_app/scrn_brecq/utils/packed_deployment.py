"""Load packed SCRN-BRECQ deployment artifacts for validation inference.

This module restores the compact ``packed_deployment`` export into a regular
PyTorch ``QuantModel`` by dequantizing packed integer weights back to FP32.
It verifies artifact completeness and numerical equivalence; it is not an
INT4/INT8 runtime kernel.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from SCRN_BRECQ_app.scrn_brecq.quant import QuantModule


SUPPORTED_FORMAT = "scrn_brecq_packed_deployment"
SUPPORTED_VERSION = 1


def unpack_uint4(values: torch.Tensor, *, num_values: int | None = None) -> torch.Tensor:
    """Unpack low-nibble-first uint4 values from a uint8 tensor."""
    packed = values.detach().cpu().to(torch.uint8).flatten()
    low = (packed & 0x0F).to(torch.int64)
    high = ((packed >> 4) & 0x0F).to(torch.int64)
    unpacked = torch.empty(packed.numel() * 2, dtype=torch.int64)
    unpacked[0::2] = low
    unpacked[1::2] = high
    if num_values is not None:
        return unpacked[: int(num_values)].contiguous()
    return unpacked.contiguous()


def unpack_unsigned_values(values: torch.Tensor, *, bits: int, num_values: int) -> torch.Tensor:
    """Unpack a packed unsigned integer payload according to bitwidth."""
    bitwidth = int(bits)
    if bitwidth == 4:
        return unpack_uint4(values, num_values=num_values)
    if bitwidth == 8:
        return values.detach().cpu().to(torch.uint8).flatten()[: int(num_values)].to(torch.int64).contiguous()
    if bitwidth == 2:
        return _unpack_uint2(values, num_values=num_values)
    raise ValueError(f"Packed deployment supports 2/4/8-bit weights, got {bits}")


def restore_packed_deployment(model: torch.nn.Module, packed_dir: str | Path) -> dict[str, Any]:
    """Restore a packed deployment directory into ``model``.

    Quantized weights are dequantized into each ``QuantModule.weight`` and
    ``QuantModule.org_weight``.  Non-quantized parameters, including biases,
    are restored from ``aux_fp32.bin``.  Weight quantization is disabled because
    the stored weights already represent the deployed quantized values.
    """
    root = Path(packed_dir)
    manifest = load_packed_manifest(root)
    weights_path = root / str(manifest["weight_file"])
    aux_path = root / str(manifest["aux_file"])
    if not weights_path.is_file():
        raise FileNotFoundError(f"Packed weight file does not exist: {weights_path}")
    if not aux_path.is_file():
        raise FileNotFoundError(f"Packed aux file does not exist: {aux_path}")

    modules = dict(model.named_modules())
    parameters = dict(model.named_parameters())
    summary = {
        "restored_quantized_layers": 0,
        "restored_non_quantized_tensors": 0,
        "restored_activation_quantizers": 0,
        "final_quant_state": manifest.get("final_quant_state") or {},
        "packed_dir": str(root),
    }

    for layer in manifest.get("layers", []):
        module_name = str(layer["name"])
        module = model if module_name == "_root" else modules.get(module_name)
        if not isinstance(module, QuantModule):
            raise KeyError(f"Manifest layer {module_name!r} does not point to a QuantModule.")
        q_int = _read_packed_weight(weights_path, layer)
        delta = _read_aux_tensor(aux_path, layer["weight_delta"])
        zero_point = _read_aux_tensor(aux_path, layer["weight_zero_point"])
        dequantized = (q_int.to(torch.float32) - zero_point) * delta
        with torch.no_grad():
            module.weight.copy_(dequantized.to(dtype=module.weight.dtype, device=module.weight.device))
            module.org_weight.copy_(dequantized.to(dtype=module.org_weight.dtype, device=module.org_weight.device))
        module.packed_weight_int = q_int.detach().clone()
        module.weight_quantizer.delta = delta.to(device=module.weight.device)
        module.weight_quantizer.zero_point = zero_point.to(device=module.weight.device)
        module.weight_quantizer.inited = True

        if layer.get("activation_delta") is not None:
            act_delta = _read_aux_tensor(aux_path, layer["activation_delta"]).to(device=module.weight.device)
            if isinstance(module.act_quantizer.delta, nn.Parameter) or "delta" in module.act_quantizer._parameters:
                module.act_quantizer.delta = nn.Parameter(act_delta)
            else:
                module.act_quantizer.delta = act_delta
            summary["restored_activation_quantizers"] += 1
        if layer.get("activation_zero_point") is not None:
            module.act_quantizer.zero_point = _read_aux_tensor(aux_path, layer["activation_zero_point"]).to(
                device=module.weight.device
            )
            module.act_quantizer.inited = True

        module.set_quant_state(False, False)
        summary["restored_quantized_layers"] += 1

    for entry in manifest.get("aux_tensors", []):
        name = str(entry["name"])
        parameter = parameters.get(name)
        if parameter is None:
            raise KeyError(f"Manifest aux tensor {name!r} does not point to a model parameter.")
        tensor = _read_aux_tensor(aux_path, entry).to(dtype=parameter.dtype, device=parameter.device)
        with torch.no_grad():
            parameter.copy_(tensor)
        summary["restored_non_quantized_tensors"] += 1

    for module in modules.values():
        if isinstance(module, QuantModule):
            with torch.no_grad():
                if module.bias is not None and module.org_bias is not None:
                    module.org_bias.copy_(module.bias.detach())
                module.org_weight.copy_(module.weight.detach())

    return summary


def load_packed_manifest(packed_dir: str | Path) -> dict[str, Any]:
    """Load and validate ``manifest.json`` from a packed deployment directory."""
    manifest_path = Path(packed_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Packed manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != SUPPORTED_FORMAT:
        raise ValueError(f"Unsupported packed deployment format: {manifest.get('format')!r}")
    if int(manifest.get("format_version", -1)) != SUPPORTED_VERSION:
        raise ValueError(f"Unsupported packed deployment version: {manifest.get('format_version')!r}")
    for key in ("weight_file", "aux_file", "layers", "aux_tensors"):
        if key not in manifest:
            raise KeyError(f"Packed manifest misses required key: {key}")
    return manifest


def _read_packed_weight(weights_path: Path, layer: Mapping[str, Any]) -> torch.Tensor:
    entry = layer["packed_weight"]
    offset = int(entry["offset_bytes"])
    num_bytes = int(entry["num_bytes"])
    with weights_path.open("rb") as handle:
        handle.seek(offset)
        payload = handle.read(num_bytes)
    if len(payload) != num_bytes:
        raise ValueError(f"Packed weight payload for {layer['name']!r} is truncated.")
    packed = torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()
    q_int = unpack_unsigned_values(packed, bits=int(layer["weight_bits"]), num_values=int(layer["weight_values"]))
    return q_int.view(*[int(dim) for dim in layer["weight_shape"]]).contiguous()


def _read_aux_tensor(aux_path: Path, entry: Mapping[str, Any]) -> torch.Tensor:
    if str(entry.get("dtype")) != "float32":
        raise ValueError(f"Only float32 aux tensors are supported, got {entry.get('dtype')!r} for {entry.get('name')!r}")
    offset = int(entry["offset_bytes"])
    num_bytes = int(entry["num_bytes"])
    with aux_path.open("rb") as handle:
        handle.seek(offset)
        payload = handle.read(num_bytes)
    if len(payload) != num_bytes:
        raise ValueError(f"Aux tensor payload for {entry.get('name')!r} is truncated.")
    tensor = torch.frombuffer(bytearray(payload), dtype=torch.float32).clone()
    shape = [int(dim) for dim in entry["shape"]]
    if not shape:
        return tensor.reshape(()).contiguous()
    return tensor.view(*shape).contiguous()


def _unpack_uint2(values: torch.Tensor, *, num_values: int) -> torch.Tensor:
    packed = values.detach().cpu().to(torch.uint8).flatten()
    unpacked = torch.empty(packed.numel() * 4, dtype=torch.int64)
    unpacked[0::4] = (packed & 0x03).to(torch.int64)
    unpacked[1::4] = ((packed >> 2) & 0x03).to(torch.int64)
    unpacked[2::4] = ((packed >> 4) & 0x03).to(torch.int64)
    unpacked[3::4] = ((packed >> 6) & 0x03).to(torch.int64)
    return unpacked[: int(num_values)].contiguous()


__all__ = [
    "load_packed_manifest",
    "restore_packed_deployment",
    "unpack_uint4",
    "unpack_unsigned_values",
]
