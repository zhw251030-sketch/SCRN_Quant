"""SCRN-BRECQ 量化代码的轻量 smoke check。

该脚本不依赖真实 checkpoint 和 calibration 数据，只用缩小版 SCRN 与合成输入验证
QuantModel 包装、量化状态切换和量化前向是否可用。
"""

from __future__ import annotations

import argparse
import json

import torch

from SCRN_BRECQ_app.scrn_brecq.quant import BaseQuantBlock, QuantModel, QuantModule
from SCRN_BRECQ_app.scrn_repro.model import SCRN
from SCRN_BRECQ_app.scrn_repro.utils import set_random_seed


def build_parser() -> argparse.ArgumentParser:
    """构建 smoke check 参数解析器。"""
    parser = argparse.ArgumentParser(description="Run a lightweight SCRN-BRECQ quantization smoke check.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=1005)
    parser.add_argument("--size", type=int, default=16, help="Synthetic square input size")
    return parser


def main() -> None:
    """执行 SCRN-BRECQ smoke check。"""
    args = build_parser().parse_args()
    set_random_seed(args.seed)
    device = select_device(args.device)
    model = SCRN(
        dim=8,
        stage_depths=(1, 1, 1, 1, 1),
        head_dim=4,
        window_size=4,
        input_resolution=args.size,
    )
    quant_model = QuantModel(
        model,
        weight_quant_params={"n_bits": 4, "channel_wise": True, "scale_method": "max"},
        act_quant_params={"n_bits": 4, "channel_wise": False, "scale_method": "max"},
    ).to(device)
    quant_model.eval()

    input_tensor = torch.randn(1, 1, args.size, args.size, device=device)
    quant_model.set_quant_state(False, False)
    with torch.no_grad():
        fp32_output = quant_model(input_tensor)
    quant_model.set_quant_state(True, False)
    with torch.no_grad():
        weight_quant_output = quant_model(input_tensor)
    quant_model.set_quant_state(True, True)
    with torch.no_grad():
        weight_act_quant_output = quant_model(input_tensor)

    expected_shape = (1, 1, args.size, args.size)
    for name, output in {
        "fp32": fp32_output,
        "weight_quant": weight_quant_output,
        "weight_act_quant": weight_act_quant_output,
    }.items():
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(f"{name} output shape mismatch: {tuple(output.shape)}")
        if not torch.isfinite(output).all():
            raise RuntimeError(f"{name} output contains NaN or Inf.")

    quant_module_count = sum(isinstance(module, QuantModule) for module in quant_model.modules())
    quant_block_count = sum(isinstance(module, BaseQuantBlock) for module in quant_model.modules())
    if quant_module_count <= 0:
        raise RuntimeError("QuantModel did not create any QuantModule.")
    if quant_block_count != 5:
        raise RuntimeError(f"Expected 5 SCRN quant blocks, got {quant_block_count}.")

    metrics = {
        "device": str(device),
        "input_shape": list(input_tensor.shape),
        "output_shape": list(weight_act_quant_output.shape),
        "quant_modules": quant_module_count,
        "quant_blocks": quant_block_count,
        "fp32_mean": float(fp32_output.mean().item()),
        "weight_quant_mean": float(weight_quant_output.mean().item()),
        "weight_act_quant_mean": float(weight_act_quant_output.mean().item()),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


def select_device(device_arg: str) -> torch.device:
    """解析 smoke check 设备参数。"""
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "cuda":
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
