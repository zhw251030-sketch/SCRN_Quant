"""SCRN 复现代码的轻量 smoke check。

该脚本不依赖真实数据和 checkpoint，只用合成 patch 验证数据退化、SCRN 小模型前向
和指标工具是否能在当前环境中正常工作。
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from SCRN_BRECQ_app.scrn_repro.data import degrade_patch
from SCRN_BRECQ_app.scrn_repro.model import SCRN
from SCRN_BRECQ_app.scrn_repro.utils import set_random_seed, snr_db, ssim_score


def build_parser() -> argparse.ArgumentParser:
    """构建 smoke check 参数解析器。"""
    parser = argparse.ArgumentParser(description="Run a lightweight SCRN reproduction smoke check.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=1005)
    parser.add_argument("--size", type=int, default=16, help="Synthetic square patch size")
    return parser


def main() -> None:
    """执行 SCRN 复现 smoke check。"""
    args = build_parser().parse_args()
    set_random_seed(args.seed)
    device = select_device(args.device)
    clean = synthetic_patch(args.size)
    degraded, mask, info = degrade_patch(
        clean,
        missing_rate=0.25,
        snr_db=5.0,
        rng=np.random.default_rng(args.seed),
    )

    model = SCRN(
        dim=8,
        stage_depths=(1, 1, 1, 1, 1),
        head_dim=4,
        window_size=4,
        input_resolution=args.size,
    ).to(device)
    model.eval()
    with torch.no_grad():
        input_tensor = torch.from_numpy(degraded).view(1, 1, args.size, args.size).float().to(device)
        output = model(input_tensor)

    if tuple(output.shape) != (1, 1, args.size, args.size):
        raise RuntimeError(f"Unexpected SCRN output shape: {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise RuntimeError("SCRN output contains NaN or Inf.")

    metrics = {
        "device": str(device),
        "input_shape": list(input_tensor.shape),
        "output_shape": list(output.shape),
        "missing_rate": info.missing_rate,
        "snr_db_setting": info.snr_db,
        "mask_keep_ratio": float(mask.mean()),
        "input_snr_db": snr_db(degraded, clean),
        "input_ssim": ssim_score(degraded, clean, win_size=15),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


def synthetic_patch(size: int) -> np.ndarray:
    """生成一个稳定的二维地震风格合成 patch。"""
    if size < 16:
        raise ValueError("--size must be at least 16 so SSIM win_size=15 is valid.")
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    patch = np.sin(4.0 * np.pi * grid_x) * np.cos(3.0 * np.pi * grid_y)
    patch += 0.25 * np.sin(10.0 * np.pi * (grid_x + grid_y))
    patch = patch.astype(np.float32)
    patch /= max(float(np.max(np.abs(patch))), 1e-6)
    return patch


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
