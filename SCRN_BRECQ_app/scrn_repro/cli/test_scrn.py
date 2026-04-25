"""SCRN 独立复现测试入口。

默认使用 SCRN-main/test_data 中的 clear/noise_and_miss 数据作为复现对照，
但不导入 SCRN-main 的任何 Python 文件。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from SCRN_BRECQ_app.scrn_repro.model import SCRN
from SCRN_BRECQ_app.scrn_repro.training import collect_environment, create_run_dir, load_checkpoint, write_json, write_summary
from SCRN_BRECQ_app.scrn_repro.utils import snr_db, ssim_score


DEFAULT_CLEAN_PATH = "SCRN-main/test_data/clear.npy"
DEFAULT_INPUT_PATH = "SCRN-main/test_data/noise_and_miss.npy"
DEFAULT_RUN_ROOT = "SCRN_BRECQ_app/scrn_repro/runs/test"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a reproduced SCRN checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="训练产生的 checkpoint 路径")
    parser.add_argument("--clean-path", default=DEFAULT_CLEAN_PATH, help="干净参考数据 .npy")
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH, help="带噪/缺道输入数据 .npy")
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT, help="测试 run 输出根目录")
    parser.add_argument("--run-name", default="scrn_test", help="run 名称，会和时间戳组合成目录名")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--save-figure", action="store_true", help="保存输入/输出/参考对比图")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_path = _require_file(args.checkpoint, "checkpoint")
    clean_path = _require_file(args.clean_path, "clean reference")
    input_path = _require_file(args.input_path, "test input")

    device = _select_device(args.device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model = _build_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    clean = np.load(clean_path).astype(np.float32)
    degraded = np.load(input_path).astype(np.float32)
    if clean.shape != degraded.shape:
        raise ValueError(f"clean and input shapes differ: {clean.shape} vs {degraded.shape}")

    run_dir = create_run_dir(args.run_root, run_name=args.run_name)
    _write_config(run_dir, args, checkpoint_path, clean_path, input_path, device, checkpoint)

    start = time.time()
    with torch.no_grad():
        tensor = torch.from_numpy(degraded).view(1, 1, degraded.shape[0], degraded.shape[1]).float().to(device)
        prediction = model(tensor).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    elapsed = time.time() - start

    metrics = {
        "before_snr_db": snr_db(degraded, clean),
        "after_snr_db": snr_db(prediction, clean),
        "before_ssim": ssim_score(degraded, clean),
        "after_ssim": ssim_score(prediction, clean),
        "inference_seconds": elapsed,
    }
    np.save(run_dir / "prediction.npy", prediction)
    write_json(run_dir / "metrics.json", metrics)
    if args.save_figure:
        _save_comparison_figure(run_dir / "comparison.png", clean, degraded, prediction, metrics)
    write_summary(
        run_dir / "summary.md",
        title="SCRN Test Run",
        sections={
            "Metrics": metrics,
            "Inputs": {
                "checkpoint": checkpoint_path,
                "clean": clean_path,
                "input": input_path,
                "prediction": run_dir / "prediction.npy",
            },
        },
    )
    print(
        "before_snr={before_snr_db:.4f} after_snr={after_snr_db:.4f} "
        "before_ssim={before_ssim:.4f} after_ssim={after_ssim:.4f} seconds={inference_seconds:.4f}".format(**metrics),
        flush=True,
    )


def _build_model_from_checkpoint(checkpoint: dict[str, Any]) -> SCRN:
    config = checkpoint.get("model_config", {})
    return SCRN(
        dim=int(config.get("dim", 64)),
        stage_depths=tuple(config.get("stage_depths", (1, 1, 1, 1, 1))),
        head_dim=int(config.get("head_dim", 32)),
        window_size=int(config.get("window_size", 8)),
        drop_path_rate=float(config.get("drop_path_rate", 0.0)),
        input_resolution=int(config.get("input_resolution", 128)),
    )


def _write_config(
    run_dir: Path,
    args: argparse.Namespace,
    checkpoint_path: Path,
    clean_path: Path,
    input_path: Path,
    device: torch.device,
    checkpoint: dict[str, Any],
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_loss": checkpoint.get("loss"),
        "clean_path": str(clean_path),
        "input_path": str(input_path),
        "environment": collect_environment(),
    }
    write_json(run_dir / "config.json", payload)


def _save_comparison_figure(path: Path, clean: np.ndarray, degraded: np.ndarray, prediction: np.ndarray, metrics: dict[str, float]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Saving comparison figures requires installing matplotlib.") from exc

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    panels = [
        (clean, "Ground Truth"),
        (degraded, f"Input SNR={metrics['before_snr_db']:.2f}dB"),
        (prediction, f"Output SNR={metrics['after_snr_db']:.2f}dB"),
    ]
    vmin = float(np.min(clean))
    vmax = float(np.max(clean))
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0

    im = None
    for axis, (image, title) in zip(axes, panels):
        im = axis.imshow(image, cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.75)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "cuda":
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device("cpu")


def _require_file(path: str, description: str) -> Path:
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(f"{description} does not exist: {candidate}")
    return candidate


if __name__ == "__main__":
    main()
