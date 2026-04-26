"""SCRN-BRECQ 量化与评估命令行入口。

本脚本把前八部分实现的组件串成完整流程：加载 SCRN checkpoint、构造 QuantModel、
收集 calibration data、执行 BRECQ reconstruction、保存量化 checkpoint，并在 SCRN
默认测试样本上计算 SNR/SSIM。脚本只组织流程，量化算法仍放在 `quant/` 子包中。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from SCRN_BRECQ_app.scrn_brecq.data import CalibrationDataConfig, load_calibration_data
from SCRN_BRECQ_app.scrn_brecq.model import load_scrn_for_brecq
from SCRN_BRECQ_app.scrn_brecq.quant import (
    BaseQuantBlock,
    QuantModel,
    QuantModule,
    block_reconstruction,
    layer_reconstruction,
)
from SCRN_BRECQ_app.scrn_brecq.utils import load_json
from SCRN_BRECQ_app.scrn_repro.training import collect_environment, create_run_dir, write_json, write_summary
from SCRN_BRECQ_app.scrn_repro.utils import set_random_seed, snr_db, ssim_score


DEFAULT_CONFIG_PATH = Path("SCRN_BRECQ_app/scrn_brecq/configs/default_quant_config.json")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    大部分参数默认从 JSON 配置读取；CLI 只覆盖用户显式传入的字段，便于把完整实验
    配置写入 run 目录复现。
    """
    parser = argparse.ArgumentParser(description="Run SCRN-BRECQ quantization and evaluation.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="SCRN-BRECQ quant config JSON")
    parser.add_argument("--scrn-checkpoint", default=None, help="SCRN FP32 checkpoint path")
    parser.add_argument("--calibration-dataset-dir", default=None, help="Calibration clean patch directory")
    parser.add_argument("--run-root", default=None, help="Quantization run output root")
    parser.add_argument("--run-name", default=None, help="Run name suffix")
    parser.add_argument("--eval-clean-path", default=None, help="Evaluation clean reference .npy")
    parser.add_argument("--eval-input-path", default=None, help="Evaluation degraded input .npy")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--init-batch-size", type=int, default=None)
    parser.add_argument("--n-bits-w", type=int, default=None)
    parser.add_argument("--n-bits-a", type=int, default=None)
    parser.add_argument("--iters-w", type=int, default=None)
    parser.add_argument("--iters-a", type=int, default=None)
    parser.add_argument("--rounding-loss-weight", type=float, default=None)
    parser.add_argument("--b-start", type=float, default=None)
    parser.add_argument("--b-end", type=float, default=None)
    parser.add_argument("--warmup", type=float, default=None)
    parser.add_argument("--activation-lr", type=float, default=None)
    parser.add_argument("--lp-norm", type=float, default=None)
    parser.add_argument("--scale-method", choices=["max", "max_scale", "mse"], default=None)
    parser.add_argument("--opt-mode", choices=["mse", "fisher_diag", "fisher_full"], default=None)

    parser.add_argument("--channel-wise", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--act-quant", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--disable-8bit-head-stem", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-figure", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--asym", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main() -> None:
    """命令行主流程。"""
    args = build_parser().parse_args()
    config = load_and_resolve_config(args)
    set_random_seed(int(config["seed"]))
    device = select_device(str(config["device"]))
    run_dir = create_run_dir(config["run_root"], run_name=str(config["run_name"]))

    print(f"[SCRN-BRECQ] run_dir={run_dir}", flush=True)
    loaded = load_scrn_for_brecq(config["scrn_checkpoint"], device=device)
    quant_model = build_quant_model(loaded.model, config).to(device)
    calibration_data = load_calibration_data(
        CalibrationDataConfig(
            dataset_dir=config["calibration_dataset_dir"],
            num_samples=int(config["num_samples"]),
            batch_size=int(config["batch_size"]),
            num_workers=int(config["num_workers"]),
            seed=int(config["seed"]),
            device=device,
            pin_memory=(device.type == "cuda"),
        )
    )

    write_json(run_dir / "config.json", build_run_config(config, loaded, device))
    run_reconstruction(quant_model, calibration_data, config)
    metrics, prediction = evaluate_quant_model(
        quant_model,
        clean_path=config["eval_clean_path"],
        input_path=config["eval_input_path"],
        device=device,
    )

    np.save(run_dir / "prediction.npy", prediction)
    write_json(run_dir / "metrics.json", metrics)
    if bool(config["save_figure"]):
        save_comparison_figure(
            run_dir / "comparison.png",
            clean=np.load(config["eval_clean_path"]).astype(np.float32),
            degraded=np.load(config["eval_input_path"]).astype(np.float32),
            prediction=prediction,
            metrics=metrics,
        )

    checkpoint_path = save_quant_checkpoint(run_dir, quant_model, loaded, config, metrics)
    write_summary(
        run_dir / "summary.md",
        title="SCRN-BRECQ Quantization Run",
        sections={
            "Metrics": metrics,
            "Artifacts": {
                "checkpoint": checkpoint_path,
                "prediction": run_dir / "prediction.npy",
                "config": run_dir / "config.json",
                "metrics": run_dir / "metrics.json",
            },
            "Quantization": {
                "n_bits_w": config["n_bits_w"],
                "n_bits_a": config["n_bits_a"],
                "act_quant": config["act_quant"],
                "num_samples": config["num_samples"],
                "iters_w": config["iters_w"],
                "iters_a": config["iters_a"],
            },
        },
    )
    print(
        "before_snr={before_snr_db:.4f} after_snr={after_snr_db:.4f} "
        "before_ssim={before_ssim:.4f} after_ssim={after_ssim:.4f} seconds={inference_seconds:.4f}".format(
            **metrics
        ),
        flush=True,
    )
    print(f"[SCRN-BRECQ] checkpoint={checkpoint_path}", flush=True)


def load_and_resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    """读取 JSON 配置，并用 CLI 显式参数覆盖。"""
    config = dict(load_json(args.config))
    config["config_path"] = str(args.config)
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        config[key] = value
    return normalize_config(config)


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """补齐和校验第九部分需要的配置字段。"""
    defaults = {
        "run_root": "SCRN_BRECQ_app/scrn_brecq/runs/quant",
        "run_name": "scrn_brecq_quant",
        "eval_clean_path": "SCRN-main/test_data/clear.npy",
        "eval_input_path": "SCRN-main/test_data/noise_and_miss.npy",
        "save_figure": False,
        "opt_mode": "mse",
        "asym": True,
        "init_batch_size": 64,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)

    int_keys = ["seed", "n_bits_w", "n_bits_a", "num_samples", "batch_size", "num_workers", "iters_w", "iters_a", "init_batch_size"]
    for key in int_keys:
        config[key] = int(config[key])
    float_keys = ["rounding_loss_weight", "b_start", "b_end", "warmup", "activation_lr", "lp_norm"]
    for key in float_keys:
        config[key] = float(config[key])
    bool_keys = ["channel_wise", "act_quant", "disable_8bit_head_stem", "save_figure", "asym"]
    for key in bool_keys:
        config[key] = bool(config[key])

    positive_keys = ["n_bits_w", "n_bits_a", "num_samples", "batch_size", "iters_w", "iters_a", "init_batch_size"]
    for key in positive_keys:
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive, got {config[key]}")
    if config["num_workers"] < 0:
        raise ValueError(f"num_workers must be non-negative, got {config['num_workers']}")
    if config["opt_mode"] not in {"mse", "fisher_diag", "fisher_full"}:
        raise ValueError(f"Unsupported opt_mode: {config['opt_mode']}")
    for path_key in ["scrn_checkpoint", "calibration_dataset_dir", "eval_clean_path", "eval_input_path"]:
        if not Path(config[path_key]).exists():
            raise FileNotFoundError(f"{path_key} does not exist: {config[path_key]}")
    return config


def select_device(device_arg: str) -> torch.device:
    """解析设备参数。"""
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "cuda":
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device("cpu")


def build_quant_model(model: torch.nn.Module, config: dict[str, Any]) -> QuantModel:
    """根据配置构造 SCRN QuantModel。"""
    weight_quant_params = {
        "n_bits": int(config["n_bits_w"]),
        "channel_wise": bool(config["channel_wise"]),
        "scale_method": str(config["scale_method"]),
    }
    act_quant_params = {
        "n_bits": int(config["n_bits_a"]),
        "channel_wise": False,
        "scale_method": str(config["scale_method"]),
        "leaf_param": bool(config["act_quant"]),
    }
    quant_model = QuantModel(model, weight_quant_params=weight_quant_params, act_quant_params=act_quant_params)
    if not bool(config["disable_8bit_head_stem"]):
        quant_model.set_first_last_layer_to_8bit()
    return quant_model


def run_reconstruction(quant_model: QuantModel, calibration_data: torch.Tensor, config: dict[str, Any]) -> None:
    """执行 W-only，并按需执行 W+A reconstruction。"""
    device = next(quant_model.parameters()).device
    init_inputs = calibration_data[: min(int(config["init_batch_size"]), int(calibration_data.size(0)))].to(device)

    quant_model.eval()
    quant_model.set_quant_state(True, False)
    with torch.no_grad():
        _ = quant_model(init_inputs)

    weight_kwargs = {
        "cali_data": calibration_data,
        "batch_size": int(config["batch_size"]),
        "iters": int(config["iters_w"]),
        "weight": float(config["rounding_loss_weight"]),
        "opt_mode": str(config["opt_mode"]),
        "asym": bool(config["asym"]),
        "b_range": (float(config["b_start"]), float(config["b_end"])),
        "warmup": float(config["warmup"]),
        "act_quant": False,
        "p": float(config["lp_norm"]),
    }
    reconstruct_model(quant_model, quant_model.model, weight_kwargs)
    quant_model.set_quant_state(True, False)

    if bool(config["act_quant"]):
        quant_model.set_quant_state(True, True)
        with torch.no_grad():
            _ = quant_model(init_inputs)
        quant_model.disable_network_output_quantization()
        act_kwargs = {
            "cali_data": calibration_data,
            "batch_size": int(config["batch_size"]),
            "iters": int(config["iters_a"]),
            "opt_mode": "mse",
            "asym": False,
            "act_quant": True,
            "lr": float(config["activation_lr"]),
            "p": float(config["lp_norm"]),
        }
        reconstruct_model(quant_model, quant_model.model, act_kwargs)
        quant_model.set_quant_state(True, True)


def reconstruct_model(quant_model: QuantModel, module: torch.nn.Module, reconstruction_kwargs: dict[str, Any], prefix: str = "") -> None:
    """递归遍历 QuantModel，执行 layer 或 block reconstruction。

    遇到 `BaseQuantBlock` 后不再进入其内部，避免 block 内 `QuantModule` 被重复重构。
    """
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, BaseQuantBlock):
            if child.ignore_reconstruction:
                print(f"[SCRN-BRECQ] skip block {full_name}", flush=True)
                continue
            print(f"[SCRN-BRECQ] reconstruct block {full_name}", flush=True)
            block_reconstruction(quant_model, child, **reconstruction_kwargs)
        elif isinstance(child, QuantModule):
            if child.ignore_reconstruction:
                print(f"[SCRN-BRECQ] skip layer {full_name}", flush=True)
                continue
            print(f"[SCRN-BRECQ] reconstruct layer {full_name}", flush=True)
            layer_reconstruction(quant_model, child, **reconstruction_kwargs)
        else:
            reconstruct_model(quant_model, child, reconstruction_kwargs, full_name)


def evaluate_quant_model(
    model: torch.nn.Module,
    *,
    clean_path: str | Path,
    input_path: str | Path,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    """在 SCRN 默认测试 `.npy` 上评估量化模型。"""
    clean = np.load(clean_path).astype(np.float32)
    degraded = np.load(input_path).astype(np.float32)
    if clean.shape != degraded.shape:
        raise ValueError(f"clean and input shapes differ: {clean.shape} vs {degraded.shape}")

    model.eval()
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
    return metrics, prediction


def save_quant_checkpoint(
    run_dir: Path,
    quant_model: QuantModel,
    loaded,
    config: dict[str, Any],
    metrics: dict[str, float],
) -> Path:
    """保存量化模型 checkpoint。"""
    checkpoint_path = run_dir / "checkpoints" / "quantized_scrn_brecq.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "quant_model_state_dict": quant_model.state_dict(),
        "model_config": asdict(loaded.config),
        "quant_config": serializable_config(config),
        "source_checkpoint": str(loaded.checkpoint_path),
        "source_checkpoint_epoch": loaded.epoch,
        "source_checkpoint_loss": loaded.loss,
        "final_quant_state": {"weight_quant": True, "act_quant": bool(config["act_quant"])},
        "metrics": metrics,
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def build_run_config(config: dict[str, Any], loaded, device: torch.device) -> dict[str, Any]:
    """构建写入 run `config.json` 的配置快照。"""
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "quant_config": serializable_config(config),
        "source_checkpoint": str(loaded.checkpoint_path),
        "source_checkpoint_epoch": loaded.epoch,
        "source_checkpoint_loss": loaded.loss,
        "model_config": asdict(loaded.config),
        "environment": collect_environment(),
    }


def serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    """把 Path/torch 类型转成 JSON 友好的普通对象。"""
    output: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, Path):
            output[key] = str(value)
        elif isinstance(value, torch.device):
            output[key] = str(value)
        else:
            output[key] = value
    return output


def save_comparison_figure(
    path: Path,
    *,
    clean: np.ndarray,
    degraded: np.ndarray,
    prediction: np.ndarray,
    metrics: dict[str, float],
) -> None:
    """保存 clean/input/output 对比图。"""
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

    image = None
    for axis, (array, title) in zip(axes, panels):
        image = axis.imshow(array, cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
    fig.colorbar(image, ax=axes, shrink=0.75)
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
