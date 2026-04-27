"""SCRN-BRECQ 量化与评估命令行入口。

本脚本把前八部分实现的组件串成完整流程：加载 SCRN checkpoint、构造 QuantModel、
收集 calibration data、执行 BRECQ reconstruction、保存量化 checkpoint，并在 SCRN
默认测试样本上计算 SNR/SSIM。脚本只组织流程，量化算法仍放在 `quant/` 子包中。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

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
    parser.add_argument("--gpus", default=None, help="Visible GPU ids, e.g. `0` or `0,1,2,3`; multi-GPU requires torchrun")

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
    parser.add_argument("--distributed", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--asym", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main() -> None:
    """命令行主流程。"""
    run_start_time = time.time()
    args = build_parser().parse_args()
    config = load_and_resolve_config(args)
    configure_visible_gpus(config)
    distributed_state = init_distributed(config)
    rank = int(distributed_state["rank"])
    world_size = int(distributed_state["world_size"])
    is_main = rank == 0
    device = select_device(str(config["device"]), distributed_state)
    set_random_seed(int(config["seed"]))

    run_dir = create_run_dir(config["run_root"], run_name=str(config["run_name"])) if is_main else None
    run_dir = share_run_dir(run_dir, bool(config["distributed"]))

    try:
        if is_main:
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
                rank=rank,
                world_size=world_size,
            )
        )

        if is_main:
            write_json(run_dir / "config.json", build_run_config(config, loaded, device, distributed_state))
            clean, degraded = load_eval_arrays(config["eval_clean_path"], config["eval_input_path"])

            quant_model.set_quant_state(False, False)
            fp32_prediction, fp32_seconds = predict_array(quant_model, degraded, device)
        else:
            clean = degraded = None
            fp32_prediction = quant_pre_recon_prediction = None
            fp32_seconds = quant_pre_recon_seconds = 0.0

        initialize_weight_quantization(quant_model, calibration_data, config)
        quant_model.set_quant_state(True, False)
        if is_main:
            quant_pre_recon_prediction, quant_pre_recon_seconds = predict_array(quant_model, degraded, device)
        barrier_if_distributed(bool(config["distributed"]))

        reconstruction_start_time = time.time()
        run_reconstruction(quant_model, calibration_data, config, is_main=is_main)
        barrier_if_distributed(bool(config["distributed"]))
        reconstruction_seconds = time.time() - reconstruction_start_time

        if is_main:
            quant_model.set_quant_state(True, bool(config["act_quant"]))
            quant_post_recon_prediction, quant_post_recon_seconds = predict_array(quant_model, degraded, device)
            metrics = build_comparison_metrics(
                clean,
                degraded,
                fp32_prediction=fp32_prediction,
                fp32_seconds=fp32_seconds,
                quant_pre_recon_prediction=quant_pre_recon_prediction,
                quant_pre_recon_seconds=quant_pre_recon_seconds,
                quant_post_recon_prediction=quant_post_recon_prediction,
                quant_post_recon_seconds=quant_post_recon_seconds,
            )
            add_timing_metrics(metrics, run_start_time=run_start_time, reconstruction_seconds=reconstruction_seconds)

            np.save(run_dir / "fp32_prediction.npy", fp32_prediction)
            np.save(run_dir / "quant_pre_recon_prediction.npy", quant_pre_recon_prediction)
            np.save(run_dir / "quant_post_recon_prediction.npy", quant_post_recon_prediction)
            np.save(run_dir / "prediction.npy", quant_post_recon_prediction)
            write_json(run_dir / "metrics.json", metrics)
            if bool(config["save_figure"]):
                save_comparison_figure(
                    run_dir / "comparison.png",
                    clean=clean,
                    degraded=degraded,
                    fp32_prediction=fp32_prediction,
                    quant_pre_recon_prediction=quant_pre_recon_prediction,
                    quant_post_recon_prediction=quant_post_recon_prediction,
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
                        "fp32_prediction": run_dir / "fp32_prediction.npy",
                        "quant_pre_recon_prediction": run_dir / "quant_pre_recon_prediction.npy",
                        "quant_post_recon_prediction": run_dir / "quant_post_recon_prediction.npy",
                        "comparison": run_dir / "comparison.png" if bool(config["save_figure"]) else "disabled",
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
                    "Distributed": distributed_state,
                },
            )
            print(
                "input_snr={input_snr_db:.4f} fp32_snr={fp32_snr_db:.4f} "
                "pre_recon_snr={quant_pre_recon_snr_db:.4f} post_recon_snr={quant_post_recon_snr_db:.4f} "
                "input_ssim={input_ssim:.4f} post_recon_ssim={quant_post_recon_ssim:.4f} "
                "inference_seconds={quant_post_recon_inference_seconds:.4f} "
                "reconstruction_seconds={reconstruction_seconds:.2f} elapsed_seconds={elapsed_seconds:.2f}".format(
                    **metrics
                ),
                flush=True,
            )
            print(f"[SCRN-BRECQ] checkpoint={checkpoint_path}", flush=True)
    finally:
        if bool(config["distributed"]) and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


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
        "distributed": False,
        "gpus": "",
    }
    for key, value in defaults.items():
        config.setdefault(key, value)

    int_keys = ["seed", "n_bits_w", "n_bits_a", "num_samples", "batch_size", "num_workers", "iters_w", "iters_a", "init_batch_size"]
    for key in int_keys:
        config[key] = int(config[key])
    float_keys = ["rounding_loss_weight", "b_start", "b_end", "warmup", "activation_lr", "lp_norm"]
    for key in float_keys:
        config[key] = float(config[key])
    bool_keys = ["channel_wise", "act_quant", "disable_8bit_head_stem", "save_figure", "asym", "distributed"]
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
    if config["distributed"] and config["act_quant"]:
        raise NotImplementedError("Distributed SCRN-BRECQ currently supports W-only reconstruction; set --no-act-quant.")
    if config["distributed"] and config["device"] == "cpu":
        raise ValueError("--distributed requires CUDA; do not use --device cpu.")
    if _visible_gpu_count_from_text(str(config.get("gpus", ""))) > 1 and not config["distributed"]:
        raise ValueError("Multiple GPUs require --distributed and torchrun. Use --gpus with --distributed.")
    for path_key in ["scrn_checkpoint", "calibration_dataset_dir", "eval_clean_path", "eval_input_path"]:
        if not Path(config[path_key]).exists():
            raise FileNotFoundError(f"{path_key} does not exist: {config[path_key]}")
    return config


def configure_visible_gpus(config: dict[str, Any]) -> None:
    """根据 `--gpus` 限制当前进程可见 CUDA 设备。"""
    gpus = str(config.get("gpus", "")).strip()
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def init_distributed(config: dict[str, Any]) -> dict[str, int]:
    """初始化 torchrun 分布式环境，单进程时返回默认 rank 信息。"""
    if not bool(config["distributed"]):
        return {"enabled": False, "rank": 0, "local_rank": 0, "world_size": 1}

    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(f"--distributed requires torchrun environment variables, missing: {missing}")
    if not torch.cuda.is_available():
        raise RuntimeError("--distributed requires CUDA but torch.cuda.is_available() is False")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    visible_count = visible_gpu_count(config)
    if visible_count < world_size:
        raise ValueError(f"Visible GPU count {visible_count} is smaller than WORLD_SIZE={world_size}")
    if local_rank >= visible_count:
        raise ValueError(f"LOCAL_RANK={local_rank} is outside visible GPU count {visible_count}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return {"enabled": True, "rank": rank, "local_rank": local_rank, "world_size": world_size}


def visible_gpu_count(config: dict[str, Any]) -> int:
    """返回当前配置下应当可见的 GPU 数量。"""
    configured = _visible_gpu_count_from_text(str(config.get("gpus", "")))
    if configured > 0:
        return configured
    env_count = _visible_gpu_count_from_text(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if env_count > 0:
        return env_count
    return int(torch.cuda.device_count())


def _visible_gpu_count_from_text(value: str) -> int:
    """解析 `0,1,2` 形式的 GPU 列表长度。"""
    value = str(value).strip()
    if not value or value == "-1":
        return 0
    return len([item for item in value.split(",") if item.strip()])


def share_run_dir(run_dir: Path | None, distributed: bool) -> Path:
    """分布式时由 rank 0 广播 run 目录。"""
    if not distributed:
        if run_dir is None:
            raise RuntimeError("run_dir was not created")
        return run_dir
    payload = [str(run_dir) if run_dir is not None else None]
    dist.broadcast_object_list(payload, src=0)
    if payload[0] is None:
        raise RuntimeError("Failed to broadcast run_dir from rank 0")
    return Path(payload[0])


def barrier_if_distributed(distributed: bool) -> None:
    """分布式流程同步点，单进程时不做处理。"""
    if distributed:
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def select_device(device_arg: str, distributed_state: dict[str, Any] | None = None) -> torch.device:
    """解析设备参数。"""
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg in {"auto", "cuda"} and torch.cuda.is_available():
        if distributed_state is not None and int(distributed_state.get("world_size", 1)) > 1:
            local_rank = int(distributed_state["local_rank"])
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)
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


def initialize_weight_quantization(
    quant_model: QuantModel,
    calibration_data: torch.Tensor,
    config: dict[str, Any],
) -> None:
    """初始化权重量化 scale/zero point，并保留 reconstruction 前状态用于评估。"""
    device = next(quant_model.parameters()).device
    init_inputs = calibration_data[: min(int(config["init_batch_size"]), int(calibration_data.size(0)))].to(device)
    quant_model.eval()
    quant_model.set_quant_state(True, False)
    with torch.no_grad():
        _ = quant_model(init_inputs)


def run_reconstruction(
    quant_model: QuantModel,
    calibration_data: torch.Tensor,
    config: dict[str, Any],
    *,
    is_main: bool = True,
) -> None:
    """执行 W-only，并按需执行 W+A reconstruction。"""
    device = next(quant_model.parameters()).device
    init_inputs = calibration_data[: min(int(config["init_batch_size"]), int(calibration_data.size(0)))].to(device)

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
        "multi_gpu": bool(config["distributed"]),
        "log_enabled": is_main,
    }
    reconstruct_model(quant_model, quant_model.model, weight_kwargs, log_enabled=is_main)
    quant_model.set_quant_state(True, False)

    if bool(config["act_quant"]):
        if bool(config["distributed"]):
            raise NotImplementedError("Distributed activation reconstruction is not supported yet.")
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
            "log_enabled": is_main,
        }
        reconstruct_model(quant_model, quant_model.model, act_kwargs, log_enabled=is_main)
        quant_model.set_quant_state(True, True)


def reconstruct_model(
    quant_model: QuantModel,
    module: torch.nn.Module,
    reconstruction_kwargs: dict[str, Any],
    prefix: str = "",
    *,
    log_enabled: bool = True,
) -> None:
    """递归遍历 QuantModel，执行 layer 或 block reconstruction。

    遇到 `BaseQuantBlock` 后不再进入其内部，避免 block 内 `QuantModule` 被重复重构。
    """
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, BaseQuantBlock):
            if child.ignore_reconstruction:
                if log_enabled:
                    print(f"[SCRN-BRECQ] skip block {full_name}", flush=True)
                continue
            if log_enabled:
                print(f"[SCRN-BRECQ] reconstruct block {full_name}", flush=True)
            block_reconstruction(quant_model, child, **reconstruction_kwargs)
        elif isinstance(child, QuantModule):
            if child.ignore_reconstruction:
                if log_enabled:
                    print(f"[SCRN-BRECQ] skip layer {full_name}", flush=True)
                continue
            if log_enabled:
                print(f"[SCRN-BRECQ] reconstruct layer {full_name}", flush=True)
            layer_reconstruction(quant_model, child, **reconstruction_kwargs)
        else:
            reconstruct_model(quant_model, child, reconstruction_kwargs, full_name, log_enabled=log_enabled)


def load_eval_arrays(clean_path: str | Path, input_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """读取 SCRN 默认测试 `.npy` 对，并检查形状一致。"""
    clean = np.load(clean_path).astype(np.float32)
    degraded = np.load(input_path).astype(np.float32)
    if clean.shape != degraded.shape:
        raise ValueError(f"clean and input shapes differ: {clean.shape} vs {degraded.shape}")
    return clean, degraded


def predict_array(model: torch.nn.Module, degraded: np.ndarray, device: torch.device) -> tuple[np.ndarray, float]:
    """对单张 degraded array 做模型推理，返回预测和耗时。"""
    model.eval()
    start = time.time()
    with torch.no_grad():
        tensor = torch.from_numpy(degraded).view(1, 1, degraded.shape[0], degraded.shape[1]).float().to(device)
        prediction = model(tensor).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    elapsed = time.time() - start
    return prediction, elapsed


def build_comparison_metrics(
    clean: np.ndarray,
    degraded: np.ndarray,
    *,
    fp32_prediction: np.ndarray,
    fp32_seconds: float,
    quant_pre_recon_prediction: np.ndarray,
    quant_pre_recon_seconds: float,
    quant_post_recon_prediction: np.ndarray,
    quant_post_recon_seconds: float,
) -> dict[str, float]:
    """构建五图对比对应的完整指标。"""
    metrics = {
        "input_snr_db": snr_db(degraded, clean),
        "input_ssim": ssim_score(degraded, clean),
        "fp32_snr_db": snr_db(fp32_prediction, clean),
        "fp32_ssim": ssim_score(fp32_prediction, clean),
        "fp32_inference_seconds": float(fp32_seconds),
        "quant_pre_recon_snr_db": snr_db(quant_pre_recon_prediction, clean),
        "quant_pre_recon_ssim": ssim_score(quant_pre_recon_prediction, clean),
        "quant_pre_recon_inference_seconds": float(quant_pre_recon_seconds),
        "quant_post_recon_snr_db": snr_db(quant_post_recon_prediction, clean),
        "quant_post_recon_ssim": ssim_score(quant_post_recon_prediction, clean),
        "quant_post_recon_inference_seconds": float(quant_post_recon_seconds),
    }
    # 保留旧字段名，方便已有 summary/脚本把最终量化结果当作 after 指标读取。
    metrics["before_snr_db"] = metrics["input_snr_db"]
    metrics["before_ssim"] = metrics["input_ssim"]
    metrics["after_snr_db"] = metrics["quant_post_recon_snr_db"]
    metrics["after_ssim"] = metrics["quant_post_recon_ssim"]
    metrics["inference_seconds"] = metrics["quant_post_recon_inference_seconds"]
    return metrics


def add_timing_metrics(metrics: dict[str, float], *, run_start_time: float, reconstruction_seconds: float) -> None:
    """补充量化流程耗时指标。

    `elapsed_seconds` 从 CLI 主流程开始计时，到最终量化推理完成后写入 metrics 前结束；
    `reconstruction_seconds` 覆盖 layer/block reconstruction，并在分布式模式下包含结束同步等待。
    """
    elapsed_seconds = time.time() - float(run_start_time)
    metrics["reconstruction_seconds"] = float(reconstruction_seconds)
    metrics["reconstruction_minutes"] = float(reconstruction_seconds) / 60.0
    metrics["elapsed_seconds"] = float(elapsed_seconds)
    metrics["elapsed_minutes"] = float(elapsed_seconds) / 60.0


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


def build_run_config(config: dict[str, Any], loaded, device: torch.device, distributed_state: dict[str, Any]) -> dict[str, Any]:
    """构建写入 run `config.json` 的配置快照。"""
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "distributed": dict(distributed_state),
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
    fp32_prediction: np.ndarray,
    quant_pre_recon_prediction: np.ndarray,
    quant_post_recon_prediction: np.ndarray,
    metrics: dict[str, float],
) -> None:
    """保存五图对比：GT、输入、FP32、量化重建前、量化重建后。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Saving comparison figures requires installing matplotlib.") from exc

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.8), constrained_layout=True)
    panels = [
        (clean, "Ground Truth"),
        (degraded, f"Input\nSNR={metrics['input_snr_db']:.2f}dB SSIM={metrics['input_ssim']:.3f}"),
        (fp32_prediction, f"FP32 SCRN\nSNR={metrics['fp32_snr_db']:.2f}dB SSIM={metrics['fp32_ssim']:.3f}"),
        (
            quant_pre_recon_prediction,
            "Quant Before Recon\n"
            f"SNR={metrics['quant_pre_recon_snr_db']:.2f}dB SSIM={metrics['quant_pre_recon_ssim']:.3f}",
        ),
        (
            quant_post_recon_prediction,
            "Quant After BRECQ\n"
            f"SNR={metrics['quant_post_recon_snr_db']:.2f}dB SSIM={metrics['quant_post_recon_ssim']:.3f}",
        ),
    ]
    vmin = float(np.min(clean))
    vmax = float(np.max(clean))
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0

    image = None
    for axis, (array, title) in zip(axes, panels):
        image = axis.imshow(array, cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax)
        # 五图并排时标题需要显式压小，避免指标文字遮挡图像区域。
        axis.set_title(title, fontsize=9, pad=6)
        axis.axis("off")
    fig.colorbar(image, ax=axes, shrink=0.72, fraction=0.02, pad=0.01)
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
