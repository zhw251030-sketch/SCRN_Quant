"""评估已保存的 SCRN-BRECQ 量化 checkpoint。

`quantize_scrn.py` 保存的 checkpoint 中包含 QuantModel 的 state_dict、SCRN 结构配置
和量化配置。本脚本负责重新构建同构量化模型、加载权重，并在指定 clean/input
测试对上输出预测、指标和 summary。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from SCRN_BRECQ_app.scrn_brecq.quant import AdaRoundQuantizer, QuantModel, QuantModule
from SCRN_BRECQ_app.scrn_brecq.quant.activation_precision import (
    apply_activation_bitwidth_overrides,
    normalize_activation_bitwidth_overrides,
)
from SCRN_BRECQ_app.scrn_brecq.utils import build_model_size_report
from SCRN_BRECQ_app.scrn_repro.model import SCRNConfig, build_scrn_from_config
from SCRN_BRECQ_app.scrn_repro.training import collect_environment, create_run_dir, write_json, write_summary
from SCRN_BRECQ_app.scrn_repro.utils import snr_db, ssim_score


DEFAULT_RUN_ROOT = "SCRN_BRECQ_app/scrn_brecq/runs/quant_eval"


def build_parser() -> argparse.ArgumentParser:
    """构建量化 checkpoint 评估参数解析器。"""
    parser = argparse.ArgumentParser(description="Evaluate a saved SCRN-BRECQ quantized checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to quantized_scrn_brecq.pth")
    parser.add_argument("--eval-clean-path", default=None, help="Clean reference .npy; defaults to checkpoint quant_config")
    parser.add_argument("--eval-input-path", default=None, help="Degraded input .npy; defaults to checkpoint quant_config")
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT, help="Evaluation run output root")
    parser.add_argument("--run-name", default="quantized_eval", help="Run name suffix")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--save-figure", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main() -> None:
    """命令行主流程。"""
    args = build_parser().parse_args()
    device = select_device(args.device)
    checkpoint_path = require_file(args.checkpoint, "quantized checkpoint")
    checkpoint = load_quant_checkpoint(checkpoint_path)
    quant_config = normalize_quant_config(checkpoint.get("quant_config", {}))
    clean_path, input_path = resolve_eval_paths(args, quant_config)
    save_figure = bool(quant_config.get("save_figure", False)) if args.save_figure is None else bool(args.save_figure)

    quant_model = build_quant_model_from_checkpoint(checkpoint)
    state_dict = checkpoint["quant_model_state_dict"]
    restore_quantizer_state_shapes(quant_model, state_dict)
    quant_model.load_state_dict(state_dict, strict=True)
    quant_model.to(device)
    quant_model.eval()

    final_state = checkpoint.get("final_quant_state", {})
    weight_quant = bool(final_state.get("weight_quant", True))
    act_quant = bool(final_state.get("act_quant", quant_config.get("act_quant", False)))
    if act_quant:
        # 原量化流程在 activation quant 时关闭网络最终输出激活量化。
        quant_model.disable_network_output_quantization()
    quant_model.set_quant_state(weight_quant, act_quant)

    clean, degraded = load_eval_arrays(clean_path, input_path)
    prediction, seconds = predict_array(quant_model, degraded, device)
    metrics = build_metrics(clean, degraded, prediction, seconds)
    metrics["model_size"] = build_model_size_report(
        quant_model,
        source_checkpoint_path=checkpoint.get("source_checkpoint"),
        quant_checkpoint_path=checkpoint_path,
    )
    metrics["activation_bitwidth_summary"] = apply_activation_bitwidth_overrides(
        quant_model,
        quant_config["activation_bitwidth_overrides"],
    )

    run_dir = create_run_dir(args.run_root, run_name=args.run_name)
    np.save(run_dir / "prediction.npy", prediction)
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "config.json",
        build_run_config(
            checkpoint_path=checkpoint_path,
            clean_path=clean_path,
            input_path=input_path,
            device=device,
            checkpoint=checkpoint,
            quant_config=quant_config,
            save_figure=save_figure,
        ),
    )
    if save_figure:
        save_evaluation_figure(run_dir / "comparison.png", clean=clean, degraded=degraded, prediction=prediction, metrics=metrics)
    write_summary(
        run_dir / "summary.md",
        title="SCRN-BRECQ Quantized Checkpoint Evaluation",
        sections={
            "Metrics": metrics,
            "Model Size": metrics["model_size"],
            "Inputs": {
                "checkpoint": checkpoint_path,
                "clean": clean_path,
                "input": input_path,
                "prediction": run_dir / "prediction.npy",
            },
            "Quantization": {
                "weight_quant": weight_quant,
                "act_quant": act_quant,
                "n_bits_w": quant_config["n_bits_w"],
                "n_bits_a": quant_config["n_bits_a"],
                "activation_bitwidth_summary": metrics["activation_bitwidth_summary"],
            },
        },
    )
    print(
        "input_snr={input_snr_db:.4f} quant_snr={quant_snr_db:.4f} "
        "input_ssim={input_ssim:.4f} quant_ssim={quant_ssim:.4f} seconds={quant_inference_seconds:.4f}".format(
            **metrics
        ),
        flush=True,
    )
    print(f"[SCRN-BRECQ] eval_run_dir={run_dir}", flush=True)


def load_quant_checkpoint(path: Path) -> dict[str, Any]:
    """读取量化 checkpoint，并检查基本字段。"""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)!r}")
    required = {"model_config", "quant_config", "quant_model_state_dict"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise KeyError(f"Quantized checkpoint misses required keys: {missing}")
    if not isinstance(checkpoint["quant_model_state_dict"], Mapping):
        raise TypeError("checkpoint['quant_model_state_dict'] must be a mapping.")
    return checkpoint


def build_quant_model_from_checkpoint(checkpoint: Mapping[str, Any]) -> QuantModel:
    """根据 checkpoint 中保存的 SCRN 和量化配置重建 QuantModel。"""
    model_config = scrn_config_from_mapping(checkpoint["model_config"])
    quant_config = normalize_quant_config(checkpoint.get("quant_config", {}))
    model = build_scrn_from_config(model_config)
    quant_model = QuantModel(
        model,
        weight_quant_params={
            "n_bits": int(quant_config["n_bits_w"]),
            "channel_wise": bool(quant_config["channel_wise"]),
            "scale_method": str(quant_config["scale_method"]),
        },
        act_quant_params={
            "n_bits": int(quant_config["n_bits_a"]),
            "channel_wise": False,
            "scale_method": str(quant_config["scale_method"]),
            "leaf_param": bool(quant_config["act_quant"]),
        },
    )
    if not bool(quant_config["disable_8bit_head_stem"]):
        quant_model.set_first_last_layer_to_8bit()
    apply_activation_bitwidth_overrides(quant_model, quant_config["activation_bitwidth_overrides"])
    return quant_model


def restore_quantizer_state_shapes(quant_model: QuantModel, state_dict: Mapping[str, torch.Tensor]) -> None:
    """按 state_dict 形态恢复 AdaRound 和激活量化器的可加载结构。

    Reconstruction 后的权重量化器会从 `UniformAffineQuantizer` 变成
    `AdaRoundQuantizer`。加载 state_dict 前必须先按 `alpha` 键替换对应模块。
    """
    modules = dict(quant_model.named_modules())
    for alpha_key in sorted(key for key in state_dict if key.endswith(".weight_quantizer.alpha")):
        module_path = alpha_key.removesuffix(".weight_quantizer.alpha")
        module = modules.get(module_path)
        if not isinstance(module, QuantModule):
            raise KeyError(f"State key {alpha_key!r} does not point to a QuantModule.")
        delta_key = f"{module_path}.weight_quantizer.delta"
        zero_point_key = f"{module_path}.weight_quantizer.zero_point"
        if delta_key not in state_dict or zero_point_key not in state_dict:
            raise KeyError(f"AdaRound state for {module_path!r} misses delta or zero_point.")

        base_quantizer = module.weight_quantizer
        base_quantizer.delta = state_dict[delta_key].detach().clone()
        base_quantizer.zero_point = state_dict[zero_point_key].detach().clone()
        base_quantizer.inited = True
        module.weight_quantizer = AdaRoundQuantizer(base_quantizer, module.weight.detach(), round_mode="learned_round_sigmoid")
        module.weight_quantizer.soft_targets = False

    modules = dict(quant_model.named_modules())

    # 新 checkpoint 会把量化器 zero_point 作为 buffer 保存。加载前先创建同形状
    # buffer，避免 strict=True 时把它识别为 unexpected key。
    for zero_point_key in sorted(key for key in state_dict if key.endswith(".zero_point")):
        quantizer_path = zero_point_key.removesuffix(".zero_point")
        quantizer = modules.get(quantizer_path)
        if quantizer is None or not hasattr(quantizer, "zero_point"):
            continue
        quantizer.zero_point = torch.zeros_like(state_dict[zero_point_key])

    # Activation quantization checkpoints contain learned delta parameters and zero points.
    # Older checkpoints did not save zero_point; those keep inited=False and retain the
    # legacy behavior of refreshing activation zero points on first evaluation forward.
    for delta_key in sorted(key for key in state_dict if key.endswith(".act_quantizer.delta")):
        module_path = delta_key.removesuffix(".act_quantizer.delta")
        module = modules.get(module_path)
        if not isinstance(module, QuantModule):
            raise KeyError(f"State key {delta_key!r} does not point to a QuantModule.")
        module.act_quantizer.delta = nn.Parameter(torch.zeros_like(state_dict[delta_key]))
        zero_point_key = f"{module_path}.act_quantizer.zero_point"
        module.act_quantizer.inited = zero_point_key in state_dict


def normalize_quant_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """补齐评估 checkpoint 需要的量化配置字段。"""
    config = dict(raw_config)
    defaults = {
        "n_bits_w": 4,
        "n_bits_a": 4,
        "channel_wise": True,
        "scale_method": "mse",
        "act_quant": False,
        "disable_8bit_head_stem": False,
        "save_figure": False,
        "activation_bitwidth_overrides": [],
    }
    for key, value in defaults.items():
        config.setdefault(key, value)
    config["n_bits_w"] = int(config["n_bits_w"])
    config["n_bits_a"] = int(config["n_bits_a"])
    config["channel_wise"] = bool(config["channel_wise"])
    config["act_quant"] = bool(config["act_quant"])
    config["disable_8bit_head_stem"] = bool(config["disable_8bit_head_stem"])
    config["save_figure"] = bool(config["save_figure"])
    config["activation_bitwidth_overrides"] = normalize_activation_bitwidth_overrides(
        config.get("activation_bitwidth_overrides")
    )
    return config


def scrn_config_from_mapping(raw_config: Mapping[str, Any]) -> SCRNConfig:
    """把 checkpoint 中的普通 mapping 转成 SCRNConfig。"""
    if not isinstance(raw_config, Mapping):
        raise TypeError(f"Expected model_config mapping, got {type(raw_config)!r}")
    return SCRNConfig(
        in_channels=int(raw_config.get("in_channels", 1)),
        dim=int(raw_config.get("dim", 64)),
        stage_depths=parse_stage_depths(raw_config.get("stage_depths", (1, 1, 1, 1, 1))),
        head_dim=int(raw_config.get("head_dim", 32)),
        window_size=int(raw_config.get("window_size", 8)),
        drop_path_rate=float(raw_config.get("drop_path_rate", 0.0)),
        input_resolution=int(raw_config.get("input_resolution", 128)),
    )


def parse_stage_depths(value: Any) -> tuple[int, int, int, int, int]:
    """解析 SCRN 的 5 段 stage depths。"""
    if isinstance(value, str):
        depths = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    else:
        depths = tuple(int(part) for part in value)
    if len(depths) != 5:
        raise ValueError(f"SCRN expects 5 stage depths, got {depths}")
    return depths


def resolve_eval_paths(args: argparse.Namespace, quant_config: Mapping[str, Any]) -> tuple[Path, Path]:
    """解析评估 clean/input 路径，CLI 参数优先于 checkpoint 配置。"""
    clean_path = Path(args.eval_clean_path or quant_config.get("eval_clean_path", ""))
    input_path = Path(args.eval_input_path or quant_config.get("eval_input_path", ""))
    return require_file(clean_path, "clean reference"), require_file(input_path, "degraded input")


def load_eval_arrays(clean_path: Path, input_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取 SCRN 测试 `.npy` 对，并检查形状一致。"""
    clean = np.load(clean_path).astype(np.float32)
    degraded = np.load(input_path).astype(np.float32)
    if clean.shape != degraded.shape:
        raise ValueError(f"clean and input shapes differ: {clean.shape} vs {degraded.shape}")
    return clean, degraded


def predict_array(model: torch.nn.Module, degraded: np.ndarray, device: torch.device) -> tuple[np.ndarray, float]:
    """对单张 degraded array 做量化模型推理。"""
    model.eval()
    start = time.time()
    with torch.no_grad():
        tensor = torch.from_numpy(degraded).view(1, 1, degraded.shape[0], degraded.shape[1]).float().to(device)
        prediction = model(tensor).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    return prediction, time.time() - start


def build_metrics(clean: np.ndarray, degraded: np.ndarray, prediction: np.ndarray, seconds: float) -> dict[str, Any]:
    """构建量化 checkpoint 评估指标，并保留 before/after 兼容字段。"""
    metrics = {
        "input_snr_db": snr_db(degraded, clean),
        "input_ssim": ssim_score(degraded, clean),
        "quant_snr_db": snr_db(prediction, clean),
        "quant_ssim": ssim_score(prediction, clean),
        "quant_inference_seconds": float(seconds),
    }
    metrics["before_snr_db"] = metrics["input_snr_db"]
    metrics["before_ssim"] = metrics["input_ssim"]
    metrics["after_snr_db"] = metrics["quant_snr_db"]
    metrics["after_ssim"] = metrics["quant_ssim"]
    metrics["inference_seconds"] = metrics["quant_inference_seconds"]
    return metrics


def build_run_config(
    *,
    checkpoint_path: Path,
    clean_path: Path,
    input_path: Path,
    device: torch.device,
    checkpoint: Mapping[str, Any],
    quant_config: Mapping[str, Any],
    save_figure: bool,
) -> dict[str, Any]:
    """构建写入 run `config.json` 的配置快照。"""
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "clean_path": str(clean_path),
        "input_path": str(input_path),
        "save_figure": bool(save_figure),
        "source_checkpoint": checkpoint.get("source_checkpoint"),
        "source_checkpoint_epoch": checkpoint.get("source_checkpoint_epoch"),
        "source_checkpoint_loss": checkpoint.get("source_checkpoint_loss"),
        "model_config": checkpoint.get("model_config", {}),
        "quant_config": dict(quant_config),
        "final_quant_state": checkpoint.get("final_quant_state", {}),
        "environment": collect_environment(),
    }


def save_evaluation_figure(
    path: Path,
    *,
    clean: np.ndarray,
    degraded: np.ndarray,
    prediction: np.ndarray,
    metrics: Mapping[str, Any],
) -> None:
    """保存 clean、input、quantized prediction 三图对比。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Saving comparison figures requires installing matplotlib.") from exc

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), constrained_layout=True)
    panels = [
        (clean, "Ground Truth"),
        (degraded, f"Input\nSNR={metrics['input_snr_db']:.2f}dB SSIM={metrics['input_ssim']:.3f}"),
        (prediction, f"Quantized SCRN\nSNR={metrics['quant_snr_db']:.2f}dB SSIM={metrics['quant_ssim']:.3f}"),
    ]
    vmin = float(np.min(clean))
    vmax = float(np.max(clean))
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0

    image = None
    for axis, (array, title) in zip(axes, panels):
        image = axis.imshow(array, cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax)
        axis.set_title(title, fontsize=9, pad=6)
        axis.axis("off")
    fig.colorbar(image, ax=axes, shrink=0.75, fraction=0.03, pad=0.01)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def select_device(device_arg: str) -> torch.device:
    """解析评估设备参数。"""
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "cuda":
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device("cpu")


def require_file(path: str | Path, description: str) -> Path:
    """检查文件存在，并返回 Path。"""
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(f"{description} does not exist: {candidate}")
    return candidate


if __name__ == "__main__":
    main()
