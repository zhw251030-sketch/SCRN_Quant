"""多样本评估已保存的 SCRN-BRECQ 量化 checkpoint。

单样本 `evaluate_quantized_scrn.py` 只评估一对 clean/input `.npy`。本脚本面向
泛化检查：从 clean patch 目录抽取多个样本，按 SCRN 训练时相同的退化方式在线生成
degraded 输入，然后在同一批输入上比较 FP32 路径和量化路径。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn import (
    build_quant_model_from_checkpoint,
    load_quant_checkpoint,
    normalize_quant_config,
    require_file,
    restore_quantizer_state_shapes,
    select_device,
)
from SCRN_BRECQ_app.scrn_brecq.utils import build_model_size_report
from SCRN_BRECQ_app.scrn_repro.data import degrade_patch, discover_patch_files
from SCRN_BRECQ_app.scrn_repro.training import collect_environment, create_run_dir, write_json, write_summary
from SCRN_BRECQ_app.scrn_repro.utils import snr_db, ssim_score


DEFAULT_EVAL_DATASET_DIR = "SCRN_BRECQ_app/scrn_repro/datasets/scrn_quant_10750_0_patches"
DEFAULT_RUN_ROOT = "SCRN_BRECQ_app/scrn_brecq/runs/generalization_eval"


def build_parser() -> argparse.ArgumentParser:
    """构建多样本量化评估参数解析器。"""
    parser = argparse.ArgumentParser(description="Evaluate a SCRN-BRECQ checkpoint on multiple degraded patches.")
    parser.add_argument("--checkpoint", required=True, help="Path to quantized_scrn_brecq.pth")
    parser.add_argument("--eval-dataset-dir", default=DEFAULT_EVAL_DATASET_DIR, help="Directory of clean eval patch .npy files")
    parser.add_argument("--num-eval-samples", type=int, default=128, help="Number of clean patches to evaluate")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--seed", type=int, default=20260427, help="Seed for sample selection and online degradation")
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT, help="Output run root")
    parser.add_argument("--run-name", default="multi_sample_eval", help="Run name suffix")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--save-figures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-figures", type=int, default=8, help="Maximum comparison figures to save")
    return parser


def main() -> None:
    """命令行主流程。"""
    args = build_parser().parse_args()
    validate_args(args)
    run_start = time.time()

    checkpoint_path = require_file(args.checkpoint, "quantized checkpoint")
    eval_dataset_dir = require_directory(args.eval_dataset_dir, "eval dataset directory")
    device = select_device(args.device)

    checkpoint = load_quant_checkpoint(checkpoint_path)
    quant_config = normalize_quant_config(checkpoint.get("quant_config", {}))
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
        # 与单样本评估保持一致：activation quant 时关闭网络最终输出量化。
        quant_model.disable_network_output_quantization()

    all_files = discover_patch_files(eval_dataset_dir)
    selected = select_eval_files(all_files, num_samples=int(args.num_eval_samples), seed=int(args.seed))
    run_dir = create_run_dir(args.run_root, run_name=args.run_name)
    per_sample_path = run_dir / "per_sample_metrics.jsonl"
    figures_dir = run_dir / "figures"
    if bool(args.save_figures) and int(args.max_figures) > 0:
        figures_dir.mkdir(parents=True, exist_ok=True)

    rows, timing = evaluate_files(
        quant_model,
        selected,
        device=device,
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        weight_quant=weight_quant,
        act_quant=act_quant,
        figures_dir=figures_dir if bool(args.save_figures) else None,
        max_figures=int(args.max_figures),
    )
    write_jsonl(per_sample_path, rows)

    metrics = build_aggregate_metrics(rows)
    metrics.update(
        {
            "sample_count": len(rows),
            "dataset_file_count": len(all_files),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "fp32_inference_seconds": timing["fp32_inference_seconds"],
            "quant_inference_seconds": timing["quant_inference_seconds"],
            "elapsed_seconds": time.time() - run_start,
            "model_size": build_model_size_report(
                quant_model,
                source_checkpoint_path=checkpoint.get("source_checkpoint"),
                quant_checkpoint_path=checkpoint_path,
            ),
        }
    )

    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "config.json",
        build_run_config(
            args=args,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            eval_dataset_dir=eval_dataset_dir,
            selected_files=selected,
            device=device,
            checkpoint=checkpoint,
            quant_config=quant_config,
            weight_quant=weight_quant,
            act_quant=act_quant,
        ),
    )
    write_summary(
        run_dir / "summary.md",
        title="SCRN-BRECQ Multi-Sample Evaluation",
        sections={
            "Metrics": summary_metrics(metrics),
            "Artifacts": {
                "config": run_dir / "config.json",
                "metrics": run_dir / "metrics.json",
                "per_sample_metrics": per_sample_path,
                "figures": figures_dir if bool(args.save_figures) and int(args.max_figures) > 0 else None,
            },
            "Inputs": {
                "checkpoint": checkpoint_path,
                "eval_dataset_dir": eval_dataset_dir,
                "num_eval_samples": int(args.num_eval_samples),
                "batch_size": int(args.batch_size),
                "seed": int(args.seed),
            },
            "Quantization": {
                "weight_quant": weight_quant,
                "act_quant": act_quant,
                "n_bits_w": quant_config["n_bits_w"],
                "n_bits_a": quant_config["n_bits_a"],
            },
        },
    )

    print(
        "samples={sample_count} input_snr_mean={input_snr_db_mean:.4f} "
        "fp32_snr_mean={fp32_snr_db_mean:.4f} quant_snr_mean={quant_snr_db_mean:.4f} "
        "quant_ssim_mean={quant_ssim_mean:.4f} elapsed={elapsed_seconds:.2f}s".format(**metrics),
        flush=True,
    )
    print(f"[SCRN-BRECQ] multi_eval_run_dir={run_dir}", flush=True)


def validate_args(args: argparse.Namespace) -> None:
    """检查命令行参数取值。"""
    if int(args.num_eval_samples) <= 0:
        raise ValueError(f"--num-eval-samples must be positive, got {args.num_eval_samples}")
    if int(args.batch_size) <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if int(args.max_figures) < 0:
        raise ValueError(f"--max-figures must be non-negative, got {args.max_figures}")


def require_directory(path: str | Path, description: str) -> Path:
    """检查目录存在，并返回 Path。"""
    candidate = Path(path)
    if not candidate.is_dir():
        raise FileNotFoundError(f"{description} does not exist: {candidate}")
    return candidate


def select_eval_files(files: list[Path], *, num_samples: int, seed: int) -> list[Path]:
    """从候选 clean patch 中按固定 seed 抽取评估样本。"""
    if num_samples > len(files):
        raise ValueError(f"Requested {num_samples} samples, but only {len(files)} files are available.")
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(files), size=int(num_samples), replace=False)
    return [files[int(index)] for index in indices]


def evaluate_files(
    model: torch.nn.Module,
    files: list[Path],
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
    weight_quant: bool,
    act_quant: bool,
    figures_dir: Path | None,
    max_figures: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """批量评估 clean patch，并返回逐样本指标和总推理耗时。"""
    rows: list[dict[str, Any]] = []
    fp32_seconds = 0.0
    quant_seconds = 0.0
    figure_count = 0

    for batch_start in range(0, len(files), batch_size):
        batch_files = files[batch_start : batch_start + batch_size]
        clean_batch, degraded_batch, degradation_infos = load_degraded_batch(batch_files, seed=seed, offset=batch_start)
        input_tensor = torch.from_numpy(degraded_batch[:, None, :, :]).float().to(device)

        model.set_quant_state(False, False)
        synchronize_if_needed(device)
        fp32_start = time.time()
        with torch.no_grad():
            fp32_batch = model(input_tensor).squeeze(1).detach().cpu().numpy().astype(np.float32)
        synchronize_if_needed(device)
        fp32_seconds += time.time() - fp32_start

        model.set_quant_state(weight_quant, act_quant)
        synchronize_if_needed(device)
        quant_start = time.time()
        with torch.no_grad():
            quant_batch = model(input_tensor).squeeze(1).detach().cpu().numpy().astype(np.float32)
        synchronize_if_needed(device)
        quant_seconds += time.time() - quant_start

        for local_index, path in enumerate(batch_files):
            global_index = batch_start + local_index
            clean = clean_batch[local_index]
            degraded = degraded_batch[local_index]
            fp32_prediction = fp32_batch[local_index]
            quant_prediction = quant_batch[local_index]
            row = build_sample_metrics(
                sample_index=global_index,
                path=path,
                degradation_info=degradation_infos[local_index],
                clean=clean,
                degraded=degraded,
                fp32_prediction=fp32_prediction,
                quant_prediction=quant_prediction,
            )
            rows.append(row)
            if figures_dir is not None and figure_count < max_figures:
                save_comparison_figure(
                    figures_dir / f"sample_{global_index:04d}.png",
                    clean=clean,
                    degraded=degraded,
                    fp32_prediction=fp32_prediction,
                    quant_prediction=quant_prediction,
                    metrics=row,
                )
                figure_count += 1

    return rows, {"fp32_inference_seconds": fp32_seconds, "quant_inference_seconds": quant_seconds}


def load_degraded_batch(files: list[Path], *, seed: int, offset: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """读取 clean patch，并按固定 seed 生成 degraded batch。"""
    clean_items: list[np.ndarray] = []
    degraded_items: list[np.ndarray] = []
    info_items: list[dict[str, float]] = []
    expected_shape: tuple[int, int] | None = None

    for local_index, path in enumerate(files):
        clean = np.load(path).astype(np.float32)
        if clean.ndim != 2:
            raise ValueError(f"Expected 2D clean patch, got {clean.shape} from {path}")
        if expected_shape is None:
            expected_shape = clean.shape
        elif clean.shape != expected_shape:
            raise ValueError(f"All patches in a batch must share shape; got {clean.shape} and {expected_shape}")

        rng = np.random.default_rng(int(seed) + int(offset) + int(local_index))
        degraded, _, info = degrade_patch(clean, rng=rng)
        clean_items.append(clean)
        degraded_items.append(degraded)
        info_items.append({"missing_rate": float(info.missing_rate), "target_snr_db": float(info.snr_db)})

    return np.stack(clean_items, axis=0), np.stack(degraded_items, axis=0), info_items


def build_sample_metrics(
    *,
    sample_index: int,
    path: Path,
    degradation_info: Mapping[str, float],
    clean: np.ndarray,
    degraded: np.ndarray,
    fp32_prediction: np.ndarray,
    quant_prediction: np.ndarray,
) -> dict[str, Any]:
    """构建单个样本的质量与输出差异指标。"""
    diff = fp32_prediction - quant_prediction
    fp32_snr = snr_db(fp32_prediction, clean)
    quant_snr = snr_db(quant_prediction, clean)
    fp32_ssim = ssim_score(fp32_prediction, clean)
    quant_ssim = ssim_score(quant_prediction, clean)
    return {
        "sample_index": int(sample_index),
        "path": str(path),
        "missing_rate": float(degradation_info["missing_rate"]),
        "target_snr_db": float(degradation_info["target_snr_db"]),
        "input_snr_db": snr_db(degraded, clean),
        "input_ssim": ssim_score(degraded, clean),
        "fp32_snr_db": fp32_snr,
        "fp32_ssim": fp32_ssim,
        "quant_snr_db": quant_snr,
        "quant_ssim": quant_ssim,
        "quant_minus_fp32_snr_db": quant_snr - fp32_snr,
        "quant_minus_fp32_ssim": quant_ssim - fp32_ssim,
        "fp32_quant_mse": float(np.mean(diff.astype(np.float64) ** 2)),
        "fp32_quant_mean_abs_diff": float(np.mean(np.abs(diff))),
        "fp32_quant_max_abs_diff": float(np.max(np.abs(diff))),
    }


def build_aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """把逐样本指标聚合成均值、标准差、最大值、最小值和最差样本列表。"""
    aggregate_keys = [
        "input_snr_db",
        "input_ssim",
        "fp32_snr_db",
        "fp32_ssim",
        "quant_snr_db",
        "quant_ssim",
        "quant_minus_fp32_snr_db",
        "quant_minus_fp32_ssim",
        "fp32_quant_mse",
        "fp32_quant_mean_abs_diff",
        "fp32_quant_max_abs_diff",
    ]
    metrics: dict[str, Any] = {}
    for key in aggregate_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        metrics[f"{key}_mean"] = float(np.mean(values))
        metrics[f"{key}_std"] = float(np.std(values))
        metrics[f"{key}_min"] = float(np.min(values))
        metrics[f"{key}_max"] = float(np.max(values))

    metrics["worst_quant_snr_samples"] = summarize_worst(rows, key="quant_snr_db", reverse=False)
    metrics["worst_quant_minus_fp32_snr_samples"] = summarize_worst(
        rows,
        key="quant_minus_fp32_snr_db",
        reverse=False,
    )
    metrics["largest_fp32_quant_mse_samples"] = summarize_worst(rows, key="fp32_quant_mse", reverse=True)
    return metrics


def summarize_worst(rows: list[dict[str, Any]], *, key: str, reverse: bool, limit: int = 5) -> list[dict[str, Any]]:
    """提取最差样本的紧凑摘要。"""
    selected = sorted(rows, key=lambda row: float(row[key]), reverse=reverse)[:limit]
    return [
        {
            "sample_index": int(row["sample_index"]),
            "path": row["path"],
            key: float(row[key]),
            "quant_snr_db": float(row["quant_snr_db"]),
            "quant_ssim": float(row["quant_ssim"]),
            "missing_rate": float(row["missing_rate"]),
            "target_snr_db": float(row["target_snr_db"]),
        }
        for row in selected
    ]


def summary_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """挑选 summary.md 中最需要快速查看的聚合指标。"""
    return {
        "sample_count": metrics["sample_count"],
        "input_snr_db_mean": metrics["input_snr_db_mean"],
        "fp32_snr_db_mean": metrics["fp32_snr_db_mean"],
        "quant_snr_db_mean": metrics["quant_snr_db_mean"],
        "quant_minus_fp32_snr_db_mean": metrics["quant_minus_fp32_snr_db_mean"],
        "input_ssim_mean": metrics["input_ssim_mean"],
        "fp32_ssim_mean": metrics["fp32_ssim_mean"],
        "quant_ssim_mean": metrics["quant_ssim_mean"],
        "quant_minus_fp32_ssim_mean": metrics["quant_minus_fp32_ssim_mean"],
        "fp32_quant_mse_mean": metrics["fp32_quant_mse_mean"],
        "elapsed_seconds": metrics["elapsed_seconds"],
        "estimated_packed_model_size_mib": metrics["model_size"]["estimated_storage"]["estimated_packed_model_size_mib"],
        "estimated_model_compression_ratio": metrics["model_size"]["estimated_storage"]["estimated_model_compression_ratio"],
    }


def build_run_config(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint_path: Path,
    eval_dataset_dir: Path,
    selected_files: list[Path],
    device: torch.device,
    checkpoint: Mapping[str, Any],
    quant_config: Mapping[str, Any],
    weight_quant: bool,
    act_quant: bool,
) -> dict[str, Any]:
    """构建多样本评估 run 的配置快照。"""
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "eval_dataset_dir": str(eval_dataset_dir),
        "num_eval_samples": int(args.num_eval_samples),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "run_dir": str(run_dir),
        "save_figures": bool(args.save_figures),
        "max_figures": int(args.max_figures),
        "selected_sample_paths": [str(path) for path in selected_files],
        "source_checkpoint": checkpoint.get("source_checkpoint"),
        "source_checkpoint_epoch": checkpoint.get("source_checkpoint_epoch"),
        "source_checkpoint_loss": checkpoint.get("source_checkpoint_loss"),
        "model_config": checkpoint.get("model_config", {}),
        "quant_config": dict(quant_config),
        "final_quant_state": {"weight_quant": bool(weight_quant), "act_quant": bool(act_quant)},
        "environment": collect_environment(),
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """写逐样本 JSONL 指标。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def save_comparison_figure(
    path: Path,
    *,
    clean: np.ndarray,
    degraded: np.ndarray,
    fp32_prediction: np.ndarray,
    quant_prediction: np.ndarray,
    metrics: Mapping[str, Any],
) -> None:
    """保存单样本 clean/input/fp32/quant 四图对比。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Saving comparison figures requires installing matplotlib.") from exc

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), constrained_layout=True)
    panels = [
        (clean, "Ground Truth"),
        (degraded, f"Input\nSNR={metrics['input_snr_db']:.2f} SSIM={metrics['input_ssim']:.3f}"),
        (fp32_prediction, f"FP32\nSNR={metrics['fp32_snr_db']:.2f} SSIM={metrics['fp32_ssim']:.3f}"),
        (quant_prediction, f"Quant\nSNR={metrics['quant_snr_db']:.2f} SSIM={metrics['quant_ssim']:.3f}"),
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
    fig.colorbar(image, ax=axes, shrink=0.75, fraction=0.025, pad=0.01)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def synchronize_if_needed(device: torch.device) -> None:
    """CUDA 计时时同步设备，CPU 下为空操作。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
