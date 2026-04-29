"""多样本评估已保存的 SCRN-BRECQ 量化 checkpoint。

单样本 `evaluate_quantized_scrn.py` 只评估一对 clean/input `.npy`。本脚本面向
泛化检查：从 clean patch 目录抽取多个样本，按 SCRN 训练时相同的退化方式在线生成
degraded 输入，然后在同一批输入上比较 FP32、量化重建前和量化重建后路径。
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
    parser.add_argument(
        "--pre-recon-checkpoint",
        default=None,
        help="Optional path to quantized_scrn_brecq_pre_recon.pth for pre/post reconstruction comparison",
    )
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

    checkpoint_path = require_file(args.checkpoint, "post-reconstruction quantized checkpoint")
    pre_recon_checkpoint_path = (
        require_file(args.pre_recon_checkpoint, "pre-reconstruction quantized checkpoint")
        if args.pre_recon_checkpoint
        else None
    )
    eval_dataset_dir = require_directory(args.eval_dataset_dir, "eval dataset directory")
    device = select_device(args.device)

    post_bundle = load_eval_model(checkpoint_path, device)
    pre_bundle = load_eval_model(pre_recon_checkpoint_path, device) if pre_recon_checkpoint_path is not None else None
    checkpoint = post_bundle["checkpoint"]
    quant_config = post_bundle["quant_config"]
    weight_quant = bool(post_bundle["weight_quant"])
    act_quant = bool(post_bundle["act_quant"])

    all_files = discover_patch_files(eval_dataset_dir)
    selected = select_eval_files(all_files, num_samples=int(args.num_eval_samples), seed=int(args.seed))
    run_dir = create_run_dir(args.run_root, run_name=args.run_name)
    per_sample_path = run_dir / "per_sample_metrics.jsonl"
    figures_dir = run_dir / "figures"
    if bool(args.save_figures) and int(args.max_figures) > 0:
        figures_dir.mkdir(parents=True, exist_ok=True)

    rows, timing = evaluate_files(
        post_bundle["model"],
        selected,
        device=device,
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        weight_quant=weight_quant,
        act_quant=act_quant,
        pre_recon_model=pre_bundle["model"] if pre_bundle is not None else None,
        pre_recon_weight_quant=bool(pre_bundle["weight_quant"]) if pre_bundle is not None else False,
        pre_recon_act_quant=bool(pre_bundle["act_quant"]) if pre_bundle is not None else False,
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
            "quant_pre_recon_inference_seconds": timing["quant_pre_recon_inference_seconds"],
            "quant_inference_seconds": timing["quant_inference_seconds"],
            "elapsed_seconds": time.time() - run_start,
            "model_size": build_model_size_report(
                post_bundle["model"],
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
            pre_recon_checkpoint_path=pre_recon_checkpoint_path,
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
                "pre_recon_checkpoint": pre_recon_checkpoint_path or "not_provided",
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

    pre_recon_snr = metrics.get("quant_pre_recon_snr_db_mean")
    pre_recon_text = f"{pre_recon_snr:.4f}" if pre_recon_snr is not None else "not_provided"
    print(
        f"samples={metrics['sample_count']} input_snr_mean={metrics['input_snr_db_mean']:.4f} "
        f"fp32_snr_mean={metrics['fp32_snr_db_mean']:.4f} "
        f"pre_recon_snr_mean={pre_recon_text} "
        f"post_recon_snr_mean={metrics['quant_post_recon_snr_db_mean']:.4f} "
        f"post_recon_ssim_mean={metrics['quant_post_recon_ssim_mean']:.4f} "
        f"elapsed={metrics['elapsed_seconds']:.2f}s",
        flush=True,
    )
    print(f"[SCRN-BRECQ] multi_eval_run_dir={run_dir}", flush=True)


def load_eval_model(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    """恢复一个量化 checkpoint 对应的 QuantModel 和最终量化状态。"""
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
    return {
        "checkpoint": checkpoint,
        "quant_config": quant_config,
        "model": quant_model,
        "weight_quant": weight_quant,
        "act_quant": act_quant,
    }


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
    pre_recon_model: torch.nn.Module | None,
    pre_recon_weight_quant: bool,
    pre_recon_act_quant: bool,
    figures_dir: Path | None,
    max_figures: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """批量评估 clean patch，并返回逐样本指标和总推理耗时。"""
    rows: list[dict[str, Any]] = []
    fp32_seconds = 0.0
    quant_pre_recon_seconds = 0.0
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

        if pre_recon_model is not None:
            pre_recon_model.set_quant_state(pre_recon_weight_quant, pre_recon_act_quant)
            synchronize_if_needed(device)
            pre_start = time.time()
            with torch.no_grad():
                quant_pre_recon_batch = pre_recon_model(input_tensor).squeeze(1).detach().cpu().numpy().astype(np.float32)
            synchronize_if_needed(device)
            quant_pre_recon_seconds += time.time() - pre_start
        else:
            quant_pre_recon_batch = None

        model.set_quant_state(weight_quant, act_quant)
        synchronize_if_needed(device)
        quant_start = time.time()
        with torch.no_grad():
            quant_post_recon_batch = model(input_tensor).squeeze(1).detach().cpu().numpy().astype(np.float32)
        synchronize_if_needed(device)
        quant_seconds += time.time() - quant_start

        for local_index, path in enumerate(batch_files):
            global_index = batch_start + local_index
            clean = clean_batch[local_index]
            degraded = degraded_batch[local_index]
            fp32_prediction = fp32_batch[local_index]
            quant_pre_recon_prediction = quant_pre_recon_batch[local_index] if quant_pre_recon_batch is not None else None
            quant_post_recon_prediction = quant_post_recon_batch[local_index]
            row = build_sample_metrics(
                sample_index=global_index,
                path=path,
                degradation_info=degradation_infos[local_index],
                clean=clean,
                degraded=degraded,
                fp32_prediction=fp32_prediction,
                quant_pre_recon_prediction=quant_pre_recon_prediction,
                quant_post_recon_prediction=quant_post_recon_prediction,
            )
            rows.append(row)
            if figures_dir is not None and figure_count < max_figures:
                save_comparison_figure(
                    figures_dir / f"sample_{global_index:04d}.png",
                    clean=clean,
                    degraded=degraded,
                    fp32_prediction=fp32_prediction,
                    quant_pre_recon_prediction=quant_pre_recon_prediction,
                    quant_post_recon_prediction=quant_post_recon_prediction,
                    metrics=row,
                )
                figure_count += 1

    return rows, {
        "fp32_inference_seconds": fp32_seconds,
        "quant_pre_recon_inference_seconds": quant_pre_recon_seconds,
        "quant_inference_seconds": quant_seconds,
    }


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
    quant_pre_recon_prediction: np.ndarray | None,
    quant_post_recon_prediction: np.ndarray,
) -> dict[str, Any]:
    """构建单个样本的质量与输出差异指标。"""
    fp32_snr = snr_db(fp32_prediction, clean)
    fp32_ssim = ssim_score(fp32_prediction, clean)
    post_diff = fp32_prediction - quant_post_recon_prediction
    post_snr = snr_db(quant_post_recon_prediction, clean)
    post_ssim = ssim_score(quant_post_recon_prediction, clean)

    result = {
        "sample_index": int(sample_index),
        "path": str(path),
        "missing_rate": float(degradation_info["missing_rate"]),
        "target_snr_db": float(degradation_info["target_snr_db"]),
        "input_snr_db": snr_db(degraded, clean),
        "input_ssim": ssim_score(degraded, clean),
        "fp32_snr_db": fp32_snr,
        "fp32_ssim": fp32_ssim,
        "quant_post_recon_snr_db": post_snr,
        "quant_post_recon_ssim": post_ssim,
        "quant_post_minus_fp32_snr_db": post_snr - fp32_snr,
        "quant_post_minus_fp32_ssim": post_ssim - fp32_ssim,
        "fp32_quant_post_recon_mse": float(np.mean(post_diff.astype(np.float64) ** 2)),
        "fp32_quant_post_recon_mean_abs_diff": float(np.mean(np.abs(post_diff))),
        "fp32_quant_post_recon_max_abs_diff": float(np.max(np.abs(post_diff))),
    }

    if quant_pre_recon_prediction is not None:
        pre_diff = fp32_prediction - quant_pre_recon_prediction
        pre_snr = snr_db(quant_pre_recon_prediction, clean)
        pre_ssim = ssim_score(quant_pre_recon_prediction, clean)
        post_minus_pre = quant_post_recon_prediction - quant_pre_recon_prediction
        result.update(
            {
                "quant_pre_recon_snr_db": pre_snr,
                "quant_pre_recon_ssim": pre_ssim,
                "quant_pre_minus_fp32_snr_db": pre_snr - fp32_snr,
                "quant_pre_minus_fp32_ssim": pre_ssim - fp32_ssim,
                "quant_post_minus_pre_snr_db": post_snr - pre_snr,
                "quant_post_minus_pre_ssim": post_ssim - pre_ssim,
                "fp32_quant_pre_recon_mse": float(np.mean(pre_diff.astype(np.float64) ** 2)),
                "fp32_quant_pre_recon_mean_abs_diff": float(np.mean(np.abs(pre_diff))),
                "fp32_quant_pre_recon_max_abs_diff": float(np.max(np.abs(pre_diff))),
                "quant_post_pre_mse": float(np.mean(post_minus_pre.astype(np.float64) ** 2)),
                "quant_post_pre_mean_abs_diff": float(np.mean(np.abs(post_minus_pre))),
                "quant_post_pre_max_abs_diff": float(np.max(np.abs(post_minus_pre))),
            }
        )
    return add_legacy_quant_aliases(result)


def add_legacy_quant_aliases(row: dict[str, Any]) -> dict[str, Any]:
    """保留旧版 `quant_*` 字段，含义明确为 post-reconstruction quant 结果。"""
    row["quant_snr_db"] = row["quant_post_recon_snr_db"]
    row["quant_ssim"] = row["quant_post_recon_ssim"]
    row["quant_minus_fp32_snr_db"] = row["quant_post_minus_fp32_snr_db"]
    row["quant_minus_fp32_ssim"] = row["quant_post_minus_fp32_ssim"]
    row["fp32_quant_mse"] = row["fp32_quant_post_recon_mse"]
    row["fp32_quant_mean_abs_diff"] = row["fp32_quant_post_recon_mean_abs_diff"]
    row["fp32_quant_max_abs_diff"] = row["fp32_quant_post_recon_max_abs_diff"]
    return row


def build_aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """把逐样本指标聚合成均值、标准差、最大值、最小值和最差样本列表。"""
    aggregate_keys = [
        "input_snr_db",
        "input_ssim",
        "fp32_snr_db",
        "fp32_ssim",
        "quant_post_recon_snr_db",
        "quant_post_recon_ssim",
        "quant_post_minus_fp32_snr_db",
        "quant_post_minus_fp32_ssim",
        "fp32_quant_post_recon_mse",
        "fp32_quant_post_recon_mean_abs_diff",
        "fp32_quant_post_recon_max_abs_diff",
    ]
    if "quant_pre_recon_snr_db" in rows[0]:
        aggregate_keys.extend(
            [
                "quant_pre_recon_snr_db",
                "quant_pre_recon_ssim",
                "quant_pre_minus_fp32_snr_db",
                "quant_pre_minus_fp32_ssim",
                "quant_post_minus_pre_snr_db",
                "quant_post_minus_pre_ssim",
                "fp32_quant_pre_recon_mse",
                "fp32_quant_pre_recon_mean_abs_diff",
                "fp32_quant_pre_recon_max_abs_diff",
                "quant_post_pre_mse",
                "quant_post_pre_mean_abs_diff",
                "quant_post_pre_max_abs_diff",
            ]
        )
    metrics: dict[str, Any] = {}
    for key in aggregate_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        metrics[f"{key}_mean"] = float(np.mean(values))
        metrics[f"{key}_std"] = float(np.std(values))
        metrics[f"{key}_min"] = float(np.min(values))
        metrics[f"{key}_max"] = float(np.max(values))

    metrics["worst_quant_post_recon_snr_samples"] = summarize_worst(rows, key="quant_post_recon_snr_db", reverse=False)
    metrics["worst_quant_post_minus_fp32_snr_samples"] = summarize_worst(
        rows,
        key="quant_post_minus_fp32_snr_db",
        reverse=False,
    )
    metrics["largest_fp32_quant_post_recon_mse_samples"] = summarize_worst(
        rows,
        key="fp32_quant_post_recon_mse",
        reverse=True,
    )
    add_legacy_aggregate_aliases(metrics)
    return metrics


def add_legacy_aggregate_aliases(metrics: dict[str, Any]) -> None:
    """保留旧版聚合字段，含义为 post-reconstruction quant 结果。"""
    alias_pairs = {
        "quant_snr_db": "quant_post_recon_snr_db",
        "quant_ssim": "quant_post_recon_ssim",
        "quant_minus_fp32_snr_db": "quant_post_minus_fp32_snr_db",
        "quant_minus_fp32_ssim": "quant_post_minus_fp32_ssim",
        "fp32_quant_mse": "fp32_quant_post_recon_mse",
        "fp32_quant_mean_abs_diff": "fp32_quant_post_recon_mean_abs_diff",
        "fp32_quant_max_abs_diff": "fp32_quant_post_recon_max_abs_diff",
    }
    for old_prefix, new_prefix in alias_pairs.items():
        for suffix in ("mean", "std", "min", "max"):
            new_key = f"{new_prefix}_{suffix}"
            if new_key in metrics:
                metrics[f"{old_prefix}_{suffix}"] = metrics[new_key]


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
        "quant_pre_recon_snr_db_mean": metrics.get("quant_pre_recon_snr_db_mean", "not_provided"),
        "quant_post_recon_snr_db_mean": metrics["quant_post_recon_snr_db_mean"],
        "quant_post_minus_pre_snr_db_mean": metrics.get("quant_post_minus_pre_snr_db_mean", "not_provided"),
        "quant_post_minus_fp32_snr_db_mean": metrics["quant_post_minus_fp32_snr_db_mean"],
        "input_ssim_mean": metrics["input_ssim_mean"],
        "fp32_ssim_mean": metrics["fp32_ssim_mean"],
        "quant_pre_recon_ssim_mean": metrics.get("quant_pre_recon_ssim_mean", "not_provided"),
        "quant_post_recon_ssim_mean": metrics["quant_post_recon_ssim_mean"],
        "quant_post_minus_pre_ssim_mean": metrics.get("quant_post_minus_pre_ssim_mean", "not_provided"),
        "quant_post_minus_fp32_ssim_mean": metrics["quant_post_minus_fp32_ssim_mean"],
        "fp32_quant_post_recon_mse_mean": metrics["fp32_quant_post_recon_mse_mean"],
        "elapsed_seconds": metrics["elapsed_seconds"],
        "estimated_packed_model_size_mib": metrics["model_size"]["estimated_storage"]["estimated_packed_model_size_mib"],
        "estimated_model_compression_ratio": metrics["model_size"]["estimated_storage"]["estimated_model_compression_ratio"],
    }


def build_run_config(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint_path: Path,
    pre_recon_checkpoint_path: Path | None,
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
        "pre_recon_checkpoint": str(pre_recon_checkpoint_path) if pre_recon_checkpoint_path is not None else None,
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
    quant_pre_recon_prediction: np.ndarray | None,
    quant_post_recon_prediction: np.ndarray,
    metrics: Mapping[str, Any],
) -> None:
    """保存单样本 clean/input/fp32/quant pre/post 对比。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Saving comparison figures requires installing matplotlib.") from exc

    panels: list[tuple[np.ndarray, str]] = [
        (clean, "Ground Truth"),
        (degraded, f"Input\nSNR={metrics['input_snr_db']:.2f} SSIM={metrics['input_ssim']:.3f}"),
        (fp32_prediction, f"FP32\nSNR={metrics['fp32_snr_db']:.2f} SSIM={metrics['fp32_ssim']:.3f}"),
    ]
    if quant_pre_recon_prediction is not None:
        panels.append(
            (
                quant_pre_recon_prediction,
                "Quant Pre-Recon\n"
                f"SNR={metrics['quant_pre_recon_snr_db']:.2f} SSIM={metrics['quant_pre_recon_ssim']:.3f}",
            )
        )
    panels.append(
        (
            quant_post_recon_prediction,
            "Quant Post-Recon\n"
            f"SNR={metrics['quant_post_recon_snr_db']:.2f} SSIM={metrics['quant_post_recon_ssim']:.3f}",
        )
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
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
