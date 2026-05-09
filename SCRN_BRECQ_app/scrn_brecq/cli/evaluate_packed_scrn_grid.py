"""Evaluate packed SCRN-BRECQ artifacts against checkpoint final outputs on a fixed grid."""

from __future__ import annotations

import argparse
from datetime import datetime
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_packed_scrn import checkpoint_like_from_manifest
from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn import (
    build_quant_model_from_checkpoint,
    normalize_quant_config,
    require_file,
)
from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_grid import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EVAL_DATASET_DIR,
    DEFAULT_MISSING_RATES,
    DEFAULT_SEED,
    DEFAULT_SNR_SETTINGS,
    DegradationCondition,
    build_degradation_conditions,
    load_degraded_batch,
    load_manifest_source_map,
    parse_float_sequence,
    require_directory,
    select_eval_files,
    synchronize_if_needed,
    write_jsonl_rows,
)
from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_multi import load_eval_model, select_eval_device
from SCRN_BRECQ_app.scrn_brecq.utils import (
    build_model_size_report,
    load_packed_manifest,
    restore_packed_deployment,
)
from SCRN_BRECQ_app.scrn_repro.data import discover_patch_files
from SCRN_BRECQ_app.scrn_repro.training import collect_environment, create_run_dir, write_json
from SCRN_BRECQ_app.scrn_repro.utils import snr_db, ssim_score


DEFAULT_RUN_ROOT = (
    "SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/"
    "NE000_1_packed_deployment_equivalence/eval"
)
DEFAULT_RUN_NAME = "packed_grid478_seed20260507"

PER_SAMPLE_FIELDS = {
    "testset_id",
    "source",
    "patch_file",
    "patch_index",
    "condition_index",
    "snr_setting_db",
    "missing_rate",
    "input_snr_db",
    "input_ssim",
    "fp32_snr_db",
    "fp32_ssim",
    "checkpoint_snr_db",
    "checkpoint_ssim",
    "packed_snr_db",
    "packed_ssim",
    "packed_minus_checkpoint_snr_db",
    "packed_minus_checkpoint_ssim",
    "packed_vs_checkpoint_mse",
    "packed_vs_checkpoint_mean_abs_diff",
    "packed_vs_checkpoint_max_abs_diff",
    "inference_seconds",
}

STAT_FIELDS = (
    "input_snr_db",
    "input_ssim",
    "fp32_snr_db",
    "fp32_ssim",
    "checkpoint_snr_db",
    "checkpoint_ssim",
    "packed_snr_db",
    "packed_ssim",
    "packed_minus_checkpoint_snr_db",
    "packed_minus_checkpoint_ssim",
    "packed_vs_checkpoint_mse",
    "packed_vs_checkpoint_mean_abs_diff",
    "packed_vs_checkpoint_max_abs_diff",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the packed grid evaluator parser."""
    parser = argparse.ArgumentParser(description="Evaluate a packed SCRN-BRECQ artifact on a fixed grid.")
    parser.add_argument("--packed-dir", required=True, help="Directory containing manifest.json, weights.bin, and aux_fp32.bin.")
    parser.add_argument("--checkpoint", required=True, help="Reference quantized_scrn_brecq.pth checkpoint.")
    parser.add_argument("--eval-dataset-dir", default=DEFAULT_EVAL_DATASET_DIR, help="Clean patch test dataset directory.")
    parser.add_argument("--testset-id", default="paper5_energy_filtered_perpatch_absmax_478")
    parser.add_argument("--num-eval-samples", type=int, default=None, help="Optional subset size; default evaluates all patches.")
    parser.add_argument("--snr-settings", default=",".join(str(value) for value in DEFAULT_SNR_SETTINGS))
    parser.add_argument("--missing-rates", default=",".join(str(value) for value in DEFAULT_MISSING_RATES))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--cuda-device-index", type=int, default=1)
    return parser


def main() -> None:
    """Run packed grid evaluation."""
    args = build_parser().parse_args()
    validate_args(args)
    run_start = time.time()

    packed_dir = require_directory(args.packed_dir, "packed deployment directory")
    checkpoint_path = require_file(args.checkpoint, "reference quantized checkpoint")
    eval_dataset_dir = require_directory(args.eval_dataset_dir, "eval dataset directory")
    snr_settings = parse_float_sequence(args.snr_settings, option_name="--snr-settings")
    missing_rates = parse_float_sequence(args.missing_rates, option_name="--missing-rates")
    conditions = build_degradation_conditions(snr_settings, missing_rates)
    device = select_eval_device(args.device, args.cuda_device_index)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    checkpoint_bundle = load_eval_model(checkpoint_path, device)
    packed_bundle = load_packed_eval_model(packed_dir, device)
    all_files = discover_patch_files(eval_dataset_dir)
    selected_files = select_eval_files(all_files, num_samples=args.num_eval_samples, seed=int(args.seed))
    source_map, manifest_warnings = load_manifest_source_map(eval_dataset_dir)

    run_dir = create_run_dir(args.run_root, run_name=args.run_name)
    rows, timing = evaluate_files(
        checkpoint_bundle["model"],
        packed_bundle["model"],
        selected_files,
        testset_id=str(args.testset_id),
        source_map=source_map,
        conditions=conditions,
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        device=device,
        checkpoint_weight_quant=bool(checkpoint_bundle["weight_quant"]),
        checkpoint_act_quant=bool(checkpoint_bundle["act_quant"]),
        packed_weight_quant=bool(packed_bundle["weight_quant"]),
        packed_act_quant=bool(packed_bundle["act_quant"]),
    )

    per_sample_path = run_dir / "per_sample_metrics.jsonl"
    write_jsonl_rows(per_sample_path, rows)
    metrics = {
        "sample_count": len(rows),
        "patch_count": len(selected_files),
        "dataset_file_count": len(all_files),
        "condition_count": len(conditions),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "groups": aggregate_rows(rows),
        "timing": timing,
        "elapsed_seconds": float(time.time() - run_start),
        "checkpoint_model_size": build_model_size_report(
            checkpoint_bundle["model"],
            source_checkpoint_path=checkpoint_bundle["checkpoint"].get("source_checkpoint"),
            quant_checkpoint_path=checkpoint_path,
        ),
        "packed_model_size": build_model_size_report(
            packed_bundle["model"],
            source_checkpoint_path=packed_bundle["manifest"].get("source_checkpoint"),
            quant_checkpoint_path=packed_bundle["manifest"].get("quant_checkpoint"),
        ),
        "packed_restore": dict(packed_bundle["restore_summary"]),
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "config.json",
        build_run_config(
            args=args,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            packed_dir=packed_dir,
            eval_dataset_dir=eval_dataset_dir,
            selected_files=selected_files,
            conditions=conditions,
            device=device,
            checkpoint_bundle=checkpoint_bundle,
            packed_bundle=packed_bundle,
            manifest_warnings=manifest_warnings,
        ),
    )
    write_summary(run_dir / "summary.md", run_dir=run_dir, metrics=metrics, config_path=run_dir / "config.json")

    overall = metrics["groups"]["overall"][0]
    print(
        f"rows={metrics['sample_count']} patches={metrics['patch_count']} conditions={metrics['condition_count']} "
        f"checkpoint_snr_mean={overall['checkpoint_snr_db_mean']:.4f} "
        f"packed_snr_mean={overall['packed_snr_db_mean']:.4f} "
        f"packed_minus_checkpoint_snr_mean={overall['packed_minus_checkpoint_snr_db_mean']:.6f} "
        f"run_dir={run_dir}",
        flush=True,
    )


def packed_runtime_quant_state(*, final_state: Mapping[str, Any], quant_config: Mapping[str, Any]) -> dict[str, bool]:
    """Return the runtime quant state for packed restored inference."""
    act_quant = bool(final_state.get("act_quant", quant_config.get("act_quant", False)))
    return {"weight_quant": False, "act_quant": act_quant}


def load_packed_eval_model(packed_dir: str | Path, device: torch.device) -> dict[str, Any]:
    """Restore a packed deployment artifact into an evaluation model bundle."""
    manifest = load_packed_manifest(packed_dir)
    checkpoint_like = checkpoint_like_from_manifest(manifest)
    quant_config = normalize_quant_config(checkpoint_like.get("quant_config", {}))
    model = build_quant_model_from_checkpoint(checkpoint_like)
    model.to(device)
    restore_summary = restore_packed_deployment(model, packed_dir)
    runtime_state = packed_runtime_quant_state(
        final_state=manifest.get("final_quant_state") or {},
        quant_config=quant_config,
    )
    if runtime_state["act_quant"]:
        model.disable_network_output_quantization()
    model.set_quant_state(runtime_state["weight_quant"], runtime_state["act_quant"])
    model.eval()
    return {
        "manifest": manifest,
        "checkpoint_like": checkpoint_like,
        "quant_config": quant_config,
        "model": model,
        "weight_quant": runtime_state["weight_quant"],
        "act_quant": runtime_state["act_quant"],
        "restore_summary": restore_summary,
    }


def build_metric_row(
    *,
    testset_id: str,
    source: str,
    patch_file: str,
    patch_index: int,
    condition_index: int,
    snr_setting_db: float,
    missing_rate: float,
    input_snr_db: float,
    input_ssim: float,
    fp32_snr_db: float,
    fp32_ssim: float,
    checkpoint_snr_db: float,
    checkpoint_ssim: float,
    packed_snr_db: float,
    packed_ssim: float,
    packed_vs_checkpoint_mse: float,
    packed_vs_checkpoint_mean_abs_diff: float,
    packed_vs_checkpoint_max_abs_diff: float,
    inference_seconds: float,
) -> dict[str, Any]:
    """Build one packed-vs-checkpoint metrics row."""
    return {
        "testset_id": str(testset_id),
        "source": str(source),
        "patch_file": str(patch_file),
        "patch_index": int(patch_index),
        "condition_index": int(condition_index),
        "snr_setting_db": float(snr_setting_db),
        "missing_rate": float(missing_rate),
        "input_snr_db": float(input_snr_db),
        "input_ssim": float(input_ssim),
        "fp32_snr_db": float(fp32_snr_db),
        "fp32_ssim": float(fp32_ssim),
        "checkpoint_snr_db": float(checkpoint_snr_db),
        "checkpoint_ssim": float(checkpoint_ssim),
        "packed_snr_db": float(packed_snr_db),
        "packed_ssim": float(packed_ssim),
        "packed_minus_checkpoint_snr_db": float(packed_snr_db - checkpoint_snr_db),
        "packed_minus_checkpoint_ssim": float(packed_ssim - checkpoint_ssim),
        "packed_vs_checkpoint_mse": float(packed_vs_checkpoint_mse),
        "packed_vs_checkpoint_mean_abs_diff": float(packed_vs_checkpoint_mean_abs_diff),
        "packed_vs_checkpoint_max_abs_diff": float(packed_vs_checkpoint_max_abs_diff),
        "inference_seconds": float(inference_seconds),
    }


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Aggregate rows by overall, source, SNR, missing rate, and condition."""
    fields = list(STAT_FIELDS)
    return {
        "overall": summarize_groups(rows, ("testset_id",), fields),
        "by_source": summarize_groups(rows, ("testset_id", "source"), fields),
        "by_snr_setting": summarize_groups(rows, ("testset_id", "snr_setting_db"), fields),
        "by_missing_rate": summarize_groups(rows, ("testset_id", "missing_rate"), fields),
        "by_condition": summarize_groups(rows, ("testset_id", "snr_setting_db", "missing_rate"), fields),
    }


def summarize_groups(
    rows: Sequence[Mapping[str, Any]],
    group_keys: Sequence[str],
    stat_fields: Sequence[str],
) -> list[dict[str, Any]]:
    """Summarize metrics for each group."""
    buckets: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        buckets.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for key, bucket_rows in sorted(buckets.items(), key=lambda item: tuple(str(part) for part in item[0])):
        summary = {group_key: value for group_key, value in zip(group_keys, key)}
        summary["sample_count"] = len(bucket_rows)
        for field in stat_fields:
            values = np.asarray([float(row[field]) for row in bucket_rows], dtype=np.float64)
            summary[f"{field}_mean"] = float(np.mean(values))
            summary[f"{field}_median"] = float(np.median(values))
            summary[f"{field}_std"] = float(np.std(values))
            summary[f"{field}_min"] = float(np.min(values))
            summary[f"{field}_max"] = float(np.max(values))
        summaries.append(summary)
    return summaries


def evaluate_files(
    checkpoint_model: torch.nn.Module,
    packed_model: torch.nn.Module,
    files: Sequence[Path],
    *,
    testset_id: str,
    source_map: Mapping[str, str],
    conditions: Sequence[DegradationCondition],
    seed: int,
    batch_size: int,
    device: torch.device,
    checkpoint_weight_quant: bool,
    checkpoint_act_quant: bool,
    packed_weight_quant: bool,
    packed_act_quant: bool,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Evaluate checkpoint and packed-restored outputs across all degradation conditions."""
    rows: list[dict[str, Any]] = []
    fp32_seconds = 0.0
    checkpoint_seconds = 0.0
    packed_seconds = 0.0

    checkpoint_model.eval()
    packed_model.eval()

    with torch.no_grad():
        for condition in conditions:
            for batch_start in range(0, len(files), batch_size):
                batch_paths = list(files[batch_start : batch_start + batch_size])
                clean_batch, degraded_batch, meta = load_degraded_batch(
                    batch_paths,
                    testset_id=testset_id,
                    condition=condition,
                    seed=seed,
                    offset=batch_start,
                    source_map=source_map,
                )
                input_tensor = torch.from_numpy(degraded_batch[:, None, :, :]).float().to(device)

                checkpoint_model.set_quant_state(False, False)
                synchronize_if_needed(device)
                start = time.time()
                fp32_batch = checkpoint_model(input_tensor).squeeze(1).detach().cpu().numpy().astype(np.float32)
                synchronize_if_needed(device)
                fp32_seconds += time.time() - start

                checkpoint_model.set_quant_state(checkpoint_weight_quant, checkpoint_act_quant)
                synchronize_if_needed(device)
                start = time.time()
                checkpoint_batch = checkpoint_model(input_tensor).squeeze(1).detach().cpu().numpy().astype(np.float32)
                synchronize_if_needed(device)
                checkpoint_seconds += time.time() - start

                packed_model.set_quant_state(packed_weight_quant, packed_act_quant)
                synchronize_if_needed(device)
                start = time.time()
                packed_batch = packed_model(input_tensor).squeeze(1).detach().cpu().numpy().astype(np.float32)
                synchronize_if_needed(device)
                elapsed = time.time() - start
                packed_seconds += elapsed
                elapsed_per_sample = elapsed / max(len(batch_paths), 1)

                for local_index, item in enumerate(meta):
                    clean = clean_batch[local_index]
                    degraded = degraded_batch[local_index]
                    fp32_prediction = fp32_batch[local_index]
                    checkpoint_prediction = checkpoint_batch[local_index]
                    packed_prediction = packed_batch[local_index]
                    diff = packed_prediction.astype(np.float32) - checkpoint_prediction.astype(np.float32)
                    abs_diff = np.abs(diff)
                    rows.append(
                        build_metric_row(
                            testset_id=testset_id,
                            source=item["source"],
                            patch_file=item["patch_file"],
                            patch_index=item["patch_index"],
                            condition_index=condition.condition_index,
                            snr_setting_db=condition.snr_setting_db,
                            missing_rate=condition.missing_rate,
                            input_snr_db=snr_db(degraded, clean),
                            input_ssim=ssim_score(degraded, clean),
                            fp32_snr_db=snr_db(fp32_prediction, clean),
                            fp32_ssim=ssim_score(fp32_prediction, clean),
                            checkpoint_snr_db=snr_db(checkpoint_prediction, clean),
                            checkpoint_ssim=ssim_score(checkpoint_prediction, clean),
                            packed_snr_db=snr_db(packed_prediction, clean),
                            packed_ssim=ssim_score(packed_prediction, clean),
                            packed_vs_checkpoint_mse=float(np.mean(diff * diff)),
                            packed_vs_checkpoint_mean_abs_diff=float(np.mean(abs_diff)),
                            packed_vs_checkpoint_max_abs_diff=float(np.max(abs_diff)),
                            inference_seconds=elapsed_per_sample,
                        )
                    )

    return rows, {
        "fp32_inference_seconds": fp32_seconds,
        "checkpoint_inference_seconds": checkpoint_seconds,
        "packed_inference_seconds": packed_seconds,
    }


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_eval_samples is not None and int(args.num_eval_samples) <= 0:
        raise ValueError("--num-eval-samples must be positive")


def build_run_config(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint_path: Path,
    packed_dir: Path,
    eval_dataset_dir: Path,
    selected_files: Sequence[Path],
    conditions: Sequence[DegradationCondition],
    device: Any,
    checkpoint_bundle: Mapping[str, Any],
    packed_bundle: Mapping[str, Any],
    manifest_warnings: Sequence[str],
) -> dict[str, Any]:
    """Build the run config snapshot."""
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "packed_dir": str(packed_dir),
        "eval_dataset_dir": str(eval_dataset_dir),
        "selected_sample_paths": [str(path) for path in selected_files],
        "condition_count": len(conditions),
        "conditions": [
            {
                "condition_index": item.condition_index,
                "snr_setting_db": item.snr_setting_db,
                "missing_rate": item.missing_rate,
            }
            for item in conditions
        ],
        "device": str(device),
        "environment": collect_environment(),
        "manifest_warnings": list(manifest_warnings),
        "checkpoint_quant_config": dict(checkpoint_bundle["quant_config"]),
        "checkpoint_final_quant_state": {
            "weight_quant": bool(checkpoint_bundle["weight_quant"]),
            "act_quant": bool(checkpoint_bundle["act_quant"]),
        },
        "packed_quant_config": dict(packed_bundle.get("quant_config", {})),
        "packed_runtime_quant_state": {
            "weight_quant": bool(packed_bundle["weight_quant"]),
            "act_quant": bool(packed_bundle["act_quant"]),
        },
        "packed_manifest": dict(packed_bundle["manifest"]),
        "packed_restore_summary": dict(packed_bundle["restore_summary"]),
        "source_checkpoint": checkpoint_bundle["checkpoint"].get("source_checkpoint"),
        "source_checkpoint_epoch": checkpoint_bundle["checkpoint"].get("source_checkpoint_epoch"),
        "source_checkpoint_loss": checkpoint_bundle["checkpoint"].get("source_checkpoint_loss"),
        "model_config": checkpoint_bundle["checkpoint"].get("model_config", {}),
    }


def write_summary(path: str | Path, *, run_dir: Path, metrics: Mapping[str, Any], config_path: Path) -> None:
    """Write a compact Markdown summary."""
    overall = metrics["groups"]["overall"][0]
    lines = [
        "# SCRN-BRECQ Packed Grid Evaluation",
        "",
        "## Run",
        "",
        f"- run_dir: `{run_dir}`",
        f"- config: `{config_path}`",
        f"- sample_count: `{metrics['sample_count']}`",
        f"- patch_count: `{metrics['patch_count']}`",
        f"- condition_count: `{metrics['condition_count']}`",
        "",
        "## Overall Metrics",
        "",
        "| metric | mean | median | std | min | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for field in STAT_FIELDS:
        lines.append(
            "| {field} | {mean:.6f} | {median:.6f} | {std:.6f} | {min_value:.6f} | {max_value:.6f} |".format(
                field=field,
                mean=overall[f"{field}_mean"],
                median=overall[f"{field}_median"],
                std=overall[f"{field}_std"],
                min_value=overall[f"{field}_min"],
                max_value=overall[f"{field}_max"],
            )
        )

    lines.extend(
        [
            "",
            "## By Source",
            "",
            "| source | count | checkpoint_snr_mean | packed_snr_mean | packed_minus_checkpoint_snr_mean | packed_vs_checkpoint_mse_mean |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics["groups"]["by_source"]:
        lines.append(
            "| {source} | {sample_count} | {checkpoint:.6f} | {packed:.6f} | {delta:.6f} | {mse:.8e} |".format(
                source=row["source"],
                sample_count=row["sample_count"],
                checkpoint=row["checkpoint_snr_db_mean"],
                packed=row["packed_snr_db_mean"],
                delta=row["packed_minus_checkpoint_snr_db_mean"],
                mse=row["packed_vs_checkpoint_mse_mean"],
            )
        )
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
