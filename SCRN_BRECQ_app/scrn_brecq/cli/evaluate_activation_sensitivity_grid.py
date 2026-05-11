"""Evaluate activation sensitivity on the normalized fixed degradation grid."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import torch

from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn import (
    build_quant_model_from_checkpoint,
    load_quant_checkpoint,
    normalize_quant_config,
    require_file,
    restore_quantizer_state_shapes,
)
from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_grid import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EVAL_DATASET_DIR,
    DEFAULT_MISSING_RATES,
    DEFAULT_SEED,
    DEFAULT_SNR_SETTINGS,
    aggregate_rows,
    build_degradation_conditions,
    evaluate_files,
    load_manifest_source_map,
    parse_float_sequence,
    require_directory,
    select_eval_files,
    write_jsonl_rows,
)
from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_multi import select_eval_device
from SCRN_BRECQ_app.scrn_brecq.quant.activation_sensitivity import (
    SENSITIVITY_MODES,
    apply_activation_sensitivity_mode,
)
from SCRN_BRECQ_app.scrn_brecq.utils import build_model_size_report
from SCRN_BRECQ_app.scrn_repro.data import discover_patch_files
from SCRN_BRECQ_app.scrn_repro.training import collect_environment, create_run_dir, write_json


DEFAULT_RUN_ROOT = "SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE004_w4a4_activation_sensitivity/eval"
DEFAULT_RUN_NAME = "activation_sensitivity_grid478_seed20260507"
DEFAULT_TESTSET_ID = "paper5_energy_filtered_perpatch_absmax_478"


def build_parser() -> argparse.ArgumentParser:
    """Build the normalized activation sensitivity grid parser."""
    parser = argparse.ArgumentParser(description="Evaluate selective activation sensitivity on a fixed grid.")
    parser.add_argument("--checkpoint", required=True, help="Quantized checkpoint to evaluate.")
    parser.add_argument("--mode", choices=sorted(SENSITIVITY_MODES), required=True, help="Sensitivity switch mode.")
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--name-contains", default=None)
    parser.add_argument("--stage", default=None)
    parser.add_argument("--branch", default=None)
    parser.add_argument("--role", default=None)
    parser.add_argument("--module-type", default=None)
    parser.add_argument("--stages", default=None, help="Comma-separated OR selector for stage labels.")
    parser.add_argument("--branches", default=None, help="Comma-separated OR selector for branch labels.")
    parser.add_argument("--roles", default=None, help="Comma-separated OR selector for role labels.")
    parser.add_argument("--module-types", default=None, help="Comma-separated OR selector for module types.")
    parser.add_argument("--include-output-quantizer", action="store_true")
    parser.add_argument("--eval-dataset-dir", default=DEFAULT_EVAL_DATASET_DIR)
    parser.add_argument("--testset-id", default=DEFAULT_TESTSET_ID)
    parser.add_argument("--num-eval-samples", type=int, default=None)
    parser.add_argument("--snr-settings", default=",".join(str(value) for value in DEFAULT_SNR_SETTINGS))
    parser.add_argument("--missing-rates", default=",".join(str(value) for value in DEFAULT_MISSING_RATES))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--cuda-device-index", type=int, default=1)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    return parser


def main() -> None:
    """Run normalized grid sensitivity evaluation."""
    args = build_parser().parse_args()
    validate_args(args)
    started = time.time()

    checkpoint_path = require_file(args.checkpoint, "quantized checkpoint")
    eval_dataset_dir = require_directory(args.eval_dataset_dir, "eval dataset directory")
    snr_settings = parse_float_sequence(args.snr_settings, option_name="--snr-settings")
    missing_rates = parse_float_sequence(args.missing_rates, option_name="--missing-rates")
    conditions = build_degradation_conditions(snr_settings, missing_rates)
    device = select_eval_device(args.device, args.cuda_device_index)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    checkpoint = load_quant_checkpoint(checkpoint_path)
    quant_config = normalize_quant_config(checkpoint.get("quant_config", {}))
    quant_model = build_quant_model_from_checkpoint(checkpoint)
    state_dict = checkpoint["quant_model_state_dict"]
    restore_quantizer_state_shapes(quant_model, state_dict)
    quant_model.load_state_dict(state_dict, strict=True)
    quant_model.to(device)
    quant_model.eval()
    quant_model.disable_network_output_quantization()

    final_state = checkpoint.get("final_quant_state", {})
    weight_quant = bool(final_state.get("weight_quant", True))
    act_quant = bool(final_state.get("act_quant", quant_config.get("act_quant", False)))

    all_files = discover_patch_files(eval_dataset_dir)
    selected_files = select_eval_files(all_files, num_samples=args.num_eval_samples, seed=int(args.seed))
    source_map, manifest_warnings = load_manifest_source_map(eval_dataset_dir)
    run_dir = create_run_dir(args.run_root, run_name=args.run_name)

    selector = selector_summary(args)
    with apply_activation_sensitivity_mode(
        quant_model,
        mode=str(args.mode),
        index=args.index,
        name_contains=args.name_contains,
        stage=args.stage,
        branch=args.branch,
        role=args.role,
        module_type=args.module_type,
        stages=selector["stages"],
        branches=selector["branches"],
        roles=selector["roles"],
        module_types=selector["module_types"],
        include_output_quantizer=bool(args.include_output_quantizer),
    ) as selected_quantizers:
        rows, timing = evaluate_files(
            quant_model,
            selected_files,
            testset_id=str(args.testset_id),
            source_map=source_map,
            conditions=conditions,
            seed=int(args.seed),
            batch_size=int(args.batch_size),
            device=device,
            weight_quant=weight_quant,
            act_quant=act_quant,
            pre_recon_model=None,
            pre_recon_weight_quant=False,
            pre_recon_act_quant=False,
            figures_dir=None,
            max_figures=0,
        )

    per_sample_path = run_dir / "per_sample_metrics.jsonl"
    selected_csv = run_dir / "selected_quantizers.csv"
    write_jsonl_rows(per_sample_path, rows)
    write_selected_quantizers_csv(selected_csv, selected_quantizers)
    metrics = {
        "sample_count": len(rows),
        "patch_count": len(selected_files),
        "dataset_file_count": len(all_files),
        "condition_count": len(conditions),
        "mode": str(args.mode),
        "selector": selector,
        "selected_quantizer_count": len(selected_quantizers),
        "selected_quantizer_names": [str(row["name"]) for row in selected_quantizers],
        "selected_quantizer_indices": [int(row["index"]) for row in selected_quantizers],
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "groups": aggregate_rows(rows),
        "timing": timing,
        "elapsed_seconds": float(time.time() - started),
        "model_size": build_model_size_report(
            quant_model,
            source_checkpoint_path=checkpoint.get("source_checkpoint"),
            quant_checkpoint_path=checkpoint_path,
        ),
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "config.json",
        build_run_config(
            args=args,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            eval_dataset_dir=eval_dataset_dir,
            selected_files=selected_files,
            conditions=conditions,
            device=device,
            checkpoint=checkpoint,
            quant_config=quant_config,
            weight_quant=weight_quant,
            act_quant=act_quant,
            selected_quantizers=selected_quantizers,
            manifest_warnings=manifest_warnings,
        ),
    )
    write_summary(run_dir / "summary.md", run_dir=run_dir, metrics=metrics, selected_csv=selected_csv)

    overall = metrics["groups"]["overall"][0]
    print(
        f"mode={args.mode} selected={len(selected_quantizers)} rows={metrics['sample_count']} "
        f"quant_snr_mean={overall['quant_post_recon_snr_db_mean']:.4f} "
        f"quant_snr_median={overall['quant_post_recon_snr_db_median']:.4f} "
        f"quant_ssim_mean={overall['quant_post_recon_ssim_mean']:.6f} "
        f"run_dir={run_dir}",
        flush=True,
    )


def parse_selector_sequence(text: str | None) -> tuple[str, ...]:
    """Parse comma-separated selector values."""
    if text is None:
        return ()
    return tuple(part.strip() for part in str(text).split(",") if part.strip())


def selector_summary(args: argparse.Namespace) -> dict[str, Any]:
    """Return normalized selector fields."""
    return {
        "index": args.index,
        "name_contains": args.name_contains,
        "stage": args.stage,
        "branch": args.branch,
        "role": args.role,
        "module_type": args.module_type,
        "stages": parse_selector_sequence(args.stages),
        "branches": parse_selector_sequence(args.branches),
        "roles": parse_selector_sequence(args.roles),
        "module_types": parse_selector_sequence(args.module_types),
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
    eval_dataset_dir: Path,
    selected_files: Sequence[Path],
    conditions: Sequence[Any],
    device: Any,
    checkpoint: Mapping[str, Any],
    quant_config: Mapping[str, Any],
    weight_quant: bool,
    act_quant: bool,
    selected_quantizers: Sequence[Mapping[str, Any]],
    manifest_warnings: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a traceable run config payload."""
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "run_dir": str(run_dir),
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "eval_dataset_dir": str(eval_dataset_dir),
        "testset_id": str(args.testset_id),
        "num_eval_samples": args.num_eval_samples,
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "mode": str(args.mode),
        "selector": selector_summary(args),
        "include_output_quantizer": bool(args.include_output_quantizer),
        "selected_quantizer_count": len(selected_quantizers),
        "selected_quantizers": [dict(row) for row in selected_quantizers],
        "selected_sample_paths": [str(path) for path in selected_files],
        "condition_count": len(conditions),
        "conditions": [
            {
                "condition_index": int(item.condition_index),
                "snr_setting_db": float(item.snr_setting_db),
                "missing_rate": float(item.missing_rate),
            }
            for item in conditions
        ],
        "source_checkpoint": checkpoint.get("source_checkpoint"),
        "source_checkpoint_epoch": checkpoint.get("source_checkpoint_epoch"),
        "source_checkpoint_loss": checkpoint.get("source_checkpoint_loss"),
        "model_config": checkpoint.get("model_config", {}),
        "quant_config": dict(quant_config),
        "final_quant_state": {"weight_quant": bool(weight_quant), "act_quant": bool(act_quant)},
        "manifest_warnings": list(manifest_warnings),
        "environment": collect_environment(),
    }


def write_selected_quantizers_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write selected quantizer rows to CSV."""
    fieldnames = [
        "index",
        "name",
        "stage",
        "branch",
        "role",
        "module_type",
        "weight_bit",
        "act_bit",
        "act_disabled",
        "act_inited",
        "act_delta_min",
        "act_delta_max",
        "act_zero_point_min",
        "act_zero_point_max",
        "weight_shape",
        "activation_numel",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(path: Path, *, run_dir: Path, metrics: Mapping[str, Any], selected_csv: Path) -> None:
    """Write a compact Markdown summary."""
    overall = metrics["groups"]["overall"][0]
    lines = [
        "# SCRN-BRECQ Activation Sensitivity Grid Evaluation",
        "",
        f"- run_dir: `{run_dir}`",
        f"- mode: `{metrics['mode']}`",
        f"- selected_quantizer_count: `{metrics['selected_quantizer_count']}`",
        f"- sample_count: `{metrics['sample_count']}`",
        f"- patch_count: `{metrics['patch_count']}`",
        f"- condition_count: `{metrics['condition_count']}`",
        f"- selected_quantizers: `{selected_csv}`",
        "",
        "## Overall Metrics",
        "",
        "| metric | mean | median | std | min | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for field in [
        "fp32_snr_db",
        "fp32_ssim",
        "quant_post_recon_snr_db",
        "quant_post_recon_ssim",
        "quant_post_minus_fp32_snr_db",
        "quant_post_minus_fp32_ssim",
    ]:
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
            "| source | count | quant_snr_mean | quant_ssim_mean | quant_minus_fp32_snr_mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in metrics["groups"]["by_source"]:
        lines.append(
            "| {source} | {count} | {snr:.6f} | {ssim:.6f} | {delta:.6f} |".format(
                source=row["source"],
                count=row["sample_count"],
                snr=row["quant_post_recon_snr_db_mean"],
                ssim=row["quant_post_recon_ssim_mean"],
                delta=row["quant_post_minus_fp32_snr_db_mean"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
