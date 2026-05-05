"""Evaluate selective activation quantizer sensitivity for SCRN-BRECQ checkpoints."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Mapping

from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn import (
    build_quant_model_from_checkpoint,
    load_quant_checkpoint,
    normalize_quant_config,
    require_file,
    restore_quantizer_state_shapes,
    select_device,
)
from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_multi import (
    DEFAULT_EVAL_DATASET_DIR,
    build_aggregate_metrics,
    evaluate_files,
    require_directory,
    select_eval_files,
    summary_metrics,
    write_jsonl,
)
from SCRN_BRECQ_app.scrn_brecq.quant.activation_sensitivity import (
    SENSITIVITY_MODES,
    apply_activation_sensitivity_mode,
)
from SCRN_BRECQ_app.scrn_brecq.utils import build_model_size_report
from SCRN_BRECQ_app.scrn_repro.data import discover_patch_files
from SCRN_BRECQ_app.scrn_repro.training import collect_environment, create_run_dir, write_json, write_summary


DEFAULT_CHECKPOINT = (
    "SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E002c_init_sensitivity/quant/"
    "20260505_150842_e002c_init_n0064/checkpoints/quantized_scrn_brecq_pre_act_recon.pth"
)
DEFAULT_RUN_ROOT = "SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E004_sensitivity/e004a_tool_smoke"


def build_parser() -> argparse.ArgumentParser:
    """Build the E004a activation sensitivity evaluation parser."""
    parser = argparse.ArgumentParser(description="Evaluate selective activation quantizer sensitivity.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Quantized checkpoint to evaluate")
    parser.add_argument("--mode", choices=sorted(SENSITIVITY_MODES), required=True, help="Sensitivity switch mode")
    parser.add_argument("--index", type=int, default=None, help="Quantizer index selector")
    parser.add_argument("--name-contains", default=None, help="Substring selector for QuantModule name")
    parser.add_argument("--stage", default=None, help="Structure stage selector")
    parser.add_argument("--branch", default=None, help="Structure branch selector")
    parser.add_argument("--role", default=None, help="Structure role selector")
    parser.add_argument("--module-type", default=None, help="Module type selector, e.g. Conv2d or Linear")
    parser.add_argument("--eval-dataset-dir", default=DEFAULT_EVAL_DATASET_DIR, help="Directory of clean eval patch .npy files")
    parser.add_argument("--num-eval-samples", type=int, default=128, help="Number of clean patches to evaluate")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--seed", type=int, default=20260427, help="Seed for sample selection and online degradation")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT, help="Output run root")
    parser.add_argument("--run-name", default="e004a_activation_sensitivity", help="Run name suffix")
    parser.add_argument("--include-output-quantizer", action="store_true", help="Allow selecting the final output quantizer")
    parser.add_argument("--save-figures", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-figures", type=int, default=8, help="Maximum comparison figures to save")
    return parser


def main() -> None:
    """Run selective activation quantizer sensitivity evaluation."""
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
    quant_model.disable_network_output_quantization()

    final_state = checkpoint.get("final_quant_state", {})
    weight_quant = bool(final_state.get("weight_quant", True))
    act_quant = bool(final_state.get("act_quant", quant_config.get("act_quant", False)))

    all_files = discover_patch_files(eval_dataset_dir)
    selected_files = select_eval_files(all_files, num_samples=int(args.num_eval_samples), seed=int(args.seed))
    run_dir = create_run_dir(args.run_root, run_name=args.run_name)
    per_sample_path = run_dir / "per_sample_metrics.jsonl"
    figures_dir = run_dir / "figures"
    if bool(args.save_figures) and int(args.max_figures) > 0:
        figures_dir.mkdir(parents=True, exist_ok=True)

    with apply_activation_sensitivity_mode(
        quant_model,
        mode=str(args.mode),
        index=args.index,
        name_contains=args.name_contains,
        stage=args.stage,
        branch=args.branch,
        role=args.role,
        module_type=args.module_type,
        include_output_quantizer=bool(args.include_output_quantizer),
    ) as selected_quantizers:
        rows, timing = evaluate_files(
            quant_model,
            selected_files,
            device=device,
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            weight_quant=weight_quant,
            act_quant=act_quant,
            pre_recon_model=None,
            pre_recon_weight_quant=False,
            pre_recon_act_quant=False,
            figures_dir=figures_dir if bool(args.save_figures) else None,
            max_figures=int(args.max_figures),
        )

    write_jsonl(per_sample_path, rows)
    selected_csv = run_dir / "selected_quantizers.csv"
    write_selected_quantizers_csv(selected_csv, selected_quantizers)

    metrics = build_aggregate_metrics(rows)
    metrics.update(
        {
            "sample_count": len(rows),
            "dataset_file_count": len(all_files),
            "mode": str(args.mode),
            "selected_quantizer_count": len(selected_quantizers),
            "selected_quantizer_names": [str(row["name"]) for row in selected_quantizers],
            "selected_quantizer_indices": [int(row["index"]) for row in selected_quantizers],
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
            selected_files=selected_files,
            device=device,
            checkpoint=checkpoint,
            quant_config=quant_config,
            weight_quant=weight_quant,
            act_quant=act_quant,
            selected_quantizers=selected_quantizers,
        ),
    )
    write_summary(
        run_dir / "summary.md",
        title="SCRN-BRECQ Activation Sensitivity Evaluation",
        sections={
            "Sensitivity": {
                "mode": str(args.mode),
                "selector": selector_summary(args),
                "selected_quantizer_count": len(selected_quantizers),
                "selected_quantizer_names": [str(row["name"]) for row in selected_quantizers],
            },
            "Metrics": summary_metrics(metrics),
            "Artifacts": {
                "config": run_dir / "config.json",
                "metrics": run_dir / "metrics.json",
                "selected_quantizers": selected_csv,
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
        },
    )
    print(
        f"mode={args.mode} selected={len(selected_quantizers)} samples={metrics['sample_count']} "
        f"quant_snr_mean={metrics['quant_snr_db_mean']:.4f} "
        f"quant_snr_median={metrics['quant_snr_db_median']:.4f} "
        f"quant_ssim_mean={metrics['quant_ssim_mean']:.4f} elapsed={metrics['elapsed_seconds']:.2f}s",
        flush=True,
    )
    print(f"[SCRN-BRECQ] activation_sensitivity_run_dir={run_dir}", flush=True)


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    if int(args.num_eval_samples) <= 0:
        raise ValueError(f"--num-eval-samples must be positive, got {args.num_eval_samples}")
    if int(args.batch_size) <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if int(args.max_figures) < 0:
        raise ValueError(f"--max-figures must be non-negative, got {args.max_figures}")


def build_run_config(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint_path: Path,
    eval_dataset_dir: Path,
    selected_files: list[Path],
    device,
    checkpoint: Mapping[str, Any],
    quant_config: Mapping[str, Any],
    weight_quant: bool,
    act_quant: bool,
    selected_quantizers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a traceable E004a run config payload."""
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
        "mode": str(args.mode),
        "selector": selector_summary(args),
        "include_output_quantizer": bool(args.include_output_quantizer),
        "selected_quantizers": selected_quantizers,
        "selected_sample_paths": [str(path) for path in selected_files],
        "source_checkpoint": checkpoint.get("source_checkpoint"),
        "source_checkpoint_epoch": checkpoint.get("source_checkpoint_epoch"),
        "source_checkpoint_loss": checkpoint.get("source_checkpoint_loss"),
        "model_config": checkpoint.get("model_config", {}),
        "quant_config": dict(quant_config),
        "final_quant_state": {"weight_quant": bool(weight_quant), "act_quant": bool(act_quant)},
        "environment": collect_environment(),
    }


def selector_summary(args: argparse.Namespace) -> dict[str, Any]:
    """Return the active selector fields."""
    return {
        "index": args.index,
        "name_contains": args.name_contains,
        "stage": args.stage,
        "branch": args.branch,
        "role": args.role,
        "module_type": args.module_type,
    }


def write_selected_quantizers_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


if __name__ == "__main__":
    main()
