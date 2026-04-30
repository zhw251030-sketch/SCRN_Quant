"""Evaluate a packed SCRN-BRECQ deployment artifact.

This validates that ``manifest.json`` plus the binary payloads can restore the
quantized model numerically.  The restored model uses dequantized FP32 weights
inside PyTorch; this script is not an INT4/INT8 deployment runtime.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn import (
    build_metrics,
    build_quant_model_from_checkpoint,
    load_eval_arrays,
    normalize_quant_config,
    predict_array,
    resolve_eval_paths,
    save_evaluation_figure,
    select_device,
)
from SCRN_BRECQ_app.scrn_brecq.utils import build_model_size_report, load_packed_manifest, restore_packed_deployment
from SCRN_BRECQ_app.scrn_repro.training import collect_environment, create_run_dir, write_json, write_summary


DEFAULT_RUN_ROOT = "SCRN_BRECQ_app/scrn_brecq/runs/packed_eval"


def build_parser() -> argparse.ArgumentParser:
    """Build packed deployment evaluation argument parser."""
    parser = argparse.ArgumentParser(description="Evaluate a packed SCRN-BRECQ deployment artifact.")
    parser.add_argument("--packed-dir", required=True, help="Directory containing manifest.json, weights.bin, and aux_fp32.bin")
    parser.add_argument("--eval-clean-path", default=None, help="Clean reference .npy; defaults to packed quant_config")
    parser.add_argument("--eval-input-path", default=None, help="Degraded input .npy; defaults to packed quant_config")
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT, help="Evaluation run output root")
    parser.add_argument("--run-name", default="packed_eval", help="Run name suffix")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--save-figure", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main() -> None:
    """CLI entrypoint."""
    args = build_parser().parse_args()
    device = select_device(args.device)
    packed_dir = require_directory(args.packed_dir, "packed deployment directory")
    manifest = load_packed_manifest(packed_dir)
    checkpoint_like = checkpoint_like_from_manifest(manifest)
    quant_config = normalize_quant_config(checkpoint_like["quant_config"])
    clean_path, input_path = resolve_eval_paths(args, quant_config)
    save_figure = bool(quant_config.get("save_figure", False)) if args.save_figure is None else bool(args.save_figure)

    quant_model = build_quant_model_from_checkpoint(checkpoint_like)
    quant_model.to(device)
    restore_summary = restore_packed_deployment(quant_model, packed_dir)

    final_state = manifest.get("final_quant_state") or {}
    act_quant = bool(final_state.get("act_quant", quant_config.get("act_quant", False)))
    if act_quant:
        quant_model.disable_network_output_quantization()
    quant_model.set_quant_state(False, act_quant)
    quant_model.eval()

    clean, degraded = load_eval_arrays(clean_path, input_path)
    prediction, seconds = predict_array(quant_model, degraded, device)
    metrics = build_metrics(clean, degraded, prediction, seconds)
    metrics["packed_restore"] = restore_summary
    metrics["model_size"] = build_model_size_report(
        quant_model,
        source_checkpoint_path=manifest.get("source_checkpoint"),
        quant_checkpoint_path=manifest.get("quant_checkpoint"),
    )

    run_dir = create_run_dir(args.run_root, run_name=args.run_name)
    np.save(run_dir / "prediction.npy", prediction)
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "config.json",
        build_run_config(
            packed_dir=packed_dir,
            clean_path=clean_path,
            input_path=input_path,
            device=device,
            manifest=manifest,
            quant_config=quant_config,
            save_figure=save_figure,
            restore_summary=restore_summary,
        ),
    )
    if save_figure:
        save_evaluation_figure(run_dir / "comparison.png", clean=clean, degraded=degraded, prediction=prediction, metrics=metrics)
    write_summary(
        run_dir / "summary.md",
        title="SCRN-BRECQ Packed Deployment Evaluation",
        sections={
            "Metrics": metrics,
            "Packed Deployment": {
                "packed_dir": packed_dir,
                "manifest_format": manifest.get("format"),
                "manifest_version": manifest.get("format_version"),
                "restore": restore_summary,
            },
            "Model Size": metrics["model_size"],
            "Inputs": {
                "packed_dir": packed_dir,
                "clean": clean_path,
                "input": input_path,
                "prediction": run_dir / "prediction.npy",
            },
            "Quantization": {
                "weight_quant": False,
                "act_quant": act_quant,
                "n_bits_w": quant_config["n_bits_w"],
                "n_bits_a": quant_config["n_bits_a"],
            },
        },
    )
    print(
        "input_snr={input_snr_db:.4f} packed_snr={quant_snr_db:.4f} "
        "input_ssim={input_ssim:.4f} packed_ssim={quant_ssim:.4f} seconds={quant_inference_seconds:.4f}".format(
            **metrics
        ),
        flush=True,
    )
    print(f"[SCRN-BRECQ] packed_eval_run_dir={run_dir}", flush=True)


def checkpoint_like_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a packed manifest into the checkpoint-like metadata used by evaluators."""
    metadata = manifest.get("checkpoint_metadata") or {}
    if not isinstance(metadata, Mapping):
        raise TypeError("manifest['checkpoint_metadata'] must be a mapping.")
    return {
        "model_config": dict(metadata.get("model_config") or {}),
        "quant_config": dict(metadata.get("quant_config") or {}),
        "source_checkpoint": manifest.get("source_checkpoint"),
        "quant_checkpoint": manifest.get("quant_checkpoint"),
        "checkpoint_stage": metadata.get("checkpoint_stage"),
        "final_quant_state": dict(manifest.get("final_quant_state") or {}),
    }


def build_run_config(
    *,
    packed_dir: Path,
    clean_path: Path,
    input_path: Path,
    device: Any,
    manifest: Mapping[str, Any],
    quant_config: Mapping[str, Any],
    save_figure: bool,
    restore_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Build packed evaluation run config."""
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "packed_dir": str(packed_dir),
        "clean_path": str(clean_path),
        "input_path": str(input_path),
        "save_figure": bool(save_figure),
        "source_checkpoint": manifest.get("source_checkpoint"),
        "quant_checkpoint": manifest.get("quant_checkpoint"),
        "manifest_format": manifest.get("format"),
        "manifest_version": manifest.get("format_version"),
        "checkpoint_metadata": manifest.get("checkpoint_metadata", {}),
        "quant_config": dict(quant_config),
        "final_quant_state": manifest.get("final_quant_state", {}),
        "restore_summary": dict(restore_summary),
        "environment": collect_environment(),
    }


def require_directory(path: str | Path, description: str) -> Path:
    """Check that a directory exists and return it as ``Path``."""
    candidate = Path(path)
    if not candidate.is_dir():
        raise FileNotFoundError(f"{description} does not exist: {candidate}")
    return candidate


if __name__ == "__main__":
    main()
