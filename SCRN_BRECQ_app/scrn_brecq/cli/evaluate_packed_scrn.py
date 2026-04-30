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
    select_device,
)
from SCRN_BRECQ_app.scrn_brecq.utils import build_model_size_report, load_packed_manifest, restore_packed_deployment
from SCRN_BRECQ_app.scrn_repro.training import collect_environment, create_run_dir, write_json, write_summary
from SCRN_BRECQ_app.scrn_repro.utils import snr_db, ssim_score


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
    reference_run_dir: Path | None = None
    reference_predictions: dict[str, np.ndarray] | None = None
    if save_figure:
        reference_run_dir = find_quant_run_dir_from_manifest(manifest)
        reference_predictions = load_deployment_reference_predictions(reference_run_dir)
        metrics.update(build_packed_checkpoint_diff_metrics(prediction, reference_predictions["checkpoint_final"]))
        metrics.update(
            {
                "fp32_snr_db": snr_db(reference_predictions["fp32"], clean),
                "fp32_ssim": ssim_score(reference_predictions["fp32"], clean),
                "checkpoint_final_snr_db": snr_db(reference_predictions["checkpoint_final"], clean),
                "checkpoint_final_ssim": ssim_score(reference_predictions["checkpoint_final"], clean),
            }
        )
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
            reference_run_dir=reference_run_dir,
        ),
    )
    if save_figure:
        if reference_predictions is None:
            raise RuntimeError("Reference predictions were not loaded for packed deployment figure.")
        save_deployment_alignment_figure(
            run_dir / "comparison.png",
            clean=clean,
            degraded=degraded,
            fp32_prediction=reference_predictions["fp32"],
            checkpoint_final_prediction=reference_predictions["checkpoint_final"],
            packed_prediction=prediction,
            quant_config=quant_config,
            metrics=metrics,
        )
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


def find_quant_run_dir_from_manifest(manifest: Mapping[str, Any]) -> Path:
    """Infer the original quant run directory from manifest['quant_checkpoint']."""
    quant_checkpoint = manifest.get("quant_checkpoint")
    if not quant_checkpoint:
        raise KeyError("Packed manifest does not include quant_checkpoint; cannot find reference run directory.")
    checkpoint_path = Path(str(quant_checkpoint))
    if checkpoint_path.parent.name != "checkpoints":
        raise ValueError(f"Expected quant_checkpoint under a checkpoints/ directory, got: {checkpoint_path}")
    return checkpoint_path.parent.parent


def load_deployment_reference_predictions(run_dir: str | Path) -> dict[str, np.ndarray]:
    """Load FP32 and checkpoint-final predictions required by the five-panel figure."""
    root = Path(run_dir)
    fp32_path = root / "fp32_prediction.npy"
    checkpoint_path = root / "quant_post_recon_prediction.npy"
    if not fp32_path.is_file():
        raise FileNotFoundError(f"Five-panel packed evaluation requires fp32_prediction.npy: {fp32_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Five-panel packed evaluation requires quant_post_recon_prediction.npy: {checkpoint_path}")
    fp32 = np.load(fp32_path).astype(np.float32)
    checkpoint_final = np.load(checkpoint_path).astype(np.float32)
    if fp32.shape != checkpoint_final.shape:
        raise ValueError(f"Reference prediction shapes differ: {fp32.shape} vs {checkpoint_final.shape}")
    return {"fp32": fp32, "checkpoint_final": checkpoint_final}


def build_packed_checkpoint_diff_metrics(packed_prediction: np.ndarray, checkpoint_prediction: np.ndarray) -> dict[str, float]:
    """Compute difference metrics between packed-restored and checkpoint-final outputs."""
    packed = packed_prediction.astype(np.float32)
    checkpoint = checkpoint_prediction.astype(np.float32)
    if packed.shape != checkpoint.shape:
        raise ValueError(f"packed and checkpoint prediction shapes differ: {packed.shape} vs {checkpoint.shape}")
    diff = packed - checkpoint
    abs_diff = np.abs(diff)
    return {
        "packed_vs_checkpoint_mse": float(np.mean(diff * diff)),
        "packed_vs_checkpoint_mean_abs_diff": float(np.mean(abs_diff)),
        "packed_vs_checkpoint_max_abs_diff": float(np.max(abs_diff)),
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
    reference_run_dir: Path | None,
) -> dict[str, Any]:
    """Build packed evaluation run config."""
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "packed_dir": str(packed_dir),
        "clean_path": str(clean_path),
        "input_path": str(input_path),
        "save_figure": bool(save_figure),
        "reference_run_dir": str(reference_run_dir) if reference_run_dir is not None else None,
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


def save_deployment_alignment_figure(
    path: Path,
    *,
    clean: np.ndarray,
    degraded: np.ndarray,
    fp32_prediction: np.ndarray,
    checkpoint_final_prediction: np.ndarray,
    packed_prediction: np.ndarray,
    quant_config: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> None:
    """Save a five-panel W4A32 packed deployment alignment figure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Saving comparison figures requires installing matplotlib.") from exc

    label = f"W{int(quant_config['n_bits_w'])}A32"
    panels = [
        (clean, "Ground Truth"),
        (degraded, f"Input\nSNR={metrics['input_snr_db']:.2f}dB SSIM={metrics['input_ssim']:.3f}"),
        (fp32_prediction, f"FP32 SCRN\nSNR={metrics['fp32_snr_db']:.2f}dB SSIM={metrics['fp32_ssim']:.3f}"),
        (
            checkpoint_final_prediction,
            f"{label} Checkpoint Final\n"
            f"SNR={metrics['checkpoint_final_snr_db']:.2f}dB SSIM={metrics['checkpoint_final_ssim']:.3f}",
        ),
        (
            packed_prediction,
            f"{label} Packed Restored\n"
            f"SNR={metrics['quant_snr_db']:.2f}dB SSIM={metrics['quant_ssim']:.3f}",
        ),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.8), constrained_layout=True)
    vmin = float(np.min(clean))
    vmax = float(np.max(clean))
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0

    image = None
    for axis, (array, title) in zip(axes, panels):
        image = axis.imshow(array, cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax)
        axis.set_title(title, fontsize=8, pad=6)
        axis.axis("off")
    fig.colorbar(image, ax=axes, shrink=0.72, fraction=0.02, pad=0.01)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def require_directory(path: str | Path, description: str) -> Path:
    """Check that a directory exists and return it as ``Path``."""
    candidate = Path(path)
    if not candidate.is_dir():
        raise FileNotFoundError(f"{description} does not exist: {candidate}")
    return candidate


if __name__ == "__main__":
    main()
