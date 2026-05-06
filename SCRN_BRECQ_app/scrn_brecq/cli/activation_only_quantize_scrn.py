"""Run activation-only SCRN-BRECQ quantization from a W4 weight-recon checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn import (
    build_quant_model_from_checkpoint,
    load_quant_checkpoint,
    restore_quantizer_state_shapes,
    scrn_config_from_mapping,
)
from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_multi import select_eval_device
from SCRN_BRECQ_app.scrn_brecq.cli.quantize_scrn import (
    initialize_activation_quantization,
    load_eval_arrays,
    predict_array,
    run_activation_reconstruction,
    save_quant_checkpoint,
    serializable_config,
)
from SCRN_BRECQ_app.scrn_brecq.data import CalibrationDataConfig, load_calibration_data
from SCRN_BRECQ_app.scrn_brecq.quant.activation_range import (
    apply_activation_ranges,
    normalize_selector_groups,
    parse_mse_shrink_ratios,
)
from SCRN_BRECQ_app.scrn_brecq.quant.activation_diagnostics import summarize_activation_quantizers
from SCRN_BRECQ_app.scrn_brecq.quant.activation_diagnostics import collect_quantizer_rows
from SCRN_BRECQ_app.scrn_brecq.utils import build_model_size_report, load_json, require_file
from SCRN_BRECQ_app.scrn_repro.training import collect_environment, create_run_dir, write_json, write_summary
from SCRN_BRECQ_app.scrn_repro.utils import set_random_seed, snr_db, ssim_score


DEFAULT_WEIGHT_RECON_CHECKPOINT = Path(
    "SCRN_BRECQ_app/scrn_brecq/runs/quant/"
    "20260504_221242_e002b_w4a8_positive_scale_1024samples_w20000_a5000/"
    "checkpoints/quantized_scrn_brecq_weight_recon.pth"
)
DEFAULT_RUN_ROOT = Path("SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E002c_init_sensitivity/quant")
DEFAULT_CALIBRATION_DATASET_DIR = Path("SCRN_BRECQ_app/scrn_repro/datasets/scrn_quant_10750_0_patches")


def build_parser() -> argparse.ArgumentParser:
    """Build the E002c activation-only quantization parser."""
    parser = argparse.ArgumentParser(description="Run activation-only SCRN-BRECQ quantization from a W4 checkpoint.")
    parser.add_argument("--config", default=None, help="Activation-only quantization config JSON")
    parser.add_argument(
        "--weight-recon-checkpoint",
        default=None,
        help="W4 weight-reconstruction checkpoint to resume from",
    )
    parser.add_argument("--calibration-dataset-dir", default=None, help="Calibration clean patch directory")
    parser.add_argument("--eval-clean-path", default=None, help="Evaluation clean reference .npy")
    parser.add_argument("--eval-input-path", default=None, help="Evaluation degraded input .npy")
    parser.add_argument("--run-root", default=None, help="Activation-only run output root")
    parser.add_argument("--run-name", default=None, help="Run name suffix")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--gpus", default=None, help="Visible GPU ids, e.g. `0`")
    parser.add_argument("--cuda-device-index", type=int, default=None, help="Explicit CUDA device index, e.g. 1 for cuda:1")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--init-batch-size", type=int, default=None)
    parser.add_argument("--iters-a", type=int, default=None)
    parser.add_argument("--activation-lr", type=float, default=None)
    parser.add_argument("--lp-norm", type=float, default=None)
    parser.add_argument("--skip-act-recon", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--activation-range-method", choices=["none", "percentile", "max", "mse_grid"], default=None)
    parser.add_argument("--activation-percentile", type=float, default=None)
    parser.add_argument("--range-mse-shrink-ratios", default=None)
    parser.add_argument("--range-loss-p", type=float, default=None)
    parser.add_argument("--range-index", type=int, default=None)
    parser.add_argument("--range-name-contains", default=None)
    parser.add_argument("--range-stage", default=None)
    parser.add_argument("--range-branch", default=None)
    parser.add_argument("--range-role", default=None)
    parser.add_argument("--range-module-type", default=None)
    parser.add_argument("--range-selector-groups-json", default=None)
    parser.add_argument("--range-exclude-selector-groups-json", default=None)
    parser.add_argument("--range-max-values-per-layer", type=int, default=None)
    parser.add_argument("--include-output-quantizer", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main() -> None:
    """Run activation initialization, optional act recon, and save traceable artifacts."""
    run_start_time = time.time()
    args = build_parser().parse_args()
    initial_config = load_initial_config(args)
    weight_recon_checkpoint = require_file(initial_config["weight_recon_checkpoint"], "weight-recon checkpoint")
    checkpoint = load_quant_checkpoint(weight_recon_checkpoint)
    config = load_and_resolve_config(args, checkpoint)
    configure_visible_gpus(config)
    device = select_eval_device(str(config["device"]), config.get("cuda_device_index"))
    set_random_seed(int(config["seed"]))

    run_dir = create_run_dir(config["run_root"], run_name=str(config["run_name"]))
    print(f"[SCRN-BRECQ] activation_only_run_dir={run_dir}", flush=True)

    quant_model = build_quant_model_from_checkpoint(checkpoint)
    state_dict = checkpoint["quant_model_state_dict"]
    restore_quantizer_state_shapes(quant_model, state_dict)
    quant_model.load_state_dict(state_dict, strict=True)
    quant_model.to(device)
    quant_model.eval()

    clean, degraded = load_eval_arrays(config["eval_clean_path"], config["eval_input_path"])
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

    quant_model.set_quant_state(True, False)
    post_weight_prediction, post_weight_seconds = predict_array(quant_model, degraded, device)

    init_start_time = time.time()
    initialize_activation_quantization(quant_model, calibration_data, config)
    activation_range_summary = apply_activation_range_calibration(quant_model, calibration_data, config)
    activation_initialization_seconds = time.time() - init_start_time

    quant_model.set_quant_state(True, True)
    quant_model.disable_network_output_quantization()
    pre_act_prediction, pre_act_seconds = predict_array(quant_model, degraded, device)
    metrics = build_activation_only_metrics(
        clean,
        degraded,
        post_weight_prediction=post_weight_prediction,
        post_weight_seconds=post_weight_seconds,
        pre_act_prediction=pre_act_prediction,
        pre_act_seconds=pre_act_seconds,
    )
    metrics["activation_initialization_seconds"] = float(activation_initialization_seconds)
    metrics["activation_initialization_minutes"] = float(activation_initialization_seconds) / 60.0
    metrics["activation_reconstruction_seconds"] = 0.0
    metrics["activation_reconstruction_minutes"] = 0.0
    metrics["elapsed_seconds"] = float(time.time() - run_start_time)
    metrics["elapsed_minutes"] = metrics["elapsed_seconds"] / 60.0
    metrics["activation_quantizer_summary"] = summarize_activation_quantizers(collect_quantizer_rows(quant_model))
    metrics["activation_range_summary"] = activation_range_summary
    metrics["source_weight_recon_metrics"] = checkpoint.get("metrics", {})
    metrics["model_size"] = build_model_size_report(
        quant_model,
        source_checkpoint_path=checkpoint.get("source_checkpoint"),
    )

    loaded = loaded_source_from_checkpoint(checkpoint, fallback_checkpoint=weight_recon_checkpoint)
    pre_act_checkpoint = save_quant_checkpoint(
        run_dir,
        quant_model,
        loaded,
        config,
        metrics,
        checkpoint_name="quantized_scrn_brecq_pre_act_recon.pth",
        checkpoint_stage="pre_activation_reconstruction",
        final_quant_state={"weight_quant": True, "act_quant": True},
    )
    final_checkpoint = None

    if not bool(config["skip_act_recon"]):
        act_start_time = time.time()
        run_activation_reconstruction(quant_model, calibration_data, config, is_main=True)
        activation_reconstruction_seconds = time.time() - act_start_time
        quant_model.set_quant_state(True, True)
        quant_model.disable_network_output_quantization()
        post_act_prediction, post_act_seconds = predict_array(quant_model, degraded, device)
        post_act_snr = snr_db(post_act_prediction, clean)
        post_act_ssim = ssim_score(post_act_prediction, clean)
        metrics.update(
            {
                "quant_post_act_recon_snr_db": post_act_snr,
                "quant_post_act_recon_ssim": post_act_ssim,
                "quant_post_act_recon_inference_seconds": float(post_act_seconds),
                "quant_act_recon_snr_gain_db": post_act_snr - metrics["quant_pre_act_recon_snr_db"],
                "quant_act_recon_ssim_gain": post_act_ssim - metrics["quant_pre_act_recon_ssim"],
                "quant_post_recon_snr_db": post_act_snr,
                "quant_post_recon_ssim": post_act_ssim,
                "quant_post_recon_inference_seconds": float(post_act_seconds),
                "activation_reconstruction_seconds": float(activation_reconstruction_seconds),
                "activation_reconstruction_minutes": float(activation_reconstruction_seconds) / 60.0,
                "activation_quantizer_summary": summarize_activation_quantizers(collect_quantizer_rows(quant_model)),
            }
        )
        metrics["elapsed_seconds"] = float(time.time() - run_start_time)
        metrics["elapsed_minutes"] = metrics["elapsed_seconds"] / 60.0
        final_checkpoint = save_quant_checkpoint(
            run_dir,
            quant_model,
            loaded,
            config,
            metrics,
            checkpoint_name="quantized_scrn_brecq.pth",
            checkpoint_stage="post_activation_reconstruction",
            final_quant_state={"weight_quant": True, "act_quant": True},
        )

    write_json(run_dir / "config.json", build_run_config(config, weight_recon_checkpoint, checkpoint, device))
    write_json(run_dir / "metrics.json", metrics)
    write_summary(
        run_dir / "summary.md",
        title="SCRN-BRECQ Activation-Only Quantization Run",
        sections={
            "Metrics": metrics,
            "Artifacts": {
                "pre_act_recon_checkpoint": pre_act_checkpoint,
                "final_checkpoint": final_checkpoint or "disabled_by_skip_act_recon",
                "config": run_dir / "config.json",
                "metrics": run_dir / "metrics.json",
            },
            "Inputs": {
                "weight_recon_checkpoint": weight_recon_checkpoint,
                "calibration_dataset_dir": config["calibration_dataset_dir"],
                "eval_clean_path": config["eval_clean_path"],
                "eval_input_path": config["eval_input_path"],
            },
            "Quantization": {
                "n_bits_w": config["n_bits_w"],
                "n_bits_a": config["n_bits_a"],
                "num_samples": config["num_samples"],
                "init_batch_size": config["init_batch_size"],
                "skip_act_recon": config["skip_act_recon"],
                "activation_range_method": config["activation_range_method"],
                "activation_percentile": config["activation_percentile"],
                "range_selector": activation_range_summary.get("selector", {}),
            },
        },
    )
    print(
        "post_weight_snr={quant_post_weight_recon_snr_db:.4f} "
        "pre_act_snr={quant_pre_act_recon_snr_db:.4f} "
        "act_init_delta={quant_act_init_snr_delta_db:.4f} "
        "non_positive_delta_count={non_positive_delta_count}".format(
            non_positive_delta_count=metrics["activation_quantizer_summary"]["non_positive_delta_count"],
            **metrics,
        ),
        flush=True,
    )
    print(f"[SCRN-BRECQ] pre_act_recon_checkpoint={pre_act_checkpoint}", flush=True)


def load_and_resolve_config(args: argparse.Namespace, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve CLI args over checkpoint quant config and local defaults."""
    config = default_config()
    if args.config:
        config.update(dict(load_json(args.config)))
    checkpoint_override_keys = {
        "calibration_dataset_dir",
        "eval_clean_path",
        "eval_input_path",
        "seed",
        "num_samples",
        "batch_size",
        "num_workers",
        "init_batch_size",
        "n_bits_w",
        "n_bits_a",
        "channel_wise",
        "act_quant",
        "disable_8bit_head_stem",
        "scale_method",
        "iters_a",
        "activation_lr",
        "lp_norm",
        "asym",
        "save_figure",
        "activation_range_method",
        "activation_percentile",
        "range_mse_shrink_ratios",
        "range_loss_p",
        "range_index",
        "range_name_contains",
        "range_stage",
        "range_branch",
        "range_role",
        "range_module_type",
        "range_selector_groups",
        "range_exclude_selector_groups",
        "range_selector_groups_json",
        "range_exclude_selector_groups_json",
        "range_max_values_per_layer",
        "include_output_quantizer",
    }
    config.update(
        {
            key: value
            for key, value in dict(checkpoint.get("quant_config", {})).items()
            if key in checkpoint_override_keys
        }
    )
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        config[key] = value
    config["act_quant"] = True
    config["distributed"] = False
    return normalize_config(config)


def load_initial_config(args: argparse.Namespace) -> dict[str, Any]:
    """Load enough config to locate the starting weight-recon checkpoint."""
    config = default_config()
    if args.config:
        config.update(dict(load_json(args.config)))
    if args.weight_recon_checkpoint is not None:
        config["weight_recon_checkpoint"] = str(args.weight_recon_checkpoint)
    return normalize_config(config)


def default_config() -> dict[str, Any]:
    """Return activation-only defaults used when no checkpoint field overrides them."""
    return {
        "weight_recon_checkpoint": str(DEFAULT_WEIGHT_RECON_CHECKPOINT),
        "calibration_dataset_dir": str(DEFAULT_CALIBRATION_DATASET_DIR),
        "eval_clean_path": "SCRN-main/test_data/clear.npy",
        "eval_input_path": "SCRN-main/test_data/noise_and_miss.npy",
        "run_root": str(DEFAULT_RUN_ROOT),
        "run_name": "activation_only_init",
        "device": "cuda",
        "gpus": "",
        "cuda_device_index": None,
        "seed": 1005,
        "num_samples": 1024,
        "batch_size": 16,
        "num_workers": 0,
        "init_batch_size": 64,
        "n_bits_w": 4,
        "n_bits_a": 8,
        "channel_wise": True,
        "act_quant": True,
        "disable_8bit_head_stem": False,
        "scale_method": "mse",
        "iters_a": 5000,
        "activation_lr": 0.0004,
        "lp_norm": 2.4,
        "skip_act_recon": True,
        "activation_range_method": "none",
        "activation_percentile": 99.99,
        "range_mse_shrink_ratios": "1.0,0.999,0.995,0.99,0.98,0.97,0.96,0.95,0.925,0.9,0.875,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5",
        "range_loss_p": 2.4,
        "range_index": None,
        "range_name_contains": None,
        "range_stage": None,
        "range_branch": None,
        "range_role": None,
        "range_module_type": None,
        "range_selector_groups": None,
        "range_exclude_selector_groups": None,
        "range_selector_groups_json": None,
        "range_exclude_selector_groups_json": None,
        "range_max_values_per_layer": 500_000,
        "include_output_quantizer": False,
        "distributed": False,
        "asym": True,
        "save_figure": False,
    }


def normalize_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and validate activation-only config values."""
    normalized = default_config()
    normalized.update(dict(config))
    normalized["weight_recon_checkpoint"] = str(require_file(normalized["weight_recon_checkpoint"], "weight-recon checkpoint"))
    normalized["calibration_dataset_dir"] = str(Path(normalized["calibration_dataset_dir"]))
    normalized["eval_clean_path"] = str(require_file(normalized["eval_clean_path"], "clean reference"))
    normalized["eval_input_path"] = str(require_file(normalized["eval_input_path"], "degraded input"))
    normalized["run_root"] = str(normalized["run_root"])
    normalized["run_name"] = str(normalized["run_name"])
    normalized["device"] = str(normalized["device"])
    normalized["gpus"] = str(normalized.get("gpus", "") or "")
    cuda_device_index = normalized.get("cuda_device_index")
    normalized["cuda_device_index"] = None if cuda_device_index is None else int(cuda_device_index)
    range_index = normalized.get("range_index")
    normalized["range_index"] = None if range_index is None else int(range_index)
    int_keys = [
        "seed",
        "num_samples",
        "batch_size",
        "num_workers",
        "init_batch_size",
        "n_bits_w",
        "n_bits_a",
        "iters_a",
        "range_max_values_per_layer",
    ]
    for key in int_keys:
        normalized[key] = int(normalized[key])
    for key in ["num_samples", "batch_size", "init_batch_size", "n_bits_w", "n_bits_a", "iters_a", "range_max_values_per_layer"]:
        if normalized[key] <= 0:
            raise ValueError(f"{key} must be positive, got {normalized[key]}")
    if normalized["num_workers"] < 0:
        raise ValueError(f"num_workers must be non-negative, got {normalized['num_workers']}")
    normalized["activation_lr"] = float(normalized["activation_lr"])
    normalized["lp_norm"] = float(normalized["lp_norm"])
    normalized["activation_percentile"] = float(normalized["activation_percentile"])
    normalized["range_loss_p"] = float(normalized["range_loss_p"])
    normalized["range_mse_shrink_ratios"] = parse_mse_shrink_ratios(normalized["range_mse_shrink_ratios"])
    normalized["range_selector_groups"] = _normalize_config_selector_groups(
        normalized.get("range_selector_groups"),
        normalized.get("range_selector_groups_json"),
        "range_selector_groups",
        "range_selector_groups_json",
    )
    normalized["range_exclude_selector_groups"] = _normalize_config_selector_groups(
        normalized.get("range_exclude_selector_groups"),
        normalized.get("range_exclude_selector_groups_json"),
        "range_exclude_selector_groups",
        "range_exclude_selector_groups_json",
    )
    if normalized["activation_lr"] <= 0.0:
        raise ValueError(f"activation_lr must be positive, got {normalized['activation_lr']}")
    if normalized["range_loss_p"] <= 0.0:
        raise ValueError(f"range_loss_p must be positive, got {normalized['range_loss_p']}")
    if normalized["activation_percentile"] <= 0.0 or normalized["activation_percentile"] >= 100.0:
        raise ValueError(f"activation_percentile must be between 0 and 100, got {normalized['activation_percentile']}")
    if normalized["activation_range_method"] not in {"none", "percentile", "max", "mse_grid"}:
        raise ValueError(f"Unsupported activation_range_method: {normalized['activation_range_method']}")
    if normalized["cuda_device_index"] is not None and normalized["cuda_device_index"] < 0:
        raise ValueError(f"cuda_device_index must be non-negative, got {normalized['cuda_device_index']}")
    for key in [
        "channel_wise",
        "act_quant",
        "disable_8bit_head_stem",
        "skip_act_recon",
        "include_output_quantizer",
        "distributed",
        "asym",
        "save_figure",
    ]:
        normalized[key] = bool(normalized[key])
    for key in ["range_name_contains", "range_stage", "range_branch", "range_role", "range_module_type"]:
        normalized[key] = None if normalized.get(key) is None else str(normalized[key])
    if not Path(normalized["calibration_dataset_dir"]).is_dir():
        raise FileNotFoundError(f"calibration_dataset_dir does not exist: {normalized['calibration_dataset_dir']}")
    return normalized


def _normalize_config_selector_groups(
    value: Any,
    json_value: Any,
    field_name: str,
    json_field_name: str,
) -> list[dict[str, Any]] | None:
    """Parse optional selector group config from JSON CLI args or config objects."""
    if json_value is not None:
        try:
            value = json.loads(str(json_value))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{json_field_name} must be valid JSON.") from exc
    return normalize_selector_groups(value, field_name)


def configure_visible_gpus(config: Mapping[str, Any]) -> None:
    """Restrict visible CUDA devices before device selection."""
    gpus = str(config.get("gpus", "")).strip()
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def apply_activation_range_calibration(
    quant_model: torch.nn.Module,
    calibration_data: torch.Tensor,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply optional post-init activation range calibration."""
    method = str(config["activation_range_method"])
    if method == "none":
        return {"method": "none"}
    if method not in {"percentile", "max", "mse_grid"}:
        raise ValueError(f"Unsupported activation_range_method: {method}")
    device = next(quant_model.parameters()).device
    init_inputs = calibration_data[: min(int(config["init_batch_size"]), int(calibration_data.size(0)))].to(device)
    return apply_activation_ranges(
        quant_model,
        init_inputs,
        method=method,
        percentile=float(config["activation_percentile"]),
        mse_shrink_ratios=config.get("range_mse_shrink_ratios"),
        loss_p=float(config["range_loss_p"]),
        index=config.get("range_index"),
        name_contains=config.get("range_name_contains"),
        stage=config.get("range_stage"),
        branch=config.get("range_branch"),
        role=config.get("range_role"),
        module_type=config.get("range_module_type"),
        selector_groups=config.get("range_selector_groups"),
        exclude_selector_groups=config.get("range_exclude_selector_groups"),
        include_output_quantizer=bool(config["include_output_quantizer"]),
        max_values_per_layer=int(config["range_max_values_per_layer"]),
        weight_quant=True,
    )


def build_activation_only_metrics(
    clean: np.ndarray,
    degraded: np.ndarray,
    *,
    post_weight_prediction: np.ndarray,
    post_weight_seconds: float,
    pre_act_prediction: np.ndarray,
    pre_act_seconds: float,
) -> dict[str, Any]:
    """Build metrics for the W4 checkpoint before and after A8 initialization."""
    post_weight_snr = snr_db(post_weight_prediction, clean)
    post_weight_ssim = ssim_score(post_weight_prediction, clean)
    pre_act_snr = snr_db(pre_act_prediction, clean)
    pre_act_ssim = ssim_score(pre_act_prediction, clean)
    metrics = {
        "input_snr_db": snr_db(degraded, clean),
        "input_ssim": ssim_score(degraded, clean),
        "quant_post_weight_recon_snr_db": post_weight_snr,
        "quant_post_weight_recon_ssim": post_weight_ssim,
        "quant_post_weight_recon_inference_seconds": float(post_weight_seconds),
        "quant_pre_act_recon_snr_db": pre_act_snr,
        "quant_pre_act_recon_ssim": pre_act_ssim,
        "quant_pre_act_recon_inference_seconds": float(pre_act_seconds),
        "quant_act_init_snr_delta_db": pre_act_snr - post_weight_snr,
        "quant_act_init_ssim_delta": pre_act_ssim - post_weight_ssim,
        "quant_post_recon_snr_db": pre_act_snr,
        "quant_post_recon_ssim": pre_act_ssim,
        "quant_post_recon_inference_seconds": float(pre_act_seconds),
    }
    metrics["before_snr_db"] = metrics["input_snr_db"]
    metrics["before_ssim"] = metrics["input_ssim"]
    metrics["after_snr_db"] = metrics["quant_post_recon_snr_db"]
    metrics["after_ssim"] = metrics["quant_post_recon_ssim"]
    metrics["inference_seconds"] = metrics["quant_post_recon_inference_seconds"]
    return metrics


def loaded_source_from_checkpoint(checkpoint: Mapping[str, Any], *, fallback_checkpoint: Path) -> SimpleNamespace:
    """Build the minimal object expected by `save_quant_checkpoint`."""
    source_checkpoint = checkpoint.get("source_checkpoint") or str(fallback_checkpoint)
    return SimpleNamespace(
        checkpoint_path=Path(source_checkpoint),
        epoch=checkpoint.get("source_checkpoint_epoch"),
        loss=checkpoint.get("source_checkpoint_loss"),
        config=scrn_config_from_mapping(checkpoint["model_config"]),
    )


def build_run_config(
    config: Mapping[str, Any],
    weight_recon_checkpoint: Path,
    checkpoint: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Build reproducibility config for activation-only runs."""
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "environment": collect_environment(),
        "device": str(device),
        "weight_recon_checkpoint": str(weight_recon_checkpoint),
        "source_checkpoint": checkpoint.get("source_checkpoint"),
        "source_checkpoint_epoch": checkpoint.get("source_checkpoint_epoch"),
        "source_checkpoint_loss": checkpoint.get("source_checkpoint_loss"),
        "weight_recon_checkpoint_stage": checkpoint.get("checkpoint_stage"),
        "weight_recon_final_quant_state": checkpoint.get("final_quant_state", {}),
        "model_config": checkpoint.get("model_config", {}),
        "checkpoint_quant_config": checkpoint.get("quant_config", {}),
        "activation_only_config": serializable_config(dict(config)),
    }


if __name__ == "__main__":
    main()
