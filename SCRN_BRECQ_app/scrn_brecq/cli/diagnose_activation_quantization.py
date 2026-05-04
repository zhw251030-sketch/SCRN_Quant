"""Diagnose SCRN-BRECQ activation quantization state and distributions."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn import (
    build_quant_model_from_checkpoint,
    load_quant_checkpoint,
    restore_quantizer_state_shapes,
    select_device,
)
from SCRN_BRECQ_app.scrn_brecq.data import CalibrationDataConfig, load_calibration_data
from SCRN_BRECQ_app.scrn_brecq.quant.activation_diagnostics import build_activation_diagnostics_report
from SCRN_BRECQ_app.scrn_brecq.utils import load_json, require_file
from SCRN_BRECQ_app.scrn_repro.training import collect_environment, create_run_dir, write_json, write_summary


DEFAULT_CONFIG_PATH = Path("SCRN_BRECQ_app/scrn_brecq/configs/activation_quantization/e001_diagnostics.json")
DEFAULT_RUN_ROOT = Path("SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E001_diagnostics")
DEFAULT_CHECKPOINT = Path(
    "SCRN_BRECQ_app/scrn_brecq/runs/quant/"
    "20260429_194908_w4a8_1024samples_w20000_a5000/checkpoints/quantized_scrn_brecq.pth"
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for E001 activation quantization diagnostics."""
    parser = argparse.ArgumentParser(description="Diagnose SCRN-BRECQ activation quantization.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="E001 diagnostics config JSON")
    parser.add_argument("--checkpoint", default=None, help="Quantized checkpoint to inspect")
    parser.add_argument("--run-root", default=None, help="Diagnostics output root")
    parser.add_argument("--run-name", default=None, help="Diagnostics run name")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--calibration-dataset-dir", default=None, help="Calibration clean patch directory")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of calibration samples to inspect")
    parser.add_argument("--batch-size", type=int, default=None, help="Calibration DataLoader batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Calibration DataLoader workers")
    parser.add_argument("--seed", type=int, default=None, help="Calibration dataset seed")
    return parser


def main() -> None:
    """Run E001 activation quantization diagnostics."""
    args = build_parser().parse_args()
    config = load_and_resolve_config(args)
    device = select_device(str(config["device"]))
    checkpoint_path = require_file(config["checkpoint"], "quantized checkpoint")
    checkpoint = load_quant_checkpoint(checkpoint_path)

    quant_model = build_quant_model_from_checkpoint(checkpoint)
    state_dict = checkpoint["quant_model_state_dict"]
    restore_quantizer_state_shapes(quant_model, state_dict)
    quant_model.load_state_dict(state_dict, strict=True)
    quant_model.to(device)
    quant_model.eval()

    quant_config = checkpoint.get("quant_config", {})
    final_state = checkpoint.get("final_quant_state", {})
    weight_quant = bool(final_state.get("weight_quant", True))
    act_quant = bool(final_state.get("act_quant", quant_config.get("act_quant", False)))
    if act_quant:
        quant_model.disable_network_output_quantization()

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
    report = build_activation_diagnostics_report(quant_model, calibration_data, weight_quant=weight_quant)
    run_dir = create_run_dir(config["run_root"], run_name=str(config["run_name"]))

    write_json(run_dir / "config.json", build_run_config(config, checkpoint_path, checkpoint, device))
    write_json(run_dir / "summary.json", report["summary"])
    write_json(run_dir / "offender_layers.json", {"offender_layers": report["offender_layers"]})
    write_csv(run_dir / "quantizers.csv", report["quantizers"])
    write_jsonl(run_dir / "activation_stats.jsonl", report["activation_stats"])
    write_summary(
        run_dir / "summary.md",
        title="SCRN-BRECQ Activation Quantization Diagnostics",
        sections={
            "Inputs": {
                "checkpoint": checkpoint_path,
                "calibration_dataset_dir": config["calibration_dataset_dir"],
                "num_samples": config["num_samples"],
                "device": str(device),
            },
            "Summary": report["summary"],
            "Artifacts": {
                "config": run_dir / "config.json",
                "summary_json": run_dir / "summary.json",
                "quantizers_csv": run_dir / "quantizers.csv",
                "activation_stats_jsonl": run_dir / "activation_stats.jsonl",
                "offender_layers_json": run_dir / "offender_layers.json",
            },
        },
    )

    print(
        "activation_quantizers={activation_quantizers} non_positive_delta_count={non_positive_delta_count} "
        "activation_stat_count={activation_stat_count} fake_quant_mse_max={fake_quant_mse_max}".format(
            **report["summary"]
        ),
        flush=True,
    )
    print(f"[SCRN-BRECQ] activation_diagnostics_run_dir={run_dir}", flush=True)


def load_and_resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    """Load JSON config and apply explicit CLI overrides."""
    config = dict(load_json(args.config))
    config.setdefault("checkpoint", str(DEFAULT_CHECKPOINT))
    config.setdefault("run_root", str(DEFAULT_RUN_ROOT))
    config.setdefault("run_name", "e001_diagnostics")
    config.setdefault("device", "cpu")
    config.setdefault("calibration_dataset_dir", "SCRN_BRECQ_app/scrn_repro/datasets/scrn_quant_10750_0_patches")
    config.setdefault("num_samples", 64)
    config.setdefault("batch_size", 16)
    config.setdefault("num_workers", 0)
    config.setdefault("seed", 1005)
    config["config_path"] = str(args.config)
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        config[key] = value
    return normalize_config(config)


def normalize_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate diagnostics config and normalize scalar types."""
    normalized = dict(config)
    normalized["checkpoint"] = str(require_file(normalized["checkpoint"], "quantized checkpoint"))
    normalized["run_root"] = str(normalized["run_root"])
    normalized["run_name"] = str(normalized["run_name"])
    normalized["device"] = str(normalized["device"])
    normalized["calibration_dataset_dir"] = str(Path(normalized["calibration_dataset_dir"]))
    for key in ("num_samples", "batch_size"):
        normalized[key] = int(normalized[key])
        if normalized[key] <= 0:
            raise ValueError(f"{key} must be positive, got {normalized[key]}")
    normalized["num_workers"] = int(normalized["num_workers"])
    if normalized["num_workers"] < 0:
        raise ValueError(f"num_workers must be non-negative, got {normalized['num_workers']}")
    normalized["seed"] = int(normalized["seed"])
    if not Path(normalized["calibration_dataset_dir"]).is_dir():
        raise FileNotFoundError(f"calibration_dataset_dir does not exist: {normalized['calibration_dataset_dir']}")
    return normalized


def build_run_config(
    config: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Build reproducibility config written into diagnostics run directory."""
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "environment": collect_environment(),
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "source_checkpoint": checkpoint.get("source_checkpoint"),
        "final_quant_state": checkpoint.get("final_quant_state", {}),
        "model_config": checkpoint.get("model_config", {}),
        "quant_config": checkpoint.get("quant_config", {}),
        "diagnostics_config": dict(config),
    }


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Write diagnostics rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write diagnostics rows to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


if __name__ == "__main__":
    main()
