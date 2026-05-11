"""Generate fixed-grid SCRN quantization comparison visuals for NE003."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn import require_file
from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_grid import (
    DegradationCondition,
    aggregate_rows,
    load_degraded_batch,
    load_eval_model,
    load_manifest_source_map,
    require_directory,
)
from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_multi import select_eval_device
from SCRN_BRECQ_app.scrn_repro.training import collect_environment, create_run_dir, write_json
from SCRN_BRECQ_app.scrn_repro.utils import snr_db, ssim_score


os.environ.setdefault(
    "MPLCONFIGDIR",
    "SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/matplotlib_cache",
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_EVAL_DATASET_DIR = "SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478"
DEFAULT_RUN_ROOT = "SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE003_fixed_grid_visual_protocol"
DEFAULT_RUN_NAME = "ne003_w4a4_w4a8_w4a32_seismic_denorm_seed20260507"
DEFAULT_TESTSET_ID = "paper5_energy_filtered_perpatch_absmax_478"
DEFAULT_SEED = 20260507
DEFAULT_CONTINUITY_SAMPLES_JSON = (
    "SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/"
    "E012_normalized_w4a32_seismic_denormalized_visuals/"
    "20260509_211000_seismic_denormalized_representative_plus_default_single/selected_samples.json"
)
DEFAULT_CLEAN = "SCRN-main/test_data/clear.npy"
DEFAULT_INPUT = "SCRN-main/test_data/noise_and_miss.npy"


def build_parser() -> argparse.ArgumentParser:
    """Build the NE003 visual protocol parser."""
    parser = argparse.ArgumentParser(description="Generate fixed-grid SCRN quantization comparison visuals.")
    parser.add_argument("--eval-dataset-dir", default=DEFAULT_EVAL_DATASET_DIR)
    parser.add_argument("--testset-id", default=DEFAULT_TESTSET_ID)
    parser.add_argument("--w4a32-checkpoint", required=True)
    parser.add_argument("--w4a32-pre-checkpoint", required=True)
    parser.add_argument("--w4a8-checkpoint", required=True)
    parser.add_argument("--w4a8-pre-checkpoint", required=True)
    parser.add_argument("--w4a4-checkpoint", required=True)
    parser.add_argument("--w4a4-pre-checkpoint", required=True)
    parser.add_argument("--w4a32-metrics", required=True)
    parser.add_argument("--w4a8-metrics", required=True)
    parser.add_argument("--w4a4-metrics", required=True)
    parser.add_argument("--continuity-samples-json", default=DEFAULT_CONTINUITY_SAMPLES_JSON)
    parser.add_argument("--default-clean", default=DEFAULT_CLEAN)
    parser.add_argument("--default-input", default=DEFAULT_INPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--cuda-device-index", type=int, default=1)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    return parser


def main() -> None:
    """Run NE003 visual generation."""
    args = build_parser().parse_args()
    validate_args(args)
    started = time.time()

    eval_dataset_dir = require_directory(args.eval_dataset_dir, "eval dataset directory")
    checkpoint_paths = {
        "w4a32_pre": require_file(args.w4a32_pre_checkpoint, "W4A32 pre checkpoint"),
        "w4a32_final": require_file(args.w4a32_checkpoint, "W4A32 final checkpoint"),
        "w4a8_pre": require_file(args.w4a8_pre_checkpoint, "W4A8 pre checkpoint"),
        "w4a8_final": require_file(args.w4a8_checkpoint, "W4A8 final checkpoint"),
        "w4a4_pre": require_file(args.w4a4_pre_checkpoint, "W4A4 pre checkpoint"),
        "w4a4_final": require_file(args.w4a4_checkpoint, "W4A4 final checkpoint"),
    }
    metric_paths = {
        "w4a32": require_file(args.w4a32_metrics, "W4A32 per-sample metrics"),
        "w4a8": require_file(args.w4a8_metrics, "W4A8 per-sample metrics"),
        "w4a4": require_file(args.w4a4_metrics, "W4A4 per-sample metrics"),
    }
    continuity_path = require_file(args.continuity_samples_json, "continuity selected samples")

    device = select_eval_device(args.device, args.cuda_device_index)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    run_dir = create_run_dir(args.run_root, run_name=args.run_name)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows_by_quant = {name: read_jsonl_rows(path) for name, path in metric_paths.items()}
    continuity_rows = load_continuity_rows(continuity_path)
    selected_rows = select_representative_rows(rows_by_quant["w4a4"], continuity_rows)
    metric_maps = {name: index_rows_by_key(rows) for name, rows in rows_by_quant.items()}
    source_map, source_warnings = load_manifest_source_map(eval_dataset_dir)
    scale_map, scale_warnings = load_manifest_scale_map(eval_dataset_dir)

    bundles = {name: load_eval_model(path, device) for name, path in checkpoint_paths.items()}
    for bundle in bundles.values():
        bundle["model"].eval()

    figure_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, selection in enumerate(selected_rows, start=1):
            figure_rows.append(
                generate_normalized_comparison(
                    index=index,
                    selection=selection,
                    metric_maps=metric_maps,
                    bundles=bundles,
                    eval_dataset_dir=eval_dataset_dir,
                    testset_id=str(args.testset_id),
                    source_map=source_map,
                    scale_map=scale_map,
                    seed=int(args.seed),
                    device=device,
                    figures_dir=figures_dir,
                )
            )
        default_row = generate_default_single_comparison(
            bundles=bundles,
            default_clean=Path(args.default_clean),
            default_input=Path(args.default_input),
            figures_dir=figures_dir,
            device=device,
        )
        if default_row is not None:
            figure_rows.append(default_row)

    metrics_summary = build_metrics_summary(metric_paths)
    write_json(run_dir / "selected_samples.json", figure_rows)
    write_json(run_dir / "metrics_summary.json", metrics_summary)
    write_json(
        run_dir / "config.json",
        {
            **build_run_config(args=args, run_dir=run_dir, device=str(device), figure_count=len(figure_rows)),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": float(time.time() - started),
            "manifest_warnings": source_warnings + scale_warnings,
            "representative_selection": {
                "normalized_count": len(selected_rows),
                "default_single_included": default_row is not None,
                "deduplication_key": ["patch_file", "condition_index"],
            },
            "environment": collect_environment(),
        },
    )
    write_summary(run_dir / "summary.md", run_dir=run_dir, rows=figure_rows, metrics_summary=metrics_summary)

    print(
        f"run_dir={run_dir} figure_count={len(figure_rows)} "
        f"normalized_representatives={len(selected_rows)} elapsed={time.time() - started:.2f}s",
        flush=True,
    )


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    if int(args.seed) < 0:
        raise ValueError("--seed must be non-negative")


def read_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL metric rows."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def load_continuity_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load E012 continuity rows and discard the raw default single sample."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for row in payload:
        if int(row.get("condition_index", -1)) < 0:
            continue
        rows.append(dict(row))
    return rows


def row_key(row: Mapping[str, Any]) -> tuple[str, int]:
    """Return the stable patch-condition key used by full-grid rows."""
    return str(row["patch_file"]), int(row["condition_index"])


def index_rows_by_key(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    """Index full-grid metrics rows by patch file and condition index."""
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = row_key(row)
        if key in indexed:
            raise ValueError(f"Duplicate metric row for {key}")
        indexed[key] = dict(row)
    return indexed


def select_representative_rows(
    w4a4_rows: Sequence[Mapping[str, Any]],
    continuity_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select deterministic NE003 representative rows with duplicate-label merging."""
    if not w4a4_rows:
        raise ValueError("w4a4_rows must not be empty")
    selected: dict[tuple[str, int], dict[str, Any]] = {}

    w4a4_by_key = index_rows_by_key(w4a4_rows)
    for row in continuity_rows:
        key = row_key(row)
        source = w4a4_by_key.get(key, dict(row))
        label = f"continuity_{row.get('condition_label', 'e012')}"
        continuity_source = dict(source)
        continuity_source.setdefault("condition_label", row.get("condition_label", condition_label(row)))
        merge_selection(selected, continuity_source, label)

    rows = [dict(row) for row in w4a4_rows]
    merge_selection(
        selected,
        min(rows, key=lambda row: (float(row["quant_post_recon_snr_db"]), row_key(row))),
        "w4a4_worst_final_snr",
    )
    merge_selection(
        selected,
        max(rows, key=lambda row: (float(row["quant_post_recon_snr_db"]), row_key(row))),
        "w4a4_best_final_snr",
    )
    post_values = np.asarray([float(row["quant_post_recon_snr_db"]) for row in rows], dtype=np.float64)
    median_value = float(np.median(post_values))
    merge_selection(
        selected,
        min(rows, key=lambda row: (abs(float(row["quant_post_recon_snr_db"]) - median_value), row_key(row))),
        "w4a4_median_final_snr",
    )
    merge_selection(
        selected,
        max(rows, key=lambda row: (float(row["quant_post_minus_pre_snr_db"]), row_key(row))),
        "w4a4_max_recon_snr_gain",
    )
    merge_selection(
        selected,
        min(rows, key=lambda row: (float(row["quant_post_minus_pre_snr_db"]), row_key(row))),
        "w4a4_worst_recon_snr_change",
    )
    merge_selection(
        selected,
        min(rows, key=lambda row: (float(row["quant_post_minus_pre_ssim"]), row_key(row))),
        "w4a4_max_ssim_drop",
    )
    return list(selected.values())


def merge_selection(
    selected: dict[tuple[str, int], dict[str, Any]],
    row: Mapping[str, Any],
    label: str,
) -> None:
    """Merge a representative row into the selected map without duplicate figures."""
    key = row_key(row)
    if key not in selected:
        selected[key] = dict(row)
        selected[key]["condition_label"] = selected[key].get("condition_label", condition_label(row))
        selected[key]["selection_labels"] = []
    labels = selected[key]["selection_labels"]
    if label not in labels:
        labels.append(label)


def condition_label(row: Mapping[str, Any]) -> str:
    """Return a compact condition label for titles and filenames."""
    return "snr{snr:g}_miss{missing:g}".format(
        snr=float(row["snr_setting_db"]),
        missing=float(row["missing_rate"]),
    )


def load_manifest_scale_map(dataset_dir: str | Path) -> tuple[dict[str, float], list[str]]:
    """Load per-patch normalization scales from manifest.json."""
    root = Path(dataset_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return {}, [f"manifest.json not found in {root}; normalization scales will be unavailable"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scale_map: dict[str, float] = {}
    warnings: list[str] = []
    for index, sample in enumerate(manifest.get("samples", [])):
        output_file = sample.get("output_file")
        normalization_scale = sample.get("normalization_scale")
        if not output_file or normalization_scale is None:
            warnings.append(f"manifest sample {index} lacks output_file or normalization_scale")
            continue
        scale_map[str(output_file)] = float(normalization_scale)
    if not scale_map:
        warnings.append(f"manifest.json in {root} did not provide any normalization scales")
    return scale_map, warnings


def restore_amplitude(image: np.ndarray, normalization_scale: float) -> np.ndarray:
    """Restore a normalized per-patch absmax image for display."""
    return np.asarray(image, dtype=np.float32) * np.float32(normalization_scale)


def symmetric_abs_limit(images: Sequence[np.ndarray]) -> float:
    """Return a nonzero symmetric display limit centered at zero."""
    absmax = 0.0
    for image in images:
        if np.asarray(image).size == 0:
            continue
        absmax = max(absmax, float(np.nanmax(np.abs(np.asarray(image, dtype=np.float32)))))
    return absmax if absmax > 0.0 else 1.0


def generate_normalized_comparison(
    *,
    index: int,
    selection: Mapping[str, Any],
    metric_maps: Mapping[str, Mapping[tuple[str, int], Mapping[str, Any]]],
    bundles: Mapping[str, Mapping[str, Any]],
    eval_dataset_dir: Path,
    testset_id: str,
    source_map: Mapping[str, str],
    scale_map: Mapping[str, float],
    seed: int,
    device: torch.device,
    figures_dir: Path,
) -> dict[str, Any]:
    """Generate one normalized representative comparison figure."""
    key = row_key(selection)
    condition = DegradationCondition(
        condition_index=int(selection["condition_index"]),
        snr_setting_db=float(selection["snr_setting_db"]),
        missing_rate=float(selection["missing_rate"]),
    )
    patch_file = str(selection["patch_file"])
    clean_batch, degraded_batch, _meta = load_degraded_batch(
        [eval_dataset_dir / patch_file],
        testset_id=testset_id,
        condition=condition,
        seed=int(seed),
        offset=int(selection["patch_index"]),
        source_map=source_map,
    )
    clean = clean_batch[0]
    degraded = degraded_batch[0]
    input_tensor = torch.from_numpy(degraded_batch[:, None, :, :]).float().to(device)

    predictions = predict_all(bundles, input_tensor)
    metrics = build_prediction_metrics(clean=clean, degraded=degraded, predictions=predictions)

    if patch_file not in scale_map:
        raise KeyError(f"Missing normalization_scale for {patch_file}")
    scale = float(scale_map[patch_file])
    display = {"clean": restore_amplitude(clean, scale), "input": restore_amplitude(degraded, scale)}
    display.update({name: restore_amplitude(value, scale) for name, value in predictions.items()})

    labels = list(selection.get("selection_labels", []))
    figure_name = "{index:02d}_{source}_{labels}_{patch}_c{condition:02d}_seismic_denorm.png".format(
        index=index,
        source=slug(str(selection.get("source", "unknown"))),
        labels=slug("__".join(labels) if labels else "selected"),
        patch=Path(patch_file).stem,
        condition=int(selection["condition_index"]),
    )
    figure_path = figures_dir / figure_name
    save_three_row_figure(
        figure_path,
        display=display,
        metrics=metrics,
        labels=labels,
        subtitle=(
            f"{selection.get('source', 'unknown')} | {patch_file} | condition={selection['condition_index']} "
            f"SNR={float(selection['snr_setting_db']):g} missing={float(selection['missing_rate']):g} "
            f"scale={scale:.6g}"
        ),
    )
    official = {
        "w4a32": dict(metric_maps["w4a32"][key]),
        "w4a8": dict(metric_maps["w4a8"][key]),
        "w4a4": dict(metric_maps["w4a4"][key]),
    }
    return {
        "selection_labels": labels,
        "testset_id": testset_id,
        "source": str(selection.get("source", "unknown")),
        "patch_file": patch_file,
        "patch_index": int(selection["patch_index"]),
        "condition_index": int(selection["condition_index"]),
        "condition_label": str(selection.get("condition_label", condition_label(selection))),
        "snr_setting_db": float(selection["snr_setting_db"]),
        "missing_rate": float(selection["missing_rate"]),
        "normalization_scale": scale,
        "amplitude_space": "restored from normalized per-patch absmax",
        "metric_amplitude_space": "normalized per-patch absmax",
        "figure": str(figure_path),
        "metrics": metrics,
        "official_full_grid_rows": official,
    }


def generate_default_single_comparison(
    *,
    bundles: Mapping[str, Mapping[str, Any]],
    default_clean: Path,
    default_input: Path,
    figures_dir: Path,
    device: torch.device,
) -> dict[str, Any] | None:
    """Generate the historical default single-sample sanity figure."""
    if not default_clean.is_file() or not default_input.is_file():
        return None
    clean = np.load(default_clean).astype(np.float32, copy=False)
    degraded = np.load(default_input).astype(np.float32, copy=False)
    input_tensor = torch.from_numpy(degraded[None, None, :, :]).float().to(device)
    predictions = predict_all(bundles, input_tensor)
    metrics = build_prediction_metrics(clean=clean, degraded=degraded, predictions=predictions)
    display = {"clean": clean, "input": degraded, **predictions}
    figure_path = figures_dir / "default_single_sample_seismic_raw_amplitude.png"
    save_three_row_figure(
        figure_path,
        display=display,
        metrics=metrics,
        labels=["default_single_sample_history_sanity"],
        subtitle="Default SCRN-main/test_data sample | raw amplitude | sanity only",
    )
    return {
        "selection_labels": ["default_single_sample_history_sanity"],
        "testset_id": "default_single_sample",
        "source": "SCRN-main/test_data",
        "patch_file": default_clean.name,
        "patch_index": -1,
        "condition_index": -1,
        "condition_label": "default_single_sample",
        "snr_setting_db": None,
        "missing_rate": None,
        "normalization_scale": None,
        "amplitude_space": "raw default SCRN sample",
        "metric_amplitude_space": "raw default SCRN sample",
        "figure": str(figure_path),
        "metrics": metrics,
    }


def predict_all(
    bundles: Mapping[str, Mapping[str, Any]],
    input_tensor: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Run FP32, W4A32, W4A8, and W4A4 pre/final predictions for one input."""
    predictions: dict[str, np.ndarray] = {}
    final_w4a32 = bundles["w4a32_final"]
    final_w4a32["model"].set_quant_state(False, False)
    predictions["fp32"] = tensor_prediction(final_w4a32["model"], input_tensor)
    for name in ["w4a32_pre", "w4a32_final", "w4a8_pre", "w4a8_final", "w4a4_pre", "w4a4_final"]:
        bundle = bundles[name]
        bundle["model"].set_quant_state(bool(bundle["weight_quant"]), bool(bundle["act_quant"]))
        predictions[name] = tensor_prediction(bundle["model"], input_tensor)
    return predictions


def tensor_prediction(model: torch.nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    """Run one model and return a 2D NumPy prediction."""
    output = model(input_tensor).squeeze(1).detach().cpu().numpy().astype(np.float32)
    return output[0]


def build_prediction_metrics(
    *,
    clean: np.ndarray,
    degraded: np.ndarray,
    predictions: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Compute SNR and SSIM for all panels."""
    metrics = {
        "input": {"snr_db": snr_db(degraded, clean), "ssim": ssim_score(degraded, clean)},
    }
    for name, prediction in predictions.items():
        metrics[name] = {"snr_db": snr_db(prediction, clean), "ssim": ssim_score(prediction, clean)}
    metrics["derived"] = {
        "w4a4_final_minus_w4a8_final_snr_db": metrics["w4a4_final"]["snr_db"] - metrics["w4a8_final"]["snr_db"],
        "w4a4_final_minus_w4a32_final_snr_db": metrics["w4a4_final"]["snr_db"] - metrics["w4a32_final"]["snr_db"],
        "w4a4_recon_gain_snr_db": metrics["w4a4_final"]["snr_db"] - metrics["w4a4_pre"]["snr_db"],
        "w4a8_recon_gain_snr_db": metrics["w4a8_final"]["snr_db"] - metrics["w4a8_pre"]["snr_db"],
        "w4a32_recon_gain_snr_db": metrics["w4a32_final"]["snr_db"] - metrics["w4a32_pre"]["snr_db"],
    }
    return metrics


def save_three_row_figure(
    path: Path,
    *,
    display: Mapping[str, np.ndarray],
    metrics: Mapping[str, Mapping[str, float]],
    labels: Sequence[str],
    subtitle: str,
) -> None:
    """Save the NE003 three-row seismic comparison figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data_keys = [
        "clean",
        "input",
        "fp32",
        "w4a32_final",
        "w4a8_final",
        "w4a4_final",
        "w4a32_pre",
        "w4a32_final",
        "w4a8_pre",
        "w4a8_final",
        "w4a4_pre",
        "w4a4_final",
    ]
    data_limit = symmetric_abs_limit([display[key] for key in data_keys])
    error_keys = ["input", "fp32", "w4a32_final", "w4a8_final", "w4a4_final"]
    errors = {key: np.asarray(display[key], dtype=np.float32) - np.asarray(display["clean"], dtype=np.float32) for key in error_keys}
    error_limit = symmetric_abs_limit(list(errors.values()))

    fig, axes = plt.subplots(3, 6, figsize=(23, 10.5), constrained_layout=True)
    title_labels = ", ".join(labels)
    fig.suptitle(f"{subtitle}\n{title_labels}", fontsize=10)

    row1 = [
        ("clean", "Clean"),
        ("input", metric_title("Input", metrics["input"])),
        ("fp32", metric_title("FP32", metrics["fp32"])),
        ("w4a32_final", metric_title("W4A32 final", metrics["w4a32_final"])),
        ("w4a8_final", metric_title("W4A8 final", metrics["w4a8_final"])),
        ("w4a4_final", metric_title("W4A4 final", metrics["w4a4_final"])),
    ]
    row2 = [
        ("w4a32_pre", metric_title("W4A32 pre", metrics["w4a32_pre"])),
        ("w4a32_final", metric_title("W4A32 final", metrics["w4a32_final"])),
        ("w4a8_pre", metric_title("W4A8 pre", metrics["w4a8_pre"])),
        ("w4a8_final", metric_title("W4A8 final", metrics["w4a8_final"])),
        ("w4a4_pre", metric_title("W4A4 pre", metrics["w4a4_pre"])),
        ("w4a4_final", metric_title("W4A4 final", metrics["w4a4_final"])),
    ]
    row3 = [
        ("input", "Input error"),
        ("fp32", "FP32 error"),
        ("w4a32_final", "W4A32 final error"),
        ("w4a8_final", "W4A8 final error"),
        ("w4a4_final", "W4A4 final error"),
    ]

    data_image = None
    for col, (key, title) in enumerate(row1):
        data_image = axes[0, col].imshow(display[key], cmap="seismic", aspect="auto", vmin=-data_limit, vmax=data_limit)
        axes[0, col].set_title(title, fontsize=8)
        axes[0, col].axis("off")
    for col, (key, title) in enumerate(row2):
        data_image = axes[1, col].imshow(display[key], cmap="seismic", aspect="auto", vmin=-data_limit, vmax=data_limit)
        axes[1, col].set_title(title, fontsize=8)
        axes[1, col].axis("off")

    error_image = None
    for col, (key, title) in enumerate(row3):
        error_image = axes[2, col].imshow(errors[key], cmap="seismic", aspect="auto", vmin=-error_limit, vmax=error_limit)
        axes[2, col].set_title(title, fontsize=8)
        axes[2, col].axis("off")
    axes[2, 5].axis("off")

    if data_image is not None:
        fig.colorbar(data_image, ax=axes[:2, :].ravel().tolist(), shrink=0.62, fraction=0.018, pad=0.01)
    if error_image is not None:
        fig.colorbar(error_image, ax=axes[2, :5].ravel().tolist(), shrink=0.72, fraction=0.018, pad=0.01)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def metric_title(label: str, metric: Mapping[str, float]) -> str:
    """Return a compact panel metric title."""
    return f"{label}\nSNR {metric['snr_db']:.2f} dB | SSIM {metric['ssim']:.4f}"


def build_metrics_summary(metric_paths: Mapping[str, Path]) -> dict[str, Any]:
    """Build a compact full-grid summary from existing metrics files."""
    summaries: dict[str, Any] = {"metric_sources": {name: str(path) for name, path in metric_paths.items()}}
    loaded: dict[str, dict[str, Any]] = {}
    for name, path in metric_paths.items():
        metrics_json = Path(path).with_name("metrics.json")
        if metrics_json.is_file():
            loaded[name] = json.loads(metrics_json.read_text(encoding="utf-8"))
        else:
            loaded[name] = {"sample_count": len(read_jsonl_rows(path)), "groups": aggregate_rows(read_jsonl_rows(path))}

    overall_w4a32 = loaded["w4a32"]["groups"]["overall"][0]
    overall_w4a8 = loaded["w4a8"]["groups"]["overall"][0]
    overall_w4a4 = loaded["w4a4"]["groups"]["overall"][0]
    summaries["objects"] = {
        "fp32": metric_object(overall_w4a32, prefix="fp32"),
        "e007_w4a32_pre": metric_object(overall_w4a32, prefix="quant_pre_recon"),
        "e007_w4a32_final": metric_object(overall_w4a32, prefix="quant_post_recon"),
        "ne000_w4a8_pre": metric_object(overall_w4a8, prefix="quant_pre_recon"),
        "ne000_w4a8_final": metric_object(overall_w4a8, prefix="quant_post_recon"),
        "ne000_2_w4a4_pre": metric_object(overall_w4a4, prefix="quant_pre_recon"),
        "ne000_2_w4a4_final": metric_object(overall_w4a4, prefix="quant_post_recon"),
    }
    summaries["reconstruction_changes"] = {
        "e007_w4a32": delta_object(overall_w4a32),
        "ne000_w4a8": delta_object(overall_w4a8),
        "ne000_2_w4a4": delta_object(overall_w4a4),
    }
    summaries["by_source"] = {
        name: metrics["groups"].get("by_source", []) for name, metrics in loaded.items()
    }
    summaries["sample_counts"] = {name: int(metrics["sample_count"]) for name, metrics in loaded.items()}
    return summaries


def metric_object(row: Mapping[str, Any], *, prefix: str) -> dict[str, float]:
    """Extract mean/median SNR and SSIM fields for one object."""
    return {
        "snr_mean": float(row[f"{prefix}_snr_db_mean"]),
        "snr_median": float(row[f"{prefix}_snr_db_median"]),
        "ssim_mean": float(row[f"{prefix}_ssim_mean"]),
        "ssim_median": float(row[f"{prefix}_ssim_median"]),
    }


def delta_object(row: Mapping[str, Any]) -> dict[str, float]:
    """Extract reconstruction delta statistics."""
    return {
        "snr_mean": float(row["quant_post_minus_pre_snr_db_mean"]),
        "snr_median": float(row["quant_post_minus_pre_snr_db_median"]),
        "ssim_mean": float(row["quant_post_minus_pre_ssim_mean"]),
        "ssim_median": float(row["quant_post_minus_pre_ssim_median"]),
    }


def build_run_config(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    device: str,
    figure_count: int,
) -> dict[str, Any]:
    """Build a NE003 run config snapshot."""
    return {
        "args": vars(args),
        "run_dir": str(run_dir),
        "eval_dataset_dir": str(args.eval_dataset_dir),
        "seed": int(args.seed),
        "device": str(device),
        "checkpoint_paths": {
            "w4a32_pre": str(args.w4a32_pre_checkpoint),
            "w4a32_final": str(args.w4a32_checkpoint),
            "w4a8_pre": str(args.w4a8_pre_checkpoint),
            "w4a8_final": str(args.w4a8_checkpoint),
            "w4a4_pre": str(args.w4a4_pre_checkpoint),
            "w4a4_final": str(args.w4a4_checkpoint),
        },
        "metric_sources": {
            "w4a32": str(args.w4a32_metrics),
            "w4a8": str(args.w4a8_metrics),
            "w4a4": str(args.w4a4_metrics),
        },
        "continuity_samples_json": str(args.continuity_samples_json),
        "default_single_sample": {
            "clean": str(getattr(args, "default_clean", DEFAULT_CLEAN)),
            "input": str(getattr(args, "default_input", DEFAULT_INPUT)),
            "role": "historical sanity only",
        },
        "display_protocol": {
            "colormap": "seismic",
            "normalized_display_formula": "display = normalized * normalization_scale",
            "prediction_scale": "symmetric per figure over clean/input/prediction panels, centered at zero",
            "error_scale": "symmetric per figure over error panels, centered at zero",
            "metric_space": "normalized per-patch absmax for full-grid rows",
        },
        "figure_count": int(figure_count),
    }


def write_summary(
    path: Path,
    *,
    run_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    metrics_summary: Mapping[str, Any],
) -> None:
    """Write a human-readable NE003 summary."""
    lines = [
        "# NE003 Fixed Grid Visual Protocol",
        "",
        "## Run",
        "",
        f"- run_dir: `{run_dir}`",
        f"- figure_count: `{len(rows)}`",
        "- colormap: `seismic`",
        "- normalized display: `display = normalized * normalization_scale`",
        "- default single sample is raw-amplitude historical sanity only.",
        "",
        "## Full-Grid Objects",
        "",
        "| object | SNR mean | SNR median | SSIM mean | SSIM median |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in metrics_summary["objects"].items():
        lines.append(
            "| {name} | {snr_mean:.6f} | {snr_median:.6f} | {ssim_mean:.6f} | {ssim_median:.6f} |".format(
                name=name,
                **metrics,
            )
        )
    lines.extend(
        [
            "",
            "## Reconstruction Changes",
            "",
            "| object | SNR mean delta | SNR median delta | SSIM mean delta | SSIM median delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, metrics in metrics_summary["reconstruction_changes"].items():
        lines.append(
            "| {name} | {snr_mean:.6f} | {snr_median:.6f} | {ssim_mean:.6f} | {ssim_median:.6f} |".format(
                name=name,
                **metrics,
            )
        )
    lines.extend(
        [
            "",
            "## Representatives",
            "",
            "| # | labels | source | patch | condition | scale | W4A4 final SNR | W4A4 gain | figure |",
            "|---:|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for index, row in enumerate(rows, start=1):
        metrics = row["metrics"]
        scale = row.get("normalization_scale")
        scale_text = "n/a" if scale is None else f"{float(scale):.6g}"
        lines.append(
            "| {index} | {labels} | {source} | {patch} | {condition} | {scale} | {snr:.3f} | {gain:.3f} | {figure} |".format(
                index=index,
                labels=", ".join(row.get("selection_labels", [])),
                source=row["source"],
                patch=row["patch_file"],
                condition=row["condition_index"],
                scale=scale_text,
                snr=metrics["w4a4_final"]["snr_db"],
                gain=metrics["derived"]["w4a4_recon_gain_snr_db"],
                figure=Path(row["figure"]).name,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def slug(text: str) -> str:
    """Return a compact filesystem-safe slug."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return cleaned.strip("_") or "item"


if __name__ == "__main__":
    main()
