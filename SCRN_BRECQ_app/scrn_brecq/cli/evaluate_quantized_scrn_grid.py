"""Evaluate a quantized SCRN-BRECQ checkpoint on a fixed degradation grid."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn import require_file
from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_multi import load_eval_model, select_eval_device
from SCRN_BRECQ_app.scrn_brecq.utils import build_model_size_report
from SCRN_BRECQ_app.scrn_repro.data import degrade_patch, discover_patch_files
from SCRN_BRECQ_app.scrn_repro.training import collect_environment, create_run_dir, write_json
from SCRN_BRECQ_app.scrn_repro.utils import snr_db, ssim_score


DEFAULT_EVAL_DATASET_DIR = "SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478"
DEFAULT_RUN_ROOT = "SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E007_normalized_w4a32_baseline/eval"
DEFAULT_RUN_NAME = "quantized_grid478_seed20260507"
DEFAULT_SEED = 20260507
DEFAULT_BATCH_SIZE = 64
DEFAULT_SNR_SETTINGS = (-2.0, -1.0, 1.0, 5.0, 10.0)
DEFAULT_MISSING_RATES = (0.02, 0.08, 0.18, 0.28, 0.38)

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
    "quant_pre_recon_snr_db",
    "quant_pre_recon_ssim",
    "quant_post_recon_snr_db",
    "quant_post_recon_ssim",
    "quant_pre_minus_fp32_snr_db",
    "quant_pre_minus_fp32_ssim",
    "quant_post_minus_fp32_snr_db",
    "quant_post_minus_fp32_ssim",
    "quant_post_minus_pre_snr_db",
    "quant_post_minus_pre_ssim",
    "inference_seconds",
}

STAT_FIELDS = (
    "input_snr_db",
    "input_ssim",
    "fp32_snr_db",
    "fp32_ssim",
    "quant_post_recon_snr_db",
    "quant_post_recon_ssim",
    "quant_post_minus_fp32_snr_db",
    "quant_post_minus_fp32_ssim",
)

OPTIONAL_PRE_STAT_FIELDS = (
    "quant_pre_recon_snr_db",
    "quant_pre_recon_ssim",
    "quant_pre_minus_fp32_snr_db",
    "quant_pre_minus_fp32_ssim",
    "quant_post_minus_pre_snr_db",
    "quant_post_minus_pre_ssim",
)


@dataclass(frozen=True)
class DegradationCondition:
    """One point in the fixed degradation grid."""

    condition_index: int
    snr_setting_db: float
    missing_rate: float


def build_parser() -> argparse.ArgumentParser:
    """Build the quantized grid evaluator parser."""
    parser = argparse.ArgumentParser(description="Evaluate a quantized SCRN-BRECQ checkpoint on a fixed grid.")
    parser.add_argument("--checkpoint", required=True, help="Post-reconstruction quantized checkpoint path.")
    parser.add_argument("--pre-recon-checkpoint", default=None, help="Optional pre-reconstruction checkpoint path.")
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
    parser.add_argument("--save-figures", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-figures", type=int, default=8)
    return parser


def main() -> None:
    """Run CLI evaluation."""
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
    snr_settings = parse_float_sequence(args.snr_settings, option_name="--snr-settings")
    missing_rates = parse_float_sequence(args.missing_rates, option_name="--missing-rates")
    conditions = build_degradation_conditions(snr_settings, missing_rates)
    device = select_eval_device(args.device, args.cuda_device_index)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    post_bundle = load_eval_model(checkpoint_path, device)
    pre_bundle = load_eval_model(pre_recon_checkpoint_path, device) if pre_recon_checkpoint_path is not None else None
    all_files = discover_patch_files(eval_dataset_dir)
    selected_files = select_eval_files(all_files, num_samples=args.num_eval_samples, seed=int(args.seed))
    source_map, manifest_warnings = load_manifest_source_map(eval_dataset_dir)

    run_dir = create_run_dir(args.run_root, run_name=args.run_name)
    figures_dir = run_dir / "figures" if args.save_figures and args.max_figures > 0 else None
    if figures_dir is not None:
        figures_dir.mkdir(parents=True, exist_ok=True)

    rows, timing = evaluate_files(
        post_bundle["model"],
        selected_files,
        testset_id=str(args.testset_id),
        source_map=source_map,
        conditions=conditions,
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        device=device,
        weight_quant=bool(post_bundle["weight_quant"]),
        act_quant=bool(post_bundle["act_quant"]),
        pre_recon_model=pre_bundle["model"] if pre_bundle is not None else None,
        pre_recon_weight_quant=bool(pre_bundle["weight_quant"]) if pre_bundle is not None else False,
        pre_recon_act_quant=bool(pre_bundle["act_quant"]) if pre_bundle is not None else False,
        figures_dir=figures_dir,
        max_figures=int(args.max_figures),
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
        "model_size": build_model_size_report(
            post_bundle["model"],
            source_checkpoint_path=post_bundle["checkpoint"].get("source_checkpoint"),
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
            pre_recon_checkpoint_path=pre_recon_checkpoint_path,
            eval_dataset_dir=eval_dataset_dir,
            selected_files=selected_files,
            conditions=conditions,
            device=device,
            post_bundle=post_bundle,
            pre_bundle=pre_bundle,
            manifest_warnings=manifest_warnings,
        ),
    )
    write_summary(run_dir / "summary.md", run_dir=run_dir, metrics=metrics, config_path=run_dir / "config.json")

    overall = metrics["groups"]["overall"][0]
    print(
        f"rows={metrics['sample_count']} patches={metrics['patch_count']} conditions={metrics['condition_count']} "
        f"fp32_snr_mean={overall['fp32_snr_db_mean']:.4f} "
        f"quant_post_snr_mean={overall['quant_post_recon_snr_db_mean']:.4f} "
        f"quant_post_minus_fp32_snr_mean={overall['quant_post_minus_fp32_snr_db_mean']:.4f} "
        f"run_dir={run_dir}",
        flush=True,
    )


def parse_float_sequence(text: str, *, option_name: str) -> tuple[float, ...]:
    """Parse a comma-separated float sequence."""
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if not parts:
        raise ValueError(f"{option_name} must contain at least one value")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"{option_name} must contain only float values: {text}") from exc


def build_degradation_conditions(
    snr_settings: Sequence[float],
    missing_rates: Sequence[float],
) -> list[DegradationCondition]:
    """Expand SNR and missing-rate settings into a stable condition list."""
    conditions: list[DegradationCondition] = []
    for snr in snr_settings:
        for missing_rate in missing_rates:
            conditions.append(
                DegradationCondition(
                    condition_index=len(conditions),
                    snr_setting_db=float(snr),
                    missing_rate=float(missing_rate),
                )
            )
    return conditions


def stable_degradation_seed(seed: int, testset_id: str, patch_index: int, condition_index: int) -> int:
    """Return the deterministic RNG seed used for one patch and condition."""
    payload = f"{seed}:{testset_id}:{patch_index}:{condition_index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


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
    quant_post_recon_snr_db: float,
    quant_post_recon_ssim: float,
    inference_seconds: float,
    quant_pre_recon_snr_db: float | None = None,
    quant_pre_recon_ssim: float | None = None,
) -> dict[str, Any]:
    """Build one JSONL metrics row."""
    row: dict[str, Any] = {
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
        "quant_post_recon_snr_db": float(quant_post_recon_snr_db),
        "quant_post_recon_ssim": float(quant_post_recon_ssim),
        "quant_post_minus_fp32_snr_db": float(quant_post_recon_snr_db - fp32_snr_db),
        "quant_post_minus_fp32_ssim": float(quant_post_recon_ssim - fp32_ssim),
        "inference_seconds": float(inference_seconds),
    }
    if quant_pre_recon_snr_db is not None and quant_pre_recon_ssim is not None:
        row.update(
            {
                "quant_pre_recon_snr_db": float(quant_pre_recon_snr_db),
                "quant_pre_recon_ssim": float(quant_pre_recon_ssim),
                "quant_pre_minus_fp32_snr_db": float(quant_pre_recon_snr_db - fp32_snr_db),
                "quant_pre_minus_fp32_ssim": float(quant_pre_recon_ssim - fp32_ssim),
                "quant_post_minus_pre_snr_db": float(quant_post_recon_snr_db - quant_pre_recon_snr_db),
                "quant_post_minus_pre_ssim": float(quant_post_recon_ssim - quant_pre_recon_ssim),
            }
        )
    return row


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Aggregate rows by overall, source, SNR, missing rate, and condition."""
    fields = list(STAT_FIELDS)
    if rows and all("quant_pre_recon_snr_db" in row for row in rows):
        fields.extend(OPTIONAL_PRE_STAT_FIELDS)
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
    model: torch.nn.Module,
    files: Sequence[Path],
    *,
    testset_id: str,
    source_map: Mapping[str, str],
    conditions: Sequence[DegradationCondition],
    seed: int,
    batch_size: int,
    device: torch.device,
    weight_quant: bool,
    act_quant: bool,
    pre_recon_model: torch.nn.Module | None = None,
    pre_recon_weight_quant: bool = False,
    pre_recon_act_quant: bool = False,
    figures_dir: Path | None = None,
    max_figures: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Evaluate selected clean patches across all degradation conditions."""
    rows: list[dict[str, Any]] = []
    fp32_seconds = 0.0
    pre_seconds = 0.0
    post_seconds = 0.0
    figure_count = 0

    model.eval()
    if pre_recon_model is not None:
        pre_recon_model.eval()

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

                model.set_quant_state(False, False)
                synchronize_if_needed(device)
                start = time.time()
                fp32_batch = model(input_tensor).squeeze(1).detach().cpu().numpy().astype(np.float32)
                synchronize_if_needed(device)
                fp32_seconds += time.time() - start

                if pre_recon_model is not None:
                    pre_recon_model.set_quant_state(pre_recon_weight_quant, pre_recon_act_quant)
                    synchronize_if_needed(device)
                    start = time.time()
                    pre_batch = pre_recon_model(input_tensor).squeeze(1).detach().cpu().numpy().astype(np.float32)
                    synchronize_if_needed(device)
                    pre_seconds += time.time() - start
                else:
                    pre_batch = None

                model.set_quant_state(weight_quant, act_quant)
                synchronize_if_needed(device)
                start = time.time()
                post_batch = model(input_tensor).squeeze(1).detach().cpu().numpy().astype(np.float32)
                synchronize_if_needed(device)
                elapsed = time.time() - start
                post_seconds += elapsed
                elapsed_per_sample = elapsed / max(len(batch_paths), 1)

                for local_index, item in enumerate(meta):
                    clean = clean_batch[local_index]
                    degraded = degraded_batch[local_index]
                    fp32_prediction = fp32_batch[local_index]
                    post_prediction = post_batch[local_index]
                    pre_prediction = pre_batch[local_index] if pre_batch is not None else None
                    pre_snr = snr_db(pre_prediction, clean) if pre_prediction is not None else None
                    pre_ssim = ssim_score(pre_prediction, clean) if pre_prediction is not None else None
                    row = build_metric_row(
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
                        quant_post_recon_snr_db=snr_db(post_prediction, clean),
                        quant_post_recon_ssim=ssim_score(post_prediction, clean),
                        inference_seconds=elapsed_per_sample,
                        quant_pre_recon_snr_db=pre_snr,
                        quant_pre_recon_ssim=pre_ssim,
                    )
                    rows.append(row)
                    if figures_dir is not None and figure_count < max_figures:
                        figure_count += 1
                        save_comparison_figure(
                            figures_dir / f"sample_{len(rows):04d}_{Path(item['patch_file']).stem}.png",
                            clean=clean,
                            degraded=degraded,
                            fp32_prediction=fp32_prediction,
                            quant_pre_recon_prediction=pre_prediction,
                            quant_post_recon_prediction=post_prediction,
                            row=row,
                        )

    return rows, {
        "fp32_inference_seconds": fp32_seconds,
        "quant_pre_recon_inference_seconds": pre_seconds,
        "quant_post_recon_inference_seconds": post_seconds,
    }


def load_degraded_batch(
    files: Sequence[Path],
    *,
    testset_id: str,
    condition: DegradationCondition,
    seed: int,
    offset: int,
    source_map: Mapping[str, str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Load clean patches and create degraded inputs for one condition."""
    clean_items: list[np.ndarray] = []
    degraded_items: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    expected_shape: tuple[int, int] | None = None
    for local_index, path in enumerate(files):
        patch_index = int(offset) + int(local_index)
        clean = np.load(path).astype(np.float32, copy=False)
        if clean.ndim != 2:
            raise ValueError(f"Expected 2D clean patch, got {clean.shape} for {path}")
        if expected_shape is None:
            expected_shape = clean.shape
        elif clean.shape != expected_shape:
            raise ValueError(f"All patches in a batch must share shape; got {clean.shape} and {expected_shape}")
        rng = np.random.default_rng(stable_degradation_seed(seed, testset_id, patch_index, condition.condition_index))
        degraded, _, _ = degrade_patch(
            clean,
            missing_rate=condition.missing_rate,
            snr_db=condition.snr_setting_db,
            rng=rng,
        )
        clean_items.append(clean)
        degraded_items.append(degraded)
        meta.append(
            {
                "patch_index": patch_index,
                "patch_file": path.name,
                "source": source_map.get(path.name, "unknown"),
            }
        )
    return np.stack(clean_items, axis=0), np.stack(degraded_items, axis=0), meta


def select_eval_files(files: Sequence[Path], *, num_samples: int | None, seed: int) -> list[Path]:
    """Select a deterministic subset, or all files when no subset size is requested."""
    sorted_files = sorted(files)
    if num_samples is None:
        return sorted_files
    if int(num_samples) <= 0:
        raise ValueError(f"--num-eval-samples must be positive, got {num_samples}")
    if int(num_samples) > len(sorted_files):
        raise ValueError(f"Requested {num_samples} samples, but only {len(sorted_files)} files are available.")
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(sorted_files), size=int(num_samples), replace=False)
    return [sorted_files[int(index)] for index in indices]


def load_manifest_source_map(dataset_dir: str | Path) -> tuple[dict[str, str], list[str]]:
    """Load patch source names from manifest.json when available."""
    root = Path(dataset_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return {}, [f"manifest.json not found in {root}; source will be unknown"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_map: dict[str, str] = {}
    warnings: list[str] = []
    for index, sample in enumerate(manifest.get("samples", [])):
        output_file = sample.get("output_file")
        source = sample.get("source")
        if not output_file or not source:
            warnings.append(f"manifest sample {index} lacks output_file or source")
            continue
        source_map[str(output_file)] = str(source)
    if not source_map:
        warnings.append(f"manifest.json in {root} did not provide any source mapping")
    return source_map, warnings


def require_directory(path: str | Path, description: str) -> Path:
    """Return a directory path or raise a clear error."""
    candidate = Path(path)
    if not candidate.is_dir():
        raise FileNotFoundError(f"{description} does not exist: {candidate}")
    return candidate


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_eval_samples is not None and int(args.num_eval_samples) <= 0:
        raise ValueError("--num-eval-samples must be positive")
    if int(args.max_figures) < 0:
        raise ValueError("--max-figures must be non-negative")


def build_run_config(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint_path: Path,
    pre_recon_checkpoint_path: Path | None,
    eval_dataset_dir: Path,
    selected_files: Sequence[Path],
    conditions: Sequence[DegradationCondition],
    device: torch.device,
    post_bundle: Mapping[str, Any],
    pre_bundle: Mapping[str, Any] | None,
    manifest_warnings: Sequence[str],
) -> dict[str, Any]:
    """Build the run config snapshot."""
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "pre_recon_checkpoint": str(pre_recon_checkpoint_path) if pre_recon_checkpoint_path is not None else None,
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
        "quant_config": dict(post_bundle["quant_config"]),
        "final_quant_state": {
            "weight_quant": bool(post_bundle["weight_quant"]),
            "act_quant": bool(post_bundle["act_quant"]),
        },
        "pre_recon_final_quant_state": None
        if pre_bundle is None
        else {
            "weight_quant": bool(pre_bundle["weight_quant"]),
            "act_quant": bool(pre_bundle["act_quant"]),
        },
        "source_checkpoint": post_bundle["checkpoint"].get("source_checkpoint"),
        "source_checkpoint_epoch": post_bundle["checkpoint"].get("source_checkpoint_epoch"),
        "source_checkpoint_loss": post_bundle["checkpoint"].get("source_checkpoint_loss"),
        "model_config": post_bundle["checkpoint"].get("model_config", {}),
    }


def write_jsonl_rows(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write per-sample metrics JSONL."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary(path: str | Path, *, run_dir: Path, metrics: Mapping[str, Any], config_path: Path) -> None:
    """Write a compact Markdown summary."""
    overall = metrics["groups"]["overall"][0]
    lines = [
        "# SCRN-BRECQ Quantized Grid Evaluation",
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
    for field in [
        "input_snr_db",
        "fp32_snr_db",
        "quant_pre_recon_snr_db",
        "quant_post_recon_snr_db",
        "quant_post_minus_fp32_snr_db",
        "quant_post_minus_pre_snr_db",
        "input_ssim",
        "fp32_ssim",
        "quant_pre_recon_ssim",
        "quant_post_recon_ssim",
        "quant_post_minus_fp32_ssim",
        "quant_post_minus_pre_ssim",
    ]:
        mean_key = f"{field}_mean"
        if mean_key not in overall:
            continue
        lines.append(
            "| {field} | {mean:.6f} | {median:.6f} | {std:.6f} | {min_value:.6f} | {max_value:.6f} |".format(
                field=field,
                mean=overall[mean_key],
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
            "| source | count | fp32_snr_mean | quant_post_snr_mean | quant_post_minus_fp32_snr_mean | quant_post_ssim_mean |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics["groups"]["by_source"]:
        lines.append(
            "| {source} | {sample_count} | {fp32:.6f} | {post:.6f} | {delta:.6f} | {ssim:.6f} |".format(
                source=row["source"],
                sample_count=row["sample_count"],
                fp32=row["fp32_snr_db_mean"],
                post=row["quant_post_recon_snr_db_mean"],
                delta=row["quant_post_minus_fp32_snr_db_mean"],
                ssim=row["quant_post_recon_ssim_mean"],
            )
        )
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def save_comparison_figure(
    path: Path,
    *,
    clean: np.ndarray,
    degraded: np.ndarray,
    fp32_prediction: np.ndarray,
    quant_pre_recon_prediction: np.ndarray | None,
    quant_post_recon_prediction: np.ndarray,
    row: Mapping[str, Any],
) -> None:
    """Save a small visual comparison for optional smoke/debug use."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Saving comparison figures requires installing matplotlib.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    panels: list[tuple[np.ndarray, str]] = [
        (clean, "Ground Truth"),
        (degraded, f"Input {row['input_snr_db']:.2f}dB"),
        (fp32_prediction, f"FP32 {row['fp32_snr_db']:.2f}dB"),
    ]
    if quant_pre_recon_prediction is not None:
        panels.append((quant_pre_recon_prediction, f"Pre {row['quant_pre_recon_snr_db']:.2f}dB"))
    panels.append((quant_post_recon_prediction, f"Post {row['quant_post_recon_snr_db']:.2f}dB"))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    vmin = float(np.min(clean))
    vmax = float(np.max(clean))
    for ax, (image, title) in zip(axes, panels):
        im = ax.imshow(image, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.72, fraction=0.02, pad=0.01)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def synchronize_if_needed(device: torch.device) -> None:
    """Synchronize CUDA timing when needed."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
