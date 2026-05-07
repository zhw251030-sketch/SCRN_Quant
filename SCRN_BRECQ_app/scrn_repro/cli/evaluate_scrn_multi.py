"""Evaluate FP32 SCRN checkpoints on clean patch test sets with a fixed degradation grid."""

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

from SCRN_BRECQ_app.scrn_repro.data import degrade_patch, discover_patch_files
from SCRN_BRECQ_app.scrn_repro.model import SCRN
from SCRN_BRECQ_app.scrn_repro.training import collect_environment, create_run_dir, load_checkpoint, write_json
from SCRN_BRECQ_app.scrn_repro.utils import snr_db, ssim_score


PRESET_FP32_TWO_MODEL_TWO_TESTSET_478 = "fp32-two-model-two-testset-478"
DEFAULT_RUN_ROOT = "SCRN_BRECQ_app/scrn_repro/runs/test_multi"
DEFAULT_RUN_NAME = "fp32_two_model_two_testset_grid478_seed20260507"
DEFAULT_SEED = 20260507
DEFAULT_BATCH_SIZE = 64
DEFAULT_SNR_SETTINGS = (-2.0, -1.0, 1.0, 5.0, 10.0)
DEFAULT_MISSING_RATES = (0.02, 0.08, 0.18, 0.28, 0.38)
DEFAULT_MODEL_SPECS = {
    "old10750_main": "SCRN_BRECQ_app/scrn_repro/runs/train/20260425_192916_four_gpu_train_quant_10750_0/checkpoints/best.pth",
    "paper5": "SCRN_BRECQ_app/scrn_repro/runs/train/20260507_170001_paper5_10750_ddp3_seed20260425/checkpoints/best.pth",
}
DEFAULT_TESTSET_SPECS = {
    "legacy478": "SCRN_BRECQ_app/scrn_repro/datasets/scrn_quant_test_478_legacy_logic",
    "paper5_478": "SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_test_478",
}
PER_SAMPLE_FIELDS = {
    "model_id",
    "testset_id",
    "source",
    "patch_file",
    "snr_setting_db",
    "missing_rate",
    "input_snr_db",
    "input_ssim",
    "output_snr_db",
    "output_ssim",
    "snr_gain_db",
    "ssim_gain",
    "inference_seconds",
}
_STAT_FIELDS = (
    "input_snr_db",
    "output_snr_db",
    "snr_gain_db",
    "input_ssim",
    "output_ssim",
    "ssim_gain",
)
_PAIRED_STAT_FIELDS = (
    "output_snr_db_delta",
    "output_ssim_delta",
    "snr_gain_db_delta",
    "ssim_gain_delta",
)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    checkpoint_path: Path


@dataclass(frozen=True)
class TestsetSpec:
    testset_id: str
    dataset_dir: Path


@dataclass(frozen=True)
class EvalJob:
    model_id: str
    checkpoint_path: Path
    testset_id: str
    dataset_dir: Path


@dataclass(frozen=True)
class DegradationCondition:
    condition_index: int
    snr_setting_db: float
    missing_rate: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate FP32 SCRN checkpoints on multiple clean patch test sets.")
    parser.add_argument(
        "--preset",
        default=PRESET_FP32_TWO_MODEL_TWO_TESTSET_478,
        choices=[PRESET_FP32_TWO_MODEL_TWO_TESTSET_478],
        help="Evaluation preset to run.",
    )
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT, help="Output run root.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Run name suffix.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--cuda-device-index", type=int, default=1, help="Physical CUDA device index to use.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--save-figures", action="store_true", help="Save a small number of comparison figures.")
    parser.add_argument("--max-figures", type=int, default=8, help="Maximum comparison figures to save.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_figures < 0:
        raise ValueError("--max-figures must be non-negative")

    jobs = build_preset_eval_matrix()
    testsets = build_preset_testsets()
    models = build_preset_models()
    conditions = build_degradation_conditions(DEFAULT_SNR_SETTINGS, DEFAULT_MISSING_RATES)
    _validate_preset_inputs(models, testsets)
    device = select_device(args.device, args.cuda_device_index)

    run_dir = create_run_dir(args.run_root, run_name=args.run_name)
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "preset": args.preset,
        "models": {item.model_id: str(item.checkpoint_path) for item in models},
        "testsets": {item.testset_id: str(item.dataset_dir) for item in testsets},
        "snr_settings": list(DEFAULT_SNR_SETTINGS),
        "missing_rates": list(DEFAULT_MISSING_RATES),
        "expected_rows": len(jobs) * 478 * len(conditions),
        "device": str(device),
        "environment": collect_environment(),
        "manifest_warnings": {},
    }
    write_json(run_dir / "config.json", config)

    all_rows: list[dict[str, Any]] = []
    input_metric_cache: dict[tuple[str, str, int], tuple[float, float]] = {}
    figure_state = {"saved": 0}
    for model in models:
        model_instance = load_scrn_model(model.checkpoint_path, device)
        for testset in testsets:
            source_map, warnings = load_manifest_source_map(testset.dataset_dir)
            if warnings:
                config["manifest_warnings"][testset.testset_id] = warnings
                write_json(run_dir / "config.json", config)
            rows = evaluate_model_on_testset(
                model_instance,
                model_id=model.model_id,
                testset_id=testset.testset_id,
                dataset_dir=testset.dataset_dir,
                source_map=source_map,
                conditions=conditions,
                seed=args.seed,
                batch_size=args.batch_size,
                device=device,
                input_metric_cache=input_metric_cache,
                figures_dir=run_dir / "figures" if args.save_figures else None,
                max_figures=args.max_figures,
                figure_state=figure_state,
            )
            all_rows.extend(rows)
            print(
                f"model={model.model_id} testset={testset.testset_id} rows={len(rows)} "
                f"total_rows={len(all_rows)}",
                flush=True,
            )
        del model_instance
        if device.type == "cuda":
            torch.cuda.empty_cache()

    per_sample_path = run_dir / "per_sample_metrics.jsonl"
    write_jsonl_rows(per_sample_path, all_rows)
    metrics = {
        "sample_count": len(all_rows),
        "condition_count": len(conditions),
        "groups": aggregate_rows(all_rows),
        "paired_comparison": build_paired_comparison(all_rows),
    }
    write_json(run_dir / "metrics.json", metrics)
    write_multi_summary(run_dir / "summary.md", run_dir=run_dir, metrics=metrics, config=config)
    print(
        f"rows={len(all_rows)} run_dir={run_dir} metrics={run_dir / 'metrics.json'} "
        f"per_sample={per_sample_path}",
        flush=True,
    )


def build_preset_models() -> list[ModelSpec]:
    return [ModelSpec(model_id=model_id, checkpoint_path=Path(path)) for model_id, path in DEFAULT_MODEL_SPECS.items()]


def build_preset_testsets() -> list[TestsetSpec]:
    return [TestsetSpec(testset_id=testset_id, dataset_dir=Path(path)) for testset_id, path in DEFAULT_TESTSET_SPECS.items()]


def build_preset_eval_matrix() -> list[EvalJob]:
    jobs = []
    for model in build_preset_models():
        for testset in build_preset_testsets():
            jobs.append(
                EvalJob(
                    model_id=model.model_id,
                    checkpoint_path=model.checkpoint_path,
                    testset_id=testset.testset_id,
                    dataset_dir=testset.dataset_dir,
                )
            )
    return jobs


def build_degradation_conditions(
    snr_settings: Sequence[float],
    missing_rates: Sequence[float],
) -> list[DegradationCondition]:
    conditions = []
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


def build_metric_row(
    *,
    model_id: str,
    testset_id: str,
    source: str,
    patch_file: str,
    patch_index: int,
    condition_index: int,
    snr_setting_db: float,
    missing_rate: float,
    input_snr_db: float,
    input_ssim: float,
    output_snr_db: float,
    output_ssim: float,
    inference_seconds: float,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "testset_id": testset_id,
        "source": source,
        "patch_file": patch_file,
        "patch_index": int(patch_index),
        "condition_index": int(condition_index),
        "snr_setting_db": float(snr_setting_db),
        "missing_rate": float(missing_rate),
        "input_snr_db": float(input_snr_db),
        "input_ssim": float(input_ssim),
        "output_snr_db": float(output_snr_db),
        "output_ssim": float(output_ssim),
        "snr_gain_db": float(output_snr_db - input_snr_db),
        "ssim_gain": float(output_ssim - input_ssim),
        "inference_seconds": float(inference_seconds),
    }


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "overall": _summarize_groups(rows, ("model_id", "testset_id"), _STAT_FIELDS),
        "by_source": _summarize_groups(rows, ("model_id", "testset_id", "source"), _STAT_FIELDS),
        "by_snr_setting": _summarize_groups(rows, ("model_id", "testset_id", "snr_setting_db"), _STAT_FIELDS),
        "by_missing_rate": _summarize_groups(rows, ("model_id", "testset_id", "missing_rate"), _STAT_FIELDS),
        "by_condition": _summarize_groups(
            rows,
            ("model_id", "testset_id", "snr_setting_db", "missing_rate"),
            _STAT_FIELDS,
        ),
    }


def build_paired_comparison(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_key: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = (row["testset_id"], row["patch_file"], row["snr_setting_db"], row["missing_rate"])
        by_key.setdefault(key, {})[str(row["model_id"])] = row

    paired_rows = []
    for model_rows in by_key.values():
        old = model_rows.get("old10750_main")
        paper = model_rows.get("paper5")
        if old is None or paper is None:
            continue
        paired_rows.append(
            {
                "testset_id": paper["testset_id"],
                "source": paper["source"],
                "patch_file": paper["patch_file"],
                "snr_setting_db": paper["snr_setting_db"],
                "missing_rate": paper["missing_rate"],
                "output_snr_db_delta": float(paper["output_snr_db"] - old["output_snr_db"]),
                "output_ssim_delta": float(paper["output_ssim"] - old["output_ssim"]),
                "snr_gain_db_delta": float(paper["snr_gain_db"] - old["snr_gain_db"]),
                "ssim_gain_delta": float(paper["ssim_gain"] - old["ssim_gain"]),
            }
        )

    return {
        "overall": _summarize_groups(paired_rows, ("testset_id",), _PAIRED_STAT_FIELDS),
        "by_source": _summarize_groups(paired_rows, ("testset_id", "source"), _PAIRED_STAT_FIELDS),
        "by_snr_setting": _summarize_groups(paired_rows, ("testset_id", "snr_setting_db"), _PAIRED_STAT_FIELDS),
        "by_missing_rate": _summarize_groups(paired_rows, ("testset_id", "missing_rate"), _PAIRED_STAT_FIELDS),
        "by_condition": _summarize_groups(
            paired_rows,
            ("testset_id", "snr_setting_db", "missing_rate"),
            _PAIRED_STAT_FIELDS,
        ),
    }


def load_manifest_source_map(dataset_dir: str | Path) -> tuple[dict[str, str], list[str]]:
    root = Path(dataset_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return {}, [f"manifest.json not found in {root}; source will be unknown"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_map: dict[str, str] = {}
    warnings = []
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


def evaluate_model_on_testset(
    model: torch.nn.Module,
    *,
    model_id: str,
    testset_id: str,
    dataset_dir: Path,
    source_map: Mapping[str, str],
    conditions: Sequence[DegradationCondition],
    seed: int,
    batch_size: int,
    device: torch.device,
    input_metric_cache: dict[tuple[str, str, int], tuple[float, float]],
    figures_dir: Path | None = None,
    max_figures: int = 0,
    figure_state: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    patch_files = discover_patch_files(dataset_dir)
    model.eval()
    rows = []
    figure_state = figure_state if figure_state is not None else {"saved": 0}
    with torch.no_grad():
        for condition in conditions:
            for batch_start in range(0, len(patch_files), batch_size):
                batch_paths = patch_files[batch_start : batch_start + batch_size]
                clean_batch = []
                degraded_batch = []
                batch_meta = []
                for offset, patch_path in enumerate(batch_paths):
                    patch_index = batch_start + offset
                    clean = np.load(patch_path).astype(np.float32, copy=False)
                    if clean.ndim != 2:
                        raise ValueError(f"Expected 2D clean patch, got {clean.shape} for {patch_path}")
                    rng = np.random.default_rng(_stable_seed(seed, testset_id, patch_index, condition.condition_index))
                    degraded, _, _ = degrade_patch(
                        clean,
                        missing_rate=condition.missing_rate,
                        snr_db=condition.snr_setting_db,
                        rng=rng,
                    )
                    cache_key = (testset_id, patch_path.name, condition.condition_index)
                    if cache_key not in input_metric_cache:
                        input_metric_cache[cache_key] = (
                            snr_db(degraded, clean),
                            ssim_score(degraded, clean),
                        )
                    clean_batch.append(clean)
                    degraded_batch.append(degraded)
                    batch_meta.append((patch_index, patch_path, source_map.get(patch_path.name, "unknown"), cache_key))

                tensor = torch.from_numpy(np.stack(degraded_batch)).unsqueeze(1).float().to(device)
                start = time.time()
                predictions = model(tensor).squeeze(1).detach().cpu().numpy().astype(np.float32)
                elapsed_per_sample = (time.time() - start) / max(len(batch_paths), 1)

                for clean, degraded, prediction, meta in zip(clean_batch, degraded_batch, predictions, batch_meta):
                    patch_index, patch_path, source, cache_key = meta
                    input_snr, input_ssim = input_metric_cache[cache_key]
                    output_snr = snr_db(prediction, clean)
                    output_ssim = ssim_score(prediction, clean)
                    row = build_metric_row(
                        model_id=model_id,
                        testset_id=testset_id,
                        source=source,
                        patch_file=patch_path.name,
                        patch_index=patch_index,
                        condition_index=condition.condition_index,
                        snr_setting_db=condition.snr_setting_db,
                        missing_rate=condition.missing_rate,
                        input_snr_db=input_snr,
                        input_ssim=input_ssim,
                        output_snr_db=output_snr,
                        output_ssim=output_ssim,
                        inference_seconds=elapsed_per_sample,
                    )
                    rows.append(row)
                    if figures_dir is not None and figure_state["saved"] < max_figures:
                        figure_state["saved"] += 1
                        _save_comparison_figure(
                            figures_dir / f"figure_{figure_state['saved']:03d}_{model_id}_{testset_id}_{patch_path.stem}.png",
                            clean=clean,
                            degraded=degraded,
                            prediction=prediction,
                            row=row,
                        )
    return rows


def load_scrn_model(checkpoint_path: str | Path, device: torch.device) -> SCRN:
    checkpoint = load_checkpoint(Path(checkpoint_path), map_location=device)
    config = checkpoint.get("model_config", {})
    model = SCRN(
        dim=int(config.get("dim", 64)),
        stage_depths=tuple(config.get("stage_depths", (1, 1, 1, 1, 1))),
        head_dim=int(config.get("head_dim", 32)),
        window_size=int(config.get("window_size", 8)),
        drop_path_rate=float(config.get("drop_path_rate", 0.0)),
        input_resolution=int(config.get("input_resolution", 128)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def select_device(device_arg: str, cuda_device_index: int) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg in {"auto", "cuda"} and torch.cuda.is_available():
        if cuda_device_index < 0 or cuda_device_index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {cuda_device_index} is outside available range 0..{torch.cuda.device_count() - 1}")
        torch.cuda.set_device(cuda_device_index)
        return torch.device("cuda", cuda_device_index)
    if device_arg == "cuda":
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device("cpu")


def write_jsonl_rows(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_multi_summary(path: str | Path, *, run_dir: Path, metrics: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    lines = [
        "# FP32 SCRN Multi-Eval Run",
        "",
        "## Run",
        "",
        f"- run_dir: `{run_dir}`",
        f"- preset: `{config['preset']}`",
        f"- sample_count: `{metrics['sample_count']}`",
        f"- condition_count: `{metrics['condition_count']}`",
        "",
        "## Overall Metrics",
        "",
        "| model | testset | count | output_snr_mean | output_ssim_mean | snr_gain_mean | ssim_gain_mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["groups"]["overall"]:
        lines.append(
            "| {model_id} | {testset_id} | {sample_count} | {output_snr_db_mean:.6f} | "
            "{output_ssim_mean:.6f} | {snr_gain_db_mean:.6f} | {ssim_gain_mean:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Paired Deltas",
            "",
            "| testset | count | paper5_minus_old_snr_mean | paper5_minus_old_ssim_mean |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in metrics["paired_comparison"]["overall"]:
        lines.append(
            "| {testset_id} | {sample_count} | {output_snr_db_delta_mean:.6f} | "
            "{output_ssim_delta_mean:.6f} |".format(**row)
        )
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _summarize_groups(
    rows: Sequence[Mapping[str, Any]],
    group_keys: Sequence[str],
    stat_fields: Sequence[str],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        buckets.setdefault(key, []).append(row)

    summaries = []
    for key, bucket_rows in sorted(buckets.items(), key=lambda item: tuple(str(part) for part in item[0])):
        summary = {group_key: value for group_key, value in zip(group_keys, key)}
        summary["sample_count"] = len(bucket_rows)
        for field in stat_fields:
            values = np.asarray([float(row[field]) for row in bucket_rows], dtype=np.float64)
            summary[f"{field}_mean"] = float(np.mean(values))
            summary[f"{field}_median"] = float(np.median(values))
            summary[f"{field}_std"] = float(np.std(values))
        summaries.append(summary)
    return summaries


def _validate_preset_inputs(models: Sequence[ModelSpec], testsets: Sequence[TestsetSpec]) -> None:
    for model in models:
        if not model.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist for {model.model_id}: {model.checkpoint_path}")
    for testset in testsets:
        if not testset.dataset_dir.is_dir():
            raise FileNotFoundError(f"Testset directory does not exist for {testset.testset_id}: {testset.dataset_dir}")
        patch_count = len(discover_patch_files(testset.dataset_dir))
        if patch_count != 478:
            raise ValueError(f"Preset expects 478 patches for {testset.testset_id}, found {patch_count}")


def _stable_seed(seed: int, testset_id: str, patch_index: int, condition_index: int) -> int:
    payload = f"{seed}:{testset_id}:{patch_index}:{condition_index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _save_comparison_figure(path: Path, *, clean: np.ndarray, degraded: np.ndarray, prediction: np.ndarray, row: Mapping[str, Any]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Saving comparison figures requires installing matplotlib.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    panels = [
        (clean, "Ground Truth"),
        (degraded, f"Input SNR={row['input_snr_db']:.2f} SSIM={row['input_ssim']:.3f}"),
        (prediction, f"Output SNR={row['output_snr_db']:.2f} SSIM={row['output_ssim']:.3f}"),
    ]
    vmin = float(np.min(clean))
    vmax = float(np.max(clean))
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0
    image = None
    for axis, (data, title) in zip(axes, panels):
        image = axis.imshow(data, cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
    fig.colorbar(image, ax=axes, shrink=0.75)
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
