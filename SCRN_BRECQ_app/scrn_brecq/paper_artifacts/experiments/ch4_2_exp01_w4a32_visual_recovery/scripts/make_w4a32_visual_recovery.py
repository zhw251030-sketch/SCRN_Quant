"""Generate W4A32 visual recovery candidates for thesis Chapter 4.2."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/scrn_brecq_matplotlib_cache")

SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_DIR = SCRIPT_PATH.parents[1]
PAPER_ARTIFACTS_DIR = SCRIPT_PATH.parents[3]
REPO_ROOT = SCRIPT_PATH.parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_grid import (  # noqa: E402
    DegradationCondition,
    load_degraded_batch,
    load_manifest_source_map,
)
from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_multi import (  # noqa: E402
    load_eval_model,
    select_eval_device,
)
from SCRN_BRECQ_app.scrn_brecq.cli.visualize_quantized_scrn_grid import (  # noqa: E402
    load_manifest_scale_map,
    restore_amplitude,
    symmetric_abs_limit,
)
from SCRN_BRECQ_app.scrn_repro.utils import snr_db, ssim_score  # noqa: E402


EXPERIMENT_INFO_PATH = EXPERIMENT_DIR / "experiment_info.json"
DEFAULT_TESTSET_ID = "paper5_energy_filtered_perpatch_absmax_478"
DEFAULT_SEED = 20260507
SOURCE_PRIORITY = ("Anisotropic", "Kerry3D", "Shots0001")
REQUIRED_MANIFEST_ROW_FIELDS = (
    "testset_id",
    "patch_index",
    "patch_file",
    "source",
    "condition_index",
    "snr_setting_db",
    "missing_rate",
    "fp32_snr_db",
    "quant_pre_recon_snr_db",
    "quant_post_recon_snr_db",
    "quant_post_minus_fp32_snr_db",
)
SCORE_FIELDS = (
    "fp32_snr_db",
    "quant_pre_minus_fp32_snr_db",
    "quant_post_minus_fp32_snr_db",
    "quant_post_minus_pre_snr_db",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-info", default=str(EXPERIMENT_INFO_PATH))
    parser.add_argument(
        "--candidate-set",
        choices=["all", "set_a_three_degradation_levels", "set_b_three_medium_samples"],
        default="all",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--cuda-device-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    started = time.time()
    info_path = Path(args.experiment_info)
    info = read_json(info_path)
    source_artifacts = info["source_artifacts"]
    metrics_path = repo_path(source_artifacts["per_sample_metrics"])
    eval_dataset_dir = repo_path(source_artifacts["test_dataset"])
    testset_id = DEFAULT_TESTSET_ID

    rows = read_jsonl_rows(metrics_path)
    selected_sets = select_requested_sets(rows, args.candidate_set)
    device = select_eval_device(args.device, args.cuda_device_index)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    source_map, source_warnings = load_manifest_source_map(eval_dataset_dir)
    scale_map, scale_warnings = load_manifest_scale_map(eval_dataset_dir)
    bundles = load_w4a32_bundles(source_artifacts, device)

    generated: list[dict[str, Any]] = []
    with torch.no_grad():
        for set_id, selected_rows in selected_sets.items():
            generated.append(
                generate_candidate_set(
                    info=info,
                    set_id=set_id,
                    selected_rows=selected_rows,
                    eval_dataset_dir=eval_dataset_dir,
                    testset_id=testset_id,
                    seed=int(args.seed),
                    source_map=source_map,
                    scale_map=scale_map,
                    bundles=bundles,
                    device=device,
                    manifest_warnings=source_warnings + scale_warnings,
                )
            )

    elapsed = time.time() - started
    print(
        "generated_sets={sets} elapsed={elapsed:.2f}s".format(
            sets=",".join(item["candidate_set"] for item in generated),
            elapsed=elapsed,
        ),
        flush=True,
    )
    for item in generated:
        print(
            "candidate_set={candidate_set} version={version} png={png} manifest={manifest}".format(**item),
            flush=True,
        )


def select_requested_sets(
    rows: Sequence[Mapping[str, Any]],
    candidate_set: str,
) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    if candidate_set in ("all", "set_a_three_degradation_levels"):
        selected["set_a_three_degradation_levels"] = [
            select_condition_representative(rows, snr_setting_db=10.0, missing_rate=0.02, row_label="light"),
            select_condition_representative(rows, snr_setting_db=1.0, missing_rate=0.18, row_label="medium"),
            select_condition_representative(rows, snr_setting_db=-2.0, missing_rate=0.38, row_label="heavy"),
        ]
    if candidate_set in ("all", "set_b_three_medium_samples"):
        selected["set_b_three_medium_samples"] = select_medium_samples(rows, snr_setting_db=1.0, missing_rate=0.18, count=3)
    return selected


def select_condition_representative(
    rows: Sequence[Mapping[str, Any]],
    *,
    snr_setting_db: float,
    missing_rate: float,
    row_label: str,
) -> dict[str, Any]:
    matching = filter_condition(rows, snr_setting_db=snr_setting_db, missing_rate=missing_rate)
    if not matching:
        raise ValueError(f"No rows found for SNR={snr_setting_db:g}, missing={missing_rate:g}")
    medians = condition_medians(matching)
    selected = min(matching, key=lambda row: representative_sort_key(row, medians))
    return attach_selection(
        selected,
        row_label=row_label,
        method="condition_median_representative",
        score=representative_score(selected, medians),
        medians=medians,
    )


def select_medium_samples(
    rows: Sequence[Mapping[str, Any]],
    *,
    snr_setting_db: float,
    missing_rate: float,
    count: int,
) -> list[dict[str, Any]]:
    matching = filter_condition(rows, snr_setting_db=snr_setting_db, missing_rate=missing_rate)
    if len(matching) < int(count):
        raise ValueError(f"Need {count} rows for medium samples, found {len(matching)}")
    medians = condition_medians(matching)
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, int]] = set()

    for source in SOURCE_PRIORITY:
        source_rows = [row for row in matching if str(row.get("source")) == source]
        if not source_rows:
            continue
        row = min(source_rows, key=lambda item: representative_sort_key(item, medians))
        selected.append(
            attach_selection(
                row,
                row_label=f"sample_{len(selected) + 1}",
                method="medium_condition_distinct_source_representative",
                score=representative_score(row, medians),
                medians=medians,
            )
        )
        selected_keys.add(row_key(row))
        if len(selected) == int(count):
            return selected

    for row in sorted(matching, key=lambda item: representative_sort_key(item, medians)):
        if row_key(row) in selected_keys:
            continue
        selected.append(
            attach_selection(
                row,
                row_label=f"sample_{len(selected) + 1}",
                method="medium_condition_representative_fallback",
                score=representative_score(row, medians),
                medians=medians,
            )
        )
        if len(selected) == int(count):
            return selected
    raise ValueError(f"Could not select {count} medium samples")


def filter_condition(
    rows: Sequence[Mapping[str, Any]],
    *,
    snr_setting_db: float,
    missing_rate: float,
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if nearly_equal(float(row["snr_setting_db"]), float(snr_setting_db))
        and nearly_equal(float(row["missing_rate"]), float(missing_rate))
    ]


def condition_medians(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    medians: dict[str, float] = {}
    for field in SCORE_FIELDS:
        medians[field] = float(np.median([float(row[field]) for row in rows]))
    return medians


def representative_sort_key(row: Mapping[str, Any], medians: Mapping[str, float]) -> tuple[float, str, int, int]:
    return (
        representative_score(row, medians),
        str(row.get("source", "")),
        int(row["patch_index"]),
        int(row["condition_index"]),
    )


def representative_score(row: Mapping[str, Any], medians: Mapping[str, float]) -> float:
    score = 0.0
    for field in SCORE_FIELDS:
        scale = abs(float(medians[field])) if abs(float(medians[field])) > 1e-6 else 1.0
        score += abs(float(row[field]) - float(medians[field])) / scale
    return float(score)


def attach_selection(
    row: Mapping[str, Any],
    *,
    row_label: str,
    method: str,
    score: float,
    medians: Mapping[str, float],
) -> dict[str, Any]:
    selected = dict(row)
    selected["selection"] = {
        "row_label": row_label,
        "method": method,
        "representative_score": float(score),
        "condition_medians": {key: float(value) for key, value in medians.items()},
    }
    return selected


def manifest_row_from_metrics(
    row: Mapping[str, Any],
    *,
    row_label: str,
    selection_method: str,
) -> dict[str, Any]:
    manifest_row = {field: row[field] for field in REQUIRED_MANIFEST_ROW_FIELDS}
    for optional_field in (
        "input_snr_db",
        "input_ssim",
        "fp32_ssim",
        "quant_pre_recon_ssim",
        "quant_post_recon_ssim",
        "quant_pre_minus_fp32_snr_db",
        "quant_post_minus_pre_snr_db",
        "quant_post_minus_pre_ssim",
    ):
        if optional_field in row:
            manifest_row[optional_field] = row[optional_field]
    manifest_row["selection"] = dict(row.get("selection", {}))
    manifest_row["selection"].setdefault("row_label", row_label)
    manifest_row["selection"].setdefault("method", selection_method)
    return manifest_row


def generate_candidate_set(
    *,
    info: Mapping[str, Any],
    set_id: str,
    selected_rows: Sequence[Mapping[str, Any]],
    eval_dataset_dir: Path,
    testset_id: str,
    seed: int,
    source_map: Mapping[str, str],
    scale_map: Mapping[str, float],
    bundles: Mapping[str, Mapping[str, Any]],
    device: torch.device,
    manifest_warnings: Sequence[str],
) -> dict[str, Any]:
    candidate_dir = EXPERIMENT_DIR / "candidates" / set_id
    version = next_version(candidate_dir)
    file_stem = candidate_file_stem(set_id, version)
    png_path = candidate_dir / f"{file_stem}.png"
    pdf_path = candidate_dir / f"{file_stem}.pdf"
    manifest_path = candidate_dir / f"manifest_{version}.json"
    summary_path = candidate_dir / f"selection_summary_{version}.md"

    display_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for selected in selected_rows:
        rendered = render_row(
            selected,
            eval_dataset_dir=eval_dataset_dir,
            testset_id=testset_id,
            seed=seed,
            source_map=source_map,
            scale_map=scale_map,
            bundles=bundles,
            device=device,
        )
        display_rows.append(rendered)
        selection = selected.get("selection", {})
        manifest_row = manifest_row_from_metrics(
            selected,
            row_label=str(selection.get("row_label", f"row_{len(manifest_rows) + 1}")),
            selection_method=str(selection.get("method", "unknown")),
        )
        manifest_row["normalization_scale"] = rendered["normalization_scale"]
        manifest_row["computed_panel_metrics"] = rendered["computed_panel_metrics"]
        manifest_rows.append(manifest_row)

    save_visual_grid(png_path, display_rows=display_rows, title=figure_title(set_id))
    save_visual_grid(pdf_path, display_rows=display_rows, title=figure_title(set_id))
    manifest = build_manifest(
        info=info,
        set_id=set_id,
        version=version,
        png_path=png_path,
        pdf_path=pdf_path,
        manifest_rows=manifest_rows,
        manifest_warnings=manifest_warnings,
    )
    write_json(manifest_path, manifest)
    write_selection_summary(summary_path, manifest=manifest)
    return {
        "candidate_set": set_id,
        "version": version,
        "png": str(png_path.relative_to(REPO_ROOT)),
        "pdf": str(pdf_path.relative_to(REPO_ROOT)),
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "summary": str(summary_path.relative_to(REPO_ROOT)),
    }


def render_row(
    selected: Mapping[str, Any],
    *,
    eval_dataset_dir: Path,
    testset_id: str,
    seed: int,
    source_map: Mapping[str, str],
    scale_map: Mapping[str, float],
    bundles: Mapping[str, Mapping[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    patch_file = str(selected["patch_file"])
    condition = DegradationCondition(
        condition_index=int(selected["condition_index"]),
        snr_setting_db=float(selected["snr_setting_db"]),
        missing_rate=float(selected["missing_rate"]),
    )
    clean_batch, degraded_batch, _meta = load_degraded_batch(
        [eval_dataset_dir / patch_file],
        testset_id=testset_id,
        condition=condition,
        seed=seed,
        offset=int(selected["patch_index"]),
        source_map=source_map,
    )
    clean = clean_batch[0]
    degraded = degraded_batch[0]
    input_tensor = torch.from_numpy(degraded_batch[:, None, :, :]).float().to(device)
    predictions = predict_w4a32(bundles, input_tensor)
    scale = float(scale_map[patch_file])
    panels = {
        "clean": restore_amplitude(clean, scale),
        "degraded": restore_amplitude(degraded, scale),
        "fp32": restore_amplitude(predictions["fp32"], scale),
        "w4a32_pre": restore_amplitude(predictions["w4a32_pre"], scale),
        "w4a32_final": restore_amplitude(predictions["w4a32_final"], scale),
    }
    metrics = {
        "degraded": metric_pair(degraded, clean),
        "fp32": metric_pair(predictions["fp32"], clean),
        "w4a32_pre": metric_pair(predictions["w4a32_pre"], clean),
        "w4a32_final": metric_pair(predictions["w4a32_final"], clean),
    }
    return {
        "row_label": selected["selection"]["row_label"],
        "source": str(selected.get("source", "unknown")),
        "patch_file": patch_file,
        "patch_index": int(selected["patch_index"]),
        "condition_index": int(selected["condition_index"]),
        "snr_setting_db": float(selected["snr_setting_db"]),
        "missing_rate": float(selected["missing_rate"]),
        "normalization_scale": scale,
        "panels": panels,
        "computed_panel_metrics": metrics,
    }


def predict_w4a32(
    bundles: Mapping[str, Mapping[str, Any]],
    input_tensor: torch.Tensor,
) -> dict[str, np.ndarray]:
    final_bundle = bundles["w4a32_final"]
    pre_bundle = bundles["w4a32_pre"]
    final_model = final_bundle["model"]
    pre_model = pre_bundle["model"]

    final_model.set_quant_state(False, False)
    fp32 = tensor_prediction(final_model, input_tensor)

    pre_model.set_quant_state(bool(pre_bundle["weight_quant"]), bool(pre_bundle["act_quant"]))
    pre = tensor_prediction(pre_model, input_tensor)

    final_model.set_quant_state(bool(final_bundle["weight_quant"]), bool(final_bundle["act_quant"]))
    final = tensor_prediction(final_model, input_tensor)

    return {"fp32": fp32, "w4a32_pre": pre, "w4a32_final": final}


def tensor_prediction(model: torch.nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    return model(input_tensor).squeeze(1).detach().cpu().numpy().astype(np.float32)[0]


def metric_pair(prediction: np.ndarray, clean: np.ndarray) -> dict[str, float]:
    return {"snr_db": float(snr_db(prediction, clean)), "ssim": float(ssim_score(prediction, clean))}


def save_visual_grid(path: Path, *, display_rows: Sequence[Mapping[str, Any]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        ("clean", "Clean"),
        ("degraded", "Degraded input"),
        ("fp32", "FP32"),
        ("w4a32_pre", "W4A32 pre"),
        ("w4a32_final", "W4A32 final"),
    ]
    fig, axes = plt.subplots(len(display_rows), len(columns), figsize=(15.0, 7.6), constrained_layout=True)
    axes_array = np.asarray(axes)
    if axes_array.ndim == 1:
        axes_array = axes_array[None, :]
    fig.suptitle(title, fontsize=13)

    for row_index, row in enumerate(display_rows):
        panels = row["panels"]
        row_limit = symmetric_abs_limit([panels[key] for key, _label in columns])
        image = None
        for col_index, (key, label) in enumerate(columns):
            axis = axes_array[row_index, col_index]
            image = axis.imshow(panels[key], cmap="seismic", aspect="auto", vmin=-row_limit, vmax=row_limit)
            title_text = label if row_index == 0 else ""
            if key != "clean":
                metric = row["computed_panel_metrics"][key]
                title_text = f"{title_text}\nSNR {metric['snr_db']:.2f} dB" if title_text else f"SNR {metric['snr_db']:.2f} dB"
            axis.set_title(title_text, fontsize=8)
            axis.axis("off")
        axes_array[row_index, 0].text(
            -0.05,
            0.5,
            row_side_label(row),
            transform=axes_array[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=8,
        )
        if image is not None:
            fig.colorbar(image, ax=axes_array[row_index, :].ravel().tolist(), shrink=0.76, fraction=0.018, pad=0.01)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def row_side_label(row: Mapping[str, Any]) -> str:
    return (
        f"{row['row_label']}\n"
        f"{row['source']}\n"
        f"{row['patch_file']}\n"
        f"SNR={float(row['snr_setting_db']):g}, miss={float(row['missing_rate']):g}"
    )


def build_manifest(
    *,
    info: Mapping[str, Any],
    set_id: str,
    version: str,
    png_path: Path,
    pdf_path: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    manifest_warnings: Sequence[str],
) -> dict[str, Any]:
    return {
        "experiment_id": info["experiment_id"],
        "candidate_set": set_id,
        "version": version,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "figure_files": {
            "png": str(png_path.relative_to(REPO_ROOT)),
            "pdf": str(pdf_path.relative_to(REPO_ROOT)),
        },
        "source_artifacts": info["source_artifacts"],
        "selection_policy": selection_policy(set_id),
        "display_protocol": {
            "layout": "3 rows x 5 columns",
            "columns": ["Clean", "Degraded input", "FP32", "W4A32 pre-reconstruction", "W4A32 final"],
            "colormap": "seismic",
            "amplitude_space": "restored from normalized per-patch absmax for display",
            "color_scaling": "symmetric per row across all five panels",
            "metric_space": "normalized per-patch absmax",
        },
        "manifest_warnings": list(manifest_warnings),
        "rows": list(manifest_rows),
    }


def selection_policy(set_id: str) -> dict[str, Any]:
    if set_id == "set_a_three_degradation_levels":
        return {
            "description": "Select one median-like representative row for light, medium, and heavy degradation.",
            "conditions": [
                {"row_label": "light", "snr_setting_db": 10.0, "missing_rate": 0.02},
                {"row_label": "medium", "snr_setting_db": 1.0, "missing_rate": 0.18},
                {"row_label": "heavy", "snr_setting_db": -2.0, "missing_rate": 0.38},
            ],
            "score_fields": list(SCORE_FIELDS),
        }
    return {
        "description": "Select three source-diverse median-like representatives under medium degradation.",
        "condition": {"snr_setting_db": 1.0, "missing_rate": 0.18},
        "source_priority": list(SOURCE_PRIORITY),
        "score_fields": list(SCORE_FIELDS),
    }


def write_selection_summary(path: Path, *, manifest: Mapping[str, Any]) -> None:
    lines = [
        f"# {manifest['candidate_set']} {manifest['version']}",
        "",
        "## Figures",
        "",
        f"- PNG: `{manifest['figure_files']['png']}`",
        f"- PDF: `{manifest['figure_files']['pdf']}`",
        "",
        "## Selected Rows",
        "",
        "| row | source | patch_index | patch_file | condition | SNR setting | missing | FP32 SNR | W4A32 pre SNR | W4A32 final SNR | final - FP32 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in manifest["rows"]:
        selection = row["selection"]
        lines.append(
            "| {label} | {source} | {patch_index} | {patch_file} | {condition_index} | {snr:.1f} | {missing:.2f} | "
            "{fp32:.4f} | {pre:.4f} | {post:.4f} | {delta:.4f} |".format(
                label=selection["row_label"],
                source=row["source"],
                patch_index=int(row["patch_index"]),
                patch_file=row["patch_file"],
                condition_index=int(row["condition_index"]),
                snr=float(row["snr_setting_db"]),
                missing=float(row["missing_rate"]),
                fp32=float(row["fp32_snr_db"]),
                pre=float(row["quant_pre_recon_snr_db"]),
                post=float(row["quant_post_recon_snr_db"]),
                delta=float(row["quant_post_minus_fp32_snr_db"]),
            )
        )
    lines.extend(
        [
            "",
            "## Selection Policy",
            "",
            json.dumps(manifest["selection_policy"], ensure_ascii=False, indent=2, sort_keys=True),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def load_w4a32_bundles(source_artifacts: Mapping[str, str], device: torch.device) -> dict[str, Mapping[str, Any]]:
    return {
        "w4a32_pre": load_eval_model(repo_path(source_artifacts["w4a32_pre_recon_checkpoint"]), device),
        "w4a32_final": load_eval_model(repo_path(source_artifacts["w4a32_final_checkpoint"]), device),
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def row_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["patch_file"]), int(row["condition_index"])


def nearly_equal(left: float, right: float, *, tolerance: float = 1e-9) -> bool:
    return abs(float(left) - float(right)) <= float(tolerance)


def next_version(candidate_dir: Path) -> str:
    existing: list[int] = []
    for path in candidate_dir.glob("manifest_v*.json"):
        match = re.fullmatch(r"manifest_v(\d{3})\.json", path.name)
        if match:
            existing.append(int(match.group(1)))
    return f"v{(max(existing) + 1 if existing else 1):03d}"


def candidate_file_stem(set_id: str, version: str) -> str:
    if set_id == "set_a_three_degradation_levels":
        return f"fig_ch4_2_w4a32_3x5_levels_{version}"
    if set_id == "set_b_three_medium_samples":
        return f"fig_ch4_2_w4a32_3x5_medium_samples_{version}"
    raise ValueError(f"Unknown candidate set: {set_id}")


def figure_title(set_id: str) -> str:
    if set_id == "set_a_three_degradation_levels":
        return "W4A32 visual recovery candidates: three degradation levels"
    return "W4A32 visual recovery candidates: three medium-degradation samples"


if __name__ == "__main__":
    main()
