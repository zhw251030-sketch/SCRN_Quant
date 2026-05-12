"""Generate W4A8/W4A4 3x6 activation visual recovery candidates."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/scrn_brecq_matplotlib_cache")

SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENTS_DIR = SCRIPT_PATH.parents[2]
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


DEFAULT_TESTSET_ID = "paper5_energy_filtered_perpatch_absmax_478"
DEFAULT_SEED = 20260507
BASE_SCRIPT_PATH = (
    EXPERIMENTS_DIR
    / "ch4_2_exp01_w4a32_visual_recovery"
    / "scripts"
    / "make_w4a32_visual_recovery.py"
)
EXPERIMENT_DIRS = {
    "w4a8": EXPERIMENTS_DIR / "ch4_3_exp01_w4a8_visual_recovery",
    "w4a4": EXPERIMENTS_DIR / "ch4_3_exp02_w4a4_visual_recovery",
}


def load_base_module():
    spec = importlib.util.spec_from_file_location("make_w4a32_visual_recovery", BASE_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-experiment", choices=["all", "w4a8", "w4a4"], default="all")
    parser.add_argument(
        "--candidate-set",
        choices=["all", "set_a_three_degradation_levels", "set_b_three_medium_samples"],
        default="all",
    )
    parser.add_argument(
        "--set-a-selection",
        choices=["condition_median", "fixed_patch_from_medium"],
        default="fixed_patch_from_medium",
    )
    parser.add_argument("--fixed-patch-file", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--cuda-device-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--row-label-style", choices=["metadata", "compact", "none"], default="none")
    parser.add_argument("--panel-metric-style", choices=["snr", "none"], default="snr")
    parser.add_argument("--column-label-style", choices=["labels", "none"], default="labels")
    parser.add_argument("--colorbar-style", choices=["per_row", "none"], default="none")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    started = time.time()
    requested = ["w4a8", "w4a4"] if args.activation_experiment == "all" else [str(args.activation_experiment)]
    device = select_eval_device(args.device, args.cuda_device_index)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    generated: list[dict[str, Any]] = []
    with torch.no_grad():
        for experiment_key in requested:
            info = read_json(EXPERIMENT_DIRS[experiment_key] / "experiment_info.json")
            generated.extend(
                generate_for_experiment(
                    info=info,
                    candidate_set=str(args.candidate_set),
                    set_a_selection=str(args.set_a_selection),
                    fixed_patch_file=args.fixed_patch_file,
                    seed=int(args.seed),
                    device=device,
                    row_label_style=str(args.row_label_style),
                    panel_metric_style=str(args.panel_metric_style),
                    column_label_style=str(args.column_label_style),
                    colorbar_style=str(args.colorbar_style),
                )
            )

    elapsed = time.time() - started
    print(
        "generated_candidates={count} elapsed={elapsed:.2f}s".format(count=len(generated), elapsed=elapsed),
        flush=True,
    )
    for item in generated:
        print(
            "experiment={experiment} candidate_set={candidate_set} version={version} png={png} manifest={manifest}".format(
                **item
            ),
            flush=True,
        )


def generate_for_experiment(
    *,
    info: Mapping[str, Any],
    candidate_set: str,
    set_a_selection: str,
    fixed_patch_file: str | None,
    seed: int,
    device: torch.device,
    row_label_style: str,
    panel_metric_style: str,
    column_label_style: str,
    colorbar_style: str,
) -> list[dict[str, Any]]:
    source_artifacts = info["source_artifacts"]
    rows = BASE.read_jsonl_rows(repo_path(source_artifacts["per_sample_metrics"]))
    selected_sets = select_requested_sets(
        rows,
        candidate_set=candidate_set,
        set_a_selection=set_a_selection,
        fixed_patch_file=fixed_patch_file,
    )
    eval_dataset_dir = repo_path(source_artifacts["test_dataset"])
    source_map, source_warnings = load_manifest_source_map(eval_dataset_dir)
    scale_map, scale_warnings = load_manifest_scale_map(eval_dataset_dir)
    bundles = load_bundles(source_artifacts, device)

    generated: list[dict[str, Any]] = []
    for set_id, selected_rows in selected_sets.items():
        generated.append(
            generate_candidate_set(
                info=info,
                set_id=set_id,
                selected_rows=selected_rows,
                eval_dataset_dir=eval_dataset_dir,
                testset_id=DEFAULT_TESTSET_ID,
                seed=seed,
                source_map=source_map,
                scale_map=scale_map,
                bundles=bundles,
                device=device,
                manifest_warnings=source_warnings + scale_warnings,
                row_label_style=row_label_style,
                panel_metric_style=panel_metric_style,
                column_label_style=column_label_style,
                colorbar_style=colorbar_style,
            )
        )
    return generated


def select_requested_sets(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_set: str,
    set_a_selection: str,
    fixed_patch_file: str | None,
) -> dict[str, list[dict[str, Any]]]:
    selected_sets = BASE.select_requested_sets(rows, candidate_set, set_a_selection=set_a_selection)
    if fixed_patch_file and candidate_set in ("all", "set_a_three_degradation_levels"):
        selected_sets["set_a_three_degradation_levels"] = select_fixed_patch_file_degradation_levels(
            rows,
            patch_file=str(fixed_patch_file),
        )
    return selected_sets


def select_fixed_patch_file_degradation_levels(
    rows: Sequence[Mapping[str, Any]],
    *,
    patch_file: str,
) -> list[dict[str, Any]]:
    conditions = (
        ("light", 10.0, 0.02),
        ("medium", 1.0, 0.18),
        ("heavy", -2.0, 0.38),
    )
    selected: list[dict[str, Any]] = []
    for row_label, snr_setting_db, missing_rate in conditions:
        matching = BASE.filter_condition(rows, snr_setting_db=snr_setting_db, missing_rate=missing_rate)
        medians = BASE.condition_medians(matching)
        row = next((item for item in matching if str(item["patch_file"]) == str(patch_file)), None)
        if row is None:
            raise ValueError(f"Patch {patch_file} lacks SNR={snr_setting_db:g}, missing={missing_rate:g}")
        selected.append(
            BASE.attach_selection(
                row,
                row_label=row_label,
                method="fixed_patch_file_degradation_levels",
                score=BASE.representative_score(row, medians),
                medians=medians,
            )
        )
    return selected


def display_columns(activation_label: str) -> list[tuple[str, str]]:
    return [
        ("clean", "Clean"),
        ("degraded", "Degraded input"),
        ("fp32", "FP32"),
        ("w4a32_final", "W4A32 final"),
        ("activation_pre", f"{activation_label} pre-act"),
        ("activation_final", f"{activation_label} final"),
    ]


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
    row_label_style: str,
    panel_metric_style: str,
    column_label_style: str,
    colorbar_style: str,
) -> dict[str, Any]:
    experiment_dir = EXPERIMENT_DIRS[str(info["activation_experiment"])]
    candidate_dir = experiment_dir / "candidates" / set_id
    version = BASE.next_version(candidate_dir)
    file_stem = candidate_file_stem(str(info["activation_experiment"]).lower(), set_id, version)
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
            w4a32_final_snr_db=float(rendered["computed_panel_metrics"]["w4a32_final"]["snr_db"]),
        )
        manifest_row["normalization_scale"] = rendered["normalization_scale"]
        manifest_row["computed_panel_metrics"] = rendered["computed_panel_metrics"]
        manifest_rows.append(manifest_row)

    save_visual_grid(
        png_path,
        display_rows=display_rows,
        activation_label=str(info["activation_label"]),
        row_label_style=row_label_style,
        panel_metric_style=panel_metric_style,
        column_label_style=column_label_style,
        colorbar_style=colorbar_style,
    )
    save_visual_grid(
        pdf_path,
        display_rows=display_rows,
        activation_label=str(info["activation_label"]),
        row_label_style=row_label_style,
        panel_metric_style=panel_metric_style,
        column_label_style=column_label_style,
        colorbar_style=colorbar_style,
    )

    manifest = build_manifest(
        info=info,
        set_id=set_id,
        version=version,
        png_path=png_path,
        pdf_path=pdf_path,
        manifest_rows=manifest_rows,
        manifest_warnings=manifest_warnings,
        row_label_style=row_label_style,
        panel_metric_style=panel_metric_style,
        column_label_style=column_label_style,
        colorbar_style=colorbar_style,
    )
    write_json(manifest_path, manifest)
    write_selection_summary(summary_path, manifest=manifest)
    return {
        "experiment": info["activation_experiment"],
        "candidate_set": set_id,
        "version": version,
        "png": str(png_path.relative_to(REPO_ROOT)),
        "pdf": str(pdf_path.relative_to(REPO_ROOT)),
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "summary": str(summary_path.relative_to(REPO_ROOT)),
    }


def manifest_row_from_metrics(
    row: Mapping[str, Any],
    *,
    row_label: str,
    selection_method: str,
    w4a32_final_snr_db: float,
) -> dict[str, Any]:
    fields = (
        "testset_id",
        "patch_index",
        "patch_file",
        "source",
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
        "quant_post_minus_fp32_snr_db",
        "quant_post_minus_pre_snr_db",
    )
    manifest_row = {field: row[field] for field in fields if field in row}
    manifest_row["w4a32_final_snr_db"] = float(w4a32_final_snr_db)
    manifest_row["activation_pre_snr_db"] = float(row["quant_pre_recon_snr_db"])
    manifest_row["activation_final_snr_db"] = float(row["quant_post_recon_snr_db"])
    manifest_row["activation_final_minus_fp32_snr_db"] = float(row["quant_post_minus_fp32_snr_db"])
    manifest_row["selection"] = dict(row.get("selection", {}))
    manifest_row["selection"].setdefault("row_label", row_label)
    manifest_row["selection"].setdefault("method", selection_method)
    return manifest_row


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
    predictions = predict_panels(bundles, input_tensor)
    scale = float(scale_map[patch_file])
    panels = {
        "clean": restore_amplitude(clean, scale),
        "degraded": restore_amplitude(degraded, scale),
        "fp32": restore_amplitude(predictions["fp32"], scale),
        "w4a32_final": restore_amplitude(predictions["w4a32_final"], scale),
        "activation_pre": restore_amplitude(predictions["activation_pre"], scale),
        "activation_final": restore_amplitude(predictions["activation_final"], scale),
    }
    metrics = {
        "degraded": metric_pair(degraded, clean),
        "fp32": metric_pair(predictions["fp32"], clean),
        "w4a32_final": metric_pair(predictions["w4a32_final"], clean),
        "activation_pre": metric_pair(predictions["activation_pre"], clean),
        "activation_final": metric_pair(predictions["activation_final"], clean),
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


def predict_panels(
    bundles: Mapping[str, Mapping[str, Any]],
    input_tensor: torch.Tensor,
) -> dict[str, np.ndarray]:
    w4a32_bundle = bundles["w4a32_final"]
    pre_bundle = bundles["activation_pre"]
    final_bundle = bundles["activation_final"]

    w4a32_model = w4a32_bundle["model"]
    w4a32_model.set_quant_state(False, False)
    fp32 = tensor_prediction(w4a32_model, input_tensor)
    w4a32_model.set_quant_state(bool(w4a32_bundle["weight_quant"]), bool(w4a32_bundle["act_quant"]))
    w4a32_final = tensor_prediction(w4a32_model, input_tensor)

    pre_model = pre_bundle["model"]
    pre_model.set_quant_state(bool(pre_bundle["weight_quant"]), bool(pre_bundle["act_quant"]))
    activation_pre = tensor_prediction(pre_model, input_tensor)

    final_model = final_bundle["model"]
    final_model.set_quant_state(bool(final_bundle["weight_quant"]), bool(final_bundle["act_quant"]))
    activation_final = tensor_prediction(final_model, input_tensor)

    return {
        "fp32": fp32,
        "w4a32_final": w4a32_final,
        "activation_pre": activation_pre,
        "activation_final": activation_final,
    }


def save_visual_grid(
    path: Path,
    *,
    display_rows: Sequence[Mapping[str, Any]],
    activation_label: str,
    row_label_style: str,
    panel_metric_style: str,
    column_label_style: str,
    colorbar_style: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = display_columns(activation_label)
    fig, axes = plt.subplots(len(display_rows), len(columns), figsize=(16.8, 6.8), constrained_layout=True)
    axes_array = np.asarray(axes)
    if axes_array.ndim == 1:
        axes_array = axes_array[None, :]

    for row_index, row in enumerate(display_rows):
        panels = row["panels"]
        row_limit = symmetric_abs_limit([panels[key] for key, _label in columns])
        image = None
        for col_index, (key, label) in enumerate(columns):
            axis = axes_array[row_index, col_index]
            image = axis.imshow(panels[key], cmap="seismic", aspect="auto", vmin=-row_limit, vmax=row_limit)
            title_text = BASE.column_title_text(label, row_index=row_index, style=column_label_style)
            if key != "clean" and panel_metric_style == "snr":
                metric = row["computed_panel_metrics"][key]
                title_text = f"{title_text}\nSNR {metric['snr_db']:.2f} dB" if title_text else f"SNR {metric['snr_db']:.2f} dB"
            axis.set_title(title_text, fontsize=8)
            axis.axis("off")
        side_label = BASE.row_side_label(row, style=row_label_style)
        if side_label:
            axes_array[row_index, 0].text(
                -0.035,
                0.5,
                side_label,
                transform=axes_array[row_index, 0].transAxes,
                ha="right",
                va="center",
                fontsize=8,
            )
        if image is not None and BASE.should_draw_colorbar(colorbar_style):
            fig.colorbar(image, ax=axes_array[row_index, :].ravel().tolist(), shrink=0.76, fraction=0.014, pad=0.01)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def build_manifest(
    *,
    info: Mapping[str, Any],
    set_id: str,
    version: str,
    png_path: Path,
    pdf_path: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    manifest_warnings: Sequence[str],
    row_label_style: str,
    panel_metric_style: str,
    column_label_style: str,
    colorbar_style: str,
) -> dict[str, Any]:
    return {
        "experiment_id": info["experiment_id"],
        "activation_experiment": info["activation_experiment"],
        "activation_label": info["activation_label"],
        "candidate_set": set_id,
        "version": version,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "figure_files": {
            "png": str(png_path.relative_to(REPO_ROOT)),
            "pdf": str(pdf_path.relative_to(REPO_ROOT)),
        },
        "source_artifacts": info["source_artifacts"],
        "selection_policy": BASE.selection_policy(set_id),
        "display_protocol": {
            "layout": "3 rows x 6 columns",
            "columns": [label for _key, label in display_columns(str(info["activation_label"]))],
            "colormap": "seismic",
            "amplitude_space": "restored from normalized per-patch absmax for display",
            "color_scaling": "symmetric per row across all six panels",
            "metric_space": "normalized per-patch absmax",
            "row_label_style": row_label_style,
            "panel_metric_style": panel_metric_style,
            "column_label_style": column_label_style,
            "colorbar_style": colorbar_style,
        },
        "manifest_warnings": list(manifest_warnings),
        "rows": list(manifest_rows),
    }


def write_selection_summary(path: Path, *, manifest: Mapping[str, Any]) -> None:
    lines = [
        f"# {manifest['activation_label']} {manifest['candidate_set']} {manifest['version']}",
        "",
        "## 图件文件",
        "",
        f"- PNG：`{manifest['figure_files']['png']}`",
        f"- PDF：`{manifest['figure_files']['pdf']}`",
        "",
        "## 选中样本",
        "",
        "| 行标签 | source | patch_index | patch_file | condition_index | SNR setting | missing | FP32 SNR | W4A32 final SNR | pre-act SNR | final SNR | final - FP32 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in manifest["rows"]:
        selection = row["selection"]
        lines.append(
            "| {label} | {source} | {patch_index} | {patch_file} | {condition_index} | {snr:.1f} | {missing:.2f} | "
            "{fp32:.4f} | {w4a32:.4f} | {pre:.4f} | {post:.4f} | {delta:.4f} |".format(
                label=selection["row_label"],
                source=row["source"],
                patch_index=int(row["patch_index"]),
                patch_file=row["patch_file"],
                condition_index=int(row["condition_index"]),
                snr=float(row["snr_setting_db"]),
                missing=float(row["missing_rate"]),
                fp32=float(row["fp32_snr_db"]),
                w4a32=float(row["w4a32_final_snr_db"]),
                pre=float(row["activation_pre_snr_db"]),
                post=float(row["activation_final_snr_db"]),
                delta=float(row["activation_final_minus_fp32_snr_db"]),
            )
        )
    lines.extend(["", "## 选样策略", "", "```json", json.dumps(manifest["selection_policy"], ensure_ascii=False, indent=2, sort_keys=True), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def load_bundles(source_artifacts: Mapping[str, str], device: torch.device) -> dict[str, Mapping[str, Any]]:
    return {
        "w4a32_final": load_eval_model(repo_path(source_artifacts["w4a32_final_checkpoint"]), device),
        "activation_pre": load_eval_model(repo_path(source_artifacts["activation_pre_act_checkpoint"]), device),
        "activation_final": load_eval_model(repo_path(source_artifacts["activation_final_checkpoint"]), device),
    }


def metric_pair(prediction: np.ndarray, clean: np.ndarray) -> dict[str, float]:
    return {"snr_db": float(snr_db(prediction, clean)), "ssim": float(ssim_score(prediction, clean))}


def tensor_prediction(model: torch.nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    return model(input_tensor).squeeze(1).detach().cpu().numpy().astype(np.float32)[0]


def candidate_file_stem(experiment_key: str, set_id: str, version: str) -> str:
    if set_id == "set_a_three_degradation_levels":
        suffix = "3x6_levels"
    elif set_id == "set_b_three_medium_samples":
        suffix = "3x6_medium_samples"
    else:
        suffix = set_id
    return f"fig_ch4_3_{experiment_key}_{suffix}_{version}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def repo_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    main()
