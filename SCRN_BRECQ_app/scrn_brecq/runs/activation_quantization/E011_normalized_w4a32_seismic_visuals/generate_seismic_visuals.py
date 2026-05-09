from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


REPO = Path(".")
sys.path.insert(0, str(REPO.resolve()))

from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_grid import (
    DegradationCondition,
    build_metric_row,
    load_degraded_batch,
    load_eval_model,
    load_manifest_source_map,
    snr_db,
    ssim_score,
)
from SCRN_BRECQ_app.scrn_repro.training import write_json


os.environ.setdefault("MPLCONFIGDIR", str(REPO / "SCRN_BRECQ_app/scrn_brecq/runs/matplotlib_cache"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


RUN_DIR = REPO / (
    "SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/"
    "E012_normalized_w4a32_seismic_denormalized_visuals/"
    "20260509_211000_seismic_denormalized_representative_plus_default_single"
)
FIGURES_DIR = RUN_DIR / "figures"
SOURCE_RUN = REPO / (
    "SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/"
    "E010_normalized_w4a32_single_gpu_visuals/"
    "20260509_204855_representative_figures_source_x_condition"
)
EVAL_DATASET_DIR = REPO / "SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478"
QUANT_RUN = REPO / "SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1"
CHECKPOINT = QUANT_RUN / "checkpoints/quantized_scrn_brecq.pth"
PRE_CHECKPOINT = QUANT_RUN / "checkpoints/quantized_scrn_brecq_pre_recon.pth"
DEFAULT_CLEAN = REPO / "SCRN-main/test_data/clear.npy"
DEFAULT_INPUT = REPO / "SCRN-main/test_data/noise_and_miss.npy"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    post_bundle = load_eval_model(CHECKPOINT, device)
    pre_bundle = load_eval_model(PRE_CHECKPOINT, device)
    model = post_bundle["model"]
    pre_model = pre_bundle["model"]
    model.eval()
    pre_model.eval()
    source_map, manifest_warnings = load_manifest_source_map(EVAL_DATASET_DIR)
    scale_map, scale_warnings = load_manifest_scale_map(EVAL_DATASET_DIR)
    selected = json.loads((SOURCE_RUN / "selected_samples.json").read_text(encoding="utf-8"))

    figure_rows: list[dict] = []
    with torch.no_grad():
        for index, row in enumerate(selected, start=1):
            figure_rows.append(
                generate_normalized_representative(
                    index=index,
                    row=row,
                    model=model,
                    pre_model=pre_model,
                    post_bundle=post_bundle,
                    pre_bundle=pre_bundle,
                    source_map=source_map,
                    scale_map=scale_map,
                    device=device,
                )
            )
    default_row = generate_default_single_sample()
    figure_rows.append(default_row)

    write_json(
        RUN_DIR / "config.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "colormap": "seismic",
            "color_scale": "symmetric per figure over all panels, centered at 0",
            "normalized_representative_display": {
                "enabled": True,
                "normalization_method": "per_patch_absmax",
                "scale_source": "manifest.json samples[].normalization_scale",
                "restoration_formula": "display = normalized * normalization_scale",
                "metric_note": "SNR/SSIM labels remain the normalized-protocol metrics; visual panels are restored for display.",
            },
            "checkpoint": str(CHECKPOINT),
            "pre_recon_checkpoint": str(PRE_CHECKPOINT),
            "representative_source_run": str(SOURCE_RUN),
            "default_single_sample": {
                "clean": str(DEFAULT_CLEAN),
                "input": str(DEFAULT_INPUT),
                "amplitude_space": "raw default SCRN sample",
            },
            "device": str(device),
            "figure_count": len(figure_rows),
            "manifest_warnings": manifest_warnings + scale_warnings,
        },
    )
    write_json(RUN_DIR / "selected_samples.json", figure_rows)
    write_summary(figure_rows)
    print(f"run_dir={RUN_DIR}")
    print(f"figure_count={len(figure_rows)}")


def generate_normalized_representative(
    *,
    index: int,
    row: dict,
    model: torch.nn.Module,
    pre_model: torch.nn.Module,
    post_bundle: dict,
    pre_bundle: dict,
    source_map: dict[str, str],
    scale_map: dict[str, float],
    device: torch.device,
) -> dict:
    condition = DegradationCondition(
        condition_index=int(row["condition_index"]),
        snr_setting_db=float(row["snr_setting_db"]),
        missing_rate=float(row["missing_rate"]),
    )
    clean_batch, degraded_batch, _meta = load_degraded_batch(
        [EVAL_DATASET_DIR / row["patch_file"]],
        testset_id=str(row["testset_id"]),
        condition=condition,
        seed=20260507,
        offset=int(row["patch_index"]),
        source_map=source_map,
    )
    input_tensor = torch.from_numpy(degraded_batch[:, None, :, :]).float().to(device)

    model.set_quant_state(False, False)
    fp32 = model(input_tensor).squeeze(1).detach().cpu().numpy().astype(np.float32)[0]
    pre_model.set_quant_state(bool(pre_bundle["weight_quant"]), bool(pre_bundle["act_quant"]))
    pre = pre_model(input_tensor).squeeze(1).detach().cpu().numpy().astype(np.float32)[0]
    model.set_quant_state(bool(post_bundle["weight_quant"]), bool(post_bundle["act_quant"]))
    post = model(input_tensor).squeeze(1).detach().cpu().numpy().astype(np.float32)[0]

    clean = clean_batch[0]
    degraded = degraded_batch[0]
    normalization_scale = scale_map.get(str(row["patch_file"]))
    if normalization_scale is None:
        raise KeyError(f"Missing normalization_scale for {row['patch_file']} in {EVAL_DATASET_DIR / 'manifest.json'}")
    metric_row = build_metric_row(
        testset_id=str(row["testset_id"]),
        source=str(row["source"]),
        patch_file=str(row["patch_file"]),
        patch_index=int(row["patch_index"]),
        condition_index=int(row["condition_index"]),
        snr_setting_db=float(row["snr_setting_db"]),
        missing_rate=float(row["missing_rate"]),
        input_snr_db=snr_db(degraded, clean),
        input_ssim=ssim_score(degraded, clean),
        fp32_snr_db=snr_db(fp32, clean),
        fp32_ssim=ssim_score(fp32, clean),
        quant_post_recon_snr_db=snr_db(post, clean),
        quant_post_recon_ssim=ssim_score(post, clean),
        inference_seconds=0.0,
        quant_pre_recon_snr_db=snr_db(pre, clean),
        quant_pre_recon_ssim=ssim_score(pre, clean),
    )
    clean_display = restore_amplitude(clean, normalization_scale)
    degraded_display = restore_amplitude(degraded, normalization_scale)
    fp32_display = restore_amplitude(fp32, normalization_scale)
    pre_display = restore_amplitude(pre, normalization_scale)
    post_display = restore_amplitude(post, normalization_scale)

    figure_name = (
        f"{index:02d}_{row['source']}_{row['condition_label']}_"
        f"{Path(row['patch_file']).stem}_seismic_denormalized.png"
    )
    figure_path = FIGURES_DIR / figure_name
    save_seismic(
        figure_path,
        [
            (clean_display, "Ground Truth"),
            (degraded_display, f"Input {metric_row['input_snr_db']:.2f}dB"),
            (fp32_display, f"FP32 {metric_row['fp32_snr_db']:.2f}dB"),
            (pre_display, f"Pre {metric_row['quant_pre_recon_snr_db']:.2f}dB"),
            (post_display, f"Post {metric_row['quant_post_recon_snr_db']:.2f}dB"),
        ],
        subtitle=(
            f"{row['source']} {row['condition_label']} restored amplitude "
            f"(scale={normalization_scale:.6g}), seismic centered at 0"
        ),
    )
    return {
        **metric_row,
        "condition_label": row["condition_label"],
        "amplitude_space": "restored from normalized per-patch absmax",
        "metric_amplitude_space": "normalized per-patch absmax",
        "normalization_scale": float(normalization_scale),
        "figure": str(figure_path),
    }


def generate_default_single_sample() -> dict:
    clean = np.load(DEFAULT_CLEAN).astype(np.float32)
    degraded = np.load(DEFAULT_INPUT).astype(np.float32)
    fp32 = np.load(QUANT_RUN / "fp32_prediction.npy").astype(np.float32)
    pre = np.load(QUANT_RUN / "quant_pre_recon_prediction.npy").astype(np.float32)
    post = np.load(QUANT_RUN / "quant_post_recon_prediction.npy").astype(np.float32)
    row = {
        "testset_id": "default_single_sample",
        "source": "SCRN-main/test_data",
        "patch_file": "clear.npy",
        "patch_index": -1,
        "condition_index": -1,
        "condition_label": "default_single_sample",
        "snr_setting_db": None,
        "missing_rate": None,
        "input_snr_db": snr_db(degraded, clean),
        "input_ssim": ssim_score(degraded, clean),
        "fp32_snr_db": snr_db(fp32, clean),
        "fp32_ssim": ssim_score(fp32, clean),
        "quant_pre_recon_snr_db": snr_db(pre, clean),
        "quant_pre_recon_ssim": ssim_score(pre, clean),
        "quant_post_recon_snr_db": snr_db(post, clean),
        "quant_post_recon_ssim": ssim_score(post, clean),
        "amplitude_space": "raw default SCRN sample",
        "metric_amplitude_space": "raw default SCRN sample",
        "normalization_scale": None,
    }
    row["quant_post_minus_fp32_snr_db"] = row["quant_post_recon_snr_db"] - row["fp32_snr_db"]
    row["quant_post_minus_pre_snr_db"] = row["quant_post_recon_snr_db"] - row["quant_pre_recon_snr_db"]
    figure_path = FIGURES_DIR / "10_default_single_sample_seismic_raw_amplitude.png"
    save_seismic(
        figure_path,
        [
            (clean, "Ground Truth"),
            (degraded, f"Input {row['input_snr_db']:.2f}dB"),
            (fp32, f"FP32 {row['fp32_snr_db']:.2f}dB"),
            (pre, f"Pre {row['quant_pre_recon_snr_db']:.2f}dB"),
            (post, f"Post {row['quant_post_recon_snr_db']:.2f}dB"),
        ],
        subtitle="Default single sample, raw amplitude, seismic centered at 0",
    )
    row["figure"] = str(figure_path)
    return row


def load_manifest_scale_map(dataset_dir: str | Path) -> tuple[dict[str, float], list[str]]:
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
    return np.asarray(image, dtype=np.float32) * np.float32(normalization_scale)


def save_seismic(path: Path, panels: list[tuple[np.ndarray, str]], *, subtitle: str) -> None:
    values = [np.asarray(image, dtype=np.float32) for image, _title in panels]
    absmax = max(float(np.nanmax(np.abs(image))) for image in values)
    if absmax <= 0:
        absmax = 1.0
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    im = None
    for ax, (image, title) in zip(axes, panels):
        im = ax.imshow(image, cmap="seismic", aspect="auto", vmin=-absmax, vmax=absmax)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(subtitle, fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.72, fraction=0.02, pad=0.01)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary(rows: list[dict]) -> None:
    lines = [
        "# Normalized W4A32 Seismic Visuals",
        "",
        f"- figure_count: `{len(rows)}`",
        "- colormap: `seismic`",
        "- color scale: symmetric per figure over all panels, centered at `0`",
        "- normalized representative panels are restored for display with `display = normalized * normalization_scale`.",
        "- metric labels keep the original normalized-protocol SNR/SSIM values.",
        "- Panels: Ground Truth, Input, FP32, W4A32 pre-recon, W4A32 post-recon.",
        "",
        "| # | source | condition | display amplitude | scale | patch | input SNR | FP32 SNR | pre SNR | post SNR | post-FP32 | figure |",
        "|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(rows, start=1):
        scale = row.get("normalization_scale")
        scale_text = "n/a" if scale is None else f"{float(scale):.6g}"
        lines.append(
            "| {idx} | {source} | {condition} | {amp} | {scale} | {patch} | {input_snr:.2f} | "
            "{fp32_snr:.2f} | {pre_snr:.2f} | {post_snr:.2f} | {delta:.3f} | {figure} |".format(
                idx=idx,
                source=row["source"],
                condition=row["condition_label"],
                amp=row["amplitude_space"],
                scale=scale_text,
                patch=row["patch_file"],
                input_snr=row["input_snr_db"],
                fp32_snr=row["fp32_snr_db"],
                pre_snr=row["quant_pre_recon_snr_db"],
                post_snr=row["quant_post_recon_snr_db"],
                delta=row["quant_post_minus_fp32_snr_db"],
                figure=Path(row["figure"]).name,
            )
        )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
