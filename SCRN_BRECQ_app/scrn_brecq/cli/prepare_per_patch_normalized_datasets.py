"""Prepare per-patch absmax-normalized SCRN clean patch datasets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

from SCRN_BRECQ_app.scrn_brecq.data.per_patch_normalization import (
    DEFAULT_NORMALIZATION_EPS,
    prepare_per_patch_absmax_calibration_dataset,
    prepare_per_patch_absmax_dataset,
)


@dataclass(frozen=True)
class NormalizationFamily:
    """One source dataset family and its per-patch normalized outputs."""

    family_id: str
    input_train_dir: Path
    input_calibration_dir: Path
    input_test_dir: Path
    output_train_dir: Path
    output_calibration_dir: Path
    output_test_dir: Path
    train_dataset_type: str
    calibration_dataset_type: str
    test_dataset_type: str


PAPER5_FAMILY = NormalizationFamily(
    family_id="paper5",
    input_train_dir=Path("SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_train_10750"),
    input_calibration_dir=Path("SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_cali_1024_stratified"),
    input_test_dir=Path("SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_test_478"),
    output_train_dir=Path("SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_perpatch_absmax_train_10750"),
    output_calibration_dir=Path("SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_perpatch_absmax_cali_1024_stratified"),
    output_test_dir=Path("SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_perpatch_absmax_test_478"),
    train_dataset_type="paper_style_train_perpatch_absmax",
    calibration_dataset_type="paper_style_calibration_perpatch_absmax",
    test_dataset_type="paper_style_test_perpatch_absmax",
)

ENERGY_FILTERED_FAMILY = NormalizationFamily(
    family_id="paper5_energy_filtered",
    input_train_dir=Path("SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_train_10750"),
    input_calibration_dir=Path("SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_cali_1024_stratified"),
    input_test_dir=Path("SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_test_478"),
    output_train_dir=Path("SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_train_10750"),
    output_calibration_dir=Path(
        "SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified"
    ),
    output_test_dir=Path("SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478"),
    train_dataset_type="paper_style_energy_filtered_train_perpatch_absmax",
    calibration_dataset_type="paper_style_energy_filtered_calibration_perpatch_absmax",
    test_dataset_type="paper_style_energy_filtered_test_perpatch_absmax",
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Prepare per-patch absmax-normalized SCRN clean patch datasets.")
    parser.add_argument(
        "--preset",
        choices=["paper5-and-energy-filtered"],
        default="paper5-and-energy-filtered",
        help="Default preset generates both paper5 and paper5 energy-filtered derivatives.",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "paper5", "energy-filtered"],
        default="all",
        help="Subset of dataset families to generate.",
    )
    parser.add_argument("--eps", type=float, default=DEFAULT_NORMALIZATION_EPS)
    parser.add_argument("--dry-run", action="store_true", help="Build manifests in memory without writing output files")
    parser.add_argument("--overwrite", action="store_true", help="Replace non-empty output directories")
    return parser


def main() -> None:
    """Prepare requested normalized dataset families and print a compact JSON summary."""
    args = build_parser().parse_args()
    families = list(_selected_families(args.mode))
    results: dict[str, dict[str, Mapping]] = {}

    for family in families:
        train_manifest = prepare_per_patch_absmax_dataset(
            family.input_train_dir,
            family.output_train_dir,
            dataset_type=family.train_dataset_type,
            eps=float(args.eps),
            dry_run=bool(args.dry_run),
            overwrite=bool(args.overwrite),
        )
        calibration_manifest = prepare_per_patch_absmax_calibration_dataset(
            family.input_calibration_dir,
            family.output_train_dir,
            family.output_calibration_dir,
            dataset_type=family.calibration_dataset_type,
            normalized_train_manifest=train_manifest if bool(args.dry_run) else None,
            eps=float(args.eps),
            dry_run=bool(args.dry_run),
            overwrite=bool(args.overwrite),
        )
        test_manifest = prepare_per_patch_absmax_dataset(
            family.input_test_dir,
            family.output_test_dir,
            dataset_type=family.test_dataset_type,
            eps=float(args.eps),
            dry_run=bool(args.dry_run),
            overwrite=bool(args.overwrite),
        )
        results[family.family_id] = {
            "train": train_manifest,
            "calibration": calibration_manifest,
            "test": test_manifest,
        }

    print(json.dumps(_compact_results(results), indent=2, sort_keys=True, ensure_ascii=False), flush=True)


def _selected_families(mode: str) -> Iterable[NormalizationFamily]:
    if mode == "paper5":
        return (PAPER5_FAMILY,)
    if mode == "energy-filtered":
        return (ENERGY_FILTERED_FAMILY,)
    return (PAPER5_FAMILY, ENERGY_FILTERED_FAMILY)


def _compact_results(results: Mapping[str, Mapping[str, Mapping]]) -> dict:
    compact: dict[str, dict[str, dict]] = {}
    for family_id, manifests in results.items():
        compact[family_id] = {}
        for split, manifest in manifests.items():
            compact[family_id][split] = {
                "dataset_type": manifest.get("dataset_type"),
                "sample_count": manifest.get("sample_count"),
                "zero_or_tiny_scale_count": manifest.get("zero_or_tiny_scale_count"),
                "scale_summary": manifest.get("scale_summary"),
                "per_source_counts": manifest.get("per_source_counts"),
            }
    return compact


if __name__ == "__main__":
    main()
