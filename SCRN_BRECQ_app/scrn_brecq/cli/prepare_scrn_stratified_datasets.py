"""Prepare stratified SCRN calibration and legacy-logic test datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from SCRN_BRECQ_app.scrn_brecq.data.stratified_scrn_datasets import (
    DEFAULT_CALIBRATION_SAMPLE_COUNT,
    DEFAULT_SEED,
    build_training_patch_hashes,
    prepare_calibration_dataset,
    prepare_test_dataset_from_segy_dir,
)


DEFAULT_TRAIN_PATCH_DIR = Path("SCRN_BRECQ_app/scrn_repro/datasets/scrn_quant_10750_0_patches")
DEFAULT_RAW_SEGY_DIR = Path("/home/data1/hanwen/project/Project/SCRN_quant/data/train")
DEFAULT_CALIBRATION_OUTPUT_DIR = Path(
    "SCRN_BRECQ_app/scrn_repro/datasets/scrn_quant_10750_0_cali_1024_stratified"
)
DEFAULT_TEST_OUTPUT_DIR = Path("SCRN_BRECQ_app/scrn_repro/datasets/scrn_quant_test_478_legacy_logic")


def build_parser() -> argparse.ArgumentParser:
    """Build the stratified SCRN dataset preparation parser."""
    parser = argparse.ArgumentParser(description="Prepare stratified SCRN calibration/test clean patch datasets.")
    parser.add_argument("--mode", choices=["calibration", "test", "both"], default="both")
    parser.add_argument("--train-patch-dir", default=str(DEFAULT_TRAIN_PATCH_DIR), help="Legacy 10750_0 clean patch dir")
    parser.add_argument("--raw-segy-dir", default=str(DEFAULT_RAW_SEGY_DIR), help="Directory containing local SEG-Y sources")
    parser.add_argument(
        "--calibration-output-dir",
        default=str(DEFAULT_CALIBRATION_OUTPUT_DIR),
        help="Output dir for 1024 stratified calibration clean patches",
    )
    parser.add_argument(
        "--test-output-dir",
        default=str(DEFAULT_TEST_OUTPUT_DIR),
        help="Output dir for 478 legacy-logic clean test patches",
    )
    parser.add_argument("--num-calibration-samples", type=int, default=DEFAULT_CALIBRATION_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true", help="Build manifests in memory without writing output files")
    parser.add_argument("--overwrite", action="store_true", help="Replace non-empty output directories")
    parser.add_argument(
        "--exclude-training-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude test candidates whose float32 array hash exactly matches a training patch",
    )
    return parser


def main() -> None:
    """Prepare requested datasets and print a compact JSON summary."""
    args = build_parser().parse_args()
    if int(args.num_calibration_samples) != DEFAULT_CALIBRATION_SAMPLE_COUNT:
        raise ValueError(
            "--num-calibration-samples currently supports only "
            f"{DEFAULT_CALIBRATION_SAMPLE_COUNT}, got {args.num_calibration_samples}"
        )

    train_patch_dir = Path(args.train_patch_dir)
    raw_segy_dir = Path(args.raw_segy_dir)
    results: dict[str, Mapping] = {}

    if args.mode in {"calibration", "both"}:
        results["calibration"] = prepare_calibration_dataset(
            train_patch_dir,
            Path(args.calibration_output_dir),
            seed=int(args.seed),
            dry_run=bool(args.dry_run),
            overwrite=bool(args.overwrite),
        )

    if args.mode in {"test", "both"}:
        train_hashes = build_training_patch_hashes(train_patch_dir) if bool(args.exclude_training_hashes) else set()
        results["test"] = prepare_test_dataset_from_segy_dir(
            raw_segy_dir,
            Path(args.test_output_dir),
            train_hashes=train_hashes,
            seed=int(args.seed),
            dry_run=bool(args.dry_run),
            overwrite=bool(args.overwrite),
        )

    print(json.dumps(_compact_results(results), indent=2, sort_keys=True, ensure_ascii=False), flush=True)


def _compact_results(results: Mapping[str, Mapping]) -> dict:
    compact: dict[str, dict] = {}
    for name, manifest in results.items():
        compact[name] = {
            "dataset_type": manifest.get("dataset_type"),
            "sample_count": manifest.get("sample_count"),
            "seed": manifest.get("seed"),
            "per_source_counts": manifest.get("per_source_counts"),
            "training_hash_excluded_count": manifest.get("training_hash_excluded_count"),
        }
    return compact


if __name__ == "__main__":
    main()
