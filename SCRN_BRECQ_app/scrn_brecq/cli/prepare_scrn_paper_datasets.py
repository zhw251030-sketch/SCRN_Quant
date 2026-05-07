"""Prepare SCRN paper-style train, calibration, and test clean patch datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from SCRN_BRECQ_app.scrn_brecq.data.paper_scrn_datasets import (
    DEFAULT_CALIBRATION_SAMPLE_COUNT,
    DEFAULT_SEED,
    build_training_patch_hashes,
    prepare_calibration_dataset,
    prepare_test_dataset_from_segy_dir,
    prepare_train_dataset_from_segy_dir,
)


DEFAULT_RAW_SEGY_DIR = Path("/home/data1/hanwen/project/Project/SCRN_quant/data/train")
DEFAULT_TRAIN_OUTPUT_DIR = Path("SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_train_10750")
DEFAULT_CALIBRATION_OUTPUT_DIR = Path("SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_cali_1024_stratified")
DEFAULT_TEST_OUTPUT_DIR = Path("SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_test_478")


def build_parser() -> argparse.ArgumentParser:
    """Build the SCRN paper-style dataset preparation parser."""
    parser = argparse.ArgumentParser(description="Prepare SCRN paper-style clean patch datasets.")
    parser.add_argument("--mode", choices=["train", "calibration", "test", "all"], default="all")
    parser.add_argument("--raw-segy-dir", default=str(DEFAULT_RAW_SEGY_DIR), help="Directory containing local SEG-Y sources")
    parser.add_argument("--train-output-dir", default=str(DEFAULT_TRAIN_OUTPUT_DIR), help="Output dir for 10750 train patches")
    parser.add_argument(
        "--calibration-output-dir",
        default=str(DEFAULT_CALIBRATION_OUTPUT_DIR),
        help="Output dir for 1024 stratified calibration patches",
    )
    parser.add_argument("--test-output-dir", default=str(DEFAULT_TEST_OUTPUT_DIR), help="Output dir for 478 test patches")
    parser.add_argument("--num-calibration-samples", type=int, default=DEFAULT_CALIBRATION_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true", help="Build manifests in memory without writing output files")
    parser.add_argument("--overwrite", action="store_true", help="Replace non-empty output directories")
    parser.add_argument(
        "--exclude-training-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude test candidates whose float32 patch hash exactly matches a new train patch",
    )
    return parser


def main() -> None:
    """Prepare requested paper-style datasets and print a compact JSON summary."""
    args = build_parser().parse_args()
    if int(args.num_calibration_samples) != DEFAULT_CALIBRATION_SAMPLE_COUNT:
        raise ValueError(
            "--num-calibration-samples currently supports only "
            f"{DEFAULT_CALIBRATION_SAMPLE_COUNT}, got {args.num_calibration_samples}"
        )

    raw_segy_dir = Path(args.raw_segy_dir)
    train_output_dir = Path(args.train_output_dir)
    calibration_output_dir = Path(args.calibration_output_dir)
    test_output_dir = Path(args.test_output_dir)
    results: dict[str, Mapping] = {}

    if args.mode in {"train", "all"}:
        results["train"] = prepare_train_dataset_from_segy_dir(
            raw_segy_dir,
            train_output_dir,
            seed=int(args.seed),
            dry_run=bool(args.dry_run),
            overwrite=bool(args.overwrite),
        )

    if args.mode in {"calibration", "all"}:
        train_manifest = results.get("train") if bool(args.dry_run) and "train" in results else None
        results["calibration"] = prepare_calibration_dataset(
            train_output_dir,
            calibration_output_dir,
            train_manifest=train_manifest,
            seed=int(args.seed),
            dry_run=bool(args.dry_run),
            overwrite=bool(args.overwrite),
        )

    if args.mode in {"test", "all"}:
        if not bool(args.exclude_training_hashes):
            train_hashes = set()
        elif bool(args.dry_run) and "train" in results:
            train_hashes = {str(sample["sha256"]) for sample in results["train"].get("samples", [])}
        else:
            train_hashes = build_training_patch_hashes(train_output_dir)
        results["test"] = prepare_test_dataset_from_segy_dir(
            raw_segy_dir,
            test_output_dir,
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
