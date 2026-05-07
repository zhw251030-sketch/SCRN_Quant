"""Paper-style SCRN 5-source dataset preparation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from typing import Mapping, Sequence

import numpy as np

from SCRN_BRECQ_app.scrn_repro.data.patches import augment_patch


DEFAULT_SEED = 20260507
DEFAULT_PATCH_SIZE = (128, 128)
DEFAULT_STRIDE = (48, 48)
DEFAULT_MIN_STD = 1e-3
DEFAULT_AUGMENT_TIMES = 4
DEFAULT_CALIBRATION_SAMPLE_COUNT = 1024


@dataclass(frozen=True)
class PaperTrainSource:
    """One locally available SCRN paper-style training source."""

    name: str
    filename: str
    kind: str
    samples: int
    traces: int
    train_shots: int
    final_patches: int


@dataclass(frozen=True)
class PaperTestSource:
    """One locally available SCRN paper-style test source."""

    name: str
    filename: str
    kind: str
    samples: int
    traces: int
    quota: int
    train_source_name: str
    train_shots_before_test: int = 0
    trace_start: int = 0


@dataclass(frozen=True)
class PaperPatch:
    """A clean patch extracted from a deterministic paper-style region."""

    data: np.ndarray
    top: int
    left: int
    sha256: str


@dataclass(frozen=True)
class SelectedPaperCalibrationPatch:
    """A selected calibration patch from a paper-style training manifest."""

    source: str
    train_file: str
    path: Path
    sha256: str
    manifest_index: int


PAPER_TRAIN_SOURCES: tuple[PaperTrainSource, ...] = (
    PaperTrainSource("1997_2.5D_shots", "1997_2.5D_shots.segy", "shot_gather", 200, 256, 10, 300),
    PaperTrainSource("7m_shots_0201", "7m_shots_0201_0329.segy", "shot_gather", 650, 3008, 1, 3355),
    PaperTrainSource(
        "Anisotropic_FD_Model",
        "Anisotropic_FD_Model_Shots_part1.sgy",
        "shot_gather",
        350,
        800,
        2,
        750,
    ),
    PaperTrainSource("Kerry3D", "Kerry3D.segy", "full_matrix", 1252, 287, 1, 480),
    PaperTrainSource("Shots0001_0200", "shots0001_0200.segy", "shot_gather", 900, 1201, 3, 5865),
)

PAPER_TEST_SOURCES: tuple[PaperTestSource, ...] = (
    PaperTestSource(
        "Anisotropic",
        "Anisotropic_FD_Model_Shots_part1.sgy",
        "shot_gather",
        350,
        800,
        75,
        train_source_name="Anisotropic_FD_Model",
        train_shots_before_test=2,
    ),
    PaperTestSource(
        "Kerry3D",
        "Kerry3D.segy",
        "full_matrix",
        1252,
        128,
        16,
        train_source_name="Kerry3D",
        trace_start=287,
    ),
    PaperTestSource(
        "Shots0001",
        "shots0001_0200.segy",
        "shot_gather",
        900,
        1201,
        387,
        train_source_name="Shots0001_0200",
        train_shots_before_test=3,
    ),
)

DEFAULT_TRAIN_QUOTAS = {source.name: source.final_patches for source in PAPER_TRAIN_SOURCES}
DEFAULT_TEST_QUOTAS = {source.name: source.quota for source in PAPER_TEST_SOURCES}
DEFAULT_CALIBRATION_QUOTAS = {
    "1997_2.5D_shots": 28,
    "7m_shots_0201": 320,
    "Anisotropic_FD_Model": 71,
    "Kerry3D": 46,
    "Shots0001_0200": 559,
}


def compute_sliding_window_count(
    height: int,
    width: int,
    *,
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE,
    stride: tuple[int, int] = DEFAULT_STRIDE,
) -> int:
    """Return the deterministic sliding-window count for a matrix."""
    patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
    stride_h, stride_w = int(stride[0]), int(stride[1])
    if patch_h <= 0 or patch_w <= 0 or stride_h <= 0 or stride_w <= 0:
        raise ValueError("patch_size and stride values must be positive")
    if int(height) < patch_h or int(width) < patch_w:
        return 0
    count_h = ((int(height) - patch_h) // stride_h) + 1
    count_w = ((int(width) - patch_w) // stride_w) + 1
    return int(count_h * count_w)


def compute_source_patch_counts(
    source: PaperTrainSource,
    *,
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE,
    stride: tuple[int, int] = DEFAULT_STRIDE,
    augment_times: int = DEFAULT_AUGMENT_TIMES,
) -> tuple[int, int]:
    """Return `(raw_patch_count, final_patch_count)` for a paper table source."""
    raw_per_region = compute_sliding_window_count(
        source.samples,
        source.traces,
        patch_size=patch_size,
        stride=stride,
    )
    region_count = int(source.train_shots) if source.kind == "shot_gather" else 1
    raw_count = raw_per_region * region_count
    return raw_count, raw_count * (int(augment_times) + 1)


def select_source_matrices_from_array(
    source_data: Sequence[np.ndarray] | np.ndarray,
    source: PaperTrainSource | PaperTestSource,
    *,
    samples: int | None = None,
    traces: int | None = None,
    shot_count: int | None = None,
    trace_start: int = 0,
    normalize: bool = True,
) -> list[np.ndarray]:
    """Select deterministic paper-style source regions from in-memory arrays."""
    sample_count = int(samples if samples is not None else source.samples)
    trace_count = int(traces if traces is not None else source.traces)
    selected: list[np.ndarray] = []

    if source.kind == "shot_gather":
        count = int(shot_count if shot_count is not None else getattr(source, "train_shots", 1))
        if isinstance(source_data, np.ndarray) and source_data.ndim == 3:
            shot_matrices = [source_data[index] for index in range(source_data.shape[0])]
        elif isinstance(source_data, np.ndarray) and source_data.ndim == 2 and count == 1:
            shot_matrices = [source_data]
        else:
            shot_matrices = list(source_data)  # type: ignore[arg-type]
        if len(shot_matrices) < count:
            raise ValueError(f"{source.name} requires {count} shot matrices, got {len(shot_matrices)}")
        for matrix in shot_matrices[:count]:
            selected.append(_crop_and_normalize(matrix, sample_count, trace_count, trace_start=0, normalize=normalize))
        return selected

    if source.kind == "full_matrix":
        matrix = np.asarray(source_data, dtype=np.float32)
        selected.append(
            _crop_and_normalize(
                matrix,
                sample_count,
                trace_count,
                trace_start=int(trace_start),
                normalize=normalize,
            )
        )
        return selected

    raise ValueError(f"Unsupported source kind for {source.name}: {source.kind}")


def apply_augmentation(
    patch: np.ndarray,
    *,
    augment_times: int = DEFAULT_AUGMENT_TIMES,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Return original patch plus seeded random geometric augmentations."""
    return [data for _, data in _apply_augmentation_with_modes(patch, augment_times=augment_times, rng=rng)]


def extract_paper_patches_from_array(
    data: np.ndarray,
    *,
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE,
    stride: tuple[int, int] = DEFAULT_STRIDE,
    min_std: float | None = DEFAULT_MIN_STD,
) -> list[PaperPatch]:
    """Extract clean paper-style patches from one selected source matrix."""
    source = np.asarray(data, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {source.shape}")
    patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
    stride_h, stride_w = int(stride[0]), int(stride[1])
    patches: list[PaperPatch] = []
    for top in range(0, source.shape[0] - patch_h + 1, stride_h):
        for left in range(0, source.shape[1] - patch_w + 1, stride_w):
            patch = np.ascontiguousarray(source[top : top + patch_h, left : left + patch_w], dtype=np.float32)
            if patch.shape != (patch_h, patch_w):
                continue
            if min_std is not None and (float(np.sum(patch)) == 0.0 or float(patch.std()) <= float(min_std)):
                continue
            patches.append(PaperPatch(data=patch, top=top, left=left, sha256=sha256_array(patch)))
    return patches


def prepare_train_dataset_from_arrays(
    source_matrices: Mapping[str, Sequence[np.ndarray] | np.ndarray],
    output_dir: str | Path,
    *,
    sources: Sequence[PaperTrainSource] = PAPER_TRAIN_SOURCES,
    source_overrides: Mapping[str, Mapping[str, int]] | None = None,
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE,
    stride: tuple[int, int] = DEFAULT_STRIDE,
    min_std: float | None = None,
    augment_times: int = DEFAULT_AUGMENT_TIMES,
    seed: int = DEFAULT_SEED,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict:
    """Write the paper-style 5-source training clean patch directory."""
    overrides = source_overrides or {}
    rng = np.random.default_rng(int(seed))
    selected_items: list[tuple[PaperTrainSource, int, PaperPatch, int, int, np.ndarray]] = []
    per_source_counts: dict[str, int] = {}
    per_source_raw_counts: dict[str, int] = {}

    for source in sources:
        if source.name not in source_matrices:
            raise ValueError(f"Missing source matrix for {source.name}")
        override = overrides.get(source.name, {})
        sample_count = int(override.get("samples", source.samples))
        trace_count = int(override.get("traces", source.traces))
        shot_count = int(override.get("train_shots", source.train_shots))
        matrices = select_source_matrices_from_array(
            source_matrices[source.name],
            source,
            samples=sample_count,
            traces=trace_count,
            shot_count=shot_count,
            normalize=True,
        )

        raw_count = 0
        final_count = 0
        for region_index, matrix in enumerate(matrices):
            raw_patches = extract_paper_patches_from_array(
                matrix,
                patch_size=patch_size,
                stride=stride,
                min_std=min_std,
            )
            raw_count += len(raw_patches)
            for raw_patch in raw_patches:
                for augmentation_index, (mode, augmented) in enumerate(
                    _apply_augmentation_with_modes(raw_patch.data, augment_times=augment_times, rng=rng)
                ):
                    selected_items.append((source, region_index, raw_patch, augmentation_index, mode, augmented))
                    final_count += 1
        per_source_raw_counts[source.name] = raw_count
        per_source_counts[source.name] = final_count

    if _should_validate_default_train_counts(sources, overrides, patch_size, stride, augment_times):
        for source in sources:
            actual = per_source_counts.get(source.name, 0)
            if actual != source.final_patches:
                raise ValueError(
                    f"{source.name} generated {actual} final patches, "
                    f"but the paper-style target requires {source.final_patches}."
                )

    manifest = _base_manifest(
        dataset_type="paper_style_train",
        seed=seed,
        sample_count=len(selected_items),
        quotas=per_source_counts,
    )
    manifest.update(
        {
            "source_protocol": "paper_style_table2_5_available_sources_deterministic_regions",
            "patch_size": list(patch_size),
            "stride": list(stride),
            "min_std": None if min_std is None else float(min_std),
            "augment_times": int(augment_times),
            "normalization": "absmax_after_region_crop",
            "per_source_raw_patch_counts": per_source_raw_counts,
            "paper_table_sources": [_train_source_dict(source) for source in sources],
            "samples": [
                {
                    "output_file": f"train_{index:06d}.npy",
                    "source": source.name,
                    "source_file": source.filename,
                    "region_index": int(region_index),
                    "top": int(raw_patch.top),
                    "left": int(raw_patch.left),
                    "augmentation_index": int(augmentation_index),
                    "augmentation_mode": int(mode),
                    "sha256": sha256_array(augmented),
                }
                for index, (source, region_index, raw_patch, augmentation_index, mode, augmented) in enumerate(
                    selected_items,
                    start=1,
                )
            ],
        }
    )

    if dry_run:
        return manifest

    output = _prepare_output_dir(output_dir, overwrite=overwrite)
    for index, (_, _, _, _, _, augmented) in enumerate(selected_items, start=1):
        np.save(output / f"train_{index:06d}.npy", np.asarray(augmented, dtype=np.float32))
    _write_json(output / "manifest.json", manifest)
    _write_readme(
        output / "README.md",
        title="SCRN Paper-Style 5-Source 10750 Training Clean Patches",
        body=(
            "This directory follows a deterministic paper-style protocol for the five locally "
            "available SCRN Table 2 training sources. It is not an exact reconstruction of the "
            "authors' private split because the paper does not publish shot IDs or spatial offsets."
        ),
        manifest=manifest,
    )
    return manifest


def prepare_train_dataset_from_segy_dir(
    segy_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict:
    """Prepare the paper-style training dataset from local SEG-Y files."""
    root = Path(segy_dir)
    matrices: dict[str, Sequence[np.ndarray] | np.ndarray] = {}
    for source in PAPER_TRAIN_SOURCES:
        path = root / source.filename
        if not path.exists():
            raise FileNotFoundError(f"SEG-Y source for {source.name} does not exist: {path}")
        if source.kind == "shot_gather":
            matrices[source.name] = read_shot_matrices_from_segy(
                path,
                shot_indices=list(range(source.train_shots)),
                samples=source.samples,
                traces=source.traces,
                normalize=True,
            )
        else:
            matrices[source.name] = read_matrix_window_from_segy(
                path,
                samples=source.samples,
                traces=source.traces,
                trace_start=0,
                normalize=True,
            )

    manifest = prepare_train_dataset_from_arrays(
        matrices,
        output_dir,
        seed=seed,
        dry_run=dry_run,
        overwrite=overwrite,
    )
    manifest["raw_segy_dir"] = str(root)
    if not dry_run:
        _write_json(Path(output_dir) / "manifest.json", manifest)
    return manifest


def select_stratified_calibration_from_manifest(
    train_manifest: Mapping,
    train_dir: str | Path,
    *,
    quotas: Mapping[str, int] | None = None,
    seed: int = DEFAULT_SEED,
) -> list[SelectedPaperCalibrationPatch]:
    """Select calibration patches by source from a paper-style training manifest."""
    selected_quotas = dict(DEFAULT_CALIBRATION_QUOTAS if quotas is None else quotas)
    samples = list(train_manifest.get("samples", []))
    grouped: dict[str, list[SelectedPaperCalibrationPatch]] = {source: [] for source in selected_quotas}
    root = Path(train_dir)
    for manifest_index, sample in enumerate(samples):
        source = str(sample.get("source", ""))
        if source not in grouped:
            continue
        train_file = str(sample["output_file"])
        grouped[source].append(
            SelectedPaperCalibrationPatch(
                source=source,
                train_file=train_file,
                path=root / train_file,
                sha256=str(sample.get("sha256", "")),
                manifest_index=manifest_index,
            )
        )

    rng = np.random.default_rng(int(seed))
    selected: list[SelectedPaperCalibrationPatch] = []
    for source_name in _ordered_train_sources(selected_quotas):
        quota = int(selected_quotas.get(source_name, 0))
        if quota <= 0:
            continue
        candidates = grouped.get(source_name, [])
        if len(candidates) < quota:
            raise ValueError(
                f"Source {source_name} has only {len(candidates)} candidate train patches, "
                f"but calibration quota requires {quota}."
            )
        positions = rng.choice(len(candidates), size=quota, replace=False)
        source_selected = sorted((candidates[int(position)] for position in positions), key=lambda item: item.train_file)
        selected.extend(source_selected)
    return selected


def prepare_calibration_dataset(
    train_dir: str | Path,
    output_dir: str | Path,
    *,
    train_manifest: Mapping | str | Path | None = None,
    quotas: Mapping[str, int] | None = None,
    seed: int = DEFAULT_SEED,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict:
    """Write a paper-style stratified calibration clean patch directory."""
    root = Path(train_dir)
    manifest_data = _load_manifest(train_manifest if train_manifest is not None else root / "manifest.json")
    selected = select_stratified_calibration_from_manifest(manifest_data, root, quotas=quotas, seed=seed)
    selected_quotas = dict(DEFAULT_CALIBRATION_QUOTAS if quotas is None else quotas)
    manifest = _base_manifest(
        dataset_type="paper_style_calibration",
        seed=seed,
        sample_count=len(selected),
        quotas=selected_quotas,
    )
    manifest.update(
        {
            "input_train_dir": str(root),
            "source_protocol": "paper_style_train_manifest_stratified_sampling",
            "samples": [
                {
                    "output_file": f"cali_{index:06d}.npy",
                    "source": item.source,
                    "train_file": item.train_file,
                    "train_manifest_index": int(item.manifest_index),
                    "sha256": item.sha256 or sha256_npy_array(item.path),
                }
                for index, item in enumerate(selected, start=1)
            ],
        }
    )

    if dry_run:
        return manifest

    output = _prepare_output_dir(output_dir, overwrite=overwrite)
    for index, item in enumerate(selected, start=1):
        if not item.path.exists():
            raise FileNotFoundError(f"Selected train patch does not exist: {item.path}")
        shutil.copy2(item.path, output / f"cali_{index:06d}.npy")
    _write_json(output / "manifest.json", manifest)
    _write_readme(
        output / "README.md",
        title="SCRN Paper-Style 1024 Stratified Calibration Clean Patches",
        body=(
            "This directory is stratified from the paper-style 5-source training clean patch set. "
            "The clean targets are intended for SCRN-BRECQ calibration pipelines that generate "
            "degraded inputs online."
        ),
        manifest=manifest,
    )
    return manifest


def prepare_test_dataset_from_arrays(
    source_matrices: Mapping[str, Sequence[np.ndarray] | np.ndarray],
    output_dir: str | Path,
    *,
    quotas: Mapping[str, int] | None = None,
    train_hashes: set[str] | None = None,
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE,
    stride: tuple[int, int] = DEFAULT_STRIDE,
    min_std: float | None = None,
    seed: int = DEFAULT_SEED,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict:
    """Write the paper-style 478 clean test patch directory from selected regions."""
    selected_quotas = dict(DEFAULT_TEST_QUOTAS if quotas is None else quotas)
    excluded_hashes = train_hashes or set()
    rng = np.random.default_rng(int(seed))
    selected_items: list[tuple[str, int, PaperPatch]] = []
    per_source_counts: dict[str, int] = {}
    per_source_candidates: dict[str, int] = {}
    per_source_excluded: dict[str, int] = {}
    per_source_regions: dict[str, int] = {}

    for source_name in _ordered_test_sources(selected_quotas):
        if source_name not in source_matrices:
            raise ValueError(f"Missing test source matrix for {source_name}")
        quota = int(selected_quotas[source_name])
        all_candidates: list[tuple[int, PaperPatch]] = []
        region_matrices = _as_region_matrices(source_matrices[source_name])
        per_source_regions[source_name] = len(region_matrices)
        for region_index, matrix in enumerate(region_matrices):
            region_candidates = extract_paper_patches_from_array(
                matrix,
                patch_size=patch_size,
                stride=stride,
                min_std=min_std,
            )
            all_candidates.extend((region_index, patch) for patch in region_candidates)
        kept: list[tuple[int, PaperPatch]] = []
        excluded_count = 0
        for region_index, patch in all_candidates:
            if patch.sha256 in excluded_hashes:
                excluded_count += 1
                continue
            kept.append((region_index, patch))
        if len(kept) < quota:
            raise ValueError(
                f"{source_name} has not enough candidate patches after filtering: "
                f"need {quota}, got {len(kept)}."
            )
        if len(kept) > quota:
            positions = rng.choice(len(kept), size=quota, replace=False)
            kept = [kept[int(position)] for position in positions]
        kept = sorted(kept, key=lambda item: (item[0], item[1].top, item[1].left))
        selected_items.extend((source_name, region_index, patch) for region_index, patch in kept)
        per_source_counts[source_name] = quota
        per_source_candidates[source_name] = len(all_candidates)
        per_source_excluded[source_name] = excluded_count

    manifest = _base_manifest(
        dataset_type="paper_style_test",
        seed=seed,
        sample_count=len(selected_items),
        quotas=selected_quotas,
    )
    manifest.update(
        {
            "source_protocol": "paper_style_table3_deterministic_non_overlapping_regions_without_augmentation",
            "patch_size": list(patch_size),
            "stride": list(stride),
            "min_std": None if min_std is None else float(min_std),
            "per_source_candidate_counts": per_source_candidates,
            "per_source_training_hash_excluded_counts": per_source_excluded,
            "per_source_region_counts": per_source_regions,
            "training_hash_excluded_count": int(sum(per_source_excluded.values())),
            "samples": [
                {
                    "output_file": f"test_{index:06d}.npy",
                    "source": source_name,
                    "source_file": _test_source_filename(source_name),
                    "region_index": int(region_index),
                    "top": int(patch.top),
                    "left": int(patch.left),
                    "sha256": patch.sha256,
                }
                for index, (source_name, region_index, patch) in enumerate(selected_items, start=1)
            ],
        }
    )

    if dry_run:
        return manifest

    output = _prepare_output_dir(output_dir, overwrite=overwrite)
    for index, (_, _, patch) in enumerate(selected_items, start=1):
        np.save(output / f"test_{index:06d}.npy", np.asarray(patch.data, dtype=np.float32))
    _write_json(output / "manifest.json", manifest)
    _write_readme(
        output / "README.md",
        title="SCRN Paper-Style 478 Clean Test Patches",
        body=(
            "This directory uses deterministic non-overlapping regions for the three locally "
            "available SCRN Table 3 test sources. Test patches are not augmented."
        ),
        manifest=manifest,
    )
    return manifest


def prepare_test_dataset_from_segy_dir(
    segy_dir: str | Path,
    output_dir: str | Path,
    *,
    train_hashes: set[str] | None = None,
    seed: int = DEFAULT_SEED,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict:
    """Prepare the paper-style test dataset from local SEG-Y files."""
    root = Path(segy_dir)
    matrices: dict[str, Sequence[np.ndarray] | np.ndarray] = {}
    excluded_hashes = train_hashes or set()
    for source in PAPER_TEST_SOURCES:
        path = root / source.filename
        if not path.exists():
            raise FileNotFoundError(f"SEG-Y source for {source.name} does not exist: {path}")
        if source.kind == "shot_gather":
            matrices[source.name] = read_consecutive_test_shots_for_quota(
                path,
                source=source,
                train_hashes=excluded_hashes,
            )
        else:
            matrices[source.name] = read_consecutive_matrix_windows_for_quota(
                path,
                source=source,
                train_hashes=excluded_hashes,
            )

    manifest = prepare_test_dataset_from_arrays(
        matrices,
        output_dir,
        train_hashes=excluded_hashes,
        seed=seed,
        dry_run=dry_run,
        overwrite=overwrite,
    )
    manifest["raw_segy_dir"] = str(root)
    manifest["paper_test_sources"] = [_test_source_dict(source) for source in PAPER_TEST_SOURCES]
    if not dry_run:
        _write_json(Path(output_dir) / "manifest.json", manifest)
    return manifest


def read_consecutive_test_shots_for_quota(
    path: str | Path,
    *,
    source: PaperTestSource,
    train_hashes: set[str] | None = None,
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE,
    stride: tuple[int, int] = DEFAULT_STRIDE,
    min_std: float | None = None,
) -> list[np.ndarray]:
    """Read consecutive post-training shots until hash-filtered candidates meet the quota."""
    excluded_hashes = train_hashes or set()
    selected: list[np.ndarray] = []
    kept_count = 0
    shot_index = int(source.train_shots_before_test)
    while kept_count < int(source.quota):
        shot = read_shot_matrices_from_segy(
            path,
            shot_indices=[shot_index],
            samples=source.samples,
            traces=source.traces,
            normalize=True,
        )[0]
        selected.append(shot)
        for patch in extract_paper_patches_from_array(shot, patch_size=patch_size, stride=stride, min_std=min_std):
            if patch.sha256 not in excluded_hashes:
                kept_count += 1
        shot_index += 1
    return selected


def read_consecutive_matrix_windows_for_quota(
    path: str | Path,
    *,
    source: PaperTestSource,
    train_hashes: set[str] | None = None,
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE,
    stride: tuple[int, int] = DEFAULT_STRIDE,
    min_std: float | None = None,
) -> list[np.ndarray]:
    """Read consecutive trace windows until hash-filtered candidates meet the quota."""
    excluded_hashes = train_hashes or set()
    selected: list[np.ndarray] = []
    kept_count = 0
    trace_start = int(source.trace_start)
    while kept_count < int(source.quota):
        matrix = read_matrix_window_from_segy(
            path,
            samples=source.samples,
            traces=source.traces,
            trace_start=trace_start,
            normalize=True,
        )
        selected.append(matrix)
        for patch in extract_paper_patches_from_array(matrix, patch_size=patch_size, stride=stride, min_std=min_std):
            if patch.sha256 not in excluded_hashes:
                kept_count += 1
        trace_start += int(source.traces)
    return selected


def build_training_patch_hashes(patch_dir: str | Path) -> set[str]:
    """Build SHA-256 hashes for every `.npy` patch in a training directory."""
    hashes: set[str] = set()
    for path in sorted(Path(patch_dir).glob("*.npy")):
        hashes.add(sha256_npy_array(path))
    return hashes


def read_shot_matrices_from_segy(
    path: str | Path,
    *,
    shot_indices: Sequence[int],
    samples: int,
    traces: int,
    normalize: bool = True,
) -> list[np.ndarray]:
    """Read selected SourceX-style shot matrices as `[samples, traces]` crops."""
    try:
        import segyio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Reading SEG-Y files requires installing segyio.") from exc

    selected: list[np.ndarray] = []
    with segyio.open(str(path), "r", ignore_geometry=True) as segy_file:
        segy_file.mmap()
        source_x = segy_file.attributes(segyio.TraceField.SourceX)[:]
        trace_num = len(source_x)
        shot_num = len(set(source_x))
        traces_per_shot = trace_num if trace_num == shot_num or shot_num <= 1 else trace_num // shot_num
        for shot_index in shot_indices:
            start = int(shot_index) * traces_per_shot
            stop = start + traces_per_shot
            if stop > trace_num:
                raise ValueError(f"Shot index {shot_index} exceeds trace count for {path}")
            data = np.asarray([np.copy(trace) for trace in segy_file.trace[start:stop]], dtype=np.float32).T
            selected.append(_crop_and_normalize(data, int(samples), int(traces), normalize=normalize))
    return selected


def read_matrix_window_from_segy(
    path: str | Path,
    *,
    samples: int,
    traces: int,
    trace_start: int = 0,
    normalize: bool = True,
) -> np.ndarray:
    """Read a trace window from a full-matrix SEG-Y source as `[samples, traces]`."""
    try:
        import segyio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Reading SEG-Y files requires installing segyio.") from exc

    with segyio.open(str(path), "r", ignore_geometry=True) as segy_file:
        segy_file.mmap()
        trace_stop = int(trace_start) + int(traces)
        if trace_stop > len(segy_file.trace):
            raise ValueError(f"Trace window [{trace_start}, {trace_stop}) exceeds trace count for {path}")
        data = np.asarray([np.copy(trace) for trace in segy_file.trace[int(trace_start) : trace_stop]], dtype=np.float32).T
    return _crop_and_normalize(data, int(samples), int(traces), normalize=normalize)


def normalize_by_absmax(data: np.ndarray) -> np.ndarray:
    """Normalize by the maximum absolute amplitude."""
    array = np.asarray(data, dtype=np.float32)
    scale = float(np.max(np.abs(array)))
    if scale < 1e-12:
        return array
    return array / scale


def sha256_npy_array(path: str | Path) -> str:
    """Hash the float32 array content stored in a `.npy` file."""
    return sha256_array(np.load(path).astype(np.float32, copy=False))


def sha256_array(array: np.ndarray) -> str:
    """Hash an array after converting it to contiguous float32 bytes."""
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _apply_augmentation_with_modes(
    patch: np.ndarray,
    *,
    augment_times: int,
    rng: np.random.Generator | None,
) -> list[tuple[int, np.ndarray]]:
    rng = rng or np.random.default_rng()
    base = np.ascontiguousarray(patch, dtype=np.float32)
    augmented: list[tuple[int, np.ndarray]] = [(0, base)]
    for _ in range(int(augment_times)):
        mode = int(rng.integers(0, 8))
        augmented.append((mode, augment_patch(base, mode)))
    return augmented


def _as_region_matrices(source_data: Sequence[np.ndarray] | np.ndarray) -> list[np.ndarray]:
    if isinstance(source_data, np.ndarray) and source_data.ndim == 2:
        return [source_data]
    if isinstance(source_data, np.ndarray) and source_data.ndim == 3:
        return [np.asarray(source_data[index], dtype=np.float32) for index in range(source_data.shape[0])]
    return [np.asarray(matrix, dtype=np.float32) for matrix in source_data]  # type: ignore[arg-type]


def _crop_and_normalize(
    data: np.ndarray,
    samples: int,
    traces: int,
    *,
    trace_start: int = 0,
    normalize: bool = True,
) -> np.ndarray:
    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D source matrix, got shape {array.shape}")
    trace_stop = int(trace_start) + int(traces)
    crop = np.ascontiguousarray(array[: int(samples), int(trace_start) : trace_stop], dtype=np.float32)
    if crop.shape != (int(samples), int(traces)):
        raise ValueError(
            f"Requested crop {(int(samples), int(traces))} at trace_start={trace_start}, "
            f"but source shape {array.shape} produced {crop.shape}."
        )
    return normalize_by_absmax(crop) if normalize else crop


def _load_manifest(manifest: Mapping | str | Path) -> Mapping:
    if isinstance(manifest, Mapping):
        return manifest
    return json.loads(Path(manifest).read_text(encoding="utf-8"))


def _base_manifest(*, dataset_type: str, seed: int, sample_count: int, quotas: Mapping[str, int]) -> dict:
    return {
        "dataset_type": dataset_type,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(seed),
        "sample_count": int(sample_count),
        "quotas": {str(name): int(count) for name, count in quotas.items()},
        "per_source_counts": {str(name): int(count) for name, count in quotas.items()},
        "notes": [
            "Data files are local artifacts and must not be committed.",
            "This is a deterministic paper-style protocol, not the authors' unpublished exact split.",
        ],
    }


def _ordered_train_sources(quotas: Mapping[str, int]) -> list[str]:
    known = [source.name for source in PAPER_TRAIN_SOURCES if source.name in quotas]
    extra = sorted(name for name in quotas if name not in known)
    return known + extra


def _ordered_test_sources(quotas: Mapping[str, int]) -> list[str]:
    known = [source.name for source in PAPER_TEST_SOURCES if source.name in quotas]
    extra = sorted(name for name in quotas if name not in known)
    return known + extra


def _test_source_filename(source_name: str) -> str:
    for source in PAPER_TEST_SOURCES:
        if source.name == source_name:
            return source.filename
    return ""


def _train_source_dict(source: PaperTrainSource) -> dict:
    raw_count, final_count = compute_source_patch_counts(source)
    return {
        "name": source.name,
        "filename": source.filename,
        "kind": source.kind,
        "samples": int(source.samples),
        "traces": int(source.traces),
        "train_shots": int(source.train_shots),
        "raw_patch_count": int(raw_count),
        "final_patch_count": int(final_count),
        "paper_final_patches": int(source.final_patches),
    }


def _test_source_dict(source: PaperTestSource) -> dict:
    return {
        "name": source.name,
        "filename": source.filename,
        "kind": source.kind,
        "samples": int(source.samples),
        "traces": int(source.traces),
        "quota": int(source.quota),
        "train_source_name": source.train_source_name,
        "train_shots_before_test": int(source.train_shots_before_test),
        "trace_start": int(source.trace_start),
    }


def _should_validate_default_train_counts(
    sources: Sequence[PaperTrainSource],
    overrides: Mapping[str, Mapping[str, int]],
    patch_size: tuple[int, int],
    stride: tuple[int, int],
    augment_times: int,
) -> bool:
    return (
        tuple(sources) == PAPER_TRAIN_SOURCES
        and not overrides
        and tuple(patch_size) == DEFAULT_PATCH_SIZE
        and tuple(stride) == DEFAULT_STRIDE
        and int(augment_times) == DEFAULT_AUGMENT_TIMES
    )


def _prepare_output_dir(output_dir: str | Path, *, overwrite: bool) -> Path:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output}. Pass overwrite=True to replace it.")
        if output.resolve() == Path.cwd().resolve():
            raise ValueError("Refusing to overwrite the current working directory.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _write_json(path: Path, data: Mapping) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_readme(path: Path, *, title: str, body: str, manifest: Mapping) -> None:
    lines = [
        f"# {title}",
        "",
        body,
        "",
        "## Summary",
        "",
        f"- Sample count: `{manifest['sample_count']}`",
        f"- Seed: `{manifest['seed']}`",
        f"- Manifest: `manifest.json`",
        "- Data files are ignored by Git and should remain local.",
        "",
        "## Per-source Counts",
        "",
    ]
    for source, count in manifest.get("per_source_counts", {}).items():
        lines.append(f"- `{source}`: `{count}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
