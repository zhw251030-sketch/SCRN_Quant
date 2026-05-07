"""Stratified SCRN calibration and test dataset preparation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Mapping, Sequence

import numpy as np


DEFAULT_SEED = 20260507
DEFAULT_PATCH_SIZE = (128, 128)
DEFAULT_STRIDE = (48, 48)
DEFAULT_MIN_STD = 1e-3
DEFAULT_CALIBRATION_SAMPLE_COUNT = 1024


@dataclass(frozen=True)
class LegacyTrainSource:
    """One source range in the legacy 10750_0 patch directory."""

    name: str
    filename: str
    start_index: int
    end_index: int

    @property
    def count(self) -> int:
        return int(self.end_index) - int(self.start_index) + 1


@dataclass(frozen=True)
class TestSource:
    """One source used by the 478-patch legacy-logic test dataset."""

    name: str
    filename: str
    quota: int


@dataclass(frozen=True)
class SelectedCalibrationPatch:
    """A selected calibration patch and its recovered legacy metadata."""

    source: str
    source_file: str
    train_index: int
    path: Path


@dataclass(frozen=True)
class LegacyPatch:
    """A patch extracted by the legacy full-file sliding-window logic."""

    data: np.ndarray
    top: int
    left: int
    sha256: str


@dataclass(frozen=True)
class LegacyPatchSelection:
    """Reservoir-sampled legacy patches and filtering statistics."""

    selected: list[LegacyPatch]
    candidate_count: int
    excluded_count: int


LEGACY_TRAIN_SOURCES: tuple[LegacyTrainSource, ...] = (
    LegacyTrainSource("1997_2.5D_shots", "1997_2.5D_shots.segy", 1, 300),
    LegacyTrainSource("7m_shots_0201", "7m_shots_0201_0329.segy", 301, 3655),
    LegacyTrainSource("Anisotropic_FD_Model", "Anisotropic_FD_Model_Shots_part1.sgy", 3656, 4405),
    LegacyTrainSource("Kerry3D", "Kerry3D.segy", 4406, 4885),
    LegacyTrainSource("Shots0001_0200", "shots0001_0200.segy", 4886, 10750),
)

TEST_SOURCES: tuple[TestSource, ...] = (
    TestSource("Anisotropic", "Anisotropic_FD_Model_Shots_part1.sgy", 75),
    TestSource("Kerry3D", "Kerry3D.segy", 16),
    TestSource("Shots0001", "shots0001_0200.segy", 387),
)

_TRAIN_PATCH_RE = re.compile(r"^train_data_(\d+)\.npy$")


def allocate_largest_remainder(weights: Mapping[str, int], *, total: int) -> dict[str, int]:
    """Allocate integer quotas with the largest-remainder method."""
    if int(total) <= 0:
        raise ValueError(f"total must be positive, got {total}")
    if not weights:
        raise ValueError("weights must not be empty")

    ordered_items = [(str(name), int(weight)) for name, weight in weights.items()]
    if any(weight <= 0 for _, weight in ordered_items):
        raise ValueError(f"weights must be positive, got {weights}")

    weight_sum = sum(weight for _, weight in ordered_items)
    exact = [(name, int(total) * weight / weight_sum, order) for order, (name, weight) in enumerate(ordered_items)]
    quotas = {name: int(value) for name, value, _ in exact}
    remaining = int(total) - sum(quotas.values())
    remainders = sorted(exact, key=lambda item: (item[1] - int(item[1]), -item[2]), reverse=True)
    for name, _, _ in remainders[:remaining]:
        quotas[name] += 1
    return quotas


DEFAULT_CALIBRATION_QUOTAS = allocate_largest_remainder(
    {source.name: source.count for source in LEGACY_TRAIN_SOURCES},
    total=DEFAULT_CALIBRATION_SAMPLE_COUNT,
)
DEFAULT_TEST_QUOTAS = {source.name: source.quota for source in TEST_SOURCES}


def parse_train_patch_index(path: str | Path) -> int:
    """Parse `train_data_N.npy` and return the one-based legacy index."""
    match = _TRAIN_PATCH_RE.match(Path(path).name)
    if match is None:
        raise ValueError(f"Expected train_data_N.npy file name, got {Path(path).name!r}")
    return int(match.group(1))


def source_for_train_index(index: int) -> LegacyTrainSource:
    """Return the legacy source range that owns a `train_data_N.npy` index."""
    numeric_index = int(index)
    for source in LEGACY_TRAIN_SOURCES:
        if source.start_index <= numeric_index <= source.end_index:
            return source
    raise ValueError(f"train_data index {numeric_index} is outside legacy 10750_0 ranges")


def select_stratified_calibration_files(
    patch_dir: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    quotas: Mapping[str, int] | None = None,
) -> list[SelectedCalibrationPatch]:
    """Select calibration clean patches while preserving legacy source quotas."""
    selected_quotas = dict(DEFAULT_CALIBRATION_QUOTAS if quotas is None else quotas)
    grouped: dict[str, list[SelectedCalibrationPatch]] = {name: [] for name in selected_quotas}
    root = Path(patch_dir)
    for path in sorted(root.glob("*.npy"), key=_numeric_train_patch_sort_key):
        train_index = parse_train_patch_index(path)
        source = source_for_train_index(train_index)
        if source.name in grouped:
            grouped[source.name].append(
                SelectedCalibrationPatch(
                    source=source.name,
                    source_file=source.filename,
                    train_index=train_index,
                    path=path,
                )
            )

    rng = np.random.default_rng(int(seed))
    selected: list[SelectedCalibrationPatch] = []
    for source in LEGACY_TRAIN_SOURCES:
        quota = int(selected_quotas.get(source.name, 0))
        if quota <= 0:
            continue
        candidates = grouped.get(source.name, [])
        if len(candidates) < quota:
            raise ValueError(
                f"Source {source.name} has only {len(candidates)} candidate patches, "
                f"but quota requires {quota}."
            )
        positions = rng.choice(len(candidates), size=quota, replace=False)
        source_selected = sorted((candidates[int(position)] for position in positions), key=lambda item: item.train_index)
        selected.extend(source_selected)
    return selected


def prepare_calibration_dataset(
    patch_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    quotas: Mapping[str, int] | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict:
    """Write a stratified calibration clean patch directory and manifest."""
    selected = select_stratified_calibration_files(patch_dir, seed=seed, quotas=quotas)
    manifest = _base_manifest(
        dataset_type="stratified_calibration",
        seed=seed,
        sample_count=len(selected),
        quotas=_counts_from_selected(selected),
    )
    manifest.update(
        {
            "input_patch_dir": str(Path(patch_dir)),
            "source_protocol": "legacy_10750_0_index_ranges",
            "samples": [
                {
                    "output_file": f"cali_{index:06d}.npy",
                    "source": item.source,
                    "source_file": item.source_file,
                    "train_index": item.train_index,
                    "input_file": str(item.path),
                    "sha256": sha256_npy_array(item.path),
                }
                for index, item in enumerate(selected, start=1)
            ],
        }
    )

    if dry_run:
        return manifest

    output = _prepare_output_dir(output_dir, overwrite=overwrite)
    for index, item in enumerate(selected, start=1):
        shutil.copy2(item.path, output / f"cali_{index:06d}.npy")
    _write_json(output / "manifest.json", manifest)
    _write_readme(
        output / "README.md",
        title="SCRN Stratified 1024 Calibration Clean Patches",
        body=(
            "This directory was generated from the legacy 10750_0 clean patch pool using "
            "source-stratified sampling. It contains clean targets; SCRNPatchDataset and "
            "SCRN-BRECQ calibration code generate degraded inputs online."
        ),
        manifest=manifest,
    )
    return manifest


def extract_legacy_patches(
    data: np.ndarray,
    *,
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE,
    stride: tuple[int, int] = DEFAULT_STRIDE,
    min_std: float = DEFAULT_MIN_STD,
    exclude_hashes: set[str] | None = None,
) -> list[LegacyPatch]:
    """Extract patches with the full-file 10750_0 sliding-window logic."""
    return list(
        iter_legacy_patches(
            data,
            patch_size=patch_size,
            stride=stride,
            min_std=min_std,
            exclude_hashes=exclude_hashes,
        )
    )


def iter_legacy_patches(
    data: np.ndarray,
    *,
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE,
    stride: tuple[int, int] = DEFAULT_STRIDE,
    min_std: float = DEFAULT_MIN_STD,
    exclude_hashes: set[str] | None = None,
):
    """Yield legacy full-file sliding-window patches without storing all candidates."""
    source = np.asarray(data, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {source.shape}")
    patch_h, patch_w = (int(patch_size[0]), int(patch_size[1]))
    stride_h, stride_w = (int(stride[0]), int(stride[1]))
    if patch_h <= 0 or patch_w <= 0 or stride_h <= 0 or stride_w <= 0:
        raise ValueError("patch_size and stride values must be positive")

    excluded = exclude_hashes or set()
    for top in range(0, source.shape[0] - patch_h + 1, stride_h):
        for left in range(0, source.shape[1] - patch_w + 1, stride_w):
            patch = np.ascontiguousarray(source[top : top + patch_h, left : left + patch_w], dtype=np.float32)
            if patch.shape != (patch_h, patch_w):
                continue
            if float(np.sum(patch)) == 0.0 or float(patch.std()) <= float(min_std):
                continue
            digest = sha256_array(patch)
            if digest in excluded:
                continue
            yield LegacyPatch(data=patch, top=top, left=left, sha256=digest)


def select_legacy_patches(
    data: np.ndarray,
    *,
    quota: int,
    rng: np.random.Generator,
    train_hashes: set[str] | None = None,
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE,
    stride: tuple[int, int] = DEFAULT_STRIDE,
    min_std: float = DEFAULT_MIN_STD,
) -> LegacyPatchSelection:
    """Select a fixed number of legacy patches with bounded memory."""
    if int(quota) <= 0:
        raise ValueError(f"quota must be positive, got {quota}")

    train_hashes = train_hashes or set()
    selected: list[LegacyPatch] = []
    candidate_count = 0
    excluded_count = 0
    for patch in iter_legacy_patches(data, patch_size=patch_size, stride=stride, min_std=min_std):
        if patch.sha256 in train_hashes:
            excluded_count += 1
            continue
        candidate_count += 1
        if len(selected) < int(quota):
            selected.append(patch)
            continue
        replacement_index = int(rng.integers(0, candidate_count))
        if replacement_index < int(quota):
            selected[replacement_index] = patch

    if candidate_count < int(quota):
        raise ValueError(
            f"not enough candidate patches after filtering: need {quota}, got {candidate_count}."
        )
    return LegacyPatchSelection(
        selected=sorted(selected, key=lambda patch: (patch.top, patch.left)),
        candidate_count=candidate_count,
        excluded_count=excluded_count,
    )


def prepare_test_dataset_from_arrays(
    arrays: Mapping[str, np.ndarray],
    output_dir: str | Path,
    *,
    quotas: Mapping[str, int] | None = None,
    train_hashes: set[str] | None = None,
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE,
    stride: tuple[int, int] = DEFAULT_STRIDE,
    min_std: float = DEFAULT_MIN_STD,
    seed: int = DEFAULT_SEED,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict:
    """Prepare the 478 clean test patch directory from already loaded arrays."""
    selected_quotas = dict(DEFAULT_TEST_QUOTAS if quotas is None else quotas)
    train_hashes = train_hashes or set()
    rng = np.random.default_rng(int(seed))
    selected_items: list[tuple[str, LegacyPatch]] = []
    per_source_counts: dict[str, int] = {}
    per_source_candidates: dict[str, int] = {}
    excluded_total = 0

    for source_name in _ordered_quota_sources(selected_quotas):
        if source_name not in arrays:
            raise ValueError(f"Missing source array for {source_name}")
        quota = int(selected_quotas[source_name])
        selection = select_legacy_patches(
            arrays[source_name],
            quota=quota,
            rng=rng,
            train_hashes=train_hashes,
            patch_size=patch_size,
            stride=stride,
            min_std=min_std,
        )
        excluded_total += selection.excluded_count
        per_source_candidates[source_name] = selection.candidate_count
        selected_items.extend((source_name, patch) for patch in selection.selected)
        per_source_counts[source_name] = quota

    manifest = _base_manifest(
        dataset_type="legacy_logic_test",
        seed=seed,
        sample_count=len(selected_items),
        quotas=selected_quotas,
    )
    manifest.update(
        {
            "source_protocol": "legacy_full_file_sliding_window_without_augmentation",
            "patch_size": list(patch_size),
            "stride": list(stride),
            "min_std": float(min_std),
            "per_source_counts": per_source_counts,
            "per_source_candidate_counts": per_source_candidates,
            "training_hash_excluded_count": int(excluded_total),
            "samples": [
                {
                    "output_file": f"test_{index:06d}.npy",
                    "source": source_name,
                    "source_file": _source_filename(source_name),
                    "top": patch.top,
                    "left": patch.left,
                    "sha256": patch.sha256,
                }
                for index, (source_name, patch) in enumerate(selected_items, start=1)
            ],
        }
    )

    if dry_run:
        return manifest

    output = _prepare_output_dir(output_dir, overwrite=overwrite)
    for index, (_, patch) in enumerate(selected_items, start=1):
        np.save(output / f"test_{index:06d}.npy", patch.data.astype(np.float32, copy=False))
    _write_json(output / "manifest.json", manifest)
    _write_readme(
        output / "README.md",
        title="SCRN 478 Legacy-Logic Clean Test Patches",
        body=(
            "This directory follows the legacy 10750_0 full-file patch extraction style for "
            "the locally available SCRN test sources. Marmousi is omitted because the matching "
            "test source is not locally available."
        ),
        manifest=manifest,
    )
    return manifest


def prepare_test_dataset_from_segy_dir(
    segy_dir: str | Path,
    output_dir: str | Path,
    *,
    quotas: Mapping[str, int] | None = None,
    train_hashes: set[str] | None = None,
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE,
    stride: tuple[int, int] = DEFAULT_STRIDE,
    min_std: float = DEFAULT_MIN_STD,
    seed: int = DEFAULT_SEED,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict:
    """Prepare the legacy-logic test patch directory from local SEG-Y files."""
    selected_quotas = dict(DEFAULT_TEST_QUOTAS if quotas is None else quotas)
    root = Path(segy_dir)
    rng = np.random.default_rng(int(seed))
    selected_items: list[tuple[str, LegacyPatch]] = []
    per_source_counts: dict[str, int] = {}
    per_source_candidates: dict[str, int] = {}
    excluded_total = 0
    for source_name in _ordered_quota_sources(selected_quotas):
        filename = _source_filename(source_name)
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f"SEG-Y source for {source_name} does not exist: {path}")
        selection = select_legacy_patches(
            read_full_segy_matrix(path, normalize=True),
            quota=int(selected_quotas[source_name]),
            rng=rng,
            train_hashes=train_hashes,
            patch_size=patch_size,
            stride=stride,
            min_std=min_std,
        )
        selected_items.extend((source_name, patch) for patch in selection.selected)
        per_source_counts[source_name] = int(selected_quotas[source_name])
        per_source_candidates[source_name] = selection.candidate_count
        excluded_total += selection.excluded_count

    manifest = _base_manifest(
        dataset_type="legacy_logic_test",
        seed=seed,
        sample_count=len(selected_items),
        quotas=selected_quotas,
    )
    manifest.update(
        {
            "raw_segy_dir": str(root),
            "source_protocol": "legacy_full_file_sliding_window_without_augmentation",
            "patch_size": list(patch_size),
            "stride": list(stride),
            "min_std": float(min_std),
            "per_source_counts": per_source_counts,
            "per_source_candidate_counts": per_source_candidates,
            "training_hash_excluded_count": int(excluded_total),
            "samples": [
                {
                    "output_file": f"test_{index:06d}.npy",
                    "source": source_name,
                    "source_file": _source_filename(source_name),
                    "top": patch.top,
                    "left": patch.left,
                    "sha256": patch.sha256,
                }
                for index, (source_name, patch) in enumerate(selected_items, start=1)
            ],
        }
    )

    if dry_run:
        return manifest

    output = _prepare_output_dir(output_dir, overwrite=overwrite)
    for index, (_, patch) in enumerate(selected_items, start=1):
        np.save(output / f"test_{index:06d}.npy", patch.data.astype(np.float32, copy=False))
    _write_readme(
        output / "README.md",
        title="SCRN 478 Legacy-Logic Clean Test Patches",
        body=(
            "This directory follows the legacy 10750_0 full-file patch extraction style for "
            "the locally available SCRN test sources. Marmousi is omitted because the matching "
            "test source is not locally available."
        ),
        manifest=manifest,
    )
    if not dry_run:
        _write_json(output / "manifest.json", manifest)
    return manifest


def read_full_segy_matrix(path: str | Path, *, normalize: bool = True) -> np.ndarray:
    """Read a whole SEG-Y file as `[samples, traces]`, matching 10750_0."""
    try:
        import segyio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Reading SEG-Y files requires installing segyio.") from exc

    with segyio.open(str(path), "r", ignore_geometry=True) as segy_file:
        segy_file.mmap()
        trace_count = len(segy_file.trace)
        if trace_count <= 0:
            raise ValueError(f"SEG-Y file has no traces: {path}")
        sample_count = len(segy_file.trace[0])
        data = np.empty((sample_count, trace_count), dtype=np.float32)
        for trace_index in range(trace_count):
            data[:, trace_index] = segy_file.trace[trace_index]
    return normalize_by_absmax(data) if normalize else data


def normalize_by_absmax(data: np.ndarray) -> np.ndarray:
    """Normalize by the maximum absolute amplitude."""
    array = np.asarray(data, dtype=np.float32)
    scale = float(np.max(np.abs(array)))
    if scale < 1e-12:
        return array
    return array / scale


def build_training_patch_hashes(patch_dir: str | Path) -> set[str]:
    """Build SHA-256 hashes for every training clean patch in a directory."""
    hashes: set[str] = set()
    for path in sorted(Path(patch_dir).glob("*.npy"), key=_numeric_train_patch_sort_key):
        hashes.add(sha256_npy_array(path))
    return hashes


def sha256_npy_array(path: str | Path) -> str:
    """Hash the float32 array content stored in a `.npy` file."""
    return sha256_array(np.load(path).astype(np.float32, copy=False))


def sha256_array(array: np.ndarray) -> str:
    """Hash an array after converting it to contiguous float32 bytes."""
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _numeric_train_patch_sort_key(path: Path) -> tuple[int, str]:
    try:
        return parse_train_patch_index(path), path.name
    except ValueError:
        return 10**18, path.name


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
            "The 10750_0 source ranges are recovered from file numbering; no original coordinates were saved.",
        ],
    }


def _counts_from_selected(selected: Sequence[SelectedCalibrationPatch]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in selected:
        counts[item.source] = counts.get(item.source, 0) + 1
    return counts


def _ordered_quota_sources(quotas: Mapping[str, int]) -> list[str]:
    known = [source.name for source in TEST_SOURCES if source.name in quotas]
    extra = sorted(name for name in quotas if name not in known)
    return known + extra


def _source_filename(source_name: str) -> str:
    for source in TEST_SOURCES:
        if source.name == source_name:
            return source.filename
    for source in LEGACY_TRAIN_SOURCES:
        if source.name == source_name:
            return source.filename
    return ""


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
