"""Per-patch normalization helpers for SCRN clean patch datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from typing import Mapping, Sequence

import numpy as np


DEFAULT_NORMALIZATION_EPS = 1e-12
NORMALIZATION_METHOD = "per_patch_absmax"
RESTORATION_FORMULA = "restored = normalized * normalization_scale"


@dataclass(frozen=True)
class PerPatchAbsmaxMetadata:
    """Metadata required to restore one per-patch absmax-normalized patch."""

    normalization_scale: float
    zero_or_tiny_scale: bool
    normalization_eps: float = DEFAULT_NORMALIZATION_EPS
    normalization_method: str = NORMALIZATION_METHOD
    restoration_formula: str = RESTORATION_FORMULA


def normalize_patch_absmax(
    patch: np.ndarray,
    *,
    eps: float = DEFAULT_NORMALIZATION_EPS,
) -> tuple[np.ndarray, PerPatchAbsmaxMetadata]:
    """Normalize one clean patch by its own absolute maximum."""
    array = np.ascontiguousarray(np.asarray(patch, dtype=np.float32))
    if not np.all(np.isfinite(array)):
        raise ValueError("patch contains non-finite values")
    scale = float(np.max(np.abs(array))) if array.size else 0.0
    zero_or_tiny_scale = scale <= float(eps)
    if zero_or_tiny_scale:
        normalized = array.copy()
    else:
        normalized = np.ascontiguousarray(array / np.float32(scale), dtype=np.float32)
    return normalized, PerPatchAbsmaxMetadata(
        normalization_scale=scale,
        zero_or_tiny_scale=bool(zero_or_tiny_scale),
        normalization_eps=float(eps),
    )


def restore_per_patch_absmax(normalized_patch: np.ndarray, normalization_scale: float) -> np.ndarray:
    """Restore a normalized patch to its original clean patch amplitude space."""
    return np.ascontiguousarray(np.asarray(normalized_patch, dtype=np.float32) * np.float32(normalization_scale))


def sha256_array(array: np.ndarray) -> str:
    """Return SHA-256 over contiguous float32 array bytes."""
    contiguous = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def prepare_per_patch_absmax_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    dataset_type: str,
    eps: float = DEFAULT_NORMALIZATION_EPS,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict:
    """Normalize every patch in a train or test dataset directory."""
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    input_manifest = _load_manifest(input_root / "manifest.json")
    input_samples = _manifest_samples(input_manifest)
    output_samples: list[dict] = []
    scales: list[float] = []
    tiny_count = 0

    if not dry_run:
        _prepare_output_dir(output_root, overwrite=overwrite)

    for index, sample in enumerate(input_samples):
        input_file = _sample_output_file(sample, index=index)
        input_path = input_root / input_file
        if not input_path.is_file():
            raise FileNotFoundError(f"Input patch not found: {input_path}")
        patch = np.load(input_path).astype(np.float32, copy=False)
        input_sha256 = sha256_array(patch)
        normalized, norm_meta = normalize_patch_absmax(patch, eps=eps)
        output_sha256 = sha256_array(normalized)
        scales.append(float(norm_meta.normalization_scale))
        tiny_count += int(norm_meta.zero_or_tiny_scale)
        if not dry_run:
            np.save(output_root / input_file, normalized)

        output_sample = dict(sample)
        output_sample.update(
            _sample_normalization_metadata(
                input_dataset_dir=input_root,
                input_file=input_file,
                input_sha256=input_sha256,
                output_sha256=output_sha256,
                metadata=norm_meta,
            )
        )
        output_sample["output_file"] = input_file
        output_sample["sha256"] = output_sha256
        output_samples.append(output_sample)

    manifest = _build_output_manifest(
        input_manifest=input_manifest,
        input_dataset_dir=input_root,
        output_dataset_dir=output_root,
        dataset_type=dataset_type,
        samples=output_samples,
        scales=scales,
        zero_or_tiny_scale_count=tiny_count,
        eps=eps,
    )
    if not dry_run:
        _write_json(output_root / "manifest.json", manifest)
        _write_readme(output_root / "README.md", manifest=manifest, title="SCRN Per-Patch Absmax Normalized Clean Patches")
    return manifest


def prepare_per_patch_absmax_calibration_dataset(
    input_cali_dir: str | Path,
    normalized_train_dir: str | Path,
    output_dir: str | Path,
    *,
    dataset_type: str,
    normalized_train_manifest: Mapping | str | Path | None = None,
    eps: float = DEFAULT_NORMALIZATION_EPS,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict:
    """Create a normalized calibration dataset by copying selected normalized train patches."""
    input_root = Path(input_cali_dir)
    train_root = Path(normalized_train_dir)
    output_root = Path(output_dir)
    input_manifest = _load_manifest(input_root / "manifest.json")
    train_manifest = _load_manifest(normalized_train_manifest if normalized_train_manifest is not None else train_root / "manifest.json")
    input_samples = _manifest_samples(input_manifest)
    train_by_file = {
        str(sample.get("output_file")): sample
        for sample in _manifest_samples(train_manifest)
        if sample.get("output_file")
    }
    output_samples: list[dict] = []
    scales: list[float] = []
    tiny_count = 0

    if not dry_run:
        _prepare_output_dir(output_root, overwrite=overwrite)

    for index, sample in enumerate(input_samples):
        input_file = _sample_output_file(sample, index=index)
        train_file = sample.get("train_file")
        if not train_file:
            raise ValueError(f"Calibration sample {index} does not include train_file")
        train_file = str(train_file)
        train_sample = train_by_file.get(train_file)
        if train_sample is None:
            raise KeyError(f"Normalized train manifest does not contain train_file={train_file}")
        train_path = train_root / train_file
        input_path = input_root / input_file
        if not dry_run and not train_path.is_file():
            raise FileNotFoundError(f"Normalized train patch not found: {train_path}")
        if not input_path.is_file():
            raise FileNotFoundError(f"Input calibration patch not found: {input_path}")

        input_patch = np.load(input_path).astype(np.float32, copy=False)
        input_sha256 = sha256_array(input_patch)
        if dry_run and not train_path.is_file():
            normalized_patch = None
            output_sha256 = str(train_sample.get("output_sha256", train_sample.get("sha256", "")))
            if not output_sha256:
                raise ValueError(f"Dry-run train sample lacks output_sha256 for {train_file}")
        else:
            normalized_patch = np.load(train_path).astype(np.float32, copy=False)
            output_sha256 = sha256_array(normalized_patch)
        scale = float(train_sample.get("normalization_scale", 0.0))
        zero_or_tiny = bool(train_sample.get("zero_or_tiny_scale", scale <= float(eps)))
        scales.append(scale)
        tiny_count += int(zero_or_tiny)
        if not dry_run:
            assert normalized_patch is not None
            np.save(output_root / input_file, normalized_patch)

        output_sample = dict(sample)
        output_sample.update(
            {
                "input_dataset_dir": str(input_root),
                "input_file": input_file,
                "input_sha256": input_sha256,
                "normalized_train_dataset_dir": str(train_root),
                "normalized_train_file": train_file,
                "normalized_train_sha256": str(train_sample.get("output_sha256", output_sha256)),
                "output_sha256": output_sha256,
                "normalization_method": NORMALIZATION_METHOD,
                "normalization_scale": scale,
                "normalization_eps": float(train_sample.get("normalization_eps", eps)),
                "zero_or_tiny_scale": zero_or_tiny,
                "restoration_formula": RESTORATION_FORMULA,
                "sha256": output_sha256,
            }
        )
        output_sample["output_file"] = input_file
        output_samples.append(output_sample)

    manifest = _build_output_manifest(
        input_manifest=input_manifest,
        input_dataset_dir=input_root,
        output_dataset_dir=output_root,
        dataset_type=dataset_type,
        samples=output_samples,
        scales=scales,
        zero_or_tiny_scale_count=tiny_count,
        eps=eps,
        extra={
            "normalized_train_dir": str(train_root),
            "normalized_train_dataset_type": train_manifest.get("dataset_type"),
            "normalization_source": "copied_from_normalized_train_by_train_file",
        },
    )
    if not dry_run:
        _write_json(output_root / "manifest.json", manifest)
        _write_readme(
            output_root / "README.md",
            manifest=manifest,
            title="SCRN Per-Patch Absmax Normalized Calibration Clean Patches",
        )
    return manifest


def _sample_normalization_metadata(
    *,
    input_dataset_dir: Path,
    input_file: str,
    input_sha256: str,
    output_sha256: str,
    metadata: PerPatchAbsmaxMetadata,
) -> dict:
    return {
        "input_dataset_dir": str(input_dataset_dir),
        "input_file": input_file,
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "normalization_method": metadata.normalization_method,
        "normalization_scale": float(metadata.normalization_scale),
        "normalization_eps": float(metadata.normalization_eps),
        "zero_or_tiny_scale": bool(metadata.zero_or_tiny_scale),
        "restoration_formula": metadata.restoration_formula,
    }


def _build_output_manifest(
    *,
    input_manifest: Mapping,
    input_dataset_dir: Path,
    output_dataset_dir: Path,
    dataset_type: str,
    samples: Sequence[Mapping],
    scales: Sequence[float],
    zero_or_tiny_scale_count: int,
    eps: float,
    extra: Mapping | None = None,
) -> dict:
    per_source_counts = _compute_per_source_counts(samples)
    manifest = {
        "dataset_type": dataset_type,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sample_count": len(samples),
        "input_dataset_type": input_manifest.get("dataset_type"),
        "input_dataset_dir": str(input_dataset_dir),
        "output_dataset_dir": str(output_dataset_dir),
        "normalization_method": NORMALIZATION_METHOD,
        "normalization_eps": float(eps),
        "restoration_formula": RESTORATION_FORMULA,
        "zero_or_tiny_scale_count": int(zero_or_tiny_scale_count),
        "scale_summary": summarize_scales(scales),
        "scale_threshold_counts": _scale_threshold_counts(scales),
        "per_source_counts": per_source_counts,
        "samples": [dict(sample) for sample in samples],
    }
    for optional_key in (
        "seed",
        "patch_size",
        "stride",
        "source_protocol",
        "energy_filter",
        "per_source_region_counts",
        "per_source_candidate_counts",
        "per_source_low_energy_rejected_counts",
        "training_hash_excluded_count",
        "original_only",
    ):
        if optional_key in input_manifest:
            manifest[f"input_{optional_key}"] = input_manifest[optional_key]
    if extra:
        manifest.update(dict(extra))
    return manifest


def summarize_scales(scales: Sequence[float]) -> dict:
    """Return compact numeric summary for normalization scales."""
    if not scales:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p01": None,
            "p99": None,
        }
    values = np.asarray(scales, dtype=np.float64)
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p01": float(np.percentile(values, 1)),
        "p99": float(np.percentile(values, 99)),
    }


def _scale_threshold_counts(scales: Sequence[float]) -> dict:
    values = np.asarray(scales, dtype=np.float64)
    thresholds = (DEFAULT_NORMALIZATION_EPS, 1e-9, 1e-6, 1e-4, 1e-3)
    return {f"<= {threshold:.0e}": int(np.sum(values <= threshold)) for threshold in thresholds}


def _compute_per_source_counts(samples: Sequence[Mapping]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        source = str(sample.get("source", "unknown"))
        counts[source] = counts.get(source, 0) + 1
    return counts


def _manifest_samples(manifest: Mapping) -> list[Mapping]:
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("manifest.json must include a list-valued samples field")
    return [dict(sample) for sample in samples]


def _sample_output_file(sample: Mapping, *, index: int) -> str:
    output_file = sample.get("output_file")
    if not output_file:
        raise ValueError(f"Manifest sample {index} does not include output_file")
    return str(output_file)


def _load_manifest(path: Mapping | str | Path) -> dict:
    if isinstance(path, Mapping):
        return dict(path)
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _write_json(path: str | Path, data: Mapping) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _prepare_output_dir(output_dir: str | Path, *, overwrite: bool) -> Path:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _write_readme(path: str | Path, *, manifest: Mapping, title: str) -> None:
    scale_summary = manifest.get("scale_summary", {})
    threshold_counts = manifest.get("scale_threshold_counts", {})
    lines = [
        f"# {title}",
        "",
        "This directory is a per-patch absmax-normalized derivative of an existing SCRN clean patch dataset.",
        "The original dataset is preserved; these files are experimental inputs for normalized FP32/BRECQ studies.",
        "",
        "## Protocol",
        "",
        "- Normalization method: `per_patch_absmax`",
        f"- Formula: `normalized = patch / max(abs(patch))` when scale > `{manifest.get('normalization_eps')}`",
        "- Tiny-scale formula: patches with zero/tiny scale are copied without division.",
        f"- Restoration: `{manifest.get('restoration_formula')}`",
        "- Restoration returns the original clean patch space, not raw SEG-Y amplitude units.",
        "",
        "## Dataset",
        "",
        f"- Dataset type: `{manifest.get('dataset_type')}`",
        f"- Input dataset type: `{manifest.get('input_dataset_type')}`",
        f"- Input dataset dir: `{manifest.get('input_dataset_dir')}`",
        f"- Sample count: `{manifest.get('sample_count')}`",
        f"- Zero/tiny scale count: `{manifest.get('zero_or_tiny_scale_count')}`",
        "",
        "## Scale Summary",
        "",
    ]
    for key in ("count", "min", "p01", "median", "mean", "p99", "max"):
        lines.append(f"- {key}: `{scale_summary.get(key)}`")
    lines.extend(["", "## Tiny-Scale Threshold Counts", ""])
    for key, value in threshold_counts.items():
        lines.append(f"- scale {key}: `{value}`")
    lines.extend(["", "## Per-Source Counts", ""])
    for source, count in manifest.get("per_source_counts", {}).items():
        lines.append(f"- {source}: `{count}`")
    lines.extend(
        [
            "",
            "## Risk Notes",
            "",
            "- Near-zero patches in the unfiltered source dataset can be numerically amplified by per-patch absmax normalization.",
            "- Each sample stores `normalization_scale` and `zero_or_tiny_scale` so later evaluation can restore amplitudes.",
            "- Calibration derivatives copy normalized train files by `train_file`; they do not perform a new random draw.",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")
