"""Generate a clean-patch atlas for the 478-patch SCRN test set."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/scrn_brecq_matplotlib_cache")

SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[6]
DEFAULT_DATASET_DIR = (
    REPO_ROOT
    / "SCRN_BRECQ_app"
    / "scrn_repro"
    / "datasets"
    / "scrn_paper5_energy_filtered_perpatch_absmax_test_478"
)
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "candidates" / "clean_patch_atlas"
DEFAULT_VERSION = 1
PAGE_ROWS = 8
PAGE_COLS = 6
PATCHES_PER_PAGE = PAGE_ROWS * PAGE_COLS
SELECTION_INDEX_FIELDS = (
    "patch_file",
    "patch_index",
    "source",
    "source_file",
    "region_index",
    "top",
    "left",
    "normalization_scale",
    "zero_or_tiny_scale",
    "input_file",
    "output_sha256",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION)
    parser.add_argument("--rows", type=int, default=PAGE_ROWS)
    parser.add_argument("--cols", type=int, default=PAGE_COLS)
    parser.add_argument("--cmap", default="seismic")
    parser.add_argument("--vmin", type=float, default=-1.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    samples = load_dataset_samples(dataset_dir)
    version = int(args.version)
    per_page = int(args.rows) * int(args.cols)
    pages = paginate(samples, per_page=per_page)
    output_dir.mkdir(parents=True, exist_ok=True)

    version_tag = f"v{version:03d}"
    pdf_path = output_dir / f"atlas_clean_test478_{version_tag}.pdf"
    png_template = output_dir / f"atlas_clean_test478_{version_tag}_page_{{page:03d}}.png"
    index_path = output_dir / f"selection_index_{version_tag}.csv"
    manifest_path = output_dir / f"manifest_{version_tag}.json"

    write_selection_index(index_path, [selection_index_row(sample) for sample in samples])
    page_paths = render_atlas(
        pages,
        dataset_dir=dataset_dir,
        pdf_path=pdf_path,
        png_template=png_template,
        rows=int(args.rows),
        cols=int(args.cols),
        cmap=str(args.cmap),
        vmin=float(args.vmin),
        vmax=float(args.vmax),
        dpi=int(args.dpi),
    )
    write_manifest(
        manifest_path,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        version=version,
        samples=samples,
        pages=pages,
        page_paths=page_paths,
        pdf_path=pdf_path,
        index_path=index_path,
        rows=int(args.rows),
        cols=int(args.cols),
        cmap=str(args.cmap),
        vmin=float(args.vmin),
        vmax=float(args.vmax),
    )

    print(
        "atlas_version={version_tag} samples={samples} pages={pages} pdf={pdf} index={index}".format(
            version_tag=version_tag,
            samples=len(samples),
            pages=len(pages),
            pdf=pdf_path,
            index=index_path,
        ),
        flush=True,
    )


def read_manifest(dataset_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(dataset_dir) / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset_samples(dataset_dir: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(dataset_dir)
    manifest = read_manifest(dataset_path)
    manifest_samples = {
        str(sample["output_file"]): sample
        for sample in manifest.get("samples", [])
        if "output_file" in sample
    }
    patch_files = sorted(dataset_path.glob("test_*.npy"), key=lambda path: patch_file_sort_key(path.name))
    if int(manifest.get("sample_count", -1)) != len(patch_files):
        raise ValueError(
            "manifest sample_count={manifest_count} does not match npy count={file_count}".format(
                manifest_count=manifest.get("sample_count"),
                file_count=len(patch_files),
            )
        )

    samples: list[dict[str, Any]] = []
    for patch_index, path in enumerate(patch_files):
        metadata = dict(manifest_samples.get(path.name, {}))
        metadata["patch_file"] = path.name
        metadata["patch_index"] = patch_index
        metadata["patch_path"] = str(path)
        samples.append(metadata)
    return samples


def patch_file_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"test_(\d+)\.npy$", name)
    if not match:
        return (math.inf, name)
    return (int(match.group(1)), name)


def paginate(items: Sequence[Any], *, per_page: int) -> list[list[Any]]:
    if per_page <= 0:
        raise ValueError("per_page must be positive")
    return [list(items[start : start + per_page]) for start in range(0, len(items), per_page)]


def selection_index_row(sample: Mapping[str, Any]) -> dict[str, Any]:
    return {field: sample.get(field, "") for field in SELECTION_INDEX_FIELDS}


def write_selection_index(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTION_INDEX_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SELECTION_INDEX_FIELDS})


def render_atlas(
    pages: Sequence[Sequence[Mapping[str, Any]]],
    *,
    dataset_dir: Path,
    pdf_path: Path,
    png_template: Path,
    rows: int,
    cols: int,
    cmap: str,
    vmin: float,
    vmax: float,
    dpi: int,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    page_paths: list[Path] = []
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        for page_index, page_samples in enumerate(pages, start=1):
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.15), constrained_layout=True)
            flat_axes = list(axes.ravel())
            for axis, sample in zip(flat_axes, page_samples):
                patch = load_patch(dataset_dir / str(sample["patch_file"]))
                axis.imshow(patch, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
                axis.set_title(panel_title(sample), fontsize=6.5, pad=2)
                axis.set_xticks([])
                axis.set_yticks([])
            for axis in flat_axes[len(page_samples) :]:
                axis.axis("off")
            fig.suptitle(f"Clean test patches 478 atlas - page {page_index:02d}/{len(pages):02d}", fontsize=12)

            page_path = Path(str(png_template).format(page=page_index))
            fig.savefig(page_path, dpi=dpi)
            pdf.savefig(fig)
            plt.close(fig)
            page_paths.append(page_path)
    return page_paths


def load_patch(path: str | Path) -> Any:
    import numpy as np

    patch = np.load(path)
    patch = np.asarray(patch)
    patch = np.squeeze(patch)
    if patch.ndim != 2:
        raise ValueError(f"Expected 2D patch at {path}, got shape {patch.shape}")
    return patch


def panel_title(sample: Mapping[str, Any]) -> str:
    return "{patch_file}\nidx={patch_index} / {source}".format(
        patch_file=sample.get("patch_file", ""),
        patch_index=sample.get("patch_index", ""),
        source=sample.get("source", ""),
    )


def write_manifest(
    path: str | Path,
    *,
    dataset_dir: Path,
    output_dir: Path,
    version: int,
    samples: Sequence[Mapping[str, Any]],
    pages: Sequence[Sequence[Mapping[str, Any]]],
    page_paths: Sequence[Path],
    pdf_path: Path,
    index_path: Path,
    rows: int,
    cols: int,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_id": "ch4_common_exp01_testset_clean_patch_atlas",
        "version": f"v{version:03d}",
        "purpose": "clean_patch_visual_browsing",
        "dataset_dir": relpath_or_str(dataset_dir),
        "dataset_manifest": relpath_or_str(dataset_dir / "manifest.json"),
        "output_dir": relpath_or_str(output_dir),
        "sample_count": len(samples),
        "page_count": len(pages),
        "rows": rows,
        "cols": cols,
        "patches_per_page": rows * cols,
        "last_page_count": len(pages[-1]) if pages else 0,
        "cmap": cmap,
        "vmin": vmin,
        "vmax": vmax,
        "pdf": relpath_or_str(pdf_path),
        "png_pages": [relpath_or_str(page_path) for page_path in page_paths],
        "selection_index": relpath_or_str(index_path),
        "samples": [selection_index_row(sample) for sample in samples],
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def relpath_or_str(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
