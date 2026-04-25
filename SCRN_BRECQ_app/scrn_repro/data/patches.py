"""SEG-Y 地震数据切分 patch 的独立实现。

该模块只提供数据准备能力：读取 SEG-Y、按滑窗切出二维 patch、做简单几何增强。
训练/测试流程会在后续步骤单独实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


PatchSize = tuple[int, int]


@dataclass(frozen=True)
class PatchExtractionConfig:
    """patch 切分默认配置。"""

    patch_size: PatchSize = (128, 128)
    stride: PatchSize = (48, 48)
    augment_times: int = 0
    max_patches: int | None = None
    min_std: float = 1e-3
    normalize: bool = True
    jump: int = 1


def iter_segy_files(data_dir: str | Path) -> list[Path]:
    """列出目录下可读取的 SEG-Y 文件。"""
    root = Path(data_dir)
    files = sorted([*root.glob("*.segy"), *root.glob("*.sgy")])
    if not files:
        raise FileNotFoundError(f"No .segy or .sgy files found in {root}")
    return files


def normalize_by_absmax(data: np.ndarray) -> np.ndarray:
    """按最大绝对振幅归一化，避免不同炮集幅值尺度差异过大。"""
    data = np.asarray(data, dtype=np.float32)
    scale = float(np.max(np.abs(data)))
    if scale < 1e-12:
        return data
    return data / scale


def augment_patch(patch: np.ndarray, mode: int) -> np.ndarray:
    """8 种常见几何增强，和公开实现中的翻转/旋转策略保持一致。"""
    if mode == 0:
        out = patch
    elif mode == 1:
        out = np.flipud(patch)
    elif mode == 2:
        out = np.rot90(patch)
    elif mode == 3:
        out = np.flipud(np.rot90(patch))
    elif mode == 4:
        out = np.rot90(patch, k=2)
    elif mode == 5:
        out = np.flipud(np.rot90(patch, k=2))
    elif mode == 6:
        out = np.rot90(patch, k=3)
    elif mode == 7:
        out = np.flipud(np.rot90(patch, k=3))
    else:
        raise ValueError(f"Unsupported augmentation mode: {mode}")
    return np.ascontiguousarray(out, dtype=np.float32)


def split_patches(
    data: np.ndarray,
    *,
    patch_size: PatchSize = (128, 128),
    stride: PatchSize = (48, 48),
    augment_times: int = 0,
    max_patches: int | None = None,
    min_std: float = 1e-3,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """按滑动窗口切分二维地震数据。

    只保留非零且方差足够的 patch，避免把空白区域加入训练集。
    """
    rng = rng or np.random.default_rng()
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data.shape}")

    patch_h, patch_w = patch_size
    stride_h, stride_w = stride
    height, width = data.shape
    patches: list[np.ndarray] = []

    for top in range(0, height - patch_h + 1, stride_h):
        for left in range(0, width - patch_w + 1, stride_w):
            patch = np.ascontiguousarray(data[top:top + patch_h, left:left + patch_w], dtype=np.float32)
            if patch.shape != patch_size or np.allclose(patch, 0.0) or float(patch.std()) <= min_std:
                continue

            patches.append(patch)
            if _reached_limit(patches, max_patches):
                return patches

            for _ in range(augment_times):
                mode = int(rng.integers(0, 8))
                patches.append(augment_patch(patch, mode))
                if _reached_limit(patches, max_patches):
                    return patches

    return patches


def read_segy_shots(path: str | Path, *, jump: int = 1, normalize: bool = True) -> Iterator[np.ndarray]:
    """逐炮读取 SEG-Y 数据并转换为二维数组。

    公开实现通过 SourceX 判断炮集数；这里保留相同意图，但把 segyio 作为可选依赖延迟导入。
    """
    try:
        import segyio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Reading SEG-Y files requires installing segyio.") from exc

    if jump < 1:
        raise ValueError(f"jump must be >= 1, got {jump}")

    with segyio.open(str(path), "r", ignore_geometry=True) as segy_file:
        segy_file.mmap()
        source_x = segy_file.attributes(segyio.TraceField.SourceX)[:]
        trace_num = len(source_x)
        shot_num = len(set(source_x))
        if trace_num == shot_num or shot_num <= 1:
            shot_num = 1
            traces_per_shot = trace_num
        else:
            traces_per_shot = trace_num // shot_num

        for shot_index in range(0, shot_num, jump):
            start = shot_index * traces_per_shot
            stop = (shot_index + 1) * traces_per_shot
            # trace 读取后是 [trace, time]，转置为 [time, trace]，便于按列模拟缺失道。
            data = np.asarray([np.copy(trace) for trace in segy_file.trace[start:stop]], dtype=np.float32).T
            yield normalize_by_absmax(data) if normalize else data


def collect_patches_from_segy_dir(
    data_dir: str | Path,
    *,
    config: PatchExtractionConfig = PatchExtractionConfig(),
    seed: int | None = None,
) -> list[np.ndarray]:
    """从目录中所有 SEG-Y 文件收集 patch。"""
    rng = np.random.default_rng(seed)
    patches: list[np.ndarray] = []
    for segy_path in iter_segy_files(data_dir):
        for shot in read_segy_shots(segy_path, jump=config.jump, normalize=config.normalize):
            remaining = None if config.max_patches is None else config.max_patches - len(patches)
            if remaining is not None and remaining <= 0:
                return patches
            patches.extend(split_patches(
                shot,
                patch_size=config.patch_size,
                stride=config.stride,
                augment_times=config.augment_times,
                max_patches=remaining,
                min_std=config.min_std,
                rng=rng,
            ))
    return patches


def save_patches_as_npy(patches: Sequence[np.ndarray], output_dir: str | Path, *, prefix: str = "patch") -> None:
    """把 patch 序列保存为单个 .npy 文件。

    该函数只被 CLI 显式调用；本轮不会自动执行，避免产生数据文件进入 Git。
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for index, patch in enumerate(patches, start=1):
        np.save(output / f"{prefix}_{index:06d}.npy", np.asarray(patch, dtype=np.float32))


def _reached_limit(patches: Sequence[np.ndarray], max_patches: int | None) -> bool:
    return max_patches is not None and len(patches) >= max_patches

