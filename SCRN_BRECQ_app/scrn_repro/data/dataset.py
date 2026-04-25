"""SCRN 训练数据集封装。

训练集目录中保存的是干净地震 patch；Dataset 在读取时在线生成缺失道和高斯噪声，
从而复现 SCRN 同时去噪与插值的监督训练形式。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .degradation import DEFAULT_MISSING_RATES, DEFAULT_SNR_DB_VALUES, DegradationInfo, degrade_patch


@dataclass(frozen=True)
class PatchSampleInfo:
    """记录一个训练样本的来源和退化参数。"""

    path: str
    missing_rate: float
    snr_db: float


def discover_patch_files(patch_dir: str | Path, *, max_samples: int | None = None) -> list[Path]:
    """按文件名排序发现 `.npy` patch 文件。"""
    root = Path(patch_dir)
    files = sorted(root.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy patch files found in {root}")
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}")
        files = files[:max_samples]
    return files


class SCRNPatchDataset(Dataset):
    """由 clean patch 生成 SCRN 训练样本。

    返回值为 `(degraded, clean)`，张量形状均为 `[1, H, W]`。
    若设置 `return_info=True`，额外返回样本路径、缺失率和 SNR，便于调试记录。
    """

    def __init__(
        self,
        patch_dir: str | Path,
        *,
        max_samples: int | None = None,
        seed: int | None = None,
        missing_rates: Sequence[float] = DEFAULT_MISSING_RATES,
        snr_db_values: Sequence[float] = DEFAULT_SNR_DB_VALUES,
        return_info: bool = False,
    ) -> None:
        self.patch_files = discover_patch_files(patch_dir, max_samples=max_samples)
        self.seed = seed
        self.missing_rates = tuple(float(value) for value in missing_rates)
        self.snr_db_values = tuple(float(value) for value in snr_db_values)
        self.return_info = return_info
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.patch_files)

    def __getitem__(self, index: int):
        path = self.patch_files[index]
        clean = np.load(path).astype(np.float32, copy=False)
        rng = self._rng_for(index)
        degraded, _, info = degrade_patch(
            clean,
            missing_rates=self.missing_rates,
            snr_db_values=self.snr_db_values,
            rng=rng,
        )

        # 模型约定输入形状为 [B, 1, H, W]，单样本在这里补上通道维。
        degraded_tensor = torch.from_numpy(degraded).unsqueeze(0).float()
        clean_tensor = torch.from_numpy(clean).unsqueeze(0).float()
        if not self.return_info:
            return degraded_tensor, clean_tensor

        sample_info = PatchSampleInfo(path=str(path), missing_rate=info.missing_rate, snr_db=info.snr_db)
        return degraded_tensor, clean_tensor, sample_info

    def set_epoch(self, epoch: int) -> None:
        """让固定 seed 的退化过程随 epoch 改变，同时保持可复现。"""
        self.epoch = int(epoch)

    def _rng_for(self, index: int) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        # 用 epoch 与 index 组合 seed，保证多轮训练中同一 patch 的退化参数会变化。
        return np.random.default_rng(int(self.seed) + self.epoch * len(self.patch_files) + int(index))


def build_train_loader(
    dataset: SCRNPatchDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    distributed: bool,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DistributedSampler | None]:
    """构建训练 DataLoader，并在 DDP 下使用 DistributedSampler。"""
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return loader, sampler
