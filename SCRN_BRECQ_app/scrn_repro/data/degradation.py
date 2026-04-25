"""地震 patch 退化样本生成工具。

SCRN 的监督目标是干净 patch；训练输入由干净 patch 在线退化得到：
先按列方向模拟缺失道，再按目标 SNR 添加高斯随机噪声。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


DEFAULT_MISSING_RATES = (0.02, 0.08, 0.18, 0.28, 0.38)
DEFAULT_SNR_DB_VALUES = (-2.0, -1.0, 1.0, 5.0, 10.0)


@dataclass(frozen=True)
class DegradationInfo:
    """记录一次退化所使用的参数，便于之后复现实验。"""

    missing_rate: float
    snr_db: float


def make_rng(seed: int | None = None) -> np.random.Generator:
    """创建 numpy 随机数发生器；seed 为 None 时保持随机。"""
    return np.random.default_rng(seed)


def choose_value(values: Iterable[float], rng: np.random.Generator) -> float:
    """从候选参数中随机取一个标量。"""
    values = tuple(float(value) for value in values)
    if not values:
        raise ValueError("values must not be empty")
    return values[int(rng.integers(0, len(values)))]


def make_random_trace_mask(shape: tuple[int, int], missing_rate: float, rng: np.random.Generator) -> np.ndarray:
    """生成列方向随机缺失道 mask。

    地震剖面通常按列表示道位置，因此这里对整列置零来模拟缺失道。
    返回形状为 [H, W] 的 float32 mask，保留道为 1，缺失道为 0。
    """
    if len(shape) != 2:
        raise ValueError(f"shape must be 2D, got {shape}")
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError(f"missing_rate must be in [0, 1), got {missing_rate}")

    _, width = shape
    mask_cols = np.ones(width, dtype=np.float32)
    num_missing = int(width * missing_rate)
    if num_missing > 0:
        missing_indices = rng.choice(width, size=num_missing, replace=False)
        mask_cols[missing_indices] = 0.0
    return np.broadcast_to(mask_cols, shape).copy()


def gaussian_noise_for_snr(clean: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """按目标 SNR 标定高斯噪声幅度。

    先生成标准正态噪声，再按 clean 的方差和目标 SNR 重新缩放，使噪声能量可控。
    """
    clean = np.asarray(clean, dtype=np.float32)
    noise = rng.standard_normal(clean.shape).astype(np.float32)
    noise = noise - np.mean(noise)

    signal_power = np.linalg.norm(clean - clean.mean()) ** 2 / clean.size
    noise_variance = signal_power / np.power(10.0, snr_db / 10.0)
    noise_std = np.std(noise)
    if noise_std < 1e-12:
        return np.zeros_like(clean, dtype=np.float32)
    return (np.sqrt(noise_variance) / noise_std * noise).astype(np.float32)


def degrade_patch(
    clean_patch: np.ndarray,
    *,
    missing_rate: float | None = None,
    snr_db: float | None = None,
    missing_rates: Iterable[float] = DEFAULT_MISSING_RATES,
    snr_db_values: Iterable[float] = DEFAULT_SNR_DB_VALUES,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, DegradationInfo]:
    """由干净 patch 生成 SCRN 输入样本。

    返回 degraded_patch、trace_mask 和退化参数。degraded_patch = clean * mask + noise。
    """
    rng = rng or make_rng()
    clean = np.asarray(clean_patch, dtype=np.float32)
    if clean.ndim != 2:
        raise ValueError(f"clean_patch must be 2D, got shape {clean.shape}")

    rate = float(missing_rate) if missing_rate is not None else choose_value(missing_rates, rng)
    snr = float(snr_db) if snr_db is not None else choose_value(snr_db_values, rng)
    mask = make_random_trace_mask(clean.shape, rate, rng)
    noise = gaussian_noise_for_snr(clean, snr, rng)
    degraded = clean * mask + noise
    return degraded.astype(np.float32), mask.astype(np.float32), DegradationInfo(rate, snr)

