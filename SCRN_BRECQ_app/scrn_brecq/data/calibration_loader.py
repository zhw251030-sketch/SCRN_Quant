"""SCRN-BRECQ 校准数据加载器。

BRECQ 重构阶段需要一小批能代表模型真实输入分布的数据。对 SCRN 来说，模型推理
输入是带噪声和缺失道的 degraded patch，因此本文件只收集 `SCRNPatchDataset`
返回的第一个张量 `degraded`，不把 clean target 放进 calibration data。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from SCRN_BRECQ_app.scrn_repro.data import SCRNPatchDataset


# 当前推荐 SCRN checkpoint 使用的 10750_0 训练 patch 数据。
# 这里选择 SCRN_BRECQ_app 内的复现数据目录，避免默认配置依赖仓库外部绝对路径。
DEFAULT_CALIBRATION_DATASET_DIR = Path("SCRN_BRECQ_app/scrn_repro/datasets/scrn_quant_10750_0_patches")


@dataclass(frozen=True)
class CalibrationDataConfig:
    """校准数据加载配置。

    参数:
        dataset_dir: clean patch `.npy` 文件目录。Dataset 会在线生成 degraded 输入。
        num_samples: 最终收集多少个 calibration 输入样本。
        batch_size: DataLoader 每批读取多少个样本。
        num_workers: DataLoader worker 数。默认 0 便于调试和保持简单可复现。
        seed: 控制在线退化过程，使相同配置下 calibration data 可复现。
        device: 返回的 calibration tensor 放置到哪个设备。
        pin_memory: CUDA 训练/量化时可开启 pinned memory；CPU 测试时保持 False。
    """

    dataset_dir: str | Path = DEFAULT_CALIBRATION_DATASET_DIR
    num_samples: int = 1024
    batch_size: int = 16
    num_workers: int = 0
    seed: int | None = 1005
    device: str | torch.device = "cpu"
    pin_memory: bool = False


def build_calibration_dataset(config: CalibrationDataConfig) -> SCRNPatchDataset:
    """构建 SCRN calibration 数据集。

    复用 `scrn_repro` 已验证的 `SCRNPatchDataset`，这样校准输入的退化方式与 SCRN
    训练阶段一致。`max_samples` 限制为 `num_samples`，避免 DataLoader 扫描和读取
    超过 BRECQ 实际需要的数据。
    """
    _validate_positive("num_samples", config.num_samples)
    return SCRNPatchDataset(
        config.dataset_dir,
        max_samples=config.num_samples,
        seed=config.seed,
        return_info=False,
    )


def build_calibration_loader(config: CalibrationDataConfig) -> DataLoader:
    """构建普通单进程/多进程 DataLoader。

    BRECQ 校准不是 SCRN 的 DDP 训练流程，因此这里不使用 DistributedSampler，也不做
    shuffle。保持固定文件顺序和固定 seed，有助于量化实验复现。
    """
    _validate_positive("batch_size", config.batch_size)
    if config.num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {config.num_workers}")

    dataset = build_calibration_dataset(config)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=bool(config.pin_memory),
        drop_last=False,
    )


def collect_calibration_inputs(
    loader: Iterable,
    num_samples: int,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """从 DataLoader 中只收集 degraded 输入，拼成 BRECQ 使用的 `cali_data`。

    `SCRNPatchDataset` 返回 `(degraded, clean)`。本函数显式只读取 batch 的第一个
    元素，避免把监督训练用的 clean target 混入后训练量化的校准输入。
    """
    _validate_positive("num_samples", num_samples)
    target_device = _select_device(device)
    batches: list[torch.Tensor] = []
    collected = 0

    for batch in loader:
        degraded = _extract_degraded_batch(batch)
        degraded = degraded.detach().to(device=target_device, dtype=torch.float32)
        batches.append(degraded)
        collected += int(degraded.size(0))
        if collected >= num_samples:
            break

    if not batches:
        raise ValueError("Calibration loader produced no batches.")

    calibration_data = torch.cat(batches, dim=0)[:num_samples]
    if calibration_data.ndim != 4:
        raise ValueError(f"Expected calibration data shape [N, C, H, W], got {tuple(calibration_data.shape)}")
    return calibration_data.contiguous()


def load_calibration_data(config: CalibrationDataConfig) -> torch.Tensor:
    """便捷入口：构建 loader 并返回 calibration 输入 tensor。"""
    loader = build_calibration_loader(config)
    return collect_calibration_inputs(loader, config.num_samples, device=config.device)


def _extract_degraded_batch(batch) -> torch.Tensor:
    """从 Dataset/DataLoader batch 中取出 degraded 输入张量。

    当前 `SCRNPatchDataset` 默认返回二元组 `(degraded, clean)`；如果后续调试时开启
    `return_info=True`，则会返回三元组 `(degraded, clean, info)`。两种形式的第一个
    元素都是 SCRN 模型输入。
    """
    if isinstance(batch, (tuple, list)) and batch:
        degraded = batch[0]
    else:
        degraded = batch
    if not isinstance(degraded, torch.Tensor):
        raise TypeError(f"Expected degraded batch tensor, got {type(degraded)!r}")
    return degraded


def _select_device(device: str | torch.device) -> torch.device:
    """解析 calibration tensor 的目标设备。"""
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(device)


def _validate_positive(name: str, value: int) -> None:
    """检查正整数配置，尽早暴露错误参数。"""
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive, got {value}")

