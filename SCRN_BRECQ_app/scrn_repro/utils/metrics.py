"""SCRN 复现实验常用评价指标。"""

from __future__ import annotations

import numpy as np


def snr_db(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    """计算信噪比，单位 dB。"""
    pred = np.asarray(prediction, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if pred.shape != tgt.shape:
        raise ValueError(f"prediction and target shapes differ: {pred.shape} vs {tgt.shape}")

    signal = np.sum(tgt**2)
    noise = np.sum((tgt - pred) ** 2)
    return float(10.0 * np.log10((signal + eps) / (noise + eps)))


def ssim_score(prediction: np.ndarray, target: np.ndarray, *, win_size: int = 15) -> float:
    """计算结构相似度 SSIM。

    skimage 是可选依赖，因此延迟导入；若环境没有安装，会给出明确错误。
    """
    try:
        from skimage.metrics import structural_similarity
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("SSIM calculation requires installing scikit-image.") from exc

    pred = np.asarray(prediction, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if pred.shape != tgt.shape:
        raise ValueError(f"prediction and target shapes differ: {pred.shape} vs {tgt.shape}")

    data_range = float(max(pred.max(), tgt.max()) - min(pred.min(), tgt.min()))
    if data_range <= 0:
        return 1.0
    return float(structural_similarity(pred, tgt, win_size=win_size, data_range=data_range))

