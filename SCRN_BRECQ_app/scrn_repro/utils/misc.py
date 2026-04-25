"""SCRN 复现流程中的基础工具。"""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np


def set_random_seed(seed: int) -> None:
    """设置 Python、NumPy 和可选 PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ModuleNotFoundError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_directory(path: str | Path) -> Path:
    """确保目录存在并返回 Path。"""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def require_directory(path: str | Path, description: str = "directory") -> Path:
    """检查目录存在性，用于 CLI 参数校验。"""
    directory = Path(path)
    if not directory.is_dir():
        raise NotADirectoryError(f"{description} does not exist or is not a directory: {directory}")
    return directory

