"""SCRN-BRECQ 文件读写辅助函数。

本文件只放通用的 I/O 小工具，避免模型加载、量化算法等模块反复编写路径检查、
JSON 读取和 checkpoint 读取代码。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def require_file(path: str | Path, description: str = "file") -> Path:
    """确认输入路径是一个已存在文件，并返回规范化后的 `Path`。

    参数:
        path: 待检查的文件路径，可以是字符串或 `Path`。
        description: 出错时用于说明文件用途的文字，例如 "checkpoint"。

    返回:
        已检查存在的 `Path` 对象。
    """
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(f"{description} does not exist: {candidate}")
    return candidate


def load_json(path: str | Path) -> dict[str, Any]:
    """读取 JSON 文件，并要求顶层内容是字典。

    BRECQ 量化流程会读取配置文件和训练 run 的 `config.json`。这些文件顶层都应
    是 key-value 字典；如果读取到列表或其他类型，通常说明传错了文件。
    """
    json_path = require_file(path, "JSON file")
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {json_path}, got {type(payload)!r}")
    return payload


def load_torch_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """读取 PyTorch checkpoint，并要求返回字典结构。

    SCRN 训练脚本保存的是包含 `model_state_dict`、`model_config`、`epoch`、`loss`
    等字段的字典。这里集中做类型检查，方便后续模型加载和量化入口复用。
    """
    checkpoint_path = require_file(path, "checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict in {checkpoint_path}, got {type(checkpoint)!r}")
    return checkpoint

