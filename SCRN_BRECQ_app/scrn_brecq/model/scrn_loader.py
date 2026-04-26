"""SCRN checkpoint 加载适配。

BRECQ 量化算法处理的是已经构建好的 `torch.nn.Module`，而 SCRN 复现训练产物是
`.pth` checkpoint 文件。本文件负责把训练产物恢复成可量化的 FP32 SCRN 模型。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from SCRN_BRECQ_app.scrn_brecq.utils.io import load_json, load_torch_checkpoint
from SCRN_BRECQ_app.scrn_repro.model import SCRN, SCRNConfig, build_scrn_from_config


# 当前复现实验中测试指标更好的 SCRN checkpoint。
# 对应测试 run: 20260425_195621_quant_10750_0_best_eval_gt_colorbar
# 指标: after_snr_db=11.7872, after_ssim=0.8700。
DEFAULT_RECOMMENDED_SCRN_CHECKPOINT = Path(
    "SCRN_BRECQ_app/scrn_repro/runs/train/"
    "20260425_192916_four_gpu_train_quant_10750_0/checkpoints/best.pth"
)


@dataclass(frozen=True)
class LoadedSCRN:
    """保存 SCRN 加载结果和 checkpoint 元信息。

    量化流程主要使用 `model` 字段；`checkpoint`、`config`、`epoch`、`loss`
    用于写运行记录和排查量化结果。
    """

    model: SCRN
    checkpoint_path: Path
    checkpoint: dict[str, Any]
    config: SCRNConfig
    epoch: int | None
    loss: float | None


def load_scrn_for_brecq(
    checkpoint_path: str | Path = DEFAULT_RECOMMENDED_SCRN_CHECKPOINT,
    *,
    config_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
    eval_mode: bool = True,
) -> LoadedSCRN:
    """加载可供 BRECQ 使用的 SCRN FP32 模型。

    参数:
        checkpoint_path: SCRN 训练保存的 `.pth` checkpoint。默认使用当前测试效果更好的
            `20260425_192916.../best.pth`。
        config_path: 可选的训练 run `config.json`。当 checkpoint 内没有 `model_config`
            时，用它恢复模型结构参数。
        device: 模型加载到的设备，通常先用 `"cpu"` 做结构检查，量化时再转到 CUDA。
        strict: 是否严格匹配 checkpoint 与模型参数名。
        eval_mode: 是否把模型切换到 `eval()`。BRECQ 是后训练量化，默认应使用推理模式。

    返回:
        `LoadedSCRN`，其中包含模型对象和 checkpoint 元信息。
    """
    checkpoint_file = Path(checkpoint_path)
    target_device = _select_device(device)
    checkpoint = load_torch_checkpoint(checkpoint_file, map_location=target_device)
    config = scrn_config_from_checkpoint(checkpoint, config_path=config_path)
    model = build_scrn_model(config).to(target_device)
    state_dict = extract_model_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=strict)
    if eval_mode:
        model.eval()

    return LoadedSCRN(
        model=model,
        checkpoint_path=checkpoint_file,
        checkpoint=checkpoint,
        config=config,
        epoch=_optional_int(checkpoint.get("epoch")),
        loss=_optional_float(checkpoint.get("loss")),
    )


def load_scrn_model(
    checkpoint_path: str | Path = DEFAULT_RECOMMENDED_SCRN_CHECKPOINT,
    *,
    config_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
    eval_mode: bool = True,
) -> SCRN:
    """只返回 SCRN 模型对象的便捷函数。

    后续如果某段 BRECQ 代码只关心 `nn.Module`，可以调用这个函数；如果需要记录
    checkpoint 的 epoch、loss 和结构配置，则应调用 `load_scrn_for_brecq`。
    """
    return load_scrn_for_brecq(
        checkpoint_path,
        config_path=config_path,
        device=device,
        strict=strict,
        eval_mode=eval_mode,
    ).model


def scrn_config_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
) -> SCRNConfig:
    """从 checkpoint 或训练配置文件恢复 SCRN 结构配置。

    优先使用 checkpoint 里的 `model_config`，因为它是保存权重时直接写入的结构参数。
    如果旧 checkpoint 缺少该字段，再从训练 run 的 `config.json` 的 `args` 中读取。
    """
    raw_config = checkpoint.get("model_config")
    if raw_config is None and config_path is not None:
        raw_config = load_json(config_path).get("args", {})
    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, Mapping):
        raise TypeError(f"Expected model_config mapping, got {type(raw_config)!r}")

    return SCRNConfig(
        in_channels=int(raw_config.get("in_channels", 1)),
        dim=int(raw_config.get("dim", 64)),
        stage_depths=_parse_stage_depths(raw_config.get("stage_depths", (1, 1, 1, 1, 1))),
        head_dim=int(raw_config.get("head_dim", 32)),
        window_size=int(raw_config.get("window_size", 8)),
        drop_path_rate=float(raw_config.get("drop_path_rate", 0.0)),
        input_resolution=int(raw_config.get("input_resolution", 128)),
    )


def build_scrn_model(config: SCRNConfig) -> SCRN:
    """根据 `SCRNConfig` 构建原始 FP32 SCRN 模型。

    这个函数只构建模型结构，不加载权重。拆出来是为了后续测试、量化包装和配置检查
    可以复用同一套构建逻辑。
    """
    return build_scrn_from_config(config)


def extract_model_state_dict(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """从 checkpoint 中取出 SCRN 的模型参数。

    训练脚本使用 `model_state_dict` 保存参数；这里也兼容少数工具常用的 `state_dict`
    字段。若参数名前带有 DDP 的 `module.` 前缀，会在加载前去掉。
    """
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None:
        raise KeyError("Checkpoint does not contain `model_state_dict` or `state_dict`.")
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Expected state dict mapping, got {type(state_dict)!r}")
    return _strip_module_prefix(state_dict)


def _select_device(device: str | torch.device) -> torch.device:
    """解析设备参数，支持 `auto`、`cpu`、`cuda` 和具体 CUDA 设备。"""
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(device)


def _parse_stage_depths(value: Any) -> tuple[int, int, int, int, int]:
    """把配置中的 stage depth 转成 SCRN 需要的 5 元组。"""
    if isinstance(value, str):
        depths = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    else:
        depths = tuple(int(part) for part in value)
    if len(depths) != 5:
        raise ValueError(f"SCRN expects 5 stage depths, got {depths}")
    return depths


def _strip_module_prefix(state_dict: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """去掉 DDP/DataParallel 可能保存出的 `module.` 参数名前缀。"""
    normalized = {}
    for key, value in state_dict.items():
        normalized_key = key.removeprefix("module.")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"State dict value for {key!r} is not a tensor: {type(value)!r}")
        normalized[normalized_key] = value
    return normalized


def _optional_int(value: Any) -> int | None:
    """把可选数字字段转成 `int`，缺失时保留为 `None`。"""
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    """把可选数字字段转成 `float`，缺失时保留为 `None`。"""
    return None if value is None else float(value)
