"""加载 SCRN 独立复现模型。

本模块只依赖 SCRN_BRECQ_app/scrn_repro 中的实现，不再导入 SCRN-main 源码。
"""

from pathlib import Path
from typing import Any, Mapping, Optional, Union

from .paths import DEFAULT_SCRN_CHECKPOINT, require_existing_path
from .scrn_repro.model import SCRN


def import_scrn_class():
    """返回独立复现的 SCRN 类，保留该函数便于兼容旧调用。"""
    return SCRN


def build_scrn_model(
    *,
    device: Optional[Union[str, "torch.device"]] = "cpu",
    eval_mode: bool = True,
    **model_kwargs: Any,
):
    """Build a fresh SCRN model using the original implementation."""
    import torch

    SCRN = import_scrn_class()
    model = SCRN(**model_kwargs)

    if device is not None:
        model = model.to(torch.device(device))
    if eval_mode:
        model.eval()
    return model


def load_scrn_model(
    checkpoint_path: Union[str, Path] = DEFAULT_SCRN_CHECKPOINT,
    *,
    device: Optional[Union[str, "torch.device"]] = "cpu",
    eval_mode: bool = True,
    strict: bool = True,
    **model_kwargs: Any,
):
    """Load a SCRN checkpoint saved either as a full module or a state dict."""
    import torch
    from torch import nn

    SCRN = import_scrn_class()
    checkpoint = _load_checkpoint(Path(checkpoint_path), torch, device)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        model = SCRN(**model_kwargs)
        state_dict = _extract_state_dict(checkpoint)
        model.load_state_dict(_strip_known_prefixes(state_dict), strict=strict)

    if device is not None:
        model = model.to(torch.device(device))
    if eval_mode:
        model.eval()
    return model


def _load_checkpoint(path: Path, torch_module, device):
    require_existing_path(path, "SCRN checkpoint")
    kwargs = {"map_location": torch_module.device(device)} if device is not None else {}
    try:
        return torch_module.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch_module.load(path, **kwargs)


def _extract_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        if all(hasattr(value, "shape") for value in checkpoint.values()):
            return checkpoint

    raise TypeError(
        "Unsupported SCRN checkpoint format. Expected a torch module, a state dict, "
        "or a dict containing state_dict/model_state_dict/model."
    )


def _strip_known_prefixes(state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned
