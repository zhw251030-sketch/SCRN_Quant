"""Integration helpers for SCRN reproduction and later BRECQ-style PTQ."""

__all__ = ["build_scrn_model", "load_scrn_model"]


def __getattr__(name):
    """延迟导入模型加载器，避免数据工具在无 PyTorch 环境下导入失败。"""
    if name in __all__:
        from .scrn_loader import build_scrn_model, load_scrn_model

        return {"build_scrn_model": build_scrn_model, "load_scrn_model": load_scrn_model}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
