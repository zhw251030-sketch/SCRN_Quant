"""SCRN 复现与 BRECQ 迁移应用包。

这里导出的 `build_scrn_model` 和 `load_scrn_model` 是早期兼容入口。
新的 BRECQ 量化流程应优先使用 `SCRN_BRECQ_app.scrn_brecq.model` 中的
`load_scrn_for_brecq`，以便同时获得 checkpoint 元信息和 SCRN 结构配置。
"""

__all__ = ["build_scrn_model", "load_scrn_model"]


def __getattr__(name):
    """延迟导入模型加载器，避免数据工具在无 PyTorch 环境下导入失败。"""
    if name in __all__:
        from .scrn_loader import build_scrn_model, load_scrn_model

        return {"build_scrn_model": build_scrn_model, "load_scrn_model": load_scrn_model}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
