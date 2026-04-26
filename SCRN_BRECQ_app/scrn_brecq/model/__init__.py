"""SCRN 模型适配子包。

后续会在这里实现 SCRN checkpoint 加载、模型构建和配置恢复逻辑。
该子包只处理 SCRN 模型本身，不放 BRECQ 量化算法。
"""

from .scrn_loader import (
    DEFAULT_RECOMMENDED_SCRN_CHECKPOINT,
    LoadedSCRN,
    build_scrn_model,
    extract_model_state_dict,
    load_scrn_for_brecq,
    load_scrn_model,
    scrn_config_from_checkpoint,
)

__all__ = [
    "DEFAULT_RECOMMENDED_SCRN_CHECKPOINT",
    "LoadedSCRN",
    "build_scrn_model",
    "extract_model_state_dict",
    "load_scrn_for_brecq",
    "load_scrn_model",
    "scrn_config_from_checkpoint",
]

