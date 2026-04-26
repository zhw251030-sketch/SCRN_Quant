"""数据适配子包。

后续会在这里实现 BRECQ calibration 数据加载逻辑。
SCRN 的输入是单通道地震 patch，因此这里会适配 `scrn_repro` 已生成的 `.npy` patch 数据。
"""

from .calibration_loader import (
    DEFAULT_CALIBRATION_DATASET_DIR,
    CalibrationDataConfig,
    build_calibration_dataset,
    build_calibration_loader,
    collect_calibration_inputs,
    load_calibration_data,
)

__all__ = [
    "DEFAULT_CALIBRATION_DATASET_DIR",
    "CalibrationDataConfig",
    "build_calibration_dataset",
    "build_calibration_loader",
    "collect_calibration_inputs",
    "load_calibration_data",
]

