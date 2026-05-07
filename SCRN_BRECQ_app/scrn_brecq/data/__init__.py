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
from .stratified_scrn_datasets import (
    DEFAULT_CALIBRATION_QUOTAS,
    DEFAULT_TEST_QUOTAS,
    allocate_largest_remainder,
    prepare_calibration_dataset,
    prepare_test_dataset_from_segy_dir,
    select_stratified_calibration_files,
)
from .paper_scrn_datasets import (
    DEFAULT_CALIBRATION_QUOTAS as PAPER_DEFAULT_CALIBRATION_QUOTAS,
    DEFAULT_TEST_QUOTAS as PAPER_DEFAULT_TEST_QUOTAS,
    DEFAULT_TRAIN_QUOTAS as PAPER_DEFAULT_TRAIN_QUOTAS,
    PAPER_TEST_SOURCES,
    PAPER_TRAIN_SOURCES,
    prepare_calibration_dataset as prepare_paper_calibration_dataset,
    prepare_test_dataset_from_segy_dir as prepare_paper_test_dataset_from_segy_dir,
    prepare_train_dataset_from_segy_dir as prepare_paper_train_dataset_from_segy_dir,
)

__all__ = [
    "DEFAULT_CALIBRATION_DATASET_DIR",
    "DEFAULT_CALIBRATION_QUOTAS",
    "PAPER_DEFAULT_CALIBRATION_QUOTAS",
    "PAPER_DEFAULT_TEST_QUOTAS",
    "PAPER_DEFAULT_TRAIN_QUOTAS",
    "DEFAULT_TEST_QUOTAS",
    "PAPER_TEST_SOURCES",
    "PAPER_TRAIN_SOURCES",
    "CalibrationDataConfig",
    "allocate_largest_remainder",
    "build_calibration_dataset",
    "build_calibration_loader",
    "collect_calibration_inputs",
    "load_calibration_data",
    "prepare_calibration_dataset",
    "prepare_paper_calibration_dataset",
    "prepare_paper_test_dataset_from_segy_dir",
    "prepare_paper_train_dataset_from_segy_dir",
    "prepare_test_dataset_from_segy_dir",
    "select_stratified_calibration_files",
]
