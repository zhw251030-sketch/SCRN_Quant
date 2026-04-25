"""SCRN 数据准备与退化样本生成模块。"""

from .degradation import (
    DEFAULT_MISSING_RATES,
    DEFAULT_SNR_DB_VALUES,
    DegradationInfo,
    degrade_patch,
    gaussian_noise_for_snr,
    make_random_trace_mask,
)
from .dataset import PatchSampleInfo, SCRNPatchDataset, build_train_loader, discover_patch_files
from .patches import (
    PatchExtractionConfig,
    augment_patch,
    collect_patches_from_segy_dir,
    iter_segy_files,
    normalize_by_absmax,
    save_patches_as_npy,
    split_patches,
)

__all__ = [
    "DEFAULT_MISSING_RATES",
    "DEFAULT_SNR_DB_VALUES",
    "DegradationInfo",
    "PatchExtractionConfig",
    "PatchSampleInfo",
    "SCRNPatchDataset",
    "augment_patch",
    "build_train_loader",
    "collect_patches_from_segy_dir",
    "degrade_patch",
    "discover_patch_files",
    "gaussian_noise_for_snr",
    "iter_segy_files",
    "make_random_trace_mask",
    "normalize_by_absmax",
    "save_patches_as_npy",
    "split_patches",
]
