# SCRN Per-Patch Absmax Normalized Calibration Clean Patches

This directory is a per-patch absmax-normalized derivative of an existing SCRN clean patch dataset.
The original dataset is preserved; these files are experimental inputs for normalized FP32/BRECQ studies.

## Protocol

- Normalization method: `per_patch_absmax`
- Formula: `normalized = patch / max(abs(patch))` when scale > `1e-12`
- Tiny-scale formula: patches with zero/tiny scale are copied without division.
- Restoration: `restored = normalized * normalization_scale`
- Restoration returns the original clean patch space, not raw SEG-Y amplitude units.

## Dataset

- Dataset type: `paper_style_energy_filtered_calibration_perpatch_absmax`
- Input dataset type: `paper_style_calibration`
- Input dataset dir: `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_cali_1024_stratified`
- Sample count: `1024`
- Zero/tiny scale count: `0`

## Scale Summary

- count: `1024`
- min: `0.004283910617232323`
- p01: `0.00479383859783411`
- median: `0.04099760018289089`
- mean: `0.10550983919802093`
- p99: `1.0`
- max: `1.0`

## Tiny-Scale Threshold Counts

- scale <= 1e-12: `0`
- scale <= 1e-09: `0`
- scale <= 1e-06: `0`
- scale <= 1e-04: `0`
- scale <= 1e-03: `0`

## Per-Source Counts

- 1997_2.5D_shots: `28`
- 7m_shots_0201: `320`
- Anisotropic_FD_Model: `71`
- Kerry3D: `46`
- Shots0001_0200: `559`

## Risk Notes

- Near-zero patches in the unfiltered source dataset can be numerically amplified by per-patch absmax normalization.
- Each sample stores `normalization_scale` and `zero_or_tiny_scale` so later evaluation can restore amplitudes.
- Calibration derivatives copy normalized train files by `train_file`; they do not perform a new random draw.
