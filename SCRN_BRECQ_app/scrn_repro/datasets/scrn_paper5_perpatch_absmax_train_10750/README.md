# SCRN Per-Patch Absmax Normalized Clean Patches

This directory is a per-patch absmax-normalized derivative of an existing SCRN clean patch dataset.
The original dataset is preserved; these files are experimental inputs for normalized FP32/BRECQ studies.

## Protocol

- Normalization method: `per_patch_absmax`
- Formula: `normalized = patch / max(abs(patch))` when scale > `1e-12`
- Tiny-scale formula: patches with zero/tiny scale are copied without division.
- Restoration: `restored = normalized * normalization_scale`
- Restoration returns the original clean patch space, not raw SEG-Y amplitude units.

## Dataset

- Dataset type: `paper_style_train_perpatch_absmax`
- Input dataset type: `paper_style_train`
- Input dataset dir: `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_train_10750`
- Sample count: `10750`
- Zero/tiny scale count: `5400`

## Scale Summary

- count: `10750`
- min: `0.0`
- p01: `0.0`
- median: `2.7119266466201354e-15`
- mean: `0.017223822481803954`
- p99: `0.187086284160614`
- max: `1.0`

## Tiny-Scale Threshold Counts

- scale <= 1e-12: `5400`
- scale <= 1e-09: `5485`
- scale <= 1e-06: `5585`
- scale <= 1e-04: `5795`
- scale <= 1e-03: `7190`

## Per-Source Counts

- 1997_2.5D_shots: `300`
- 7m_shots_0201: `3355`
- Anisotropic_FD_Model: `750`
- Kerry3D: `480`
- Shots0001_0200: `5865`

## Risk Notes

- Near-zero patches in the unfiltered source dataset can be numerically amplified by per-patch absmax normalization.
- Each sample stores `normalization_scale` and `zero_or_tiny_scale` so later evaluation can restore amplitudes.
- Calibration derivatives copy normalized train files by `train_file`; they do not perform a new random draw.
