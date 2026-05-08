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

- Dataset type: `paper_style_test_perpatch_absmax`
- Input dataset type: `paper_style_test`
- Input dataset dir: `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_test_478`
- Sample count: `478`
- Zero/tiny scale count: `18`

## Scale Summary

- count: `478`
- min: `3.1507986106588796e-24`
- p01: `4.578341337869511e-24`
- median: `0.038313694298267365`
- mean: `0.06918392573312036`
- p99: `0.7501971244812066`
- max: `1.0`

## Tiny-Scale Threshold Counts

- scale <= 1e-12: `18`
- scale <= 1e-09: `27`
- scale <= 1e-06: `33`
- scale <= 1e-04: `39`
- scale <= 1e-03: `139`

## Per-Source Counts

- Anisotropic: `75`
- Kerry3D: `16`
- Shots0001: `387`

## Risk Notes

- Near-zero patches in the unfiltered source dataset can be numerically amplified by per-patch absmax normalization.
- Each sample stores `normalization_scale` and `zero_or_tiny_scale` so later evaluation can restore amplitudes.
- Calibration derivatives copy normalized train files by `train_file`; they do not perform a new random draw.
