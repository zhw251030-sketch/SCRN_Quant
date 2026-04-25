# SCRN Training Patch Dataset

本目录用于保存 SCRN 独立复现流程的训练 patch 数据集。目录中的 `.npy` 文件是本地生成的数据产物，已由仓库根目录 `.gitignore` 的 `*.npy` 规则忽略，不应提交到 Git。

## Generation

- Generated at: 2026-04-25 16:16:25 CST
- Conda environment: `quant`
- Python: `/home/hanwen/anaconda3/envs/quant/bin/python`
- Python version: `3.10.20`
- NumPy version: `2.2.6`
- SEG-Y reader: `segyio`

Command:

```bash
conda run -n quant python -m SCRN_BRECQ_app.scrn_repro.cli.prepare_patches \
  --input-dir SCRN-main/data/train \
  --output-dir SCRN_BRECQ_app/scrn_repro/datasets/scrn_train_patches \
  --patch-size 128 128 \
  --stride 48 48 \
  --augment-times 4 \
  --max-patches 10000 \
  --seed 20260425 \
  --prefix scrn_train
```

## Input Files

The source directory was `SCRN-main/data/train/`. Only decompressed SEG-Y files were used:

| File | Size |
| --- | ---: |
| `1997_2.5D_shots.segy` | 170944 KB |
| `7m_shots_0201_0329.segy` | 925676 KB |
| `shots0001_0200.segy` | 1933808 KB |

The `.segy.gz` files in the same source directory were ignored.

## Patch Settings

- Patch size: `128 x 128`
- Stride: `48 x 48`
- Augment times: `4`
- Max patches: `10000`
- Random seed: `20260425`
- Normalization: enabled, using maximum absolute amplitude per shot gather
- Shot jump: `1`
- Output file pattern: `scrn_train_000001.npy` through `scrn_train_010000.npy`

## Verification

Generated file count:

```text
10000
```

Sample checks:

| File | Shape | Dtype | Min | Max | Mean | Std |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `scrn_train_000001.npy` | `(128, 128)` | `float32` | -1.000000 | 0.906063 | 0.000000 | 0.019959 |
| `scrn_train_005001.npy` | `(128, 128)` | `float32` | -0.013244 | 0.017942 | -0.000000 | 0.001597 |
| `scrn_train_010000.npy` | `(128, 128)` | `float32` | -0.009398 | 0.011642 | 0.000005 | 0.001106 |

This step only prepares clean training patches. Degraded inputs for simultaneous denoising and interpolation are generated later by applying missing-trace masks and Gaussian noise during dataset loading.
