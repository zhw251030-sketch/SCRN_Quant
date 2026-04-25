# SCRN Quant 10750_0 Training Patch Dataset

本目录保存 SCRN 复现与后续 BRECQ 量化整合中优先使用的一份 clean patch 训练集。
`.npy` 数据文件是本地产物，受仓库根目录 `.gitignore` 的 `*.npy` 规则忽略，不应提交到 Git。

## Source

- Copied at: 2026-04-25 CST
- Source directory: `/home/data1/hanwen/project/Project/SCRN_quant/train_data/10750_0`
- Destination directory: `SCRN_BRECQ_app/scrn_repro/datasets/scrn_quant_10750_0_patches`
- File pattern: `train_data_*.npy`
- File count: `10750`
- Directory size after copy: approximately `715M`

## Format

Each file is a clean seismic patch used as the supervised target. During SCRN training,
`SCRNPatchDataset` generates degraded inputs online by applying random missing-trace masks
and Gaussian noise, while keeping the clean patch as the target.

Sample checks:

| File | Shape | Dtype | Min | Max | Mean | Std |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `train_data_1.npy` | `(128, 128)` | `float32` | -0.242879 | 0.199932 | 0.000000 | 0.004427 |
| `train_data_10.npy` | `(128, 128)` | `float32` | -0.012652 | 0.016928 | 0.000000 | 0.001831 |
| `train_data_100.npy` | `(128, 128)` | `float32` | -0.019376 | 0.021564 | 0.000003 | 0.002155 |
| `train_data_5160.npy` | `(128, 128)` | `float32` | -0.011288 | 0.007226 | -0.000000 | 0.001768 |
| `train_data_9998.npy` | `(128, 128)` | `float32` | -0.007000 | 0.006701 | -0.000004 | 0.001662 |
| `train_data_9999.npy` | `(128, 128)` | `float32` | -0.007007 | 0.007619 | 0.000001 | 0.001420 |

## Reproduction Command

The dataset was validated with the SCRN reproduction training entrypoint:

```bash
conda run -n quant python -m torch.distributed.run --nproc_per_node=4 \
  -m SCRN_BRECQ_app.scrn_repro.cli.train_scrn \
  --dataset-dir SCRN_BRECQ_app/scrn_repro/datasets/scrn_quant_10750_0_patches \
  --run-name four_gpu_train_quant_10750_0 \
  --distributed \
  --epochs 80 \
  --batch-size 32 \
  --num-workers 2 \
  --seed 20260425
```

Reference run using the original source path:

- Train run: `SCRN_BRECQ_app/scrn_repro/runs/train/20260425_192916_four_gpu_train_quant_10750_0`
- Test run: `SCRN_BRECQ_app/scrn_repro/runs/test/20260425_195621_quant_10750_0_best_eval_gt_colorbar`
- Test metrics: `after_snr_db=11.7872`, `after_ssim=0.8700`

## Notes

- Keep this directory separate from `scrn_train_patches`, which was generated from SEG-Y files.
- Do not commit `.npy` files from this directory.
- Use this directory as the preferred local SCRN calibration/training patch source for upcoming BRECQ integration experiments.
