# SCRN 复现目录

本目录保存 SCRN 的独立复现代码，为 `scrn_brecq/` 量化迁移提供模型、数据读取和评估支撑。

## 设计边界

- 不导入 `SCRN-main/` 的 Python 源码，模型结构在 `model/scrn.py` 中独立复现。
- 训练和测试 run 产物写入 `SCRN_BRECQ_app/scrn_repro/runs/`，该目录被 `.gitignore` 忽略。
- 数据 patch、checkpoint、预测数组和日志不提交到 Git。
- `scrn_brecq/` 可以复用这里的模型、Dataset、metrics 和 run 管理工具，但 BRECQ 量化算法不放在本目录。

## 当前目录结构

```text
scrn_repro/
├── README.md
├── __init__.py
├── cli/
│   ├── __init__.py
│   ├── prepare_patches.py
│   ├── smoke_check.py
│   ├── test_scrn.py
│   └── train_scrn.py
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   ├── degradation.py
│   └── patches.py
├── datasets/
│   ├── scrn_quant_10750_0_patches/
│   │   └── README.md
│   └── scrn_train_patches/
│       └── README.md
├── model/
│   ├── __init__.py
│   └── scrn.py
├── training/
│   ├── __init__.py
│   ├── checkpoint.py
│   └── run_manager.py
└── utils/
    ├── __init__.py
    ├── metrics.py
    └── misc.py
```

## 复现流程

准备 clean patch 数据:

```bash
conda run -n quant python -m SCRN_BRECQ_app.scrn_repro.cli.prepare_patches \
  --input-dir <segy_dir> \
  --output-dir SCRN_BRECQ_app/scrn_repro/datasets/scrn_train_patches
```

训练 SCRN:

```bash
conda run -n quant python -m SCRN_BRECQ_app.scrn_repro.cli.train_scrn \
  --dataset-dir SCRN_BRECQ_app/scrn_repro/datasets/scrn_train_patches \
  --device auto
```

测试 SCRN checkpoint:

```bash
conda run -n quant python -m SCRN_BRECQ_app.scrn_repro.cli.test_scrn \
  --checkpoint <run_dir>/checkpoints/best.pth \
  --device auto \
  --save-figure
```

无数据、无权重依赖的 smoke check:

```bash
conda run -n quant python -m SCRN_BRECQ_app.scrn_repro.cli.smoke_check --device cpu
```

## 默认数据和权重

- `datasets/scrn_train_patches/` 用于 SCRN 训练 clean patch。
- `datasets/scrn_quant_10750_0_patches/` 用于 BRECQ calibration clean patch。
- 当前 BRECQ 默认 checkpoint 指向 `runs/train/20260425_192916_four_gpu_train_quant_10750_0/checkpoints/best.pth`。
- 上述 `.npy` 数据和 `.pth` 权重是本地复现实验依赖，默认不进入 Git。
