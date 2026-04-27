# SCRN BRECQ 应用目录

本目录用于存放将 BRECQ 后训练量化算法迁移应用到 SCRN 模型的独立实现代码。

## 设计原则

- 只参考 `BRECQ-main/` 中的算法思想和实现结构，不直接导入其中的源码模块。
- 优先复用 `SCRN_BRECQ_app/scrn_repro/` 中已经完成的 SCRN 复现模型、数据读取和评估逻辑。
- BRECQ 相关适配代码集中放在本目录，避免污染 `SCRN-main/`、`BRECQ-main/` 和 `scrn_repro/`。
- 运行产物、模型权重、缓存、日志和数据文件不提交到 Git。

## 当前目录结构

```text
scrn_brecq/
├── README.md
├── DEVELOPMENT_LOG.md
├── __init__.py
├── configs/
│   └── default_quant_config.json
├── cli/
│   ├── __init__.py
│   ├── evaluate_quantized_scrn.py
│   ├── quantize_scrn.py
│   ├── smoke_check.py
│   └── verify_quantized_scrn.py
├── data/
│   ├── __init__.py
│   └── calibration_loader.py
├── model/
│   ├── __init__.py
│   └── scrn_loader.py
├── quant/
│   ├── __init__.py
│   ├── adaptive_rounding.py
│   ├── block_recon.py
│   ├── data_utils.py
│   ├── fold_bn.py
│   ├── layer_recon.py
│   ├── quant_block.py
│   ├── quant_layer.py
│   └── quant_model.py
├── utils/
│   ├── __init__.py
│   └── io.py
└── runs/
    └── README.md
```

## 模块职责

- `configs/`: 默认量化配置，包含 checkpoint、calibration 数据、评估输入、bitwidth 和 reconstruction 参数。
- `data/`: 从 `scrn_repro` clean patch 数据中收集 SCRN degraded calibration 输入。
- `model/`: 将 SCRN 训练 checkpoint 恢复成 BRECQ 可处理的 FP32 `nn.Module`。
- `quant/`: BRECQ 迁移核心，包括量化层、AdaRound、BN folding、QuantModel、SCRN block 适配和 layer/block reconstruction。
- `cli/quantize_scrn.py`: 执行完整 SCRN-BRECQ 量化、重构、评估和 checkpoint 保存。
- `cli/evaluate_quantized_scrn.py`: 重新加载已保存的 `quantized_scrn_brecq.pth` 并评估量化模型。
- `cli/smoke_check.py`: 无权重依赖的快速结构检查，用合成输入验证 QuantModel 包装和量化前向。
- `cli/verify_quantized_scrn.py`: 检查已保存 checkpoint 是否真正启用量化，包括 bit 分布、离散等级和 FP32/量化输出差异。
- `runs/`: 量化运行产物目录。实际 run 输出被 `.gitignore` 忽略。

## 常用命令

完整量化 smoke run:

```bash
conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.quantize_scrn \
  --num-samples 2 \
  --batch-size 1 \
  --iters-w 1 \
  --run-name smoke_w_only \
  --device auto
```

重新评估已保存的量化 checkpoint:

```bash
conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn \
  --checkpoint SCRN_BRECQ_app/scrn_brecq/runs/quant/20260426_212245_smoke_w_only/checkpoints/quantized_scrn_brecq.pth \
  --device cpu
```

验证已保存 checkpoint 的量化真实性:

```bash
conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.verify_quantized_scrn \
  --checkpoint SCRN_BRECQ_app/scrn_brecq/runs/quant/20260426_212245_smoke_w_only/checkpoints/quantized_scrn_brecq.pth \
  --device cpu
```

无数据、无权重依赖的结构 smoke check:

```bash
conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.smoke_check --device cpu
```

## 输出产物

- `quantize_scrn.py` 默认写入 `SCRN_BRECQ_app/scrn_brecq/runs/quant/`。
- `evaluate_quantized_scrn.py` 默认写入 `SCRN_BRECQ_app/scrn_brecq/runs/quant_eval/`。
- 运行产物通常包括 `config.json`、`metrics.json`、`summary.md`、`prediction.npy`、可选 `comparison.png` 和 checkpoint。
- 正式 W4A32 重建建议从默认配置开始，即 `num_samples=1024`、`batch_size=16`、`iters_w=20000`、`act_quant=false`。
- `.npy`、`.pth`、运行目录、缓存和日志不应提交到 Git。
