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
│   ├── evaluate_quantized_scrn_multi.py
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
│   ├── io.py
│   └── model_size.py
└── runs/
    └── README.md
```

## 模块职责

- `configs/`: 默认量化配置，包含 checkpoint、calibration 数据、评估输入、bitwidth 和 reconstruction 参数。
- `data/`: 从 `scrn_repro` clean patch 数据中收集 SCRN degraded calibration 输入。
- `model/`: 将 SCRN 训练 checkpoint 恢复成 BRECQ 可处理的 FP32 `nn.Module`。
- `quant/`: BRECQ 迁移核心，包括量化层、AdaRound、BN folding、QuantModel、SCRN block 适配和 layer/block reconstruction。
- `cli/quantize_scrn.py`: 执行完整 SCRN-BRECQ 量化、重构、评估和 checkpoint 保存；会同时保存重建前与重建后的量化 checkpoint。
- `cli/evaluate_quantized_scrn.py`: 重新加载已保存的 `quantized_scrn_brecq.pth` 并评估量化模型。
- `cli/evaluate_quantized_scrn_multi.py`: 对已保存量化 checkpoint 做多样本评估，输出逐样本 JSONL 和聚合泛化指标；可额外加载重建前 checkpoint 对比 BRECQ reconstruction 前后变化。
- `cli/smoke_check.py`: 无权重依赖的快速结构检查，用合成输入验证 QuantModel 包装和量化前向。
- `cli/verify_quantized_scrn.py`: 检查已保存 checkpoint 是否真正启用量化，包括 bit 分布、离散等级和 FP32/量化输出差异。
- `utils/model_size.py`: 统计 checkpoint 文件大小、权重 bit 分布和理论 packed 模型大小。
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

指定单张 GPU 运行量化:

```bash
conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.quantize_scrn \
  --gpus 1 \
  --device cuda
```

使用 torchrun 做 W-only 多卡 BRECQ 重建:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n quant torchrun --standalone --nproc_per_node=4 \
  -m SCRN_BRECQ_app.scrn_brecq.cli.quantize_scrn \
  --distributed \
  --num-samples 1024 \
  --batch-size 16 \
  --iters-w 20000 \
  --run-name w4_recon_1024samples_20000iters_dist4 \
  --device cuda
```

重新评估已保存的量化 checkpoint:

```bash
conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn \
  --checkpoint SCRN_BRECQ_app/scrn_brecq/runs/quant/20260426_212245_smoke_w_only/checkpoints/quantized_scrn_brecq.pth \
  --device cpu
```

多样本泛化评估已保存的量化 checkpoint:

```bash
conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_multi \
  --checkpoint SCRN_BRECQ_app/scrn_brecq/runs/quant/20260427_192819_w4_recon_1024samples_20000iters_dist4_bsz32_global128/checkpoints/quantized_scrn_brecq.pth \
  --num-eval-samples 128 \
  --batch-size 16 \
  --run-name global128_quant10750_eval128 \
  --device auto
```

如果对应量化 run 保存了 `quantized_scrn_brecq_pre_recon.pth`，可以同时评估重建前和重建后结果:

```bash
conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_multi \
  --checkpoint SCRN_BRECQ_app/scrn_brecq/runs/quant/<run>/checkpoints/quantized_scrn_brecq.pth \
  --pre-recon-checkpoint SCRN_BRECQ_app/scrn_brecq/runs/quant/<run>/checkpoints/quantized_scrn_brecq_pre_recon.pth \
  --num-eval-samples 128 \
  --batch-size 16 \
  --run-name global128_prepost_eval128 \
  --device auto
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
- `evaluate_quantized_scrn_multi.py` 默认写入 `SCRN_BRECQ_app/scrn_brecq/runs/generalization_eval/`。
- 运行产物通常包括 `config.json`、`metrics.json`、`summary.md`、`prediction.npy`、可选 `comparison.png` 和 checkpoint。
- `quantize_scrn.py` 在权重量化初始化后保存 `checkpoints/quantized_scrn_brecq_pre_recon.pth`，在 BRECQ reconstruction 后保存 `checkpoints/quantized_scrn_brecq.pth`。前者用于分析“只量化、不重建”的基线，后者是最终量化模型。
- 多样本评估产物包括 `config.json`、`metrics.json`、`summary.md`、`per_sample_metrics.jsonl` 和可选 `figures/`，默认不保存全部预测 `.npy`。
- 多样本评估未传 `--pre-recon-checkpoint` 时，`quant_*` 旧字段和图中的 `Quant Post-Recon` 都表示最终重建后 checkpoint；传入后会额外输出 `quant_pre_recon_*`、`quant_post_recon_*` 和 `quant_post_minus_pre_*` 指标，并保存五图对比。
- `metrics.json` 会记录推理耗时、BRECQ reconstruction 耗时、本次量化流程总耗时、checkpoint 文件大小、权重 bit 参数分布和理论打包模型大小。
- 当前 `.pth` 是可恢复的 PyTorch checkpoint，不是 bit-packed 部署文件；真实 4bit 压缩收益应优先看 `model_size.estimated_storage`。
- 正式 W4A32 重建建议从默认配置开始，即 `num_samples=1024`、`batch_size=16`、`iters_w=20000`、`act_quant=false`。
- 分布式量化当前只支持 W-only；`--distributed` 与 `--act-quant` 同时使用会报错。
- `.npy`、`.pth`、运行目录、缓存和日志不应提交到 Git。
