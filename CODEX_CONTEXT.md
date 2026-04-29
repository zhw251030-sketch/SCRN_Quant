# SCRN_BRECQ Codex 共享上下文

最后更新：2026-04-29

本文件是在同一服务器、同一系统用户下，不同 Codex/OpenAI 账号协作时使用的固定交接上下文。开启新 Codex 对话时请先阅读本文件，再检查当前 Git 状态；因为本文件可能落后于最新提交。

## 项目概况

- 仓库根目录：`/home/data1/hanwen/project/Project/SCRN_Quant`
- 目标：将 BRECQ 后训练量化算法迁移并验证到 SCRN 地震重建模型上。
- 原始参考仓库：
  - `SCRN-main/`：原始 SCRN 源码和本地数据产物。
  - `BRECQ-main/`：原始 BRECQ 源码。
- 集成代码：
  - `SCRN_BRECQ_app/scrn_repro/`：SCRN 复现代码。
  - `SCRN_BRECQ_app/scrn_brecq/`：BRECQ 迁移和 SCRN 量化代码。
- 除非用户明确要求，否则不要修改 `SCRN-main/` 或 `BRECQ-main/`。

## 工作流规则

- 每次任务开始前执行：
  - `git status`
  - `git branch --show-current`
  - `git rev-parse --show-toplevel`
- 优先在 `SCRN_BRECQ_app/` 下修改。
- 修改 `SCRN_BRECQ_app/scrn_brecq/` 下文件时，同步更新 `SCRN_BRECQ_app/scrn_brecq/DEVELOPMENT_LOG.md`。
- 不要使用 `git add .`。
- 不要提交数据集、权重、日志、缓存、`.npy`、`.segy`、`.pth`、`.pt`、`.ckpt`、`__pycache__` 或 `.ipynb_checkpoints`。
- 每次提交前展示：
  - `git status`
  - `git diff`
  - `git diff --staged`
- 提交前运行最小检查，例如 `py_compile`、`json.tool`、CLI `--help` 或小型 smoke run。
- 除非用户明确要求，不要执行 `git push`。
- 避免危险命令，例如 `git reset --hard`、`git clean -fd`、删除数据目录的 `rm -rf` 或 force push。

## 当前代码能力

- SCRN 复现：
  - 模型实现在 `SCRN_BRECQ_app/scrn_repro/model/`。
  - clean patch 数据集和在线退化逻辑在 `SCRN_BRECQ_app/scrn_repro/data/`。
  - 训练和单样本测试 CLI 在 `SCRN_BRECQ_app/scrn_repro/cli/`。
- BRECQ 迁移：
  - SCRN 加载器在 `SCRN_BRECQ_app/scrn_brecq/model/scrn_loader.py`。
  - calibration loader 使用 `SCRNPatchDataset` 生成的 degraded 输入。
  - 量化模块、AdaRound、BN folding、QuantModel 和 reconstruction 逻辑在 `SCRN_BRECQ_app/scrn_brecq/quant/`。
  - 主量化 CLI：`SCRN_BRECQ_app.scrn_brecq.cli.quantize_scrn`。
  - 量化 checkpoint 评估 CLI：`SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn`。
  - 量化真实性验证 CLI：`SCRN_BRECQ_app.scrn_brecq.cli.verify_quantized_scrn`。
  - 多样本泛化评估 CLI：`SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_multi`。
- 分布式 BRECQ：
  - `quantize_scrn.py` 支持使用 torchrun 做多 GPU W-only reconstruction。
  - 分布式激活量化当前有意不支持。

## 重要实验 run

- 单卡 W4A32 基线：
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260427_152554_w4_recon_1024samples_20000iters`
  - post-recon SNR `11.5909`，SSIM `0.8385`。
- 四卡 W4A32，全局 batch 64：
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260427_165123_w4_recon_1024samples_20000iters_dist4_compare`
  - post-recon SNR `11.6952`，SSIM `0.8608`。
- 四卡 W4A32，全局 batch 128：
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260427_192819_w4_recon_1024samples_20000iters_dist4_bsz32_global128`
  - post-recon SNR `11.7469`，SSIM `0.8675`。
- 在 128 个 `scrn_quant_10750_0_patches` 样本上的多样本泛化评估：
  - `SCRN_BRECQ_app/scrn_brecq/runs/generalization_eval/20260427_222925_global128_quant10750_eval128`
  - FP32 平均 SNR `6.0901`，量化平均 SNR `4.8802`，平均 SNR gap `-1.2099 dB`。
  - FP32 平均 SSIM `0.7562`，量化平均 SSIM `0.7161`，平均 SSIM gap `-0.0401`。

## 模型大小指标解释

- 当前 `quantized_scrn_brecq.pth` 是可恢复的 PyTorch checkpoint，不是 bit-packed 部署文件。
- 实际量化 checkpoint 大小约为 `5.07 MiB`，接近 FP32 checkpoint。
- 理论 packed 模型大小约为 `0.2427 MiB`。
- 估算模型压缩率约为 `6.77x`。
- 估算量化权重压缩率约为 `7.98x`。
- 典型 W4A32 bit 分布为 `{"4": 50, "8": 2}`。

## 常用命令

运行 W4A32 四卡 reconstruction：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n quant torchrun --standalone --nproc_per_node=4 \
  -m SCRN_BRECQ_app.scrn_brecq.cli.quantize_scrn \
  --distributed \
  --num-samples 1024 \
  --batch-size 32 \
  --iters-w 20000 \
  --run-name w4_recon_1024samples_20000iters_dist4_bsz32_global128 \
  --device cuda
```

验证量化 checkpoint：

```bash
conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.verify_quantized_scrn \
  --checkpoint <run>/checkpoints/quantized_scrn_brecq.pth \
  --output-json <run>/verification.json \
  --device cpu
```

运行多样本泛化评估：

```bash
conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_multi \
  --checkpoint <run>/checkpoints/quantized_scrn_brecq.pth \
  --num-eval-samples 128 \
  --batch-size 16 \
  --device auto
```

## 交接备注

- 创建本文件时最近的重要提交：
  - `be85cbc Add multi-sample quantized SCRN evaluation`
  - `b938329 Add quantized model size metrics`
  - `ced6c09 Add quantization runtime metrics`
  - `34a9ad1 Add distributed BRECQ reconstruction support`
- 本地分支可能领先 `origin/main`；除非用户明确要求，不要 push。
- 之后跨账号交接时，请更新本文件中的：
  - 新提交 hash，
  - 重要 run 目录，
  - 关键指标，
  - 未解决问题，
  - 下一步建议任务。
