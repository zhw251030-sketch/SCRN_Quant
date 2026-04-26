# SCRN-BRECQ 开发记录

本文档用于记录 `SCRN_BRECQ_app/scrn_brecq/` 目录内每次代码和文档修改的内容、原因、参考来源和验证方式。

写这个文档的目的有三个：

1. 让后续阅读者知道每个文件为什么存在、做了什么改动。
2. 区分“参考 BRECQ 原算法”与“为 SCRN 做的迁移适配”。
3. 在量化实验出现问题时，能够回溯每一步代码变化和验证结果。

## 2026-04-26 初始化目录骨架

### 修改内容

- 新增 `README.md`，说明本目录的目标、设计原则和初始目录结构。
- 新增 `DEVELOPMENT_LOG.md`，作为后续每次修改 `scrn_brecq/` 的开发记录。
- 新增 `configs/default_quant_config.json`，保存后续 SCRN-BRECQ 量化实验的默认参数占位。
- 新增各 Python 子包的 `__init__.py`，先用中文模块注释说明职责，不实现算法逻辑。
- 新增 `runs/README.md`，说明运行产物目录用途，并强调权重、日志、数据等不提交。

### 参考来源

- `BRECQ-main/main_imagenet.py`：参考其整体量化流程参数命名，例如 `n_bits_w`、`n_bits_a`、`num_samples`、`iters_w`、`iters_a`。
- `BRECQ-main/quant/`：参考后续需要迁移的模块边界，包括量化层、量化模型、重构逻辑和 BN 折叠。
- `SCRN_BRECQ_app/scrn_repro/`：参考现有 SCRN 复现目录的分层方式。

### 验证方式

- 本次只创建文档、配置和包初始化文件，不包含可运行算法。
- 后续提交前会对新增 Python 文件执行 `python -m py_compile`。

## 2026-04-26 SCRN-BRECQ 后续任务计划（九部分）

### 整体拆分（已从八部分修正为九部分）

1. SCRN 模型加载适配：把 SCRN 训练 checkpoint 恢复成 BRECQ 可处理的 FP32 `nn.Module`。
2. 校准数据加载器：从 SCRN patch 数据中采样 calibration data，提供给 BRECQ 重构使用。
3. 基础量化层：迁移并重写 BRECQ 的均匀仿射量化、STE 和 `QuantModule`。
4. SCRN 量化模型包装：递归替换 SCRN 中的 `Conv2d`、`Linear`，并处理 Conv-BN 折叠。
5. SCRN 专用 QuantBlock 适配：让 `FeatureFusionBlock` 成为可识别的 block reconstruction 单元。
6. AdaRound 权重量化：实现自适应 rounding 参数和 soft/hard rounding 策略。
7. 重构数据缓存：通过 hook 保存目标层/块的输入输出，为 reconstruction 提供数据。
8. Layer/Block Reconstruction：实现 BRECQ 的层重构和适配 SCRN `FeatureFusionBlock` 的块重构。
9. 命令行量化与评估：串联加载模型、校准数据、量化重构、保存结果和评估指标。

### 计划修订说明

- 原计划按八部分推进，后来为了单独处理 SCRN 的 `FeatureFusionBlock`，新增第五部分 `SCRN 专用 QuantBlock 适配`。
- 因此 AdaRound 顺延为第六部分，后续 reconstruction 数据缓存、Layer/Block Reconstruction 和 CLI/评估分别顺延为第七到第九部分。

### 训练结果选择

- 当前已有测试结果中，`20260425_195621_quant_10750_0_best_eval_gt_colorbar` 的指标更好：
  - `after_snr_db`: `11.78722661219287`
  - `after_ssim`: `0.8699862043155245`
- 因此后续默认示例优先使用其对应 checkpoint：
  - `SCRN_BRECQ_app/scrn_repro/runs/train/20260425_192916_four_gpu_train_quant_10750_0/checkpoints/best.pth`

## 2026-04-26 第一部分：SCRN 模型加载适配

### 修改内容

- 新增 `utils/io.py`，集中实现文件存在检查、JSON 读取和 PyTorch checkpoint 读取。
- 新增 `model/scrn_loader.py`，实现 SCRN checkpoint 到 FP32 SCRN 模型的加载适配。
- 更新 `model/__init__.py`，导出 SCRN 加载相关函数和默认推荐 checkpoint。
- 更新 `utils/__init__.py`，导出通用 I/O 函数，方便后续模块复用。
- 更新 `configs/default_quant_config.json`，加入当前效果更好的 SCRN checkpoint 路径。

### 参考来源

- `SCRN_BRECQ_app/scrn_repro/cli/test_scrn.py`：参考其 checkpoint 加载和 SCRN 构建方式。
- `SCRN_BRECQ_app/scrn_repro/cli/train_scrn.py`：参考 checkpoint 中 `model_config` 和 `model_state_dict` 的保存格式。
- `SCRN_BRECQ_app/scrn_repro/runs/test/*/metrics.json`：比较现有测试结果，选择默认 checkpoint。

### 验证方式

- 待提交前执行 `python -m py_compile` 检查新增 Python 文件。
- 使用 `conda run -n quant python` 实际加载推荐 checkpoint，确认模型结构、epoch、loss 和参数数量。

## 2026-04-26 第二部分：SCRN Calibration 数据加载器

### 修改内容

- 新增 `data/calibration_loader.py`，实现 SCRN-BRECQ calibration data 的数据集、DataLoader 和 tensor 收集入口。
- 更新 `data/__init__.py`，导出 calibration data 相关配置和函数。
- 更新 `configs/default_quant_config.json`，加入默认 `calibration_dataset_dir` 和 `num_workers`。
- `collect_calibration_inputs` 明确只收集 `SCRNPatchDataset` 返回的 degraded 输入，不收集 clean target。

### 参考来源

- `SCRN_BRECQ_app/scrn_repro/data/dataset.py`：复用 `SCRNPatchDataset` 的 `(degraded, clean)` 样本生成逻辑。
- `SCRN_BRECQ_app/scrn_repro/data/degradation.py`：沿用训练阶段的缺失道和高斯噪声退化方式。
- `BRECQ-main/main_imagenet.py`：参考其 `get_train_samples` 将 DataLoader 输入拼接成 calibration tensor 的接口形态。

### 默认数据选择

- 默认校准目录为 `SCRN_BRECQ_app/scrn_repro/datasets/scrn_quant_10750_0_patches`。
- 该目录与当前推荐 checkpoint 的训练数据规模一致，且位于 `SCRN_BRECQ_app` 下，便于后续脚本使用相对路径复现实验。

### 验证方式

- `python -m py_compile SCRN_BRECQ_app/scrn_brecq/data/calibration_loader.py SCRN_BRECQ_app/scrn_brecq/data/__init__.py`
- `python -m json.tool SCRN_BRECQ_app/scrn_brecq/configs/default_quant_config.json`
- `conda run -n quant python` 实际加载 `num_samples=8`、`batch_size=4` 的 calibration tensor，确认形状、dtype 和同 seed 可复现。

## 2026-04-26 第三部分：BRECQ 基础量化层

### 修改内容

- 新增 `quant/quant_layer.py`，实现 `StraightThrough`、`round_ste`、`lp_loss`、`UniformAffineQuantizer` 和 `QuantModule`。
- 更新 `quant/__init__.py`，导出基础量化接口，供后续 QuantModel、AdaRound 和 reconstruction 复用。
- `UniformAffineQuantizer` 支持 2 到 8 bit、tensor-wise/channel-wise、`max`/`mse` scale 初始化和可学习 activation scale。
- `QuantModule` 支持包装 `nn.Conv2d` 和 `nn.Linear`，并保留后续 AdaRound 需要的 `org_weight`、`org_bias`、`weight_quantizer`、`act_quantizer` 等字段。

### 参考来源

- `BRECQ-main/quant/quant_layer.py`：参考基础量化层接口、STE round、Lp loss 和量化前向流程。
- `BRECQ-main/quant/adaptive_rounding.py`：检查后续 AdaRound 依赖的 quantizer 字段，避免接口缺失。
- `BRECQ-main/quant/layer_recon.py` 与 `block_recon.py`：检查 reconstruction 阶段对 `QuantModule` 状态字段的调用方式。

### 验证方式

- `python -m py_compile SCRN_BRECQ_app/scrn_brecq/quant/quant_layer.py SCRN_BRECQ_app/scrn_brecq/quant/__init__.py`
- `conda run -n quant python` 验证 `UniformAffineQuantizer` 输出 shape/dtype、channel-wise scale 形状、`QuantModule` 三种量化状态前向和 `round_ste` 反向梯度。

## 2026-04-26 第四部分：SCRN 量化模型包装 QuantModel

### 计划修正

- BRECQ 源码中有 `quant_block.py`，但原实现面向 ResNet/MobileNet/RegNet 的 block。
- SCRN 的核心 block 是 `FeatureFusionBlock`，结构包含 CNN 分支和 Swin Transformer 分支，不能直接复用原 BRECQ block 类型。
- 因此后续新增独立的第五部分：实现 SCRN 专用 `quant_block.py`，第四部分先完成基础 QuantModel 层级替换。

### 修改内容

- 新增 `quant/fold_bn.py`，在量化包装前把 SCRN `conv_branch` 中的 `Conv2d + BatchNorm2d` 折叠到 Conv 参数中。
- 新增 `quant/quant_model.py`，递归把 SCRN 内的 `nn.Conv2d` 和 `nn.Linear` 替换成 `QuantModule`。
- `QuantModel` 支持统一设置量化状态、首尾层 8bit 和关闭网络输出激活量化。
- 更新 `quant/__init__.py`，导出 `QuantModel` 和 BN folding 工具。
- 更新 `quant/quant_layer.py`，将 `QuantModule` 中保存的 FP32 权重和 bias 副本注册为 buffer，保证 `QuantModel.to(device)` 时设备迁移一致。

### 参考来源

- `BRECQ-main/quant/quant_model.py`：参考递归替换 Conv/Linear、合并 ReLU 和量化状态控制接口。
- `BRECQ-main/quant/fold_bn.py`：参考 Conv-BN 折叠公式和递归查找方式。
- `SCRN_BRECQ_app/scrn_repro/model/scrn.py`：确认 SCRN 的 BN 主要存在于 `FeatureFusionBlock.conv_branch`。

### 验证方式

- `python -m py_compile SCRN_BRECQ_app/scrn_brecq/quant/fold_bn.py SCRN_BRECQ_app/scrn_brecq/quant/quant_model.py SCRN_BRECQ_app/scrn_brecq/quant/__init__.py`
- `conda run -n quant python` 加载推荐 SCRN checkpoint，构造 `QuantModel`，验证 FP32、W quant、W+A quant 三种状态均可前向。
- 验证 `QuantModule` 数量、首尾层 8bit 设置和最后输出激活量化关闭状态。

## 2026-04-26 第五部分：SCRN 专用 QuantBlock 适配

### 修改内容

- 新增 `quant/quant_block.py`，实现 `BaseQuantBlock`、`QuantFeatureFusionBlock` 和 `specials` 映射。
- 更新 `quant/quant_model.py`，新增 `wrap_quant_blocks` 参数，默认识别并包装 SCRN `FeatureFusionBlock`。
- `QuantModel.set_quant_state` 同时支持量化 block 和普通 `QuantModule`，并新增 `quant_blocks()` 迭代器。
- 更新 `quant/__init__.py`，导出 SCRN block 相关接口。

### 参考来源

- `BRECQ-main/quant/quant_block.py`：参考 block 级量化状态控制和 `specials` 映射设计。
- `SCRN_BRECQ_app/scrn_repro/model/scrn.py`：根据 `FeatureFusionBlock` 的 CNN/Transformer 双分支结构实现 SCRN 专用包装。
- `SCRN_BRECQ_app/scrn_brecq/quant/quant_model.py`：在现有 Conv/Linear 递归替换基础上增加 block 包装。

### 与原 BRECQ quant_block 的差异

- 原 BRECQ block 面向 ResNet/MobileNet/RegNet，不能直接用于 SCRN。
- 当前 `QuantFeatureFusionBlock` 保持 forward 委托原 `FeatureFusionBlock`，第五部分只建立 block 级识别和状态控制接口。
- 激活量化中残差相加后的额外 block 输出量化，留到后续 reconstruction/activation quant 阶段再细化。

### 验证方式

- `python -m py_compile SCRN_BRECQ_app/scrn_brecq/quant/quant_block.py SCRN_BRECQ_app/scrn_brecq/quant/quant_model.py SCRN_BRECQ_app/scrn_brecq/quant/__init__.py`
- `conda run -n quant python` 加载推荐 SCRN checkpoint，构造默认 `QuantModel(wrap_quant_blocks=True)`，确认 `BaseQuantBlock` 数量为 5。
- 验证 block 包装后 quant off、W quant、W+A quant 三种状态均可前向，并与 `wrap_quant_blocks=False` 的 quant off 输出一致。

## 2026-04-26 第六部分：AdaRound 权重量化

### 修改内容

- 新增 `quant/adaptive_rounding.py`，实现 `AdaRoundQuantizer`。
- 更新 `quant/__init__.py`，导出 `AdaRoundQuantizer`。
- 修正顶部整体任务计划，加入已完成的 SCRN 专用 `quant_block.py`，并将 AdaRound 标为第六部分。

### 参考来源

- `BRECQ-main/quant/adaptive_rounding.py`：参考 AdaRound 的 rounding 模式、hard-sigmoid 参数和 `alpha` 初始化方式。
- `BRECQ-main/quant/layer_recon.py` 与 `block_recon.py`：确认后续 reconstruction 会替换 `weight_quantizer` 并优化 `alpha`。
- `SCRN_BRECQ_app/scrn_brecq/quant/quant_layer.py`：复用 `UniformAffineQuantizer` 的 `delta`、`zero_point` 和 `round_ste`。

### 实现说明

- `AdaRoundQuantizer` 依赖已初始化的 `UniformAffineQuantizer`，否则会抛出明确错误。
- 兼容原 BRECQ 中的 `learned_round_sigmoid` 名称，内部按 `learned_hard_sigmoid` 处理。
- 对 rounding remainder 做轻微 clamp，避免 `alpha` 初始化时出现 NaN/Inf。

### 验证方式

- `python -m py_compile SCRN_BRECQ_app/scrn_brecq/quant/adaptive_rounding.py SCRN_BRECQ_app/scrn_brecq/quant/__init__.py`
- `conda run -n quant python` 验证 `alpha` 类型、soft target 范围、soft/hard 前向、nearest/nearest_ste 前向、无 NaN/Inf 和未初始化量化器错误。

## 2026-04-26 第七部分：Reconstruction 数据缓存工具

### 修改内容

- 新增 `quant/data_utils.py`，实现 reconstruction 前缓存目标层/块输入输出和输出梯度的工具。
- 更新 `quant/__init__.py`，导出 `save_inp_oup_data`、`save_grad_data`、hook 类和 `quantize_model_till`。
- `save_inp_oup_data` 支持 `QuantModule` 与 `BaseQuantBlock`，并保留最后不足一个 batch 的 calibration 样本。
- `save_grad_data` 使用 SCRN 量化输出与 FP32 输出之间的 MSE loss 计算目标层/块输出梯度。

### 参考来源

- `BRECQ-main/quant/data_utils.py`：参考 forward hook 截断、输入输出缓存、梯度缓存和 `quantize_model_till` 的整体结构。
- `BRECQ-main/quant/layer_recon.py` 与 `block_recon.py`：确认后续 reconstruction 会调用 `save_inp_oup_data` 和 `save_grad_data`。
- `SCRN_BRECQ_app/scrn_brecq/quant/quant_model.py` 与 `quant_block.py`：适配当前 `QuantModel`、`QuantModule`、`BaseQuantBlock` 的遍历和量化状态控制。

### SCRN 适配说明

- 原 BRECQ 面向分类模型，梯度缓存使用 FP32 logits 与量化 logits 的 KL loss。
- SCRN 是单通道图像恢复模型，输出不是类别分布，因此第七部分改用 `MSE(out_q, out_fp)` 计算梯度。
- `quantize_model_till` 对 SCRN block 做了额外处理：当目标是 block 内部的 `QuantModule` 时，不提前打开同一 block 后面的量化层。

### 验证方式

- `python -m py_compile SCRN_BRECQ_app/scrn_brecq/quant/data_utils.py SCRN_BRECQ_app/scrn_brecq/quant/__init__.py`
- `conda run -n quant python` 加载推荐 SCRN checkpoint，构造 `QuantModel` 和小规模 calibration tensor。
- 验证 `QuantModule` 与 `BaseQuantBlock` 的输入输出缓存、`asym=True` 路径、MSE 梯度缓存和 `quantize_model_till` 状态控制。

## 2026-04-26 第八部分：Layer/Block Reconstruction

### 修改内容

- 新增 `quant/layer_recon.py`，实现单个 `QuantModule` 的 BRECQ reconstruction 优化循环。
- 新增 `quant/block_recon.py`，实现 SCRN `BaseQuantBlock` 的 block reconstruction、loss 和 `LinearTempDecay`。
- 更新 `quant/__init__.py`，导出 `layer_reconstruction`、`block_reconstruction`、loss 类和温度衰减器。
- 权重量化阶段会把目标权重量化器替换为 `AdaRoundQuantizer`，训练时使用 soft rounding，结束后切换为 hard rounding。

### 参考来源

- `BRECQ-main/quant/layer_recon.py`：参考 layer reconstruction 的 AdaRound 替换、rounding 正则和优化流程。
- `BRECQ-main/quant/block_recon.py`：参考 block reconstruction、`LinearTempDecay` 和 block 内多层 AdaRound 参数收集方式。
- `SCRN_BRECQ_app/scrn_brecq/quant/data_utils.py`：复用第七部分的输入输出缓存和 SCRN MSE 梯度缓存。

### SCRN 适配说明

- 不导入原 BRECQ 使用的 `linklink`，`multi_gpu=True` 会抛出明确错误。
- Fisher 类 reconstruction loss 对 batch 以外维度统一展平，兼容 SCRN 的 `[N, C, H, W]` 中间特征。
- 激活量化接口已保留，但要求 `act_quant_params["leaf_param"] = True` 才能优化 activation `delta`。
- 当前 `QuantFeatureFusionBlock` 的 block-level `act_quantizer` 不参与 forward，因此 block 激活重构只收集内部 `QuantModule` 的 activation scale。

### 验证方式

- `python -m py_compile SCRN_BRECQ_app/scrn_brecq/quant/layer_recon.py SCRN_BRECQ_app/scrn_brecq/quant/block_recon.py SCRN_BRECQ_app/scrn_brecq/quant/__init__.py`
- `conda run -n quant python` 加载推荐 SCRN checkpoint，构造 `QuantModel` 和小规模 calibration tensor。
- 验证 layer/block reconstruction 两次迭代可运行，AdaRound 参数正确创建，soft targets 最终关闭，量化前向输出有限。
- 单独验证 `mse`、`fisher_diag`、`fisher_full` loss 和 `multi_gpu=True` 错误路径。

## 2026-04-26 第九部分：命令行量化与评估入口

### 修改内容

- 新增 `cli/quantize_scrn.py`，把模型加载、calibration data、QuantModel、BRECQ reconstruction、checkpoint 保存和 SCRN 测试评估串成一个命令行入口。
- 更新 `configs/default_quant_config.json`，增加 run 输出目录、评估数据路径、`opt_mode`、`asym`、`init_batch_size` 等完整运行参数。
- 更新 `.gitignore`，忽略 `SCRN_BRECQ_app/scrn_brecq/runs/quant/`，避免提交量化运行产物。
- CLI 默认执行 W-only BRECQ；当 `act_quant=true` 时，继续执行 activation reconstruction。

### 参考来源

- `SCRN_BRECQ_app/scrn_repro/cli/test_scrn.py`：复用单张 `.npy` 输入输出评估方式和 SNR/SSIM 指标。
- `SCRN_BRECQ_app/scrn_repro/training/run_manager.py`：复用 run 目录、JSON 和 summary 写入工具。
- `BRECQ-main/main_imagenet.py`：参考量化流程顺序，即初始化量化参数、执行 reconstruction、切换最终量化状态并评估。

### 实现说明

- 量化 checkpoint 保存为 `run_dir/checkpoints/quantized_scrn_brecq.pth`，包含 `quant_model_state_dict`、`model_config`、`quant_config`、源 checkpoint 信息和评估指标。
- `reconstruct_model` 遇到 `BaseQuantBlock` 后不再进入其内部，避免 block 内 `QuantModule` 被重复执行 layer reconstruction。
- 运行产物包括 checkpoint、`prediction.npy`、`metrics.json`、`summary.md` 和 `config.json`，均写入被忽略的 quant run 目录。

### 验证方式

- `python -m py_compile SCRN_BRECQ_app/scrn_brecq/cli/quantize_scrn.py`
- `python -m json.tool SCRN_BRECQ_app/scrn_brecq/configs/default_quant_config.json`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.quantize_scrn --num-samples 2 --batch-size 1 --iters-w 1 --run-name smoke_w_only --device auto`
- 验证 smoke run 能完成 W-only 量化、保存 checkpoint、写出 metrics/summary，且 `git status` 不出现 `.pth`、`.npy` 或 quant run 产物。
