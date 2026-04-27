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

## 2026-04-26 第九部分补充：五图对比和重建前后指标

### 修改内容

- 更新 `cli/quantize_scrn.py`，在 reconstruction 前后分别评估量化模型，并保留 FP32 SCRN 输出。
- metrics 扩展为 `input`、`fp32`、`quant_pre_recon`、`quant_post_recon` 四组 SNR/SSIM，同时保留旧的 `before_*`、`after_*` 字段兼容 summary。
- 保存 `fp32_prediction.npy`、`quant_pre_recon_prediction.npy`、`quant_post_recon_prediction.npy`，并继续用 `prediction.npy` 指向最终重建后输出。
- 更新五图对比图：Ground Truth、Degraded Input、FP32 SCRN、Quant Before Reconstruction、Quant After BRECQ。
- 更新默认配置，将 `save_figure` 改为 `true`，默认运行会保存对比图。

### 设计原因

- 只看最终量化结果无法判断 BRECQ reconstruction 是否有效。
- 加入量化重建前结果后，可以直接比较“直接量化损失”和“BRECQ 重建恢复幅度”。

### 验证方式

- `python -m py_compile SCRN_BRECQ_app/scrn_brecq/cli/quantize_scrn.py`
- `python -m json.tool SCRN_BRECQ_app/scrn_brecq/configs/default_quant_config.json`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.quantize_scrn --num-samples 2 --batch-size 1 --iters-w 1 --run-name smoke_five_panel --device auto`
- 验证 run 目录中生成五图 `comparison.png`，metrics 包含 `fp32_*`、`quant_pre_recon_*`、`quant_post_recon_*`。

## 2026-04-27 文件结构合理性检查

### 检查范围

- 检查仓库根目录 `/home/data1/hanwen/project/Project/SCRN_Quant` 的顶层结构。
- 检查 `BRECQ-main/`、`SCRN-main/` 和 `SCRN_BRECQ_app/` 三个主要目录的职责边界。
- 检查 `SCRN_BRECQ_app/scrn_brecq/` 内部模块划分是否覆盖 SCRN 上迁移 BRECQ 的复现流程。
- 检查 `.gitignore` 是否覆盖数据、权重、运行日志、缓存和量化输出等不应提交的运行产物。

### 检查结论

- 当前 Git 实际只有仓库根目录下一个 `.git`，`BRECQ-main/`、`SCRN-main/` 和 `SCRN_BRECQ_app/` 是同一 Git 仓库内的三个顶层代码区域；如果后续文档中继续称为“三个仓库”，应理解为三个代码来源/应用区域，而不是三个嵌套 Git 仓库。
- `BRECQ-main/` 保留原 BRECQ 参考源码，`SCRN-main/` 保留原 SCRN 参考源码，新的迁移实现集中在 `SCRN_BRECQ_app/scrn_brecq/`，符合“不直接污染参考源码仓库”的复现应用要求。
- `SCRN_BRECQ_app/scrn_repro/` 保存 SCRN 复现模型、数据处理、训练和测试支撑代码；`SCRN_BRECQ_app/scrn_brecq/` 保存 BRECQ 迁移量化代码，两者职责边界清楚。
- `scrn_brecq/` 当前模块划分为 `configs/`、`cli/`、`data/`、`model/`、`quant/`、`utils/` 和 `runs/`，可以对应配置、命令入口、校准数据、SCRN 模型加载、量化算法、通用 I/O 和运行输出，结构能够支撑现有复现流程。
- `quant/` 下已经按 BRECQ 迁移流程拆分出量化层、BN folding、量化模型包装、SCRN block 适配、AdaRound、数据缓存、layer reconstruction 和 block reconstruction，模块边界与后训练量化流程基本一致。
- `.gitignore` 已覆盖 `SCRN-main` 下的大数据/权重目录、`SCRN_BRECQ_app/scrn_repro/runs/`、`SCRN_BRECQ_app/scrn_brecq/runs/quant/`、`*.npy`、`*.pth`、`*.ckpt`、`__pycache__/` 等运行产物，当前结构不会把 smoke run、checkpoint、预测数组和缓存提交进 Git。

### 待澄清或后续优化点

- `SCRN_BRECQ_app/scrn_loader.py` 和 `SCRN_BRECQ_app/paths.py` 是顶层兼容/路径辅助文件，其中 `scrn_loader.py` 与 `scrn_brecq/model/scrn_loader.py` 在职责上有重叠；目前不影响量化 CLI，但后续应明确保留为兼容入口，或在确认没有旧调用后清理。
- `SCRN_BRECQ_app/scrn_brecq/README.md` 中的目录树仍是初始化骨架，已经落后于当前实现状态；后续做复现文档整理时应同步更新为当前完整结构。
- `SCRN_BRECQ_app/scrn_repro/datasets/` 和若干训练/测试产物在本地存在但被 Git 忽略；这对本机复现实验可用，但从干净 clone 复现时需要单独准备数据和 checkpoint。
- 根目录下存在本地 `application/` 空目录，当前没有被 Git 跟踪，也没有参与 SCRN-BRECQ 复现流程；暂不处理。

### 验证方式

- `git status`
- `git branch --show-current`
- `git rev-parse --show-toplevel`
- `find . -maxdepth 2 -type d`
- `find . -name .git -type d`
- `rg --files SCRN_BRECQ_app`
- `git ls-files`
- `git check-ignore -v SCRN_BRECQ_app/__pycache__ SCRN_BRECQ_app/scrn_brecq/runs/quant SCRN_BRECQ_app/scrn_repro/runs SCRN-main/trained_model/model.pth SCRN-main/test_data/clear.npy`
- `conda run -n quant python -m compileall -q SCRN_BRECQ_app/scrn_brecq SCRN_BRECQ_app/scrn_repro`
- `conda run -n quant python -m json.tool SCRN_BRECQ_app/scrn_brecq/configs/default_quant_config.json`

## 2026-04-27 SCRN 复现与 BRECQ 迁移文件齐备性详查

### 检查目标

- `SCRN_BRECQ_app/scrn_repro/` 应能独立支撑 SCRN 复现：模型定义、数据准备、在线退化、训练、测试、指标、checkpoint 和 run 记录都应有明确文件。
- `SCRN_BRECQ_app/scrn_brecq/` 应能支撑在 SCRN 上应用 BRECQ：SCRN checkpoint 加载、calibration 输入、量化模型包装、基础量化器、AdaRound、layer/block reconstruction、命令行量化评估、配置和运行产物隔离都应有明确文件。

### `scrn_repro/` 文件齐备性

- 包入口：已有 `__init__.py`，目录可以作为 `SCRN_BRECQ_app.scrn_repro` 包导入。
- 模型复现：已有 `model/scrn.py` 和 `model/__init__.py`，包含 `SCRNConfig`、`SCRN`、`FeatureFusionBlock`、`SwinBlock`、窗口注意力、窗口 padding/reverse 和 `build_scrn_from_config`，满足复现 SCRN 网络结构的基本文件要求。
- 数据准备：已有 `data/patches.py`，覆盖 SEG-Y 文件发现、读取、归一化、patch 切分、增强和 `.npy` 保存；已有 `cli/prepare_patches.py` 作为数据准备命令行入口。
- 在线退化：已有 `data/degradation.py`，覆盖随机缺失道 mask、目标 SNR 高斯噪声和 `degrade_patch`；已有 `data/dataset.py`，用 clean patch 在线生成 `(degraded, clean)` 训练样本。
- 训练入口：已有 `cli/train_scrn.py`，覆盖单卡/CPU 训练、DDP 训练、Adam、MSE loss、MultiStepLR、checkpoint、metrics 和 summary 输出。
- 测试入口：已有 `cli/test_scrn.py`，覆盖从 checkpoint 恢复模型、读取 clean/input `.npy`、输出 prediction、SNR/SSIM 指标和可选对比图。
- 实验记录：已有 `training/checkpoint.py` 和 `training/run_manager.py`，覆盖 checkpoint 读写、run 目录、JSON/JSONL/CSV/summary 和环境信息记录。
- 指标与工具：已有 `utils/metrics.py` 和 `utils/misc.py`，覆盖 SNR、SSIM、随机种子和目录检查。
- 数据占位：已有 `datasets/scrn_train_patches/README.md` 和 `datasets/scrn_quant_10750_0_patches/README.md`；本地 `.npy` 数据存在但被 Git 忽略，符合不提交数据的要求。

结论：`scrn_repro/` 已具备 SCRN 复现所需的主要代码文件，能覆盖“准备数据 -> 训练 SCRN -> 测试 SCRN -> 保存结果”的闭环。当前更像研究脚本型复现目录，不是完整发布包；后续为了干净环境复现，应补充根目录 README 或复现说明，并考虑加入固定训练/测试配置文件。

### `scrn_brecq/` 文件齐备性

- 包入口与文档：已有 `__init__.py`、`README.md` 和 `DEVELOPMENT_LOG.md`，能说明目录目标并记录迁移过程。
- 默认配置：已有 `configs/default_quant_config.json`，覆盖 seed、device、run_root、checkpoint、calibration 数据、评估输入、量化 bit、BRECQ reconstruction 迭代数、loss 参数和图像保存开关。
- SCRN 模型加载：已有 `model/scrn_loader.py` 和 `model/__init__.py`，覆盖 checkpoint 读取、`model_config` 恢复、state dict 前缀处理、默认推荐 checkpoint 和 `LoadedSCRN` 元信息。
- calibration 数据：已有 `data/calibration_loader.py` 和 `data/__init__.py`，复用 `SCRNPatchDataset`，只收集 degraded 输入作为 BRECQ calibration data。
- 基础量化层：已有 `quant/quant_layer.py`，覆盖 STE、Lp loss、`UniformAffineQuantizer`、`QuantModule` 和 Conv/Linear 包装。
- BN folding：已有 `quant/fold_bn.py`，覆盖 Conv-BN 参数折叠、BN reset/remove 和递归搜索。
- 量化模型包装：已有 `quant/quant_model.py`，覆盖 SCRN 中 Conv/Linear 的递归替换、量化状态开关、首尾层 8bit 和输出激活量化关闭。
- SCRN block 适配：已有 `quant/quant_block.py`，覆盖 `BaseQuantBlock`、`QuantFeatureFusionBlock` 和 SCRN `FeatureFusionBlock` 的 specials 映射。
- AdaRound：已有 `quant/adaptive_rounding.py`，覆盖 soft/hard rounding、alpha 初始化和与 `UniformAffineQuantizer` 的连接。
- 重构缓存：已有 `quant/data_utils.py`，覆盖目标层/块输入输出缓存、梯度缓存、hook 截断和 `quantize_model_till`。
- 重构算法：已有 `quant/layer_recon.py` 和 `quant/block_recon.py`，覆盖 layer reconstruction、block reconstruction、rounding regularization、Fisher/MSE loss 和温度衰减。
- 量化入口：已有 `cli/quantize_scrn.py`，覆盖加载 SCRN、加载 calibration data、构造 QuantModel、初始化量化参数、执行 reconstruction、保存量化 checkpoint、输出 metrics/summary/prediction 和五图对比。
- 运行产物隔离：已有 `runs/README.md`，`.gitignore` 已忽略 `scrn_brecq/runs/quant/`、`.npy`、`.pth`、缓存和日志文件。

结论：`scrn_brecq/` 已具备把 BRECQ 后训练量化迁移到 SCRN 的主要实现文件，能覆盖“加载 FP32 SCRN -> 构造量化模型 -> 收集校准输入 -> BRECQ 权重量化重构 -> 评估保存”的闭环。当前缺口主要不是核心算法文件，而是复现实用性文件：缺少单独加载已保存量化 checkpoint 再评估的入口，`README.md` 的目录树仍停留在初始化状态，尚无独立测试目录或固定 smoke test 脚本。

### 本次发现的具体待办

- 更新 `scrn_brecq/README.md` 当前结构树，让它与已实现文件一致。
- 为 `scrn_repro/` 增加根 README 或复现说明，说明数据准备、训练、测试和默认 checkpoint 来源。
- 考虑为 `scrn_repro` 和 `scrn_brecq` 增加最小测试目录或 smoke test 脚本，避免只靠开发日志中的手动命令验证。
- 考虑新增 `scrn_brecq/cli/evaluate_quantized_scrn.py`，用于从保存的 `quantized_scrn_brecq.pth` 重新加载并评估量化模型，增强复现闭环。
- 明确 `SCRN_BRECQ_app/scrn_loader.py` 与 `scrn_brecq/model/scrn_loader.py` 的关系，避免之后读代码时误用旧兼容入口。

### 验证方式

- `git ls-files SCRN_BRECQ_app/scrn_repro SCRN_BRECQ_app/scrn_brecq`
- `find SCRN_BRECQ_app/scrn_repro SCRN_BRECQ_app/scrn_brecq -maxdepth 3 -type d`
- `find SCRN_BRECQ_app/scrn_repro SCRN_BRECQ_app/scrn_brecq -maxdepth 3 -type f ! -path '*/__pycache__/*' ! -name '*.npy' ! -name '*.pth' ! -name '*.pt' ! -name '*.ckpt'`
- `rg -n "^(class|def) " SCRN_BRECQ_app/scrn_repro SCRN_BRECQ_app/scrn_brecq`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_repro.cli.prepare_patches --help`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_repro.cli.train_scrn --help`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_repro.cli.test_scrn --help`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.quantize_scrn --help`
- `conda run -n quant python -m compileall -q SCRN_BRECQ_app/scrn_repro SCRN_BRECQ_app/scrn_brecq`
- `conda run -n quant python -m json.tool SCRN_BRECQ_app/scrn_brecq/configs/default_quant_config.json`
- `conda run -n quant python -c "...关键包导入检查..."`
- `conda run -n quant python -c "...SCRN 小输入前向检查..."`
- `conda run -n quant python -c "...QuantModel 包装和小输入前向检查..."`

### 轻量验证结果

- `prepare_patches.py`、`train_scrn.py`、`test_scrn.py` 和 `quantize_scrn.py` 的 `--help` 均能正常导入和显示参数。
- `compileall` 通过，说明两个目录下的 Python 文件语法可编译。
- `default_quant_config.json` 可被 `json.tool` 解析，且默认 checkpoint、calibration 数据目录、评估输入和 run 目录路径能被 `quantize_scrn.py` 的配置解析逻辑识别。
- 关键包导入通过：`scrn_repro.model`、`scrn_repro.data`、`scrn_repro.training`、`scrn_brecq.data`、`scrn_brecq.model` 和 `scrn_brecq.quant` 的核心对象均可导入。
- 使用缩小版 SCRN 做 `[1, 1, 16, 16]` 输入前向，输出形状为 `[1, 1, 16, 16]`。
- 使用缩小版 SCRN 构造 `QuantModel` 后做同样输入前向，输出形状为 `[1, 1, 16, 16]`，并识别出 52 个 `QuantModule` 和 5 个 `BaseQuantBlock`。

## 2026-04-27 补齐复现文档、smoke check 和量化 checkpoint 评估入口

### 修改内容

- 更新 `scrn_brecq/README.md`，把初始化目录树替换为当前完整结构，并补充模块职责、完整量化命令、量化 checkpoint 评估命令、smoke check 命令和运行产物说明。
- 新增 `scrn_repro/README.md`，说明 SCRN 复现目录的代码边界、数据准备、训练、测试、smoke check、默认数据和默认 checkpoint 来源。
- 更新 `SCRN_BRECQ_app/__init__.py` 和 `SCRN_BRECQ_app/scrn_loader.py`，明确它们是旧兼容入口；新的 BRECQ 量化流程应使用 `scrn_brecq/model/scrn_loader.py`。
- 新增 `scrn_repro/cli/smoke_check.py`，用合成 patch 验证在线退化、SCRN 小模型前向和 SNR/SSIM 指标工具。
- 新增 `scrn_brecq/cli/smoke_check.py`，用缩小版 SCRN 验证 `QuantModel` 包装、量化状态切换、量化前向、`QuantModule` 数量和 `BaseQuantBlock` 数量。
- 新增 `scrn_brecq/cli/evaluate_quantized_scrn.py`，用于加载已保存的 `quantized_scrn_brecq.pth`，重新构造量化模型并输出 `prediction.npy`、`metrics.json`、`config.json`、`summary.md` 和可选 `comparison.png`。
- 更新 `.gitignore`，忽略 `SCRN_BRECQ_app/scrn_brecq/runs/quant_eval/`，避免默认评估 run 产物进入 Git。

### 设计说明

- smoke check 采用脚本型 CLI，不引入 pytest，保持当前项目的研究脚本风格。
- `evaluate_quantized_scrn.py` 只评估量化 checkpoint 本身，不额外加载 FP32 checkpoint 做对比。
- 量化 checkpoint 中重构后的权重量化器是 `AdaRoundQuantizer`，而新构造的 `QuantModel` 默认使用 `UniformAffineQuantizer`。因此评估入口会先扫描 state dict 中的 `weight_quantizer.alpha` 键，按 `delta`、`zero_point` 和 `alpha` 恢复对应层的 AdaRound 结构，再执行严格 `load_state_dict`。
- 当前 activation quant checkpoint 格式没有保存 activation `zero_point`。评估入口为可能存在的 `act_quantizer.delta` 预创建参数以允许严格加载，但 activation zero point 会在首次前向时按评估输入刷新；当前已验证的 smoke checkpoint 是 W-only。

### 验证方式

- `git status`
- `git branch --show-current`
- `git rev-parse --show-toplevel`
- `conda run -n quant python -m compileall -q SCRN_BRECQ_app/scrn_repro SCRN_BRECQ_app/scrn_brecq`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_repro.cli.smoke_check --help`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.smoke_check --help`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn --help`
- `conda run -n quant python -m json.tool SCRN_BRECQ_app/scrn_brecq/configs/default_quant_config.json`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_repro.cli.smoke_check --device cpu`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.smoke_check --device cpu`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn --checkpoint SCRN_BRECQ_app/scrn_brecq/runs/quant/20260426_212245_smoke_w_only/checkpoints/quantized_scrn_brecq.pth --device cpu --run-root /tmp/scrn_brecq_quant_eval_smoke --run-name smoke_eval --no-save-figure`
- 量化 checkpoint 评估 smoke 输出目录为 `/tmp/scrn_brecq_quant_eval_smoke/20260427_143652_smoke_eval`，包含 `config.json`、`metrics.json`、`prediction.npy` 和 `summary.md`。
- 评估 smoke 指标：`input_snr=3.9693`、`quant_snr=11.4205`、`input_ssim=0.6053`、`quant_ssim=0.8270`。

## 2026-04-27 新增量化真实性验证 CLI

### 修改内容

- 新增 `cli/verify_quantized_scrn.py`，用于读取已保存的 `quantized_scrn_brecq.pth` 并验证量化是否真实生效。
- 更新 `README.md`，补充 `verify_quantized_scrn.py` 的目录说明、使用命令和正式 W4A32 重建建议参数。

### 验证口径

- 复用 `evaluate_quantized_scrn.py` 的 checkpoint 加载逻辑，避免重复实现 AdaRound 结构恢复。
- 报告 `final_quant_state`、`QuantModule` 数量、AdaRound 数量、权重量化 bit 分布、每个 bit 下的最大整数等级数，以及是否存在超过 `2**bit` 的异常层。
- 使用同一 checkpoint 比较 FP32 路径和量化路径输出，报告 SNR/SSIM、最大绝对差、平均绝对差和 MSE。
- `passed=true` 需要同时满足：存在量化模块、无离散等级异常、FP32 与量化输出存在非零差异。

### 正式重建检查计划

- 先提交验证 CLI 与文档改动，避免正式 run 产物和代码改动混在一个提交里。
- 随后运行 W4A32 正式检查：
  - `num_samples=1024`
  - `batch_size=16`
  - `iters_w=20000`
  - `act_quant=false`
  - `run_name=w4_recon_1024samples_20000iters`
- 正式 run 完成后，对生成的 `quantized_scrn_brecq.pth` 运行 `verify_quantized_scrn.py`，并把验证 JSON 写到对应 run 目录。

### 验证方式

- 待提交前执行：
  - `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/cli/verify_quantized_scrn.py`
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.verify_quantized_scrn --help`
  - 使用已有 smoke checkpoint 运行 `verify_quantized_scrn.py` 做功能验证。
- 已用 `20260426_214141_smoke_five_panel_layout` 的 checkpoint 验证：
  - `passed=true`
  - `weight_bit_counts={"4": 50, "8": 2}`
  - `max_unique_int_levels_per_channel_by_bit={"4": 16, "8": 214}`
  - `level_offender_count=0`
  - `fp32_quant_max_abs_diff=0.06882372498512268`

## 2026-04-27 正式 W4A32 重建与真实性验证

### 运行命令

- 正式重建：
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.quantize_scrn --num-samples 1024 --batch-size 16 --iters-w 20000 --run-name w4_recon_1024samples_20000iters --device auto`
- 量化真实性验证：
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.verify_quantized_scrn --checkpoint SCRN_BRECQ_app/scrn_brecq/runs/quant/20260427_152554_w4_recon_1024samples_20000iters/checkpoints/quantized_scrn_brecq.pth --output-json SCRN_BRECQ_app/scrn_brecq/runs/quant/20260427_152554_w4_recon_1024samples_20000iters/verification.json --device cpu`

### 运行产物

- run 目录：`SCRN_BRECQ_app/scrn_brecq/runs/quant/20260427_152554_w4_recon_1024samples_20000iters`
- 已生成 `metrics.json`、`summary.md`、`comparison.png`、`prediction.npy`、`fp32_prediction.npy`、`quant_pre_recon_prediction.npy`、`quant_post_recon_prediction.npy`、`checkpoints/quantized_scrn_brecq.pth` 和 `verification.json`。
- 这些产物位于已忽略的 `runs/quant/`，不提交到 Git。

### 指标结果

- 输入退化：`SNR=3.9693 dB`、`SSIM=0.6053`
- FP32 SCRN：`SNR=11.7869 dB`、`SSIM=0.8697`
- W4A32 重建前：`SNR=11.4071 dB`、`SSIM=0.8255`
- W4A32 BRECQ 重建后：`SNR=11.5909 dB`、`SSIM=0.8385`
- BRECQ 重建带来约 `+0.1838 dB` SNR 和 `+0.0130` SSIM 提升。
- 重建后相对 FP32 约下降 `0.1960 dB` SNR 和 `0.0312` SSIM。

### 真实性验证结果

- `passed=true`
- `final_quant_state={"weight_quant": true, "act_quant": false}`，符合 W4A32。
- `quant_modules=52`，`adaround_modules=51`。
- `weight_bit_counts={"4": 50, "8": 2}`，首尾层保持 8bit，其余权重量化为 4bit。
- `level_offender_count=0`。
- `max_unique_int_levels_per_channel_by_bit={"4": 16, "8": 219}`，4bit 层未超过 16 个整数等级，8bit 层未超过 256 个整数等级。
- FP32 路径与量化路径存在非零差异：`fp32_quant_max_abs_diff=0.03247570991516113`、`fp32_quant_mean_abs_diff=0.003559230826795101`。

### 结论

- 本次正式检查确认 checkpoint 不是只包了一层 `QuantModel` 的假量化：权重量化器落在目标整数网格，最终状态为 W-only，并且量化前向与 FP32 前向存在可测差异。
- 4bit 掉点较小在该单张默认测试样本上是可观察结果，但仍需用更多测试样本或完整测试集统计均值后再判断泛化稳定性。

## 2026-04-27 增加 torchrun 多卡 W-only BRECQ 重建

### 修改内容

- `cli/quantize_scrn.py` 新增 `--distributed` 和 `--gpus`，支持用 torchrun 启动多进程量化。
- 分布式模式读取 `RANK`、`LOCAL_RANK` 和 `WORLD_SIZE`，每个 rank 绑定一张 CUDA 设备；rank 0 负责创建 run 目录、评估和保存产物，其他 rank 只参与 reconstruction。
- `data/calibration_loader.py` 增加 rank/world_size 分片，`num_samples` 在分布式下保持全局语义。例如 `num_samples=1024`、`world_size=4` 时每个 rank 读取约 256 个 calibration 样本。
- `quant/block_recon.py` 和 `quant/layer_recon.py` 支持 `multi_gpu=True`，在 `backward()` 后对 AdaRound 参数梯度执行 `torch.distributed.all_reduce` 并除以 `world_size`。
- 分布式模式当前只支持 W-only；`--distributed --act-quant` 会直接报错，避免 activation scale 初始化和同步不一致。
- 更新 `configs/default_quant_config.json` 和 `README.md`，记录默认单卡行为、单卡指定 GPU 命令和四卡 torchrun 示例。

### 验证方式

- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/cli/quantize_scrn.py SCRN_BRECQ_app/scrn_brecq/data/calibration_loader.py SCRN_BRECQ_app/scrn_brecq/quant/block_recon.py SCRN_BRECQ_app/scrn_brecq/quant/layer_recon.py`
- `conda run -n quant python -m json.tool SCRN_BRECQ_app/scrn_brecq/configs/default_quant_config.json`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.quantize_scrn --help`
- 单卡回归 smoke：
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.quantize_scrn --num-samples 2 --batch-size 1 --iters-w 1 --run-root /tmp/scrn_brecq_single_gpu_smoke --run-name single_gpu_smoke --device auto --no-save-figure`
  - 输出目录：`/tmp/scrn_brecq_single_gpu_smoke/20260427_163038_single_gpu_smoke`
  - 指标：`fp32_snr=11.7869`、`pre_recon_snr=11.4071`、`post_recon_snr=11.4199`。
- 2 卡分布式 smoke：
  - `CUDA_VISIBLE_DEVICES=0,1 conda run -n quant torchrun --standalone --nproc_per_node=2 -m SCRN_BRECQ_app.scrn_brecq.cli.quantize_scrn --distributed --num-samples 4 --batch-size 1 --iters-w 1 --run-root /tmp/scrn_brecq_dist_smoke --run-name dist_smoke --device cuda --no-save-figure`
  - 在默认沙箱内 torchrun 本地 rendezvous 因 TCPStore 受限失败；按权限规则提升后通过。
  - 输出目录：`/tmp/scrn_brecq_dist_smoke/20260427_163715_dist_smoke`
  - `config.json` 记录 `distributed.enabled=true`、`world_size=2`、`device=cuda:0`。
  - 指标：`fp32_snr=11.7869`、`pre_recon_snr=11.4071`、`post_recon_snr=11.4198`。
- 对分布式 smoke checkpoint 运行真实性验证：
  - `passed=true`
  - `weight_bit_counts={"4": 50, "8": 2}`
  - `level_offender_count=0`
  - `fp32_quant_max_abs_diff=0.06889426708221436`

### 后续建议

- 正式四卡命令建议：
  - `CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n quant torchrun --standalone --nproc_per_node=4 -m SCRN_BRECQ_app.scrn_brecq.cli.quantize_scrn --distributed --num-samples 1024 --batch-size 16 --iters-w 20000 --run-name w4_recon_1024samples_20000iters_dist4 --device cuda`
- 若后续需要 W4A4，需要单独设计分布式 activation quantizer 初始化和同步。

## 2026-04-27 增加量化重建耗时指标

### 修改内容

- `cli/quantize_scrn.py` 在 `metrics.json` 中新增耗时字段：
  - `reconstruction_seconds` / `reconstruction_minutes`：layer/block reconstruction 墙钟耗时；分布式模式下包含结束同步等待。
  - `elapsed_seconds` / `elapsed_minutes`：CLI 主流程开始到最终量化推理完成、写入 metrics 前的总墙钟耗时。
- 终端最终输出从单一 `seconds` 改为同时显示 `inference_seconds`、`reconstruction_seconds` 和 `elapsed_seconds`，避免把最终推理耗时误读为整次量化耗时。

### 验证方式

- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/cli/quantize_scrn.py`
- 后续新 run 可直接从 `metrics.json` 读取耗时，不再需要用 run 目录文件时间戳估算。

## 2026-04-27 增加模型大小与理论压缩率指标

### 修改内容

- 新增 `utils/model_size.py`，统一统计 checkpoint 文件大小、权重参数 bit 分布和理论打包模型大小。
- `cli/quantize_scrn.py` 的 `metrics.json` 新增 `model_size`：
  - `checkpoint_files`：FP32 源 checkpoint 与量化 checkpoint 的实际文件大小。
  - `parameters`：总参数量、参与权重量化的参数量、按 bit 统计的层数和参数量。
  - `estimated_storage`：FP32 参数大小、理论量化权重大小、scale/zero point 开销、估算 packed 模型大小和压缩率。
- `cli/verify_quantized_scrn.py` 和 `cli/evaluate_quantized_scrn.py` 同步输出 `model_size`，方便对已有 checkpoint 追加分析。
- 明确当前 `.pth` 是可恢复 PyTorch checkpoint，不是 bit-packed 部署文件；真实压缩收益应看理论打包大小，后续如需文件真正变小，需要单独实现 packed/int checkpoint 导出。

### 验证方式

- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/utils/model_size.py SCRN_BRECQ_app/scrn_brecq/cli/quantize_scrn.py SCRN_BRECQ_app/scrn_brecq/cli/verify_quantized_scrn.py SCRN_BRECQ_app/scrn_brecq/cli/evaluate_quantized_scrn.py`
- 用已有 checkpoint 运行 `verify_quantized_scrn.py`，确认报告包含 `model_size.checkpoint_files`、`model_size.parameters` 和 `model_size.estimated_storage`。
- 用 `--num-samples 2 --batch-size 1 --iters-w 1` smoke run 确认 `metrics.json` 也包含 `model_size`。
- 在已有 global128 checkpoint 上的统计结果显示：实际量化 checkpoint 约 `5.0707 MiB`，理论 packed 模型约 `0.2427 MiB`，估算模型压缩率约 `6.77x`，量化权重参数占比约 `98.995%`。
