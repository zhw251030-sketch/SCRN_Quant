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

## 2026-04-27 增加多样本量化泛化评估入口

### 修改内容

- 新增 `cli/evaluate_quantized_scrn_multi.py`，用于对已保存 `quantized_scrn_brecq.pth` 做多样本评估。
- 默认评估集为 `SCRN_BRECQ_app/scrn_repro/datasets/scrn_quant_10750_0_patches`，从 clean patch 中按固定 seed 抽样，并复用 `scrn_repro.data.degrade_patch()` 在线生成 degraded 输入。
- 每个样本在同一 degraded 输入上分别运行 FP32 路径和量化路径，记录 input/FP32/quant 的 SNR、SSIM，以及 FP32 与量化输出的 MSE、平均绝对差和最大绝对差。
- 新增 `runs/generalization_eval/README.md`，`.gitignore` 忽略该目录下真实 run 产物但保留 README。
- 更新 `README.md` 和 `runs/README.md`，记录多样本评估命令、默认输出目录和产物边界。

### 验证方式

- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/cli/evaluate_quantized_scrn_multi.py`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_multi --help`
- 使用已有 global128 checkpoint 运行 smoke：
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_multi --checkpoint SCRN_BRECQ_app/scrn_brecq/runs/quant/20260427_192819_w4_recon_1024samples_20000iters_dist4_bsz32_global128/checkpoints/quantized_scrn_brecq.pth --num-eval-samples 4 --batch-size 2 --device cpu --run-root /tmp/scrn_brecq_generalization_smoke --no-save-figures`
  - 验证生成 `config.json`、`metrics.json`、`summary.md`、`per_sample_metrics.jsonl`，且 `metrics.json` 包含 `sample_count=4` 和 `model_size`。
- 正式 128 样本试跑：
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_multi --checkpoint SCRN_BRECQ_app/scrn_brecq/runs/quant/20260427_192819_w4_recon_1024samples_20000iters_dist4_bsz32_global128/checkpoints/quantized_scrn_brecq.pth --num-eval-samples 128 --batch-size 16 --device auto --run-name global128_quant10750_eval128`
  - 输出目录：`SCRN_BRECQ_app/scrn_brecq/runs/generalization_eval/20260427_222925_global128_quant10750_eval128`
  - 聚合结果：`input_snr_db_mean=0.9709`、`fp32_snr_db_mean=6.0901`、`quant_snr_db_mean=4.8802`、`quant_minus_fp32_snr_db_mean=-1.2099`、`fp32_ssim_mean=0.7562`、`quant_ssim_mean=0.7161`。

## 2026-04-29 整理 Codex 聊天记录归档文件

### 修改内容

- 将通用聊天归档文件调整为按具体对话主题保存，便于同一服务器不同 Codex/OpenAI 账号交接时查阅。
- 当前保留并纳入 Git 的聊天记录文件包括：
  - `CODEX_CHAT_BRECQ应用SCRN01_文件编写.md`
  - `CODEX_CHAT_BRECQ应用SCRN02_代码查阅检验.md`
  - `CODEX_CHAT_SCRN复现测试260425.md`
  - `CODEX_CHAT_安装插件skill.md`
- 移除此前仅作为模板的通用/占位归档文件：
  - `CODEX_CHAT_ARCHIVE.md`
  - `CODEX_CHAT_BRECQ应用SCRN03_量化实验记录.md`
  - `CODEX_CHAT_BRECQ应用SCRN04_泛化评估记录.md`
  - `CODEX_CHAT_BRECQ应用SCRN05_后续优化讨论.md`
- `CODEX_CONTEXT.md` 仍作为短交接上下文保留。

### 验证方式

- `git status` 检查新增和删除文件范围。
- `git diff --check` 检查 Markdown 空白格式。
- 本次只调整归档文档，不涉及 Python 代码或运行产物。

## 2026-04-29 增加重建前 checkpoint 与多样本 pre/post 对比

### 修改内容

- `cli/quantize_scrn.py` 在权重量化初始化后、BRECQ reconstruction 前新增保存
  `checkpoints/quantized_scrn_brecq_pre_recon.pth`。
- 最终 checkpoint 仍为 `checkpoints/quantized_scrn_brecq.pth`，并通过 `checkpoint_stage` 区分
  `pre_reconstruction` 和 `post_reconstruction`。
- `cli/evaluate_quantized_scrn_multi.py` 新增 `--pre-recon-checkpoint`：
  - 未传该参数时，兼容旧行为，`quant_*` 字段表示最终重建后量化模型。
  - 传入该参数时，在同一批 degraded 输入上同时评估 FP32、量化重建前、量化重建后路径。
  - 逐样本和聚合指标新增 `quant_pre_recon_*`、`quant_post_recon_*` 和
    `quant_post_minus_pre_*`，用于判断 BRECQ reconstruction 是否带来稳定提升。
  - 可视化从四图扩展为最多五图：clean、input、FP32、Quant Pre-Recon、Quant Post-Recon。
- 更新 `README.md` 和 `runs/README.md`，明确多样本评估中 `Quant` 默认含义是最终重建后结果；
  历史 run 如果没有保存 `quantized_scrn_brecq_pre_recon.pth`，不能事后恢复重建前模型，只能重新跑量化流程。

### 验证方式

- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/cli/quantize_scrn.py SCRN_BRECQ_app/scrn_brecq/cli/evaluate_quantized_scrn_multi.py`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_multi --help`
- 使用小规模 `quantize_scrn.py` smoke run 验证同时生成 pre/post 两个 checkpoint。
- 使用 smoke run 的两个 checkpoint 运行多样本评估，验证 `metrics.json` 同时包含
  `quant_pre_recon_snr_db_mean`、`quant_post_recon_snr_db_mean` 和
  `quant_post_minus_pre_snr_db_mean`。

## 2026-04-29 修复 W4A8 激活量化 checkpoint 与阶段输出

### 修改内容

- `quant/quant_layer.py` 将 `UniformAffineQuantizer.zero_point` 注册为 buffer，使 W+A
  checkpoint 能保存 activation quantizer 的零点。
- `cli/evaluate_quantized_scrn.py` 的 checkpoint 恢复逻辑会在 `strict=True` 加载前恢复
  activation quantizer 的 `delta` 和 `zero_point` 形状；新 checkpoint 不再依赖 eval 输入
  重新初始化激活量化状态。
- `cli/verify_quantized_scrn.py` 新增 `activation_quantization` 报告：
  - `activation_delta_count`
  - `activation_zero_point_count`
  - `initialized_activation_quantizers`
  - `missing_activation_state_count`
  - `activation_quantizers_restored`
- `cli/quantize_scrn.py` 将 reconstruction 拆成权重重建和激活重建两个阶段：
  - W-only 权重初始化后保存 `quantized_scrn_brecq_pre_recon.pth`。
  - W-only 权重重建后保存 `quantized_scrn_brecq_weight_recon.pth`。
  - W+A 激活量化初始化后保存 `quantized_scrn_brecq_pre_act_recon.pth`。
  - 最终保存 `quantized_scrn_brecq.pth`。
- W-only run 仍输出 5 图；W+A run 输出 7 图：
  Ground Truth、Input、FP32、W-only Pre Weight Recon、W-only Post Weight Recon、
  W+A Pre Act Recon、W+A Post Act Recon。
- `metrics.json` 新增阶段指标：
  - `quant_pre_weight_recon_*`
  - `quant_post_weight_recon_*`
  - `quant_pre_act_recon_*`
  - `quant_post_act_recon_*`
  - `quant_weight_recon_*_gain`
  - `quant_act_init_*_delta`
  - `quant_act_recon_*_gain`
  - `weight_reconstruction_seconds`
  - `activation_reconstruction_seconds`
- 更新 `README.md` 和 `runs/README.md`，记录 W4A8 smoke 命令、checkpoint 约定和 7 图含义。

### 验证方式

- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/quant/quant_layer.py SCRN_BRECQ_app/scrn_brecq/cli/evaluate_quantized_scrn.py SCRN_BRECQ_app/scrn_brecq/cli/verify_quantized_scrn.py SCRN_BRECQ_app/scrn_brecq/cli/quantize_scrn.py`
- W4A8 smoke：
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.quantize_scrn --n-bits-w 4 --n-bits-a 8 --act-quant --num-samples 2 --batch-size 1 --iters-w 1 --iters-a 1 --run-name smoke_w4a8_stage_fix --device cuda --gpus 0`
  - 输出目录：`SCRN_BRECQ_app/scrn_brecq/runs/quant/20260429_172312_smoke_w4a8_stage_fix`
- 真实性验证：
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.verify_quantized_scrn --checkpoint SCRN_BRECQ_app/scrn_brecq/runs/quant/20260429_172312_smoke_w4a8_stage_fix/checkpoints/quantized_scrn_brecq.pth --output-json SCRN_BRECQ_app/scrn_brecq/runs/quant/20260429_172312_smoke_w4a8_stage_fix/verification.json --device cpu`
  - `passed=true`，`activation_delta_count=52`，`activation_zero_point_count=52`，
    `missing_activation_state_count=0`。
- 重载一致性：
  - run 内 `metrics.json` 的 `quant_post_recon_snr_db=7.8479`。
  - `verify_quantized_scrn.py` 重载同一 checkpoint 后 `quant_snr_db=7.8386`。
  - `evaluate_quantized_scrn.py` 重载同一 checkpoint 后 `quant_snr=7.8386`。
  - 差异约 `0.009 dB`，已消除旧 checkpoint 因激活零点丢失导致的明显口径不一致。

### 已知边界

- 本次只修单卡 W+A checkpoint 和阶段评估口径；分布式 `--distributed --act-quant` 仍不支持。
- W4A8 smoke 只用于链路验证，`iters_w=1`、`iters_a=1` 不代表正式精度。

## 2026-04-29 增加 packed deployment 导出工具

### 修改内容

- 新增 `utils/packed_export.py`，将可恢复量化 checkpoint 中的权重导出为部署视角的紧凑格式：
  - `weights.bin` 保存按 bitwidth 打包后的整数权重，W4 使用两个 4-bit 值合并到一个 `uint8`。
  - `aux_fp32.bin` 保存权重量化 `delta`/`zero_point`、activation quantizer 状态和少量未量化 FP32 参数。
  - `manifest.json` 保存每层权重 shape、bitwidth、二进制 offset、Conv/Linear 参数和量化元数据。
  - `summary.json` 汇总原始 payload 大小、导出目录实际文件大小，以及与 `model_size.estimated_storage`
    的理论 packed 大小对比。
- 新增 `cli/export_quantized_scrn.py`，从 `quantized_scrn_brecq.pth` 恢复 `QuantModel` 后导出
  `<run_dir>/packed_deployment/`；也可用 `--output-dir` 指定目录。
- 更新 `utils/__init__.py` 导出 packed export 公共函数。
- 新增 `tests/test_packed_export.py`，用标准库 `unittest` 覆盖 uint4 打包和最小 QuantModule 导出。
- 真实 W4A8 checkpoint 中 `model.head` 是跳过 AdaRound 的 8bit Uniform 层；旧 checkpoint
  没有保存该层 weight `delta`。导出工具会仅对这种缺少权重量化状态的 Uniform 层按当前权重
  重新计算 deterministic scale，并在 `summary.json` 中记录 `recomputed_weight_quantizer_count`。

### 验证方式

- 先运行测试确认缺少 `utils.packed_export` 时失败。
- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_packed_export`
  - `Ran 3 tests ... OK`
- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/utils/packed_export.py SCRN_BRECQ_app/scrn_brecq/cli/export_quantized_scrn.py SCRN_BRECQ_app/scrn_brecq/utils/__init__.py SCRN_BRECQ_app/scrn_brecq/tests/test_packed_export.py`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.export_quantized_scrn --help`
- 对正式 W4A8 run 执行：
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.export_quantized_scrn --checkpoint SCRN_BRECQ_app/scrn_brecq/runs/quant/20260429_194908_w4a8_1024samples_w20000_a5000/checkpoints/quantized_scrn_brecq.pth`
  - 输出目录：`SCRN_BRECQ_app/scrn_brecq/runs/quant/20260429_194908_w4a8_1024samples_w20000_a5000/packed_deployment`
  - `weights.bin=213632 bytes`，`aux_fp32.bin=41276 bytes`，`manifest.json=120476 bytes`，`summary.json=2737 bytes`。
  - `raw_deployment_payload_mib=0.243099`，理论 `estimated_packed_model_size_mib=0.242702`，
    `raw_payload_to_estimated_packed_ratio=1.001635`。
  - 包含 JSON manifest/summary 后，`total_export_file_size_mib=0.360579`。

## 2026-04-30 增加 packed deployment 验证型评估脚本

### 修改内容

- 新增 `utils/packed_deployment.py`，支持读取 packed deployment 目录：
  - 解包 `uint4_lownibble_first`、`uint2_lownibble_first` 和 `uint8` 权重 payload。
  - 从 `aux_fp32.bin` 按 manifest offset/shape 读取 FP32 `delta`、`zero_point`、bias 和未量化参数。
  - 将 packed 整数权重反量化为 FP32 后写回 `QuantModule.weight/org_weight`，
    并关闭 weight quant 前向路径，用于验证 packed 文件能否复现 checkpoint 量化输出。
  - 若 manifest 中存在 activation quantizer 状态，则恢复 activation `delta/zero_point`。
- 新增 `cli/evaluate_packed_scrn.py`：
  - 输入 `--packed-dir`，从 `manifest.json` 的 `checkpoint_metadata` 重建 SCRN/QuantModel。
  - 调用 packed loader 恢复权重后，复用单样本评估逻辑输出 `prediction.npy`、`metrics.json`、
    `config.json`、`summary.md` 和可选 `comparison.png`。
  - 明确这是验证型 PyTorch 反量化评估，不是 INT4/INT8 kernel 部署 runtime。
- 更新 `.gitignore`，忽略 `SCRN_BRECQ_app/scrn_brecq/runs/packed_eval/` 产物目录。
- 更新 `utils/__init__.py` 导出 packed deployment loader 公共函数。
- 新增测试：
  - `tests/test_packed_deployment.py` 覆盖 uint4 解包和最小 QuantModule 的 packed restore。
  - `tests/test_evaluate_packed_scrn.py` 覆盖 manifest 到 checkpoint-like metadata 的转换。

### 验证方式

- 先运行测试确认缺少 `utils.packed_deployment` / `cli.evaluate_packed_scrn` 时失败。
- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_evaluate_packed_scrn SCRN_BRECQ_app.scrn_brecq.tests.test_packed_deployment`
  - `Ran 3 tests ... OK`
- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/utils/packed_deployment.py SCRN_BRECQ_app/scrn_brecq/cli/evaluate_packed_scrn.py SCRN_BRECQ_app/scrn_brecq/tests/test_packed_deployment.py SCRN_BRECQ_app/scrn_brecq/tests/test_evaluate_packed_scrn.py SCRN_BRECQ_app/scrn_brecq/utils/__init__.py`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_packed_scrn --help`
- 对 W4A32 global128 packed deployment 执行：
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_packed_scrn --packed-dir SCRN_BRECQ_app/scrn_brecq/runs/quant/20260427_192819_w4_recon_1024samples_20000iters_dist4_bsz32_global128/packed_deployment --run-name w4_global128_packed_eval --device cpu --no-save-figure`
  - 输出目录：`SCRN_BRECQ_app/scrn_brecq/runs/packed_eval/20260430_215420_w4_global128_packed_eval`
  - `packed_snr=11.7469`，`packed_ssim=0.8675`，与该 W4A32 checkpoint 的重载评估指标对齐。
  - `restored_quantized_layers=52`，`restored_non_quantized_tensors=70`，
    `restored_activation_quantizers=0`，符合 W-only packed artifact 预期。

### 已知边界

- 当前 loader 会把 packed 整数权重反量化回 FP32 后交给 PyTorch Conv/Linear 评估；
  它验证文件完整性和数值一致性，但不提供真正的 INT4/INT8 算子部署加速。

## 2026-04-30 增强 W4A32 packed deployment 五图验证

### 修改内容

- 增强 `cli/evaluate_packed_scrn.py`：
  - `--save-figure` 时从 manifest 的 `quant_checkpoint` 推导原量化 run 目录。
  - 强制读取原 run 下 `fp32_prediction.npy` 和 `quant_post_recon_prediction.npy`；
    若缺失则报错，不降级成三图。
  - 输出部署对齐五图 `comparison.png`：
    Ground Truth、Input、FP32 SCRN、W4A32 Checkpoint Final、W4A32 Packed Restored。
  - `metrics.json` 新增 packed restored 与 checkpoint final 的差异指标：
    `packed_vs_checkpoint_mse`、`packed_vs_checkpoint_mean_abs_diff`、
    `packed_vs_checkpoint_max_abs_diff`。
- 更新 `tests/test_evaluate_packed_scrn.py`，覆盖：
  - 从 `quant_checkpoint` 推导原 run 目录。
  - 五图所需 `.npy` 文件缺失时报错。
  - packed/checkpoint final 差异指标计算。

### 验证方式

- 先运行测试确认新增 helper 缺失时失败。
- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_evaluate_packed_scrn SCRN_BRECQ_app.scrn_brecq.tests.test_packed_deployment`
  - `Ran 6 tests ... OK`
- W4A32 global128 packed 五图评估：
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_packed_scrn --packed-dir SCRN_BRECQ_app/scrn_brecq/runs/quant/20260427_192819_w4_recon_1024samples_20000iters_dist4_bsz32_global128/packed_deployment --run-name w4_global128_packed_five_panel_eval --device cpu --save-figure`
  - 输出目录：`SCRN_BRECQ_app/scrn_brecq/runs/packed_eval/20260430_220931_w4_global128_packed_five_panel_eval`
  - 生成 `comparison.png`、`prediction.npy`、`metrics.json`、`config.json`、`summary.md`。
  - `packed_snr=11.7469`，`packed_ssim=0.8675`。
  - `checkpoint_final_snr_db=11.7469`，`checkpoint_final_ssim=0.8675`。
  - `packed_vs_checkpoint_mse=3.47e-09`，
    `packed_vs_checkpoint_mean_abs_diff=4.01e-05`，
    `packed_vs_checkpoint_max_abs_diff=4.66e-04`。

### 已知边界

- 五图验证仍是 packed 文件可恢复性的 PyTorch 反量化评估，不是 INT4 runtime kernel。

## 2026-04-30 W4A8 packed deployment 五图验证

### 修改内容

- 修复 `utils/packed_deployment.py` 的 W4A8 restore 问题：
  - activation `delta/zero_point` 在 manifest 中是标量 shape `[]`，读取时需要恢复为 0 维 tensor。
  - `leaf_param=True` 的 activation `delta` 注册为 `nn.Parameter`，恢复时需保持 Parameter 类型。
- 更新 `cli/evaluate_packed_scrn.py` 的五图标题标签：
  - W-only artifact 显示 `W4 weights / FP32 activations`。
  - W+A artifact 显示实际 activation bit，例如 `W4 weights / A8 activations`。
- 更新测试：
  - `tests/test_packed_deployment.py` 覆盖标量 activation quantizer 状态恢复。
  - `tests/test_evaluate_packed_scrn.py` 覆盖 W/A 标签生成。

### 验证方式

- 先运行 W4A8 packed eval 触发标量 activation 状态 restore 失败。
- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_packed_deployment SCRN_BRECQ_app.scrn_brecq.tests.test_evaluate_packed_scrn`
  - `Ran 8 tests ... OK`
- W4A8 packed 五图评估：
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_packed_scrn --packed-dir SCRN_BRECQ_app/scrn_brecq/runs/quant/20260429_194908_w4a8_1024samples_w20000_a5000/packed_deployment --run-name w4a8_packed_five_panel_eval --device cpu --save-figure`
  - 输出目录：`SCRN_BRECQ_app/scrn_brecq/runs/packed_eval/20260430_222334_w4a8_packed_five_panel_eval`
  - 生成 `comparison.png`、`prediction.npy`、`metrics.json`、`config.json`、`summary.md`。
  - `packed_snr=5.2274`，`packed_ssim=0.6626`。
  - `checkpoint_final_snr_db=5.2277`，`checkpoint_final_ssim=0.6618`。
  - `restored_activation_quantizers=52`，`restored_quantized_layers=52`。
  - `packed_vs_checkpoint_mse=5.26e-05`，
    `packed_vs_checkpoint_mean_abs_diff=5.78e-03`，
    `packed_vs_checkpoint_max_abs_diff=2.79e-02`。

### 结论

- W4A8 packed artifact 能恢复 activation quantizer 状态并完成五图评估。
- 指标层面 packed restored 与 checkpoint final 对齐，SNR 差约 `0.0003 dB`；
  逐像素差异大于 W4A32，但不改变此前结论：W4A8 的主要问题是激活量化本身导致精度明显下降。

## 2026-04-30 W4 dist4 compare packed deployment 五图验证

### 修改内容

- 将 `cli/evaluate_packed_scrn.py` 的五图标题从短标签 `W4A32/W4A8`
  泛化为 `W{bits} weights / FP32 activations` 或
  `W{bits} weights / A{bits} activations`，避免后续 A12/A16 或其他组合时标题含义不清。
- 更新 `tests/test_evaluate_packed_scrn.py`，覆盖新的泛化标题标签。

### 验证方式

- 对 W4 dist4 compare run 导出 packed deployment：
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.export_quantized_scrn --checkpoint SCRN_BRECQ_app/scrn_brecq/runs/quant/20260427_165123_w4_recon_1024samples_20000iters_dist4_compare/checkpoints/quantized_scrn_brecq.pth`
  - 输出目录：`SCRN_BRECQ_app/scrn_brecq/runs/quant/20260427_165123_w4_recon_1024samples_20000iters_dist4_compare/packed_deployment`
  - `raw_deployment_payload_mib=0.242702`，`raw_payload_to_estimated_packed_ratio=1.0`。
- 运行 packed 五图评估：
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_packed_scrn --packed-dir SCRN_BRECQ_app/scrn_brecq/runs/quant/20260427_165123_w4_recon_1024samples_20000iters_dist4_compare/packed_deployment --run-name w4_dist4_compare_packed_five_panel_eval --device cpu --save-figure`
  - 输出目录：`SCRN_BRECQ_app/scrn_brecq/runs/packed_eval/20260430_223157_w4_dist4_compare_packed_five_panel_eval`
  - 生成 `comparison.png`、`prediction.npy`、`metrics.json`、`config.json`、`summary.md`。
  - `packed_snr=11.6951`，`packed_ssim=0.8610`。
  - `checkpoint_final_snr_db=11.6952`，`checkpoint_final_ssim=0.8608`。
  - `restored_activation_quantizers=0`，`restored_quantized_layers=52`。
  - `packed_vs_checkpoint_mse=3.29e-09`，
    `packed_vs_checkpoint_mean_abs_diff=3.92e-05`，
    `packed_vs_checkpoint_max_abs_diff=4.69e-04`。

### 结论

- W4 dist4 compare 的 packed artifact 和前两个 run 一样能恢复并完成五图评估。
- packed restored 与 checkpoint final 对齐，说明 W-only packed 导出/恢复链路稳定。

## 2026-05-04 建立激活量化研究日志

### 修改内容

- 新增 `ACTIVATION_QUANTIZATION_LOG.md`，专门记录 SCRN-BRECQ 中 W4A8 激活量化失败问题的分析、修复设计和实验过程。
- 在激活量化研究日志中预留用户填写区，用于先记录潜在原因、潜在解决方案和优先验证方向。
- 明确后续规则：凡是涉及 activation quantization 的代码、配置、诊断、实验或结论，除继续更新本开发日志外，也必须在 `ACTIVATION_QUANTIZATION_LOG.md` 中追加更细的实验记录。
- 在激活量化研究日志中记录当前基线事实、候选问题池、实验索引和实验记录模板，为后续小范围诊断和正式实验提供统一记录格式。

### 设计原因

- W4A8 当前已确认不是 packed/export/restore 链路主导问题，而是 activation quantization 本身导致明显掉点。
- 激活量化修复过程可能形成后续工作的创新点，需要比普通开发日志更细地记录每次假设、实验、指标和结论。
- 当前已观察到 final W4A8 checkpoint 中存在非法负 activation `delta`，后续修复需要保留完整证据链。

### 验证方式

- 本次只新增和更新 Markdown 文档，不涉及 Python 代码。
- 后续提交前应执行 `git diff --check` 检查 Markdown 空白格式，并用 `git status`、`git diff` 确认仅包含本次文档改动。

## 2026-05-04 深度整理激活量化失败原因与实验路线

### 修改内容

- 更新 `ACTIVATION_QUANTIZATION_LOG.md`，在用户初步总结之后追加 Codex 深度分析。
- 将 W4A8 激活量化失败拆解为量化器合法性、离群值与量化粒度、量化插入位置、`zero_point` 固定、混合精度、重建目标、校准覆盖和工程约束等问题层。
- 明确后续实验路线：
  - E001：activation diagnostics 工具与基线报告。
  - E002：正 scale 约束最小修复。
  - E003：初始化覆盖与 activation learning rate sweep。
  - E004：量化插入位置与敏感性图谱。
  - E005：离群值处理和结构化量化粒度实验。
  - E006：重建目标与教师模型实验。
- 扩展激活量化候选问题池，新增初始化覆盖、`asym=False`、教师模型选择、量化位置过密、A16 fallback 限制和 activation learning rate 等候选问题。

### 设计原因

- 用户已经在激活量化日志中写入初步判断，需要转化成可执行、可验证、可复盘的实验路线。
- W4A8 失败可能成为本工作的创新点，因此需要把“现象描述”升级为“假设-诊断-实验-结论”的研究记录。
- 当前不应直接启动长实验，应先补齐诊断工具和合法性闭环。

### 验证方式

- 本次只更新 Markdown 文档，不涉及 Python 代码。
- 待提交前执行 `git diff --check`，并检查 `git status`、`git diff`、`git diff --staged`。

## 2026-05-04 建立激活量化实验目录规范

### 修改内容

- 新增 `configs/activation_quantization/`，用于保存后续激活量化诊断和实验的可复现文本配置。
- 新增 `runs/activation_quantization/`，用于保存后续 E001-E006 激活量化诊断、smoke 和实验产物。
- 在两个目录中分别添加 `README.md`，说明目录用途、建议分组和禁止提交的运行产物类型。
- 更新 `runs/README.md`，将 `runs/activation_quantization/` 纳入当前约定的 run 根目录。
- 同步更新 `ACTIVATION_QUANTIZATION_LOG.md`，把目录结构作为正式实验前的工程准备项记录下来。

### 设计原因

- 后续 W4A8 激活量化修复会产生诊断报告、配置、统计表、图、checkpoint 和对比摘要，提前隔离配置与运行产物可以避免和既有 `runs/quant/`、`runs/quant_eval/` 混杂。
- Git 不能直接跟踪空目录，因此用小型 README 固定目录结构；真正的实验输出仍按规则不纳入 Git。
- 当前只建立最小目录骨架，不提前创建 E001-E006 的具体代码文件，避免过早框架化。

### 验证方式

- 本次只新增和更新 Markdown 文档，不涉及 Python 代码。
- 提交前执行 `git diff --check` 和 `git diff --check --cached` 检查 Markdown 空白格式。

## 2026-05-04 E001 激活量化诊断工具

### 修改内容

- 新增 `quant/activation_diagnostics.py`，提供 activation quantizer 状态扫描、结构标签推断、forward hook 分布统计、fake-quant MSE 和有效 int level 统计。
- 新增 `cli/diagnose_activation_quantization.py`，复用已保存 checkpoint 的 QuantModel 重建和 quantizer state restore 逻辑，输出 E001 诊断 run。
- 新增 `configs/activation_quantization/e001_diagnostics.json`，默认使用当前 W4A8 final checkpoint，采用 CPU、64 个 calibration 样本的小规模诊断配置。
- 更新 `quant/__init__.py`，导出激活量化诊断公共函数。
- 新增 `tests/test_activation_diagnostics.py`，用小型 `QuantModule` 验证 quantizer row、负 delta offender、activation stats 和有效 int level。
- 同步更新 `ACTIVATION_QUANTIZATION_LOG.md`，记录 E001 smoke 结果和后续实验入口。

### 设计原因

- 当前 W4A8 失败需要先建立可复现诊断证据，而不是直接修改 `delta` 或启动长时间 reconstruction。
- 诊断工具只读取模型状态和 forward hook 输出，不改变量化算法行为。
- 输出文件拆分为 `summary.json`、`quantizers.csv`、`activation_stats.jsonl`、`offender_layers.json`，便于后续按层排序、筛选和跨 checkpoint 对比。

### 验证方式

- TDD 红灯：
  - `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_diagnostics`
  - 初始失败原因为缺少 `SCRN_BRECQ_app.scrn_brecq.quant.activation_diagnostics`。
- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/quant/activation_diagnostics.py SCRN_BRECQ_app/scrn_brecq/cli/diagnose_activation_quantization.py`
- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_diagnostics`
  - `Ran 3 tests ... OK`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.diagnose_activation_quantization --help`
- E001 2-sample CPU smoke，输出放到 `/tmp`，避免提交 run 产物：
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.diagnose_activation_quantization --num-samples 2 --batch-size 1 --device cpu --run-name smoke_e001 --run-root /tmp/scrn_brecq_e001_diagnostics`
  - `activation_quantizers=52`
  - `non_positive_delta_count=2`
  - `activation_stat_count=52`
  - `fake_quant_mse_max=0.003358484013006091`

## 2026-05-04 约束激活量化实验产物位置

### 修改内容

- 更新仓库根目录 `.gitignore`，忽略 `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/*`。
- 保留 `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/README.md` 继续纳入 Git，用于说明该目录用途和产物不提交规则。

### 设计原因

- 激活量化实验结果必须保存在 `/home/data1/hanwen/project/Project/SCRN_Quant` 项目内的约定 run 目录中，不应再写入 `/tmp` 或其他项目外路径。
- 通过 `.gitignore` 明确保护后，后续正式 E001/E002 run 可以放回项目目录，同时避免误提交 JSON、CSV、JSONL、图片、checkpoint 等产物。

### 验证方式

- 本次只更新 `.gitignore` 和 Markdown 日志。
- 提交前执行 `git diff --check` 和 `git diff --check --cached`。

## 2026-05-04 E001a 补齐激活量化诊断指标

### 修改内容

- 扩展 `quant/activation_diagnostics.py` 的 activation row 字段：
  - 新增 per-channel absmax 统计：`per_channel_absmax_max`、`per_channel_absmax_median`、`per_channel_absmax_ratio`、`per_channel_count`。
  - 对不支持 per-channel 统计的 activation shape 写入 `per_channel_absmax_skip_reason`。
- 扩展 summary 输出：
  - `top_outlier_layers`
  - `lowest_effective_level_layers`
  - `worst_fake_quant_mse_layers`
  - `worst_relative_mse_layers`
  - `top_per_channel_imbalance_layers`
- 新增结构分组统计：
  - `branch_summary`
  - `stage_summary`
  - `role_summary`
  - `module_type_summary`
- 更新 `tests/test_activation_diagnostics.py`，覆盖 4D activation per-channel ratio、top-k 摘要和 branch/role 分组统计。
- 同步更新 `ACTIVATION_QUANTIZATION_LOG.md`，明确 E001a 只补齐工具指标，正式 baseline 留给 E001b。

### 设计原因

- 原 E001 验收标准要求 per-channel absmax 差异和 CNN branch vs Transformer branch 等结构分组统计，第一版诊断工具只完成了部分字段。
- E001a 只补齐工具能力，不运行正式 checkpoint 诊断，避免把工具开发和实验结论混在一起。

### 验证方式

- TDD 红灯：
  - `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_diagnostics`
  - 初始失败字段：`per_channel_count`、`top_outlier_layers`。
- 修复测试输入后，`conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_diagnostics`
  - `Ran 5 tests ... OK`
- 提交前继续执行 `py_compile`、CLI `--help`、`git diff --check` 和 `git diff --check --cached`。

## 2026-05-04 E001b 运行 final W4A8 64 样本激活诊断

### 修改内容

- 运行 E001b final W4A8 64-sample baseline，输出目录：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E001_diagnostics/20260504_203753_e001_diagnostics`
- 修复 E001b 首次正式运行暴露出的诊断工具问题：
  - `torch.quantile` 在大 tensor 上报错 `RuntimeError: quantile() input tensor is too large`。
  - `quant/activation_diagnostics.py` 对超过 16,000,000 个元素的 tensor 使用 `torch.kthvalue` fallback 计算分位数。
  - 小 tensor 仍使用原 `torch.quantile` 路径，保持原测试语义。
- 更新 `tests/test_activation_diagnostics.py`，新增大 tensor 分位数回归测试。
- 同步更新 `ACTIVATION_QUANTIZATION_LOG.md`，记录 E001b 正式 baseline 结果。

### 设计原因

- 64 个 `128x128` calibration 输入经过 SCRN head 后形成约 67M 个 activation 值，超过当前 PyTorch `torch.quantile` 单 tensor 限制。
- 该修复只改变诊断统计实现，不修改模型、量化算法、checkpoint 或 reconstruction 行为。
- E001b 的正式产物保存在项目内 run 目录，且受 `.gitignore` 保护，不纳入 Git。

### E001b 关键结果

- activation quantizers：52
- activation delta count：52
- activation zero point count：52
- non-positive delta count：2
- offender layers：
  - `model.stage4.0.block.trans_branch.attn.proj`
  - `model.stage5.0.block.trans_branch.attn.proj`
- activation stat count：52
- fake quant MSE max：`0.003323915181681514`
- fake quant MSE mean：`0.00010568322308295127`
- effective int levels：min `17`，max `256`
- absmax over p99 max：`34.627632811323394`

### 结构诊断摘要

- `transformer` branch：
  - count：20
  - effective int levels min：`17`
  - fake quant MSE max：`0.003323915181681514`
  - relative MSE max：`0.020646367816721696`
- `cnn` branch：
  - count：15
  - effective int levels min：`231`
  - fake quant MSE max：`0.0001195866207126528`
  - relative MSE max：`0.0003560360421608271`
- `Linear` module：
  - effective int levels min：`17`
  - relative MSE max：`0.020646367816721696`
- `Conv2d` module：
  - effective int levels min：`188`
  - relative MSE max：`0.014584982809199652`

### 验证方式

- TDD 红灯：
  - `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_diagnostics`
  - 新增测试触发 `RuntimeError: quantile() input tensor is too large`。
- 单元测试转绿：
  - `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_diagnostics`
  - `Ran 6 tests ... OK`
- E001b 正式诊断：
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.diagnose_activation_quantization --config SCRN_BRECQ_app/scrn_brecq/configs/activation_quantization/e001_diagnostics.json`
  - 输出：`activation_quantizers=52 non_positive_delta_count=2 activation_stat_count=52 fake_quant_mse_max=0.003323915181681514`
- 确认 run 产物被 `.gitignore` 忽略：
  - `git check-ignore -v SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E001_diagnostics/20260504_203753_e001_diagnostics/summary.json`

## 2026-05-04 E001c 运行 pre-act-recon 64 样本激活诊断

### 修改内容

- 运行 E001c pre-act-recon 64-sample baseline，输出目录：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E001_diagnostics/20260504_205237_e001c_pre_act_recon`
- 使用同一诊断工具和同一 calibration 配置，对比 E001b final W4A8 baseline。
- 本步骤不修改量化算法和诊断工具代码，只更新开发日志和激活量化日志。

### 设计原因

- E001b 已证明 final checkpoint 存在 2 个非法 activation `delta`。
- E001c 用 activation reconstruction 前的 checkpoint 做对照，用来判断非法 `delta` 是初始化/权重量化阶段已经存在，还是 activation reconstruction 优化后引入。

### E001b vs E001c 关键对比

- E001b final：
  - non-positive delta count：2
  - fake quant MSE max：`0.003323915181681514`
  - fake quant MSE mean：`0.00010568322308295127`
  - effective int levels min：17
  - transformer effective int levels min：17
  - transformer relative MSE max：`0.020646367816721696`
- E001c pre-act-recon：
  - non-positive delta count：0
  - fake quant MSE max：`0.00011542496213223785`
  - fake quant MSE mean：`0.000033345033807013605`
  - effective int levels min：188
  - transformer effective int levels min：202
  - transformer relative MSE max：`0.00021580944014857998`
- 结论：非法 activation `delta` 和 transformer/Linear 的有效 level 崩塌出现在 activation reconstruction 之后。

### 验证方式

- E001c 正式诊断：
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.diagnose_activation_quantization --config SCRN_BRECQ_app/scrn_brecq/configs/activation_quantization/e001_diagnostics.json --checkpoint SCRN_BRECQ_app/scrn_brecq/runs/quant/20260429_194908_w4a8_1024samples_w20000_a5000/checkpoints/quantized_scrn_brecq_pre_act_recon.pth --run-name e001c_pre_act_recon`
  - 输出：`activation_quantizers=52 non_positive_delta_count=0 activation_stat_count=52 fake_quant_mse_max=0.00011542496213223785`
- 确认 run 产物被 `.gitignore` 忽略：
  - `git check-ignore -v SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E001_diagnostics/20260504_205237_e001c_pre_act_recon/summary.json`

## 2026-05-04 E001d 汇总激活量化诊断结论

### 修改内容

- 汇总 E001 smoke、E001a、E001b 和 E001c 的诊断阶段结论。
- 在 `ACTIVATION_QUANTIZATION_LOG.md` 追加可读性结论，明确 W4A8 激活量化失败的当前证据链和下一步 E002 修复方向。
- 本步骤不修改代码、不运行新 checkpoint、不生成新 run 产物。

### 结论摘要

- final W4A8 checkpoint 中有 2 个 activation `delta` 为负，均位于 transformer branch 的 attention projection：
  - `model.stage4.0.block.trans_branch.attn.proj`
  - `model.stage5.0.block.trans_branch.attn.proj`
- pre-act-recon checkpoint 中 `non_positive_delta_count=0`，说明非法 `delta` 是 activation reconstruction 阶段引入。
- activation reconstruction 后，transformer/Linear 的有效 int level 和局部误差明显恶化：
  - transformer effective level min 从 202 降到 17。
  - transformer relative MSE max 从 `0.00021580944014857998` 升到 `0.020646367816721696`。
- E002 应优先实现 activation `delta` 正值约束，再用 E001 工具复核 `non_positive_delta_count=0` 以及 transformer/Linear 指标是否恢复。

### 验证方式

- 本步骤只更新 Markdown 日志。
- 提交前执行 `git diff --check` 和 `git diff --check --cached`。

## 2026-05-04 记录 E001 诊断指标与样本数关系

### 修改内容

- 在 `ACTIVATION_QUANTIZATION_LOG.md` 追加 E001 方法澄清，记录 checkpoint 固定参数与 calibration 样本统计指标的区别。
- 说明为什么 64/1024/full calibration 会影响 activation 分布、fake-quant MSE、effective int level 和 top-k 排名，但不会改变 checkpoint 本身是否存在负 `delta`。
- 本步骤不修改代码、不运行新 checkpoint、不生成新 run 产物。

### 设计原因

- 之前容易误解为“W4A8 checkpoint 已经包含所有 activation 分布和误差统计”。
- 实际上 checkpoint 固定的是模型权重和 quantizer 参数；activation 分布和基于 activation 的误差指标需要在给定输入样本上 forward 后统计。

### 验证方式

- 本步骤只更新 Markdown 日志。
- 提交前执行 `git diff --check` 和 `git diff --check --cached`。

## 2026-05-04 记录 E002 正 scale 约束初步计划

### 修改内容

- 在 `ACTIVATION_QUANTIZATION_LOG.md` 追加 E002 初步计划，明确正 scale 约束是必要的合法性修复，但不一定充分解释全部 W4A8 精度损失。
- 将 E002 拆分为：
  - E002a：post-step clamp 最小修复。
  - E002b：修复后 W4A8 复现实验。
  - E002c：若 clamp 后仍差，再进入 delta ratio 限制、log-scale/softplus 参数化、学习率/迭代数调整和 attention 层选择性冻结等方向。
- 本步骤只整理计划，不修改量化算法、不运行新实验、不生成 run 产物。

### 设计原因

- E001b/E001c 已经证明负 `delta` 是 activation reconstruction 后引入的确定问题。
- 但 optimizer 能把 scale 推到负数，说明仅有合法性约束可能不足以解决全部精度问题；E002 需要用最小修复先建立因果验证闭环。

### 验证方式

- 本步骤只更新 Markdown 日志。
- 提交前执行 `git diff --check` 和 `git diff --check --cached`。

## 2026-05-04 E002a activation delta 正值投影最小修复

### 修改内容

- 在 `block_recon.py` 增加 `ACTIVATION_DELTA_MIN = 1e-8` 和 `_project_activation_delta_params_positive(...)`。
- 在 block/layer activation reconstruction 的 `optimizer.step()` 后，对 learnable activation `delta` 执行 post-step clamp。
- 新增 `test_activation_scale_constraints.py`，覆盖负/零 delta clamp、正 delta 保持不变、非法 eps 报错。
- 同步更新 `ACTIVATION_QUANTIZATION_LOG.md`，记录 E002a 只做合法性最小修复，正式 W4A8 复现实验留给 E002b。

### 设计原因

- E001b/E001c 已证明 final W4A8 的负 `delta` 由 activation reconstruction 引入。
- activation scale 必须为正，因此优化后投影是最小合法性修复；暂不修改 quantizer forward 或 checkpoint 结构，降低兼容性风险。

### 验证方式

- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_scale_constraints`
- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_diagnostics`
- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/quant/block_recon.py SCRN_BRECQ_app/scrn_brecq/quant/layer_recon.py`
- `git diff --check`
- `git diff --check --cached`

## 2026-05-04 E002b positive-scale W4A8 复现实验记录

### 修改内容

- 运行 E002b smoke，确认 E002a 修复后的 W4A8 小样本链路可生成 checkpoint，且 smoke diagnostics 中 `non_positive_delta_count=0`。
- 运行正式 W4A8 复现实验：
  - `num_samples=1024`
  - `batch_size=16`
  - `iters_w=20000`
  - `iters_a=5000`
  - `device=cuda`
  - `gpus=0`
- 对正式 checkpoint 运行 64-sample activation diagnostics，并做单样本 checkpoint reload eval。
- 本步骤不修改代码，只更新开发日志和激活量化日志；run 产物不纳入 Git。

### 关键产物

- smoke quant run：
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260504_221053_e002b_smoke_positive_scale`
- smoke diagnostics：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E002_positive_scale/diagnostics/20260504_221230_e002b_smoke_diagnostics`
- formal quant run：
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260504_221242_e002b_w4a8_positive_scale_1024samples_w20000_a5000`
- formal diagnostics：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E002_positive_scale/diagnostics/20260504_232451_e002b_final_diagnostics`
- formal reload eval：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E002_positive_scale/eval/20260504_232504_e002b_final_single_eval`

### 结论摘要

- E002b formal checkpoint：
  - `activation_quantizers=52`
  - `non_positive_delta_count=0`
  - offender layers 为空
  - 最小 activation `delta=0.000576426915358752`
  - `eps=1e-8` 边界命中数为 0
- 正式 run 内 SNR：
  - `quant_post_weight_recon_snr_db=11.696099054461113`
  - `quant_pre_act_recon_snr_db=4.9874515693637465`
  - `quant_post_act_recon_snr_db=5.236280200086368`
  - `quant_act_recon_snr_gain_db=0.2488286307226213`
- checkpoint reload 单样本评估：
  - `quant_snr_db=5.230110892430229`
  - `quant_ssim=0.6613561048695519`
- 与 E001b old final 对比：
  - 非正 delta 从 2 降到 0。
  - final run 内 SNR 仅从 `5.227702998470372` 到 `5.236280200086368`，提升约 `0.0086 dB`。
  - transformer / Linear effective level min 从 17 到 30，但仍然很低。
  - transformer / Linear relative MSE max 从 `0.020646367816721696` 到 `0.010542650171161586`，局部指标改善但未带来最终 SNR 恢复。

### 设计判断

- E002a 的正 scale 投影是必要修复，已经成功消除非法 checkpoint 状态。
- 但 E002b 表明负 `delta` 不是 W4A8 低 SNR 的唯一或主要瓶颈；后续应继续分析 activation range、learning rate、attention qkv/proj 敏感性和 reconstruction 目标。

### 验证方式

- E002b smoke quantize。
- E002b smoke diagnostics。
- E002b formal quantize。
- E002b formal 64-sample diagnostics。
- E002b formal single-sample reload eval。
- 提交前执行 `git diff --check` 和 `git diff --check --cached`。

## 2026-05-05 E002c activation-only 初始化敏感性工作流

### 修改内容

- 新增 `cli/activation_only_quantize_scrn.py`，可从已有 W4 weight-recon checkpoint 继续做 A8 activation 初始化，默认跳过 activation reconstruction。
- 新增 `tests/test_activation_only_quantize_scrn.py`，覆盖：
  - activation-only 默认配置是 init-only run。
  - checkpoint 内旧 `run_root/run_name` 不会覆盖 E002c 输出目录。
  - activation-only metrics 会记录 `quant_pre_act_recon_snr_db` 和 A8 init SNR delta。
- 修正配置合并边界：只从 checkpoint 继承量化/数据相关字段，不继承旧 run 输出路径。
- 本步骤不修改量化公式、不修改 reconstruction 算法、不提交 run 产物。

### 实验记录

- smoke run：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E002c_init_sensitivity/quant/20260505_150649_e002c_smoke_init_n0002`
  - `quant_pre_act_recon_snr_db=8.380215760145743`
  - `non_positive_delta_count=0`
- smoke diagnostics：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E002c_init_sensitivity/diagnostics/20260505_150741_e002c_smoke_diag_n0002`
  - `activation_quantizers=52`
  - `non_positive_delta_count=0`
- init-only sweep：
  - `n=2`：`20260505_150800_e002c_init_n0002`
  - `n=8`：`20260505_150814_e002c_init_n0008`
  - `n=16`：`20260505_150826_e002c_init_n0016`
  - `n=64`：`20260505_150842_e002c_init_n0064`
  - `n=256`：`20260505_150917_e002c_init_n0256`
  - `n=1024`：`20260505_150952_e002c_init_n1024`
- fixed 64-sample CUDA diagnostics：
  - `n=2`：`20260505_151932_e002c_diag_cuda_n0002`
  - `n=8`：`20260505_152323_e002c_diag_cuda_n0008`
  - `n=16`：`20260505_152708_e002c_diag_cuda_n0016`
  - `n=64`：`20260505_153054_e002c_diag_cuda_n0064`
  - `n=256`：`20260505_153439_e002c_diag_cuda_n0256`
  - `n=1024`：`20260505_153825_e002c_diag_cuda_n1024`

### 关键结论

- 当前 activation 初始化实际使用 `min(num_samples, init_batch_size)` 个样本；默认 `init_batch_size=64`，因此 `n=64/256/1024` 三组得到完全一致的 A8 初始化状态和 SNR。
- 单样本 SNR 随有效 init 样本数从 2 增加到 64 明显下降：
  - `2 -> 8 -> 16 -> 64`：`8.3802 -> 7.1866 -> 6.4882 -> 4.9875 dB`
- 所有 init-only checkpoint 的 activation `delta` 都合法：`non_positive_delta_count=0`。
- 尝试真正 `--init-batch-size 256` 时，CUDA 0 在 activation MSE scale 初始化中 OOM；因此全 256/1024 activation init 需要后续单独优化初始化策略或更大显存。

### 验证方式

- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_only_quantize_scrn`
- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/cli/activation_only_quantize_scrn.py`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.activation_only_quantize_scrn --help`
- E002c smoke quantize + smoke diagnostics。
- E002c init-only sweep + fixed 64-sample CUDA diagnostics。

## 2026-05-05 E002 阶段收束记录

### 修改内容

- 更新 `ACTIVATION_QUANTIZATION_LOG.md`，补充 E002 阶段收束判断。
- 本步骤不修改代码、不运行新实验、不生成 run 产物。

### 结论摘要

- E002a 的正 scale clamp 已作为必要合法性修复保留。
- E002b 已验证该修复能让 final checkpoint 的 `non_positive_delta_count=0`，但 final SNR 只提升约 `0.0086 dB`。
- E002c 已验证当前 A8 init 主要受 `init_batch_size` 控制，默认 64；继续只扩大 `num_samples` 不会改变 activation init 状态。
- 后续不再优先推进 delta ratio、log-scale/softplus、activation_lr sweep、attention freeze 等 reconstruction trick。
- 下一阶段 E003 优先进入 activation initialization、range/clipping、calibration subset 和 multi-sample eval 方向。

### 后续注意事项

- 不要把 `num_samples=1024` 误读为 activation init 使用 1024 个样本；当前实际由 `init_batch_size` 截断。
- 不要只凭单张 eval SNR 判断小样本 init 更好；E002c 显示小样本高 SNR 更像 subset 对单张图的偶然匹配。
- 不要直接启动长时间 A5000 reconstruction sweep；应先定位 A8 init 打开后从约 `11.696 dB` 掉到约 `4.987 dB` 的原因。

## 2026-05-05 E003 初步实验计划记录

### 修改内容

- 更新 `ACTIVATION_QUANTIZATION_LOG.md`，记录 E003 的阶段目标、实验顺序和变量优先级。
- 本步骤不修改代码、不运行实验、不生成 run 产物。

### 计划摘要

- E003 目标：验证 W4A8 失败主要来自 activation 初始化覆盖/range 问题，还是 activation reconstruction 优化稳定性问题。
- E003a：先建立 multi-sample eval，避免继续被单张 eval SNR 误导。
- E003b：把 `init_batch_size` 当作真实变量，优先测试 `2/8/16/32/64`；`256/1024` 因 E002c OOM 暂不作为第一轮必跑项。
- E003c：在 eval 口径稳定后再做 `activation_lr=4e-4/1e-4/4e-5` sweep。

### 设计判断

- 原始设想中的 `init_batch_size=64/256/1024` 需要根据 E002c 结果调整：当前 256 full-init 已在 CUDA 0 上 OOM。
- reconstruction 相关 trick 暂时低于 E003 主线优先级，因为最大掉点已经发生在 A8 init 阶段。
- E003 的关键验收不是单张图 SNR，而是 multi-sample eval、fixed diagnostics 和 activation quantizer 状态的共同结论。

## 2026-05-05 E003a multi-sample eval baseline

### 修改内容

- 复用已有 `cli/evaluate_quantized_scrn_multi.py` 完成 E003a 多样本评估。
- 补齐 aggregate median 字段，满足 E003a 对 mean / median / min / max 的记录要求。
- 新增 `tests/test_evaluate_quantized_scrn_multi.py`，验证 median 聚合和 legacy alias。
- 更新 `ACTIVATION_QUANTIZATION_LOG.md` 记录 E003a 的 run 目录、评估设置、结果表格和结论。

### 实验设置

- `num_eval_samples=128`
- `batch_size=16`
- `seed=20260427`
- `device=cuda`
- `--no-save-figures`
- run root：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E003_multi_sample_eval`
- selected sample list hash：
  - `cf3b4fe1a094`

### 关键结果

- W4 weight-recon：
  - SNR mean：`4.79725691949495`
  - SNR median：`4.309259459875541`
- A8 init n=2：
  - SNR mean：`-7.0231250370839735`
  - SNR median：`-7.857011917587894`
- A8 init n=8：
  - SNR mean：`-7.023006163652582`
  - SNR median：`-7.7914053078130605`
- A8 init n=16：
  - SNR mean：`-7.047440279252236`
  - SNR median：`-7.8041388297714125`
- A8 init n=64：
  - SNR mean：`-7.102088710746793`
  - SNR median：`-7.850034466981217`
- E002b positive-scale final：
  - SNR mean：`-7.071334905403255`
  - SNR median：`-7.811254783170435`

### 结论摘要

- E003a 推翻了“小样本 A8 init 更好”的单张 eval 解释；2/8/16/64 样本 init 在 128-sample eval 上全部约 `-7 dB`。
- E002b final activation reconstruction 在 128-sample eval 上也没有恢复，SNR mean 仍为 `-7.0713 dB`。
- 后续 E003b 必须继续使用 multi-sample eval 作为主指标，不能再依赖单张 eval SNR。

### 验证方式

- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/cli/evaluate_quantized_scrn_multi.py`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_multi --help`
- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_evaluate_quantized_scrn_multi`
- E003a 4-sample smoke。
- E003a 128-sample baseline runs。

## 2026-05-05 GPU resource usage principle

### 修改内容

- 更新 `ACTIVATION_QUANTIZATION_LOG.md`，记录后续实验优先充分利用 GPU 的原则。
- 本步骤不修改代码、不运行实验、不生成 run 产物。

### 原则摘要

- 后续量化、评估、诊断和 sweep 默认优先使用 `--device cuda`。
- 单卡实验应明确指定目标 GPU，例如 `--gpus 0`。
- CUDA 不可用或显存不足时，不自动无记录切换 CPU；先记录原因，再决定是否改 CPU 或换 GPU。
- 若为了历史口径必须用 CPU，需要在日志中明确说明。

## 2026-05-05 Multi-GPU resource usage principle

### 修改内容

- 更新 `ACTIVATION_QUANTIZATION_LOG.md`，补充后续实验应充分利用多卡资源，而不只是使用单张 GPU。
- 本步骤不修改代码、不运行实验、不生成 run 产物。

### 原则摘要

- 独立实验配置优先按 job-level 分配到多张 GPU 并行运行。
- 对不支持内部分布式的脚本，使用 `CUDA_VISIBLE_DEVICES=<gpu_id>` 或 `--gpus <gpu_id>` 绑定单个 run。
- 每个并行 run 必须使用独立 `--run-name` 和清晰输出目录，避免产物覆盖。
- 正式并行实验前检查 `nvidia-smi`，记录可用 GPU 和显存余量。
- 已知 activation reconstruction 路径不应强行单任务 distributed；更适合多进程多卡并行跑不同配置。

## 2026-05-05 E003 phase summary and handoff

### 修改内容

- 更新 `ACTIVATION_QUANTIZATION_LOG.md`，追加 E003 阶段性总结。
- 本步骤不修改代码、不运行实验、不生成 run 产物。

### 结论摘要

- E003a 已完成 multi-sample eval 口径建立，并证明小样本 A8 init 的单张高 SNR 不具备泛化意义。
- A8 init n=2/8/16/64 在 128-sample eval 上全部约 `-7 dB`，E002b final activation reconstruction 也没有恢复。
- E003b 低样本 `init_batch_size` sweep 暂缓；`256/1024` 需要先解决 memory-safe activation init，当前多卡不能自动分摊单个 init batch 的显存峰值。
- E003c activation reconstruction 学习率 sweep 暂缓；如果后续需要，只建议先做 short sweep，不直接跑 A5000。
- 下一阶段建议进入 E004，优先研究 activation range / clipping / scale_method / percentile calibration。

## 2026-05-05 E004 sensitivity plan review

### 修改内容

- 更新 `ACTIVATION_QUANTIZATION_LOG.md`，对 E004 插入位置敏感性图谱计划做合理性分析和阶段拆分。
- 本步骤不修改代码、不运行实验、不生成 run 产物。

### 计划结论

- E004 有必要先定位 52 个 activation quantizer 中是否存在少数主导崩坏的位置。
- E004 应先做选择性 activation quantizer 开关工具，再做 sentinel 小规模验证，最后才跑完整 52 层 sweep。
- 第一轮优先使用 A8 init / pre-act-recon checkpoint，避免把 activation reconstruction 的二次影响混入插入位置敏感性分析。
- 所有正式 ranking 必须使用固定 multi-sample eval subset，并记录 sample list hash。
- E004 只回答“哪里敏感”和“保留高精度是否值得”；percentile clipping、scale_method 改造和 mixed precision 修复应留到 E005/E006。

## 2026-05-05 E004a activation sensitivity tool

### 修改内容

- 新增 `quant/activation_sensitivity.py`，提供 activation quantizer selector 和临时开关 context。
- 新增 `cli/evaluate_activation_sensitivity.py`，对已保存 checkpoint 做选择性 activation quantizer multi-sample eval。
- 新增 `tests/test_activation_sensitivity.py`，覆盖 selector、模式开关、默认排除最终输出 quantizer 和状态恢复。
- 更新 `ACTIVATION_QUANTIZATION_LOG.md`，记录 E004a 工具能力、smoke 结果和 E004b 入口。

### Smoke 结果

- `all_on`：4-sample CUDA smoke，51 个候选 quantizer，SNR mean `-9.4503 dB`。
- `all_off`：4-sample CUDA smoke，51 个候选 quantizer，SNR mean `3.2151 dB`。
- `disable_one --index 1`：CSV 只选中 1 个 quantizer，SNR mean `-9.3920 dB`。
- `disable_group --branch transformer`：CSV 选中 20 个 transformer quantizers，SNR mean `-9.4168 dB`。

### 验证方式

- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_sensitivity`
- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_evaluate_quantized_scrn_multi`
- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_diagnostics`
- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/quant/activation_sensitivity.py SCRN_BRECQ_app/scrn_brecq/cli/evaluate_activation_sensitivity.py`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_activation_sensitivity --help`
- E004a CUDA smoke runs under ignored `runs/activation_quantization/E004_sensitivity/e004a_tool_smoke/`.

## 2026-05-05 E004b sentinel activation sensitivity

### 修改内容

- 使用 E004a 工具完成 128-sample sentinel 单点关闭验证。
- 更新 `ACTIVATION_QUANTIZATION_LOG.md`，记录 sentinel 选择、run 口径、结果表和结论。
- 本步骤不修改代码、不生成 tracked 产物。

### 实验设置

- checkpoint：E002c A8 init n=64 / pre-act-recon checkpoint。
- eval：128 samples，batch size 16，seed `20260427`。
- device：CUDA。
- run root：
  `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E004_sensitivity/e004b_sentinel/`

### 关键结果

- `all_on`：SNR mean `-7.1021 dB`，SSIM mean `0.1945`。
- `all_off`：SNR mean `4.7973 dB`，SSIM mean `0.7049`。
- 最大单点恢复来自 `index=20 model.stage2.1`，SNR mean `-6.8101 dB`，相对 all_on 提升 `+0.2920 dB`。
- stage4/stage5 attention qkv/proj 单点关闭基本无恢复，ΔSNR mean 约 `-0.0018` 到 `+0.0027 dB`。

### 结论摘要

- E004b 没有发现单个 sentinel quantizer 主导 W4A8 崩坏。
- 全部 activation quantizer 关闭可恢复约 `+11.90 dB`，但任一 sentinel 单点关闭都远不能解释该差距。
- 后续不建议直接跑完整 52 层单点关闭 sweep；更合理的是先做 fusion/CNN/transformer/module type/stage 的分组关闭实验。
- `CUDA_VISIBLE_DEVICES=<id>` 绑定在当前 shell 中会导致子进程 CUDA 不可用，本轮改为无绑定 `--device cuda`，未回退 CPU。

## 2026-05-05 E004d group activation sensitivity

### 修改内容

- 使用 E004a 工具完成 128-sample 结构分组关闭验证。
- 更新 `ACTIVATION_QUANTIZATION_LOG.md`，记录 group ranking、结构定位和后续 E005 方向。
- 本步骤不修改代码、不生成 tracked 产物。

### 实验设置

- checkpoint：E002c A8 init n=64 / pre-act-recon checkpoint。
- eval：128 samples，batch size 16，seed `20260427`。
- device：CUDA。
- run root：
  `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E004_sensitivity/e004d_group/`

### 关键结果

- `all_on` baseline：SNR mean `-7.1021 dB`。
- `all_off` baseline：SNR mean `4.7973 dB`。
- 关闭全部 `module_type=Conv2d` activation quantizers：SNR mean `4.5293 dB`，恢复 `+11.6314 dB`。
- 关闭全部 `module_type=Linear` 或 `branch=transformer`：SNR mean `-7.0812 dB`，仅恢复 `+0.0209 dB`。
- `role=unknown` 五个 stage transition / downsample-like Conv2d modules 恢复 `+2.8913 dB`。
- `branch=fusion` 恢复 `+1.6960 dB`，其中 `merge_proj` 恢复 `+1.3489 dB`。

### 结论摘要

- W4A8 A8 init 崩坏主要由 Conv2d activation quantization 的结构组累积误差导致。
- Transformer / Linear / attention qkv/proj 不是当前 A8 init 崩坏主因。
- 后续不应优先继续完整 52 层单点 sweep；主线应转入 E005 Conv2d activation range / clipping / calibration 策略。

## 2026-05-05 Explicit CUDA device index for eval CLIs

### 修改内容

- `evaluate_quantized_scrn_multi.py` 新增 `--cuda-device-index`，支持直接选择 `cuda:<index>`。
- `evaluate_activation_sensitivity.py` 新增 `--cuda-device-index`，复用相同设备选择逻辑。
- `test_evaluate_quantized_scrn_multi.py` 增加设备选择单元测试。
- 更新 `ACTIVATION_QUANTIZATION_LOG.md`，记录 E004/E005 后续避开 0 卡的使用原则。

### 背景

- 当前 shell 中 `CUDA_VISIBLE_DEVICES=<id>` 会导致子进程内 CUDA 不可用。
- 0 卡可能被其他任务占用，因此需要不依赖 `CUDA_VISIBLE_DEVICES` 的显式设备选择。

### 验证方式

- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_evaluate_quantized_scrn_multi`
- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/cli/evaluate_quantized_scrn_multi.py SCRN_BRECQ_app/scrn_brecq/cli/evaluate_activation_sensitivity.py`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_activation_sensitivity --help`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_multi --help`
- CUDA index smoke：
  - `--device cuda --cuda-device-index 1`
  - run 成功记录 `device: cuda:1` 和 `cuda_device_index: 1`。

## 2026-05-05 E004 follow-up route after group sensitivity

### 修改内容

- 更新 `ACTIVATION_QUANTIZATION_LOG.md`，记录 E004d 之后的 E004 后续路线重规划。
- 本步骤只整理实验路线，不修改代码、不运行新实验、不生成 tracked 产物。

### 核心判断

- E004b sentinel 单点关闭最大只恢复 `+0.2920 dB`。
- E004d 关闭全部 `module_type=Conv2d` activation quantizers 恢复 `+11.6314 dB`。
- E004d 关闭全部 `module_type=Linear` 或 `branch=transformer` 只恢复 `+0.0209 dB`。
- 因此，E004 后续不应把完整 52 层单点关闭 sweep 作为主线，而应转向 Conv2d 子组定位。

### 后续路线

- E004e：Conv2d 子组关闭细分，包括 stage、fusion、cnn、split_proj、merge_proj、unknown、head、stage5 等。
- E004f：如 E004e 发现强子组，再做 Conv2d-only reopen / leave-one-out 验证。
- E004g：汇总 sensitivity vs resource benefit 策略表，为 E005 Conv2d activation range / clipping / calibration 提供输入。

## 2026-05-05 E004e Conv2d subgroup sensitivity

### 修改内容

- 使用 E004a 工具完成 Conv2d 子组关闭细分实验。
- 更新 `ACTIVATION_QUANTIZATION_LOG.md`，记录 E004e run 口径、结果表、关键子组和后续判断。
- 本步骤不修改代码、不生成 tracked 产物。

### 实验设置

- checkpoint：E002c A8 init n=64 / pre-act-recon checkpoint。
- eval：128 samples，batch size 16，seed `20260427`。
- device：CUDA，使用 `--cuda-device-index 1/2/3` 分批并行。
- run root：
  `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E004_sensitivity/e004e_conv2d_subgroups/`

### 关键结果

- `all_on`：SNR mean `-7.1021 dB`。
- `all_off`：SNR mean `4.7973 dB`。
- `role=unknown + Conv2d`：5 个 quantizers，SNR mean `-4.2108 dB`，恢复 `+2.8913 dB`。
- `stage5 + Conv2d`：6 个 quantizers，SNR mean `-5.0410 dB`，恢复 `+2.0611 dB`。
- `branch=fusion + Conv2d`：10 个 quantizers，SNR mean `-5.4061 dB`，恢复 `+1.6960 dB`。
- `role=merge_proj + Conv2d`：5 个 quantizers，SNR mean `-5.7532 dB`，恢复 `+1.3489 dB`。
- 单点最强 `model.stage5.1`：SNR mean `-6.0764 dB`，恢复 `+1.0256 dB`。
- `branch=cnn + Conv2d`：15 个 quantizers，SNR mean `-6.6676 dB`，只恢复 `+0.4345 dB`。

### 结论摘要

- E004e 进一步把 Conv2d 问题定位到 stage transition / downsample-like modules、stage5、fusion 和 merge projection。
- 普通 CNN branch Conv2d 不是当前 A8 init 崩坏的主因。
- 仍不存在单点完全主导；当前问题是 Conv2d 多点累积误差。
- 后续可直接进入 E005 Conv2d activation range / clipping / calibration；若继续 E004，应只做 Conv2d 全关后的 reopen 子组验证，而不是完整 52 层 sweep。

## 2026-05-05 E004f skip decision

### 修改内容

- 更新 `ACTIVATION_QUANTIZATION_LOG.md`，记录不按完整 E004f reopen 计划继续的原因。
- 本步骤只记录实验路线决策，不修改代码、不运行新实验、不生成 tracked 产物。

### 决策

- 不执行完整 E004f。
- E004f 只作为可选补充保留；如后续报告需要更强反事实证据，再实现 `disable_group + reopen_group` 组合模式。
- 当前主线改为先做 E004g 策略表收束，然后进入 E005 Conv2d activation range / clipping / calibration。

### 原因

- E004b 已证明没有单点主导层。
- E004d 已证明 Conv2d activation quantization 是 A8 init 崩坏主因。
- E004e 已把 Conv2d 敏感结构进一步定位到 `role=unknown`、`stage5`、`fusion`、`merge_proj`。
- 完整 E004f 的边际收益有限，而且需要扩展现有工具的组合开关语义。

## 2026-05-05 E004g sensitivity strategy table

### 修改内容

- 更新 `ACTIVATION_QUANTIZATION_LOG.md`，完成 E004g 策略表收束。
- 汇总 E004b/E004d/E004e 的 sensitivity ranking、resource-benefit 粗 proxy 和 E005/E006 输入。
- 记录 E004 原始目标完成度和仍未完成/不再执行的项目。
- 本步骤不修改代码、不运行新实验、不生成 tracked 产物。

### 核心结果

- E004g 使用 `selected_count`、`count_share`、`benefit_per_quantizer` 作为 activation-volume / resource 粗 proxy。
- 精确 runtime `activation_numel` 尚未采集，因此严谨 memory ranking 仍需后续 tooling。
- 最优先修复结构：
  - all Conv2d：恢复 `+11.6314 dB`
  - `role=unknown`：恢复 `+2.8913 dB`
  - `stage5 + Conv2d`：恢复 `+2.0611 dB`
  - `branch=fusion + Conv2d`：恢复 `+1.6960 dB`
  - `role=merge_proj + Conv2d`：恢复 `+1.3489 dB`
  - `model.stage5.1`：恢复 `+1.0256 dB`

### E004 收束判断

- E004 已足够回答“哪些 activation quantizer 最该保留高精度或单独处理”。
- 不再执行完整 52 层单点关闭 sweep 或完整单点开启 sweep。
- E004 输出已足够支撑进入 E005 Conv2d activation range / clipping / calibration。

## 2026-05-06 E005 activation range experiment plan

### 修改内容

- 更新 `ACTIVATION_QUANTIZATION_LOG.md`，结合 E001-E004 结果重构 E005 实验计划。
- 本步骤只记录实验设计，不修改代码、不运行新实验、不生成 tracked 产物。

### 核心判断

- E005 不应泛泛从所有 activation quantizers 开始，而应优先针对 Conv2d activation。
- 第一阶段不应直接做 per-channel / group-wise / SmoothQuant。
- 最小验证路径应是：
  1. Conv2d range diagnostics 增强。
  2. Conv2d-only percentile clipping。
  3. Conv2d-only MSE range calibration。
  4. 对 `role=unknown`、`stage5`、`fusion`、`merge_proj` 做结构化局部对照。
  5. 只有在 clipping/MSE 不足或 diagnostics 显示强 channel imbalance 时，再进入 per-channel / group-wise。

### 评估口径

- 沿用 E003a/E004 128-sample eval。
- baseline all_on A8 init：SNR mean `-7.1021 dB`。
- all_off / W4A32 近似：SNR mean `4.7973 dB`。
- E005 成败必须看 128-sample SNR / SSIM 和 E001 diagnostics，不能回到单样本 SNR。

## 2026-05-06 E005a Conv2d range diagnostics

### 修改内容

- 扩展 `quant/activation_diagnostics.py`：
  - 新增 `p99_99/p99_999` 与 `abs_p99_99/abs_p99_999`。
  - 新增 `absmax_over_p99_99/absmax_over_p99_999`。
  - 新增 `conv2d_range_summary`，按 stage、branch、role、module type 汇总 Conv2d activation range、relative MSE、effective levels 和 per-channel imbalance。
  - 将 `model.stage1.1` 到 `model.stage5.1` 标记为 `branch=stage_output`、`role=stage_output_conv`，替代旧诊断中的 `unknown/unknown`。
- 扩展 `cli/diagnose_activation_quantization.py`：
  - 新增 `--cuda-device-index`，复用 multi-eval 的 device selection，支持显式选择 `cuda:1/2/3`。
- 新增 E005a 配置：
  - `configs/activation_quantization/e005a_conv2d_range_diagnostics.json`
- 更新 `tests/test_activation_diagnostics.py`，覆盖 stage output 标签、高分位字段和 Conv2d summary。
- 为避免 128-sample 大张量诊断时间过长，diagnostics 对大张量高分位和 fake-quant 局部误差使用确定性 stride sampling；小张量仍使用精确统计。本改动只影响诊断开销，不改变模型推理或量化算法。

### 验证

- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_diagnostics`
  - 结果：7 tests passed。
- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/quant/activation_diagnostics.py SCRN_BRECQ_app/scrn_brecq/cli/diagnose_activation_quantization.py`
  - 结果：通过。
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.diagnose_activation_quantization --help`
  - 结果：通过，help 中包含 `--cuda-device-index`。

### 实验产物

- smoke：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E005_range_clipping/E005a_diagnostics/20260506_141020_e005a_smoke_conv2d_range_diagnostics_fast/`
  - `activation_quantizers=52`
  - `non_positive_delta_count=0`
- formal 128-sample：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E005_range_clipping/E005a_diagnostics/20260506_141901_e005a_conv2d_range_diagnostics/`
  - device：`cuda:1`
  - `activation_quantizers=52`
  - `non_positive_delta_count=0`
  - `effective_int_levels_min=124`
  - `conv2d_range_summary` 已写入 `summary.json`
- run 产物位于 `.gitignore` 保护目录，不纳入 Git。

### 初步结论

- E005a 支持 E004 的判断：A8 init 崩坏主信号集中在 Conv2d activation，而不是 Linear / transformer。
- Conv2d 的 worst relative MSE 和 outlier ratio 明显高于 Linear。
- 旧 `role=unknown` 已明确为 stage output 3x3 Conv2d，后续统一称为 `stage_output_conv`。
- 下一步进入 E005b：先做 Conv2d tensor-wise percentile clipping，而不是直接做 per-channel/group-wise 或 attention SmoothQuant。

## 2026-05-06 E004-E005a readable conclusion note

### 修改内容

- 更新 `ACTIVATION_QUANTIZATION_LOG.md`，追加 E004-E005a 可读结论整理。
- 本步骤只整理已有实验结论，不修改代码、不运行新实验、不生成 run 产物。

### 记录重点

- E004 sensitivity 证明 A8 init 崩坏主因是 Conv2d activation quantization 的多点累积误差。
- E004 中旧 `role=unknown` 已修正为 `stage_output_conv`。
- E005a 进一步解释 Conv2d 问题由 outlier range、relative MSE 偏高和部分 channel imbalance 混合构成。
- Linear / transformer 在 A8 init 阶段不是第一修复对象，但保留为后续 sanity check。
- E005b 入口明确为 Conv2d tensor-wise percentile clipping。

## 2026-05-06 E005b percentile clipping workflow

### 修改内容

- 新增 `quant/activation_range.py`，支持对选中 activation quantizer 执行 two-sided percentile range calibration。
- 扩展 `cli/activation_only_quantize_scrn.py`：
  - 支持 `--config`。
  - 支持 `--cuda-device-index`。
  - 支持 `--activation-range-method percentile`、`--activation-percentile` 和 range selector 参数。
  - 在 A8 init 后、保存 pre-act-recon checkpoint 前写入 percentile-calibrated `delta/zero_point`。
- 新增 E005b 配置：
  - `configs/activation_quantization/e005b_conv2d_percentile_clipping.json`
- 新增/更新单元测试：
  - `tests/test_activation_range.py`
  - `tests/test_activation_only_quantize_scrn.py`

### 验证

- TDD red：
  - `test_activation_range` 初始失败于缺少 `activation_range` 模块。
  - `test_activation_only_quantize_scrn` 初始失败于缺少新增 CLI/config 参数。
- Green checks：
  - `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_range`
  - `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_only_quantize_scrn`
  - `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_diagnostics`
  - `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/quant/activation_range.py SCRN_BRECQ_app/scrn_brecq/cli/activation_only_quantize_scrn.py`
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.activation_only_quantize_scrn --help`

### Smoke

- run dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E005_range_clipping/E005b_percentile/quant/20260506_163005_e005b_smoke_percentile_p999/`
- device：`cuda:1`
- config：
  - `num_samples=2`
  - `init_batch_size=2`
  - `activation_percentile=99.9`
  - `range_module_type=Conv2d`
- 结果：
  - `activation_quantizers=52`
  - `non_positive_delta_count=0`
  - selected Conv2d quantizers：31
  - single-sample `quant_pre_act_recon_snr_db=0.4362 dB`
- 该 smoke 只验证工具链和 checkpoint 保存，不作为正式 E005b 结论。

## 2026-05-06 E005b percentile clipping experiments

### 执行内容

- 完成 E005b-1 all Conv2d percentile sweep：
  - `99.9`
  - `99.99`
  - `99.995`
  - `99.999`
- 完成 E005b-2 结构组对照：
  - `branch=fusion`
  - `role=split_proj`
  - `role=merge_proj`
  - `role=stage_output_conv`
  - `stage=stage5 + module_type=Conv2d`
- 每个 run 均从同一个 W4 weight-recon checkpoint 出发，只做 A8 init + percentile range，不跑 activation reconstruction。
- 每个正式 checkpoint 均完成：
  - 128-sample multi eval。
  - 128-sample E005a diagnostics。
- 所有产物均写入：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E005_range_clipping/E005b_percentile/`
- run 产物处于 `.gitignore` 保护目录，不纳入 Git。

### 关键结果

baseline：

- all_on A8 init：SNR mean `-7.1021 dB`。
- all_off / W4A32 近似：SNR mean `4.7973 dB`。

all Conv2d sweep：

- p99.9：128-sample SNR mean `-11.5661 dB`。
- p99.99：128-sample SNR mean `-8.0514 dB`。
- p99.995：128-sample SNR mean `-8.0202 dB`。
- p99.999：128-sample SNR mean `-7.9668 dB`。

结构组对照：

- fusion p99.999：`-7.1144 dB`。
- split_proj p99.999：`-7.0859 dB`。
- merge_proj p99.999：`-7.1121 dB`。
- stage_output_conv p99.999：`-7.0469 dB`。
- stage5 Conv2d p99.999：`-8.1822 dB`。

### 结论

- E005b 未产生有效恢复；all Conv2d percentile clipping 全部弱于原始 all_on。
- 局部结构组最多只有 `+0.0552 dB` 的弱恢复，不能作为有效修复证据。
- p99.999 all Conv2d 虽然保持 `non_positive_delta_count=0`，但 Conv2d `fake_quant_mse_max` 从 E005a 原始 A8 init 的 `9.827e-05` 放大到 `0.007526`，说明简单 tensor-wise clipping 容易把 outlier 问题转成饱和误差。
- 后续 E005 应优先转入 MSE range calibration，而不是继续扩大 percentile sweep。

## 2026-05-06 E005c Conv2d MSE range calibration

### 修改内容

- 扩展 `quant/activation_range.py`：
  - 新增通用入口 `apply_activation_ranges()`。
  - 保留 `apply_percentile_activation_ranges()` 作为兼容接口。
  - 新增 `max` range method。
  - 新增 `mse_grid` range method。
  - 新增 `parse_mse_shrink_ratios()`。
  - 每层记录 chosen range、best shrink ratio、candidate scores、sample count 和 range shrink ratio。
- 扩展 `cli/activation_only_quantize_scrn.py`：
  - `--activation-range-method` 支持 `{none,percentile,max,mse_grid}`。
  - 新增 `--range-mse-shrink-ratios`。
  - 新增 `--range-loss-p`。
- 新增配置：
  - `configs/activation_quantization/e005c_conv2d_mse_range.json`
- 更新测试：
  - `tests/test_activation_range.py`
  - `tests/test_activation_only_quantize_scrn.py`

### 验证

- TDD red：
  - `test_activation_range` 初始失败于缺少 `apply_activation_ranges` / `parse_mse_shrink_ratios`。
  - `test_activation_only_quantize_scrn` 初始失败于 CLI 不接受 `max/mse_grid`。
- Green checks：
  - `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_range`
  - `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_only_quantize_scrn`
  - `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_diagnostics`
  - `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/quant/activation_range.py SCRN_BRECQ_app/scrn_brecq/cli/activation_only_quantize_scrn.py`
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.activation_only_quantize_scrn --help`

### Smoke

- run dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E005_range_clipping/E005c_mse_range/quant/20260506_190919_e005c_smoke_mse_grid/`
- device：`cuda:1`
- result：
  - `activation_quantizers=52`
  - `non_positive_delta_count=0`
  - selected Conv2d quantizers：31
  - single-sample `quant_pre_act_recon_snr_db=8.7106 dB`
- 该 smoke 只验证工具链和 checkpoint 保存。

### 正式实验结果

baseline：

- all_on A8 init：SNR mean `-7.1021 dB`。
- all_off / W4A32 近似：SNR mean `4.7973 dB`。

all Conv2d controls：

- max：128-sample SNR mean `-7.2963 dB`。
- MSE conservative：128-sample SNR mean `-7.1224 dB`。
- MSE standard：128-sample SNR mean `-7.7571 dB`。

结构组 MSE conservative：

- fusion：`-7.1603 dB`。
- split_proj：`-7.1614 dB`。
- merge_proj：`-7.0997 dB`。
- stage_output_conv：`-7.1444 dB`。
- stage5 Conv2d：`-7.1600 dB`。

### 结论

- E005c 未产生有效恢复。
- MSE conservative 最接近 all_on，但仍低 `0.0203 dB`，基本只能视为持平或轻微变差。
- `merge_proj` 结构组只提升 `+0.0024 dB`，属于噪声级别。
- standard MSE grid 允许更强 shrink，反而下降 `0.6550 dB`，说明局部 fake-quant loss 搜索更激进时不等于最终 SNR 更好。
- E005b/E005c 合起来基本否定了 Conv2d tensor-wise percentile/MSE range calibration 作为主修复路径。
- 后续应转入 activation per-channel / group-wise feasibility，而不是继续扩大 tensor-wise range sweep。
