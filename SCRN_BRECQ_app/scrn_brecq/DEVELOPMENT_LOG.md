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

## 2026-05-06 E005D structured Conv2d clipping selectors

### 修改内容

- 扩展 `quant/activation_range.py`：
  - 新增 `range_selector_groups`，支持多个 selector 取并集。
  - 新增 `range_exclude_selector_groups`，支持从候选集合中排除指定结构。
  - selector 支持 `index/name_contains/stage/branch/role/module_type`。
  - `range_selector_groups` 与旧单 selector 字段互斥，避免组合语义不清。
- 扩展 `cli/activation_only_quantize_scrn.py`：
  - 新增 `--range-selector-groups-json`。
  - 新增 `--range-exclude-selector-groups-json`。
  - config normalize 阶段解析和校验 selector group JSON / list。
- 新增配置：
  - `configs/activation_quantization/e005d_structured_clipping.json`
- 更新测试：
  - `tests/test_activation_range.py`
  - `tests/test_activation_only_quantize_scrn.py`

### 验证

- TDD red：
  - `test_activation_range` 初始失败于 `apply_activation_ranges()` 不接受 `selector_groups` / `exclude_selector_groups`。
  - `test_activation_only_quantize_scrn` 初始失败于 CLI 不接受 selector group JSON 参数，normalize config 不解析 selector groups。
- Green checks：
  - `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_range`
  - `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_only_quantize_scrn`

### Smoke

- run dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E005_range_clipping/E005d_structured_clipping/quant/20260506_195451_e005d_smoke_p99999_merge_stageout/`
- device：`cuda:1`
- result：
  - selected Conv2d quantizers：10
  - `activation_quantizers=52`
  - `non_positive_delta_count=0`
  - single-sample `quant_pre_act_recon_snr_db=8.3784 dB`
- 该 smoke 只验证 selector union、checkpoint 保存和正 scale 状态。

### 正式实验结果

baseline：

- all_on A8 init：SNR mean `-7.1021 dB`。
- all_off / W4A32 近似：SNR mean `4.7973 dB`。
- all_on 到 all_off gap：`11.8993 dB`。

| run | method | selected | single-sample SNR | 128-sample SNR mean | median | min | SSIM mean | delta vs all_on | recovery |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p99.999 merge + stage_output | percentile | 10 | 4.9862 | -7.0700 | -7.8076 | -14.3644 | 0.2147 | +0.0321 | 0.003 |
| p99.999 split + merge + stage_output | percentile | 15 | 4.8327 | -7.0633 | -7.8080 | -14.3794 | 0.2144 | +0.0388 | 0.003 |
| p99.999 all Conv2d except stage5 | percentile | 25 | 3.0929 | -7.5204 | -8.1969 | -14.7713 | 0.2039 | -0.4183 | -0.035 |
| p99.999 all Conv2d except head + stage5 | percentile | 24 | 4.8333 | -7.4515 | -8.2368 | -14.6998 | 0.2082 | -0.3494 | -0.029 |
| MSE conservative merge + stage_output | mse_grid | 10 | 4.9797 | -7.1323 | -7.9026 | -14.4646 | 0.2017 | -0.0302 | -0.003 |
| MSE conservative all Conv2d except stage5 | mse_grid | 25 | 8.8828 | -7.0689 | -7.9072 | -14.3853 | 0.1948 | +0.0332 | 0.003 |

### 结论

- E005D 组合 / 排除结构实验没有产生超过 `+0.2 dB` 的弱有效恢复。
- 最好的 128-sample SNR mean 是 `p99.999 split+merge+stage_output` 的 `-7.0633 dB`，仅比 all_on 高 `+0.0388 dB`。
- 排除 stage5 的 all Conv2d percentile 反而下降到 `-7.5204 dB`，说明“排除有害 stage5 后扩大 clipping 覆盖面”并不能修复 A8 init。
- MSE conservative 的 all Conv2d except stage5 单样本达到 `8.8828 dB`，但 128-sample mean 只有 `-7.0689 dB`，再次确认单样本 SNR 不可靠。
- E005b/E005c/E005D 合起来应停止 tensor-wise clipping 主线，后续进入 activation per-channel / group-wise 或 E006 mixed precision。

## 2026-05-06 E005/E006 experiment numbering update

### 记录内容

- 将 E005 明确收束为离群值、range、clipping 实验线。
- 将 E006 明确定义为 activation 量化粒度实验线。
- 前文中提到的 “E005E：activation per-channel / group-wise feasibility” 统一重命名为 “E006a：Conv2d activation per-channel feasibility”。

### 原因

- E005b/E005c/E005D 已经覆盖 tensor-wise percentile、MSE/max range、结构化 clipping 组合与排除实验。
- 所有 tensor-wise clipping / range calibration 方案均没有产生超过 `+0.2 dB` 的有效恢复。
- 继续扩大 E005 tensor-wise sweep 价值较低；下一阶段应验证新的假设：activation tensor-wise 粒度过粗是否是主要瓶颈。

### 下一步入口

- E006a 应先做 feasibility，不直接做完整部署策略。
- E006a 重点验证 Conv2d activation per-channel 的 scale shape、forward 广播、checkpoint 保存/恢复、diagnostics 兼容和 128-sample eval。
- 注意不能直接复用权重量化的 `channel_wise=True` 语义；Conv2d activation per-channel 应按 `[N, C, H, W]` 的 `C` 维，目标 `delta/zero_point` 形状为 `[1, C, 1, 1]`。

## 2026-05-06 E006 granularity experiment roadmap

### 记录内容

- 在 `ACTIVATION_QUANTIZATION_LOG.md` 中补充 E006 总体展开计划。
- E006 被定义为 activation 量化粒度实验线，用于验证 tensor-wise activation 粒度是否是 W4A8 失败主因。

### 阶段划分

- E006a：Conv2d activation per-channel feasibility。
  - 目标是验证 `[1, C, 1, 1]` activation `delta/zero_point` 的广播、保存/恢复、diagnostics 和 128-sample eval。
- E006b：Conv2d activation group-wise feasibility。
  - 目标是验证 group size 4/8/16 是否能以较低复杂度接近 per-channel 收益。
- E006c：结构化粒度对照。
  - 目标是判断是否需要 all Conv2d 细粒度，还是只处理 fusion / merge_proj / split_proj / stage_output_conv / stage5 即可。
- E006d：Linear / transformer sanity check。
  - 目标是确认 Conv2d 粒度策略不会把瓶颈转移到 attention / Linear。
- E006e：策略收束。
  - 输出 per-channel 上限、group-wise 折中、结构化粒度策略表，并决定是否进入 mixed precision / selective FP32。

### 执行原则

- E006 第一阶段不跑 activation reconstruction，只看 A8 init 粒度本身。
- 不直接复用权重量化 `channel_wise=True`。
- 正式结论继续使用 128-sample eval；single-sample 只作为 smoke。
- 如果 per-channel / group-wise 均无效，则停止粒度方向，转入 mixed precision / selective FP32 / reconstruction objective。

## 2026-05-06 E006a Conv2d activation per-channel feasibility

### 修改内容

- 扩展 `quant/activation_range.py`：
  - 新增 `activation_granularity`，支持 `tensor` 和 `per_channel`。
  - `per_channel` 第一版只支持 `mse_grid`，用于隔离 activation 粒度变量。
  - 对 4D Conv2d activation `[N, C, H, W]` 按 C 维逐通道做 MSE grid range search，并写入 `[1, C, 1, 1]` 形状的 activation `delta/zero_point`。
  - 当原 activation `delta` 是 scalar `nn.Parameter` 且 shape 不一致时，替换为新的 per-channel `nn.Parameter`，避免错误 `copy_`。
- 扩展 `cli/activation_only_quantize_scrn.py`：
  - 新增 `--activation-granularity {tensor,per_channel}`。
  - 默认保持 `tensor`，兼容 E005 range/clipping 配置。
  - range calibration summary 记录 `activation_granularity`。
- 扩展 `quant/activation_diagnostics.py`：
  - per-channel activation fake-quant diagnostics 对大 4D tensor 使用保持 C 维的确定性采样，避免 128-sample diagnostics 对 per-channel quantizer 退化为全量计算。
- 新增配置：
  - `configs/activation_quantization/e006a_conv2d_per_channel.json`
- 新增/更新测试：
  - `tests/test_activation_range.py`
  - `tests/test_activation_only_quantize_scrn.py`
  - `tests/test_activation_diagnostics.py`
  - `tests/test_evaluate_quantized_scrn.py`

### 验证

- TDD red：
  - `test_activation_range` 初始失败于 `apply_activation_ranges()` 不接受 `activation_granularity`。
  - `test_activation_only_quantize_scrn` 初始失败于 CLI 不接受 `--activation-granularity`、config 缺少 `activation_granularity`。
  - `test_activation_diagnostics` 新增 per-channel 大输出采样测试，初始失败于 `fake_quant_sampled=False`。
- Green checks：
  - `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_range`
  - `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_only_quantize_scrn`
  - `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_diagnostics SCRN_BRECQ_app.scrn_brecq.tests.test_evaluate_quantized_scrn`
  - `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/quant/activation_range.py SCRN_BRECQ_app/scrn_brecq/quant/activation_diagnostics.py SCRN_BRECQ_app/scrn_brecq/cli/activation_only_quantize_scrn.py`
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.activation_only_quantize_scrn --help`
  - `conda run -n quant python -m json.tool SCRN_BRECQ_app/scrn_brecq/configs/activation_quantization/e006a_conv2d_per_channel.json`

### Smoke

- run dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006a_conv2d_per_channel/quant/20260506_210819_e006a_smoke_conv2d_per_channel_mse/`
- device：`cuda:1`
- config：
  - `num_samples=2`
  - `init_batch_size=2`
  - `activation_range_method=mse_grid`
  - `activation_granularity=per_channel`
  - `range_module_type=Conv2d`
- result：
  - selected Conv2d quantizers：31
  - first selected layer `delta/zero_point` shape：`[1, 64, 1, 1]`
  - `activation_quantizers=52`
  - `activation_delta_count=52`
  - `activation_zero_point_count=52`
  - `non_positive_delta_count=0`
  - single-sample `quant_pre_act_recon_snr_db=5.1128 dB`
- reload eval：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006a_conv2d_per_channel/eval/20260506_210901_e006a_smoke_reload_eval2/`
  - 2-sample reload eval completed on `cuda:1`，确认 per-channel activation checkpoint 可恢复并 forward。

### 正式实验结果

baseline：

- all_on A8 init：128-sample SNR mean `-7.1021 dB`。
- all_off / W4A32 近似：128-sample SNR mean `4.7973 dB`。

E006a all Conv2d per-channel MSE：

- quant run：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006a_conv2d_per_channel/quant/20260506_210914_e006a_conv2d_per_channel_mse/`
- eval run：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006a_conv2d_per_channel/eval/20260506_211010_e006a_conv2d_per_channel_eval128/`
- diagnostics run：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006a_conv2d_per_channel/diagnostics/20260506_211053_e006a_conv2d_per_channel_diagnostics128/`

| run | selected | single-sample SNR | 128-sample SNR mean | median | min | max | SSIM mean | delta vs all_on |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all Conv2d per-channel MSE | 31 | 3.2283 | -5.3817 | -5.9146 | -12.3092 | 5.9752 | 0.5549 | +1.7203 |

Diagnostics：

- `activation_quantizers=52`
- `activation_delta_count=52`
- `activation_zero_point_count=52`
- `non_positive_delta_count=0`
- selected Conv2d channel counts：min `32`，max `64`
- selected Conv2d `delta/zero_point` shape：`[1, C, 1, 1]`
- tail output quantizer remains scalar and disabled.
- `effective_int_levels_min=139`

### 结论

- E006a 证明 Conv2d activation per-channel 粒度有强信号：128-sample SNR mean 从 all_on `-7.1021 dB` 恢复到 `-5.3817 dB`，提升 `+1.7203 dB`。
- 这超过 E006a 预设的 `+1 dB` 强信号阈值，说明 tensor-wise activation 粒度过粗是当前 W4A8 A8 init 崩坏的重要原因之一。
- 但 per-channel 仍远低于 all_off / W4A32 的 `4.7973 dB`，只恢复了 all_on 到 all_off gap 的一部分，不能视为完整修复。
- 下一步应进入 E006b group-wise feasibility，验证 group size 4/8/16 是否能以更低复杂度接近 per-channel 收益；同时 E006c 需要做结构化 per-channel 对照。

## 2026-05-06 E006a default single-sample sanity check

### 目的

- 使用默认单图测试对：
  - `SCRN-main/test_data/clear.npy`
  - `SCRN-main/test_data/noise_and_miss.npy`
- 对 E006a formal checkpoint 做单样本 reload eval。
- 生成七面板图，对比旧 tensor-wise W4A8 init 和 E006a Conv2d per-channel W4A8 init。

### 运行

- E006a checkpoint：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006a_conv2d_per_channel/quant/20260506_210914_e006a_conv2d_per_channel_mse/checkpoints/quantized_scrn_brecq_pre_act_recon.pth`
- eval run：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006a_conv2d_per_channel/single_eval/20260506_213432_e006a_clear_noise_single_cpu/`
- CLI：
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn --checkpoint ... --eval-clean-path SCRN-main/test_data/clear.npy --eval-input-path SCRN-main/test_data/noise_and_miss.npy --device cpu --save-figure`
- 说明：
  - `evaluate_quantized_scrn.py` 暂无 `--cuda-device-index`。
  - 单图很小，本次用 CPU 跑，避免默认占用 GPU 0。

### 结果

| panel | SNR dB | SSIM |
|---|---:|---:|
| Input | 3.9693 | 0.6053 |
| FP32 | 11.7869 | 0.8697 |
| W4A32 pre weight recon | 11.4071 | 0.8255 |
| W4A32 post weight recon | 11.6961 | 0.8660 |
| W4A8 tensor-wise init | 4.9875 | 0.6576 |
| W4A8 E006a per-channel init | 3.2318 | 0.5931 |

- E006a per-channel vs tensor-wise init：`-1.7556 dB`。
- E006a per-channel vs W4A32 post weight recon：`-8.4643 dB`。
- 七面板图：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006a_conv2d_per_channel/single_eval/20260506_213432_e006a_clear_noise_single_cpu/seven_panel_tensor_vs_e006a.png`
- 七面板指标：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006a_conv2d_per_channel/single_eval/20260506_213432_e006a_clear_noise_single_cpu/seven_panel_metrics.json`

### 结论

- 默认单图上，E006a per-channel A8 init 低于旧 tensor-wise A8 init。
- 这与 128-sample eval 的结论相反：E006a 在 128-sample mean 上相对 all_on 提升 `+1.7203 dB`。
- 因此该单图 sanity check 再次确认：单样本 SNR 只能用于可视化和 smoke，不能作为 E006 粒度策略的正式判断依据。

## 2026-05-06 E006b Conv2d activation group-wise implementation and smoke

代码改动：

- `quant/activation_range.py`
  - 扩展 `activation_granularity`，新增 `group_wise`。
  - 新增 `activation_group_size`，仅在 `group_wise + mse_grid` 下启用。
  - Conv2d activation group-wise 按 `[N, C, H, W]` 的 C 维连续分组，最后一组允许不足 group size。
  - 写入的 activation `delta/zero_point` 仍保持 `[1, C, 1, 1]`，同组 channel 重复同一组 scale / zero point，复用 E006a checkpoint restore 和 forward 广播路径。
- `cli/activation_only_quantize_scrn.py`
  - 新增 `--activation-granularity group_wise`。
  - 新增 `--activation-group-size`。
  - config normalize 默认保持 `activation_granularity=tensor`、`activation_group_size=None`，旧 E005/E006a 配置不受影响。
- 新增 E006b 配置：
  - `configs/activation_quantization/e006b_conv2d_group_wise_g4.json`
  - `configs/activation_quantization/e006b_conv2d_group_wise_g8.json`
  - `configs/activation_quantization/e006b_conv2d_group_wise_g16.json`

测试：

- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_range`
  - 19 tests OK。
- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_only_quantize_scrn`
  - 15 tests OK。
- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_diagnostics SCRN_BRECQ_app.scrn_brecq.tests.test_evaluate_quantized_scrn`
  - 11 tests OK。
- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/quant/activation_range.py SCRN_BRECQ_app/scrn_brecq/cli/activation_only_quantize_scrn.py SCRN_BRECQ_app/scrn_brecq/quant/activation_diagnostics.py`
  - passed。
- CLI `--help` 已确认出现：
  - `--activation-granularity {tensor,per_channel,group_wise}`
  - `--activation-group-size ACTIVATION_GROUP_SIZE`

E006b-0 smoke：

- config：`e006b_conv2d_group_wise_g8.json`
- override：`num_samples=2`、`init_batch_size=2`、`batch_size=2`、`cuda_device_index=1`
- quant run：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/quant/20260506_220432_e006b_smoke_group_wise_g8_mse/`
- checkpoint：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/quant/20260506_220432_e006b_smoke_group_wise_g8_mse/checkpoints/quantized_scrn_brecq_pre_act_recon.pth`
- smoke metrics：
  - `post_weight_snr=11.6961`
  - `pre_act_snr=7.3547`
  - `selected_count=31`
  - `activation_quantizers=52`
  - `activation_delta_count=52`
  - `activation_zero_point_count=52`
  - `non_positive_delta_count=0`
  - selected Conv2d `delta_shape/zero_point_shape=[1,C,1,1]`
- reload eval：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/eval/20260506_220506_e006b_smoke_g8_reload_eval2/`
  - 2-sample reload eval completed on `cuda:1`，确认 group-wise activation checkpoint 可恢复并 forward。
- diagnostics：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/diagnostics/20260506_220522_e006b_smoke_g8_diagnostics2/`
  - `activation_quantizers=52`
  - `non_positive_delta_count=0`
  - `activation_stat_count=52`
  - `fake_quant_mse_max=7.977941277204081e-05`

当前结论：

- E006b group-wise 实现、checkpoint 保存/恢复和 diagnostics 兼容性 smoke 已通过。
- 该 smoke 不能作为正式效果结论；下一步必须跑 g4/g8/g16 固定 128-sample eval 和 diagnostics。

## 2026-05-06 E006b formal group-wise 128-sample eval

实验口径：

- 起点：E002b W4 weight-recon checkpoint。
- 不跑 activation reconstruction。
- activation range：`mse_grid`。
- activation granularity：`group_wise`。
- selected quantizers：all Conv2d activation quantizers，`selected_count=31`。
- eval：128 samples，`seed=20260427`，`batch_size=16`。
- baseline：
  - all_on tensor-wise A8 init：`-7.1021 dB`。
  - E006a all Conv2d per-channel MSE：`-5.3817 dB`。
  - E006a per-channel recovery：`+1.7203 dB`。
  - E006b 70% threshold：`+1.204 dB`。

Runs：

- g4 quant：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/quant/20260506_220727_e006b_conv2d_group_wise_g4_mse/`
- g8 quant：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/quant/20260506_220727_e006b_conv2d_group_wise_g8_mse/`
- g16 quant：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/quant/20260506_220727_e006b_conv2d_group_wise_g16_mse/`
- eval：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/eval/20260506_220836_e006b_group_wise_g4_eval128/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/eval/20260506_220837_e006b_group_wise_g8_eval128/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/eval/20260506_220837_e006b_group_wise_g16_eval128/`
- diagnostics：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/diagnostics/20260506_220911_e006b_group_wise_g4_diagnostics128/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/diagnostics/20260506_220912_e006b_group_wise_g8_diagnostics128/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006b_conv2d_group_wise/diagnostics/20260506_220912_e006b_group_wise_g16_diagnostics128/`

Formal results：

| group size | SNR mean | median | min | max | SSIM mean | delta vs all_on | recovery vs E006a gain | improved vs all_on |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | -6.1885 | -6.8713 | -13.2416 | 6.2019 | 0.4033 | +0.9136 | 53.1% | 120 / 128 |
| 8 | -8.5523 | -9.4064 | -15.9452 | 6.6961 | 0.3927 | -1.4502 | -84.3% | 3 / 128 |
| 16 | -7.4376 | -8.2121 | -14.7473 | 6.7325 | 0.2547 | -0.3355 | -19.5% | 1 / 128 |

Diagnostics summary：

| group size | activation quantizers | activation stats | non-positive delta count | fake quant MSE max |
|---:|---:|---:|---:|---:|
| 4 | 52 | 52 | 0 | 0.023934995755553246 |
| 8 | 52 | 52 | 0 | 0.023966144770383835 |
| 16 | 52 | 52 | 0 | 0.009504307061433792 |

Conclusion：

- g4 是 E006b 中唯一有正收益的 group-wise 设置，128-sample SNR mean 相对 all_on 提升 `+0.9136 dB`。
- 但 g4 只达到 E006a per-channel 恢复量的约 `53.1%`，没有达到预设 `70%` / `+1.204 dB` 部署价值阈值。
- g8 和 g16 明显不稳定，平均 SNR 低于 all_on；较粗 group-wise 不能替代 per-channel。
- E006b 结论：group-wise 有方向信号，但当前简单连续分组不是足够强的折中方案。后续应进入 E006c 结构化粒度对照，优先测试 selective per-channel / selective group-wise，而不是把 all Conv2d group-wise 作为主策略。

## 2026-05-07 E006c structured activation granularity configs and smoke

目标：

- 在 E006a all Conv2d per-channel `+1.7203 dB` 强信号、E006b all Conv2d g4 仅恢复 `53.1%` 的背景下，进入结构化粒度对照。
- E006c 第一阶段不改核心量化代码，复用：
  - `activation_granularity=per_channel`
  - `activation_granularity=group_wise`
  - `activation_group_size=4`
  - `range_selector_groups`

新增配置：

- selective per-channel：
  - `configs/activation_quantization/e006c_pc_fusion.json`
  - `configs/activation_quantization/e006c_pc_split_proj.json`
  - `configs/activation_quantization/e006c_pc_merge_proj.json`
  - `configs/activation_quantization/e006c_pc_stage_output_conv.json`
  - `configs/activation_quantization/e006c_pc_stage5.json`
  - `configs/activation_quantization/e006c_pc_split_merge_stage_output.json`
- selective group-wise g4 supplement：
  - `configs/activation_quantization/e006c_g4_split_merge_stage_output.json`

配置统一口径：

- 起点：E002b W4 weight-recon checkpoint。
- calibration：`num_samples=64`、`init_batch_size=64`。
- activation range：`mse_grid`。
- `skip_act_recon=true`。
- run root：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/`

测试：

- 先新增 E006c config parse 测试，并确认红灯：
  - `test_activation_only_quantize_scrn` 因 7 个 E006c config 文件不存在失败。
- 添加配置后复跑：
  - `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_only_quantize_scrn`
  - 17 tests OK。
- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/cli/activation_only_quantize_scrn.py SCRN_BRECQ_app/scrn_brecq/quant/activation_range.py`
  - passed。
- CLI `--help` 已确认 selector / granularity 参数仍可用。

E006c-0 smoke：

- config：`e006c_pc_split_merge_stage_output.json`
- override：`num_samples=2`、`init_batch_size=2`、`batch_size=2`、`cuda_device_index=1`
- quant：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/quant/20260507_140733_e006c_smoke_pc_split_merge_stage_output_mse/`
- checkpoint：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/quant/20260507_140733_e006c_smoke_pc_split_merge_stage_output_mse/checkpoints/quantized_scrn_brecq_pre_act_recon.pth`
- reload eval：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/eval/20260507_140757_e006c_smoke_pc_split_merge_stage_output_reload_eval2/`
- diagnostics：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/diagnostics/20260507_140828_e006c_smoke_pc_split_merge_stage_output_diagnostics2/`

Smoke 结果：

| item | value |
|---|---:|
| selected Conv2d quantizers | 15 |
| activation quantizers | 52 |
| activation delta count | 52 |
| activation zero point count | 52 |
| non-positive delta count | 0 |
| smoke single pre-act SNR | 6.9413 dB |
| reload eval post-recon SNR mean | -1.7166 dB |
| diagnostics activation stat count | 52 |
| diagnostics fake quant MSE max | 9.835336823016405e-05 |

复核：

- selected 15 个 Conv2d 对应 `split_proj + merge_proj + stage_output_conv`。
- selected Conv2d activation `delta/zero_point` shape 为 `[1, C, 1, 1]`。
- checkpoint reload eval completed on `cuda:1`，确认 selective per-channel activation checkpoint 可恢复并 forward。

当前结论：

- E006c 配置、parser 测试、selective per-channel smoke、checkpoint restore 和 diagnostics 兼容性已通过。
- smoke 单图不作为正式效果证据。
- 下一步进入 E006c-1：6 个 selective per-channel 配置的固定 128-sample eval 和 diagnostics。

## 2026-05-07 E006c structured activation granularity formal results

补充配置：

- E006c-1 single-structure per-channel 结果触发了 g4 supplement 条件：
  - fusion per-channel gain `+1.3722 dB`
  - merge_proj per-channel gain `+1.1253 dB`
  - stage_output_conv per-channel gain `+2.1301 dB`
- 因此新增：
  - `configs/activation_quantization/e006c_g4_fusion.json`
  - `configs/activation_quantization/e006c_g4_merge_proj.json`
  - `configs/activation_quantization/e006c_g4_stage_output_conv.json`
- `test_activation_only_quantize_scrn.py` 同步扩展 E006c g4 config parse 测试。

追加测试：

- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_only_quantize_scrn`
  - 17 tests OK。
- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/cli/activation_only_quantize_scrn.py SCRN_BRECQ_app/scrn_brecq/quant/activation_range.py`
  - passed。

统一口径：

- 起点：E002b W4 weight-recon checkpoint。
- calibration：`num_samples=64`、`init_batch_size=64`。
- activation range：`mse_grid`。
- `skip_act_recon=true`。
- eval：128 samples，`seed=20260427`，`batch_size=16`。
- improved count：按 `path` 对齐 E004b all_on baseline per-sample metrics。
- baseline：
  - all_on tensor-wise A8 init：`-7.1021 dB`。
  - E006a all Conv2d per-channel：`-5.3817 dB`。
  - E006b all Conv2d g4：`-6.1885 dB`。
  - 70% threshold：`+1.204 dB`。

Selective per-channel formal results：

| run | selected | SNR mean | median | min | max | SSIM mean | delta vs all_on | recovery vs E006a gain | improved vs all_on |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fusion | 10 | -5.7299 | -6.3070 | -12.8216 | 6.4301 | 0.1551 | +1.3722 | 79.8% | 124 / 128 |
| split_proj | 5 | -6.8865 | -7.6169 | -14.0823 | 6.4158 | 0.1817 | +0.2156 | 12.5% | 122 / 128 |
| merge_proj | 5 | -5.9768 | -6.6454 | -13.1202 | 6.7905 | 0.1649 | +1.1253 | 65.4% | 128 / 128 |
| stage_output_conv | 5 | -4.9720 | -5.5819 | -11.9419 | 6.8071 | 0.2423 | +2.1301 | 123.8% | 128 / 128 |
| stage5 | 6 | -10.7413 | -11.7262 | -18.3867 | 6.7568 | 0.4661 | -3.6392 | -211.5% | 1 / 128 |
| split + merge + stage_output | 15 | -2.1925 | -2.4806 | -8.5691 | 6.8545 | 0.2729 | +4.9096 | 285.4% | 124 / 128 |

Selective g4 supplement results：

| run | selected | SNR mean | median | min | max | SSIM mean | delta vs all_on | recovery vs E006a gain | improved vs all_on |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fusion g4 | 10 | -6.1183 | -6.7607 | -13.2373 | 6.7701 | 0.1693 | +0.9838 | 57.2% | 125 / 128 |
| merge_proj g4 | 5 | -6.3012 | -7.0148 | -13.4791 | 6.7940 | 0.1712 | +0.8009 | 46.6% | 128 / 128 |
| stage_output_conv g4 | 5 | -5.8190 | -6.4788 | -12.8934 | 6.7978 | 0.2231 | +1.2831 | 74.6% | 127 / 128 |
| split + merge + stage_output g4 | 15 | -4.3776 | -4.8955 | -11.2607 | 6.7751 | 0.2353 | +2.7245 | 158.4% | 127 / 128 |

Diagnostics summary：

| run | activation quantizers | activation stat count | non-positive delta count | fake quant MSE max |
|---|---:|---:|---:|---:|
| pc fusion | 52 | 52 | 0 | 9.82716228463687e-05 |
| pc split_proj | 52 | 52 | 0 | 9.82716228463687e-05 |
| pc merge_proj | 52 | 52 | 0 | 9.82716228463687e-05 |
| pc stage_output_conv | 52 | 52 | 0 | 9.82716228463687e-05 |
| pc stage5 | 52 | 52 | 0 | 0.05554213374853134 |
| pc split + merge + stage_output | 52 | 52 | 0 | 9.82716228463687e-05 |
| g4 fusion | 52 | 52 | 0 | 9.82716228463687e-05 |
| g4 merge_proj | 52 | 52 | 0 | 9.82716228463687e-05 |
| g4 stage_output_conv | 52 | 52 | 0 | 9.82716228463687e-05 |
| g4 split + merge + stage_output | 52 | 52 | 0 | 9.82716228463687e-05 |

Run paths：

- per-channel quant:
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/quant/20260507_141108_e006c_pc_fusion_mse/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/quant/20260507_141108_e006c_pc_split_proj_mse/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/quant/20260507_141108_e006c_pc_merge_proj_mse/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/quant/20260507_141227_e006c_pc_stage_output_conv_mse/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/quant/20260507_141227_e006c_pc_stage5_mse/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/quant/20260507_141227_e006c_pc_split_merge_stage_output_mse/`
- g4 quant:
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/quant/20260507_141721_e006c_g4_fusion_mse/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/quant/20260507_141721_e006c_g4_merge_proj_mse/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/quant/20260507_141721_e006c_g4_stage_output_conv_mse/`
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E006_granularity/E006c_structured_granularity/quant/20260507_141820_e006c_g4_split_merge_stage_output_mse/`

结论：

- E006c 明确通过 acceptance criteria：
  - `fusion` selective per-channel 达到 `+1.3722 dB`，超过 `+1.204 dB`。
  - `stage_output_conv` selective per-channel 达到 `+2.1301 dB`。
  - `split + merge + stage_output` selective per-channel 达到 `+4.9096 dB`，显著超过 all Conv2d per-channel。
- `stage5` selective per-channel 明显有害，SNR mean `-10.7413 dB`，后续不应把 stage5 作为独立细粒度策略。
- g4 仍有部署价值信号：
  - `stage_output_conv g4` 达到 `+1.2831 dB`，超过 70% threshold。
  - `split + merge + stage_output g4` 达到 `+2.7245 dB`，也明显超过 all Conv2d per-channel。
- E006c 反驳了“必须 all Conv2d per-channel”的假设：收益主要可由结构化 selective 方案恢复，且过度扩展到 stage5 / all Conv2d 可能引入额外误差。
- 后续更合理的收束方向是 selective per-channel / selective g4，而不是 all Conv2d per-channel。

## 2026-05-07 Stratified calibration/test dataset preparation

### 修改内容

- 新增 `data/stratified_scrn_datasets.py`，把 `10750_0` 的五个来源区间显式固化：
  - `1-300`
  - `301-3655`
  - `3656-4405`
  - `4406-4885`
  - `4886-10750`
- 新增最大余数法配额计算和分层抽样工具。对 1024 个 calibration patch，严格按总数约束得到：
  - `1997_2.5D_shots`: `28`
  - `7m_shots_0201`: `320`
  - `Anisotropic_FD_Model`: `71`
  - `Kerry3D`: `46`
  - `Shots0001_0200`: `559`
- 新增 `cli/prepare_scrn_stratified_datasets.py`，支持：
  - `--mode calibration|test|both`
  - `--seed`
  - `--dry-run`
  - `--overwrite`
  - 默认从 `scrn_quant_10750_0_patches` 生成 stratified calibration clean patch 目录。
  - 默认从旧工程本地 SEG-Y 源生成 478 个 legacy-logic clean test patch。
- 新增 `tests/test_stratified_scrn_datasets.py`，覆盖配额计算、编号到来源映射、分层抽样、manifest 输出、patch 切分和训练 hash 排除。
- 更新 `data/__init__.py` 导出新数据准备入口。

### 设计说明

- `10750_0` 原始生成脚本没有保存 patch 坐标，因此 calibration 来源只能通过 `train_data_N.npy` 的编号区间恢复。
- 原计划里 `29/320/71/46/559` 合计为 `1025`，与 1024 calibration 目标矛盾；实现中以“总数必须为 1024”和最大余数法为准，修正为 `28/320/71/46/559`。
- 测试集生成复刻 `10750_0` 的整文件读取逻辑：SEG-Y 全文件读为 `[samples, traces]`，absmax 归一化，`128x128` patch，stride `48`，无增强，过滤低方差 patch。
- 测试集默认按 float32 patch 内容的 SHA-256 排除与训练 patch 完全相同的候选；这只能避免直接重复，不能证明空间区域严格隔离。
- 真实 SEG-Y 测试集生成采用 reservoir sampling，只保留最终需要的 478 个 patch，避免把全部候选 patch 常驻内存。

### 生成结果

- Calibration output:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_quant_10750_0_cali_1024_stratified`
  - `.npy` count: `1024`
  - manifest counts: `28 / 320 / 71 / 46 / 559`
- Test output:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_quant_test_478_legacy_logic`
  - `.npy` count: `478`
  - manifest counts: `75 / 16 / 387`
  - `training_hash_excluded_count`: `10676`

### 验证方式

- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_stratified_scrn_datasets -v`
  - 7 tests OK。
- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/data/stratified_scrn_datasets.py SCRN_BRECQ_app/scrn_brecq/cli/prepare_scrn_stratified_datasets.py`
  - passed。
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.prepare_scrn_stratified_datasets --help`
  - passed。
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.prepare_scrn_stratified_datasets --mode calibration`
  - generated `1024` clean calibration patches。
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.prepare_scrn_stratified_datasets --mode test`
  - generated `478` clean test patches。

## 2026-05-07 Paper-style 5-source SCRN dataset rebuild

### 修改内容

- 新增 `data/paper_scrn_datasets.py`，实现区别于旧 `10750_0` 的 paper-style 5-source 数据准备流程。
- 新增 `cli/prepare_scrn_paper_datasets.py`，支持：
  - `--mode train|calibration|test|all`
  - `--seed`
  - `--dry-run`
  - `--overwrite`
  - `--exclude-training-hashes / --no-exclude-training-hashes`
- 新增 `tests/test_paper_scrn_datasets.py`，覆盖：
  - Table 2 几何计数：`60*5=300`、`671*5=3355`、`150*5=750`、`96*5=480`、`1173*5=5865`
  - SourceX-style shot 选择和 Kerry3D full-matrix trace window
  - original + 4 seeded augmentation
  - paper-style train manifest 的 1024 分层 calibration 抽样
  - test quota、exact hash 排除、quota 不足报错
- 更新 `data/__init__.py`，导出 paper-style 数据集准备入口。

### 协议说明

- 输出目录：
  - train: `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_train_10750`
  - calibration: `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_cali_1024_stratified`
  - test: `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_test_478`
- train 使用本地可用的 5 个 SCRN Table 2 来源：
  - `1997_2.5D_shots`
  - `7m_shots_0201`
  - `Anisotropic_FD_Model`
  - `Kerry3D`
  - `Shots0001_0200`
- shot-gather 源按 SourceX-style 连续 trace block 读取最前面的 table shot；Kerry3D 按 migration image full-matrix trace window 读取。
- patch size 为 `128x128`，stride 为 `48x48`。
- train 每个 raw sliding-window patch 保存 original + `4` 个 seeded random augmentation，seed 为 `20260507`。
- 为严格匹配 Table 2 / Table 3 的 patch 数量，paper-style train/test 默认使用几何窗口口径，不套用旧 `10750_0` 的 `std > 1e-3` 低方差过滤。旧 legacy-logic 数据集仍保持原过滤口径。
- test 默认对新 train set 的 float32 patch SHA-256 做 exact hash 排除；如果训练后第一个 deterministic region 排除后不足 quota，则按文件顺序继续读取后续 SourceX shot 或 trace window，直到满足固定 quota。manifest 记录 `per_source_region_counts`、候选数量和 hash 排除数量。
- 该实现只能声明为 deterministic paper-style protocol，不能声明完全复现论文作者未公开的 shot 编号和空间起点。

### 生成结果

- Train output:
  - `.npy` count: `10750`
  - manifest counts: `300 / 3355 / 750 / 480 / 5865`
- Calibration output:
  - `.npy` count: `1024`
  - manifest counts: `28 / 320 / 71 / 46 / 559`
- Test output:
  - `.npy` count: `478`
  - manifest counts: `75 / 16 / 387`
  - `training_hash_excluded_count`: `1079`
  - `per_source_region_counts`: `Anisotropic=3, Kerry3D=7, Shots0001=3`
  - `per_source_candidate_counts`: `Anisotropic=225, Kerry3D=168, Shots0001=1173`
  - `per_source_training_hash_excluded_counts`: `Anisotropic=150, Kerry3D=144, Shots0001=785`

### 验证方式

- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_paper_scrn_datasets -v`
  - 11 tests OK。
- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/data/paper_scrn_datasets.py SCRN_BRECQ_app/scrn_brecq/cli/prepare_scrn_paper_datasets.py`
  - passed。
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.prepare_scrn_paper_datasets --help`
  - passed。
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.prepare_scrn_paper_datasets --mode all --overwrite`
  - generated train `10750`、calibration `1024`、test `478`。
- `pytest` note:
  - `conda run -n quant python -m pytest ...` cannot run because the `quant` environment does not have `pytest` installed; equivalent `unittest` verification passed.

## 2026-05-07 Paper-style 10750 SCRN FP32 retraining

### 目的

- 使用新生成的 paper-style 5-source train set 重新训练一个 FP32 SCRN candidate：
  - dataset: `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_train_10750`
- 尽量对齐历史 app DDP baseline：
  - old run: `SCRN_BRECQ_app/scrn_repro/runs/train/20260425_192916_four_gpu_train_quant_10750_0`
  - old dataset: `/home/data1/hanwen/project/Project/SCRN_quant/train_data/10750_0`
  - old best loss: `1.3390747353301518`
  - old best epoch: `78`
  - old single eval: `after_snr_db=11.78722661219287`, `after_ssim=0.8699862043155245`

### 执行记录

- 首次在 sandbox 内直接跑 `torchrun --standalone` 失败，原因是 torch distributed local TCP rendezvous 被 sandbox 拒绝：
  - `Operation not permitted`
  - `RendezvousConnectionError`
- 随后在用户批准的外部执行权限下运行同一训练命令。
- GPU scope:
  - `CUDA_VISIBLE_DEVICES=1,2,3`
  - `torchrun --nproc_per_node=3`
  - config 记录 `--gpus 1,2,3`
- 与历史 run 的主要差异：
  - dataset 从旧 `10750_0` 换成 `scrn_paper5_train_10750`
  - world size 从 `4` 变为 `3`
  - per-GPU batch size 保持 `32`
  - global batch 从 `128` 变为 `96`
- 其余关键参数保持一致：
  - `epochs=80`
  - `lr=0.001`
  - `weight_decay=0.0`
  - `milestones=20,40,60`
  - `gamma=0.2`
  - `seed=20260425`
  - `num_workers=2`
  - `dim=64`
  - `stage_depths=1,1,1,1,1`
  - `head_dim=32`
  - `window_size=8`
  - `drop_path_rate=0.0`
  - `input_resolution=128`
- Train run:
  - `SCRN_BRECQ_app/scrn_repro/runs/train/20260507_170001_paper5_10750_ddp3_seed20260425`
  - git commit recorded by config: `58aa379ff9a871e41ec03decb0f6cce324f93ecd`
  - `best_loss=0.028283805948116685`
  - `best_epoch=74`
  - `last_loss=0.03233760286976966`
  - checkpoint: `checkpoints/best.pth`

### 单样本旧口径评估

- Eval command 使用历史单样本数据：
  - clean: `SCRN-main/test_data/clear.npy`
  - input: `SCRN-main/test_data/noise_and_miss.npy`
  - device: `cpu`
- Eval run:
  - `SCRN_BRECQ_app/scrn_repro/runs/test/20260507_180355_paper5_10750_ddp3_seed20260425_best_eval_gt_colorbar`
- Metrics:
  - `before_snr_db=3.969324203252889`
  - `before_ssim=0.6052755957782698`
  - `after_snr_db=8.286237604245681`
  - `after_ssim=0.7869134500690693`
  - `inference_seconds=0.26703453063964844`

### 结论

- 本次只生成新的 paper-style FP32 SCRN training candidate，没有重跑 BRECQ weight reconstruction 或 activation quantization。
- 相比旧 `10750_0` baseline，训练 loss 明显更低，但旧单样本 eval 的 `after_snr_db` 和 `after_ssim` 更低：
  - SNR: `8.286237604245681` vs `11.78722661219287`
  - SSIM: `0.7869134500690693` vs `0.8699862043155245`
- 因此该 checkpoint 需要后续在 paper-style 478 test set 和任务协议上做多样本评估，不能仅凭旧单样本图判断是否替代历史 FP32 baseline。

## 2026-05-07 FP32 SCRN two-model two-testset 478 multi-eval

### 修改内容

- 新增 `SCRN_BRECQ_app/scrn_repro/cli/evaluate_scrn_multi.py`。
- 新增 `SCRN_BRECQ_app/scrn_repro/tests/test_evaluate_scrn_multi.py` 和测试包初始化文件。
- CLI preset:
  - `fp32-two-model-two-testset-478`
- 固定两个 FP32 checkpoint：
  - `old10750_main`: `SCRN_BRECQ_app/scrn_repro/runs/train/20260425_192916_four_gpu_train_quant_10750_0/checkpoints/best.pth`
  - `paper5`: `SCRN_BRECQ_app/scrn_repro/runs/train/20260507_170001_paper5_10750_ddp3_seed20260425/checkpoints/best.pth`
- 固定两个 478 clean patch test sets：
  - `legacy478`: `SCRN_BRECQ_app/scrn_repro/datasets/scrn_quant_test_478_legacy_logic`
  - `paper5_478`: `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_test_478`
- 固定退化网格：
  - SNR settings: `-2,-1,1,5,10`
  - missing rates: `0.02,0.08,0.18,0.28,0.38`
  - seed: `20260507`
- 每行 per-sample 记录完整 input/output SNR/SSIM：
  - `input_snr_db`
  - `input_ssim`
  - `output_snr_db`
  - `output_ssim`
  - `snr_gain_db`
  - `ssim_gain`
- 汇总包含 overall、by source、by SNR、by missing rate、by condition，以及 `paper5 - old10750_main` paired comparison。

### 正式运行

- Command:
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_repro.cli.evaluate_scrn_multi --preset fp32-two-model-two-testset-478 --device cuda --cuda-device-index 1 --batch-size 64 --seed 20260507`
- Run:
  - `SCRN_BRECQ_app/scrn_repro/runs/test_multi/20260507_184439_fp32_two_model_two_testset_grid478_seed20260507`
- Output:
  - `per_sample_metrics.jsonl`: `47800` rows
  - `metrics.json`
  - `summary.md`
  - `config.json`
- Manifest source mapping:
  - no warnings。

### Overall 结果

| model | testset | count | output SNR mean / median | output SSIM mean / median | SNR gain mean | SSIM gain mean |
|---|---|---:|---:|---:|---:|---:|
| old10750_main | legacy478 | 11950 | 5.6730 / 5.4099 | 0.7527 / 0.7519 | 4.6241 | 0.2214 |
| old10750_main | paper5_478 | 11950 | -6.5491 / 5.1644 | 0.8096 / 0.7965 | -7.5467 | 0.2663 |
| paper5 | legacy478 | 11950 | 4.7196 / 3.8869 | 0.6787 / 0.6592 | 3.6707 | 0.1475 |
| paper5 | paper5_478 | 11950 | -3.0017 / 6.5355 | 0.8821 / 0.8976 | -3.9993 | 0.3388 |

### Paired comparison

- `legacy478`:
  - `paper5 - old10750_main` SNR mean / median: `-0.9534 / -1.2126`
  - `paper5 - old10750_main` SSIM mean / median: `-0.0740 / -0.0802`
- `paper5_478`:
  - `paper5 - old10750_main` SNR mean / median: `+3.5473 / +2.9373`
  - `paper5 - old10750_main` SSIM mean / median: `+0.0725 / +0.0703`

### Source-level notes

- `legacy478`:
  - `paper5` beats `old10750_main` on `Anisotropic` by SNR mean `+1.6871` and SSIM mean `+0.0583`。
  - `paper5` is worse on `Kerry3D` by SNR mean `-4.0864` and SSIM mean `-0.1987`。
  - `paper5` is worse on `Shots0001` by SNR mean `-1.3356` and SSIM mean `-0.0945`。
- `paper5_478`:
  - `paper5` beats `old10750_main` on `Anisotropic` by SNR mean `+3.9038` and SSIM mean `+0.0700`。
  - `paper5` is worse on `Kerry3D` by SNR mean `-2.8750` but SSIM mean is slightly higher by `+0.0121`。
  - `paper5` beats `old10750_main` on `Shots0001` by SNR mean `+3.7437` and SSIM mean `+0.0755`。
- `paper5_478` contains `33` near-zero-energy clean patches, while `legacy478` contains `0`。
  - These near-zero patches can drive output SNR mean to very negative values even when output SSIM is high.
  - For paper5_478, median SNR is therefore more interpretable than mean SNR unless near-zero patches are filtered or reported separately.

### 验证方式

- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_repro.tests.test_evaluate_scrn_multi -v`
  - 6 tests OK。
- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_repro/cli/evaluate_scrn_multi.py`
  - passed。
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_repro.cli.evaluate_scrn_multi --help`
  - passed。

## 手动记录一下第一次的按原论文设定的数据集效果差主要原因

“我刚看了 clean patch 的能量分布：旧 10750_0 训练集里没有 std <= 1e-3 的 patch；但新的 paper5_train_10750 里有 8400/10750 个 std <= 1e-3，其中 5715 个甚至 std <= 1e-6。这不是“少量低能量背景”，而是训练集主体被近空白 patch 占了。

直接原因也找到了：paper_scrn_datasets.py 里虽然定义了 DEFAULT_MIN_STD = 1e-3，但生成训练集和测试集时默认传的是 min_std=None，也就是为了凑齐论文表格 count，实际关闭了低方差过滤。再加上每个 raw patch 会生成 original + 4 个增强，空白 patch 也被放大成 5 份”

因此决定加上过滤再试一次

## 2026-05-07 Paper5 energy-filtered dataset rebuild

目的：

- 保留旧 `scrn_paper5_*` unfiltered 数据集作为 diagnostic artifact，不覆盖、不删除。
- 新建 `paper5_energy_filtered` 数据协议，完全过滤近零 / 无效 clean patch，同时保持 paper-style 5-source 数量口径：
  - train: `10750`
  - calibration: `1024`
  - test: `478`

实现：

- 扩展 `SCRN_BRECQ_app/scrn_brecq/data/paper_scrn_datasets.py`：
  - 新增 `EnergyFilter` / `DEFAULT_ENERGY_FILTER`。
  - hard reject:
    - `std <= 1e-3`
    - `absmax <= 1e-3`
    - non-finite patch
    - all-zero / near-zero patch
  - 新增 energy-filtered train/test builders。
  - train 按 source 连续扫描 shot/window，先过滤，再固定 seed source-wise 精确抽样 raw patch，最后保存 original + 4 个增强。
  - calibration 支持 `original_only=True`，只从 `augmentation_index=0` 的原始 train patch 抽样，避免同一 raw patch 的增强副本重复进入校准集。
  - test 从 energy-filtered train 实际使用区域之后开始扫描，并继续做 train hash exact exclusion。
- 扩展 `SCRN_BRECQ_app/scrn_brecq/cli/prepare_scrn_paper_datasets.py`：
  - 新增 `--protocol paper5-energy-filtered`。
  - 默认输出到新目录，不覆盖旧 `scrn_paper5_*`。

输出数据集：

- Train:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_train_10750`
  - per-source counts: `300 / 3355 / 750 / 480 / 5865`
  - scanned regions:
    - `1997_2.5D_shots=42`
    - `7m_shots_0201=5`
    - `Anisotropic_FD_Model=7`
    - `Kerry3D=5`
    - `Shots0001_0200=15`
  - low-energy rejected:
    - `192 / 2573 / 357 / 384 / 4635`
- Calibration:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_cali_1024_stratified`
  - per-source counts: `28 / 320 / 71 / 46 / 559`
  - all samples copied from original, non-augmented train patches。
- Test:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_test_478`
  - per-source counts: `Anisotropic=75, Kerry3D=16, Shots0001=387`
  - test start boundaries:
    - `Anisotropic`: shot index `7`
    - `Kerry3D`: trace start `1435`
    - `Shots0001`: shot index `15`
  - low-energy rejected:
    - `Anisotropic=204`
    - `Kerry3D=3`
    - `Shots0001=1545`

生成与验证：

- Generation command:
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.prepare_scrn_paper_datasets --protocol paper5-energy-filtered --mode all --seed 20260507 --overwrite`
- Count / energy validation:
  - train: `10750` files, `std <= 1e-3` count `0`, min std `0.0010023288`
  - calibration: `1024` files, `std <= 1e-3` count `0`, min std `0.0010070483`
  - test: `478` files, `std <= 1e-3` count `0`, min std `0.0010794682`
- Unit / CLI checks:
  - `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_paper_scrn_datasets -v`
    - 15 tests OK。
  - `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/data/paper_scrn_datasets.py SCRN_BRECQ_app/scrn_brecq/cli/prepare_scrn_paper_datasets.py`
    - passed。
  - `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.prepare_scrn_paper_datasets --help`
    - passed。

后续建议：

- 旧 `scrn_paper5_*` 结果只作为 unfiltered diagnostic 参考。
- 若继续比较 FP32 或 W4A8，应优先使用 `scrn_paper5_energy_filtered_*` 建立新 benchmark。

## 2026-05-07 Paper5 energy-filtered FP32 SCRN training run

目的：

- 在不改训练代码的前提下，用 `scrn_paper5_energy_filtered_train_10750` 重新训练一个 FP32 SCRN checkpoint。
- 配置对齐首版 `paper5_unfiltered` 三卡 run，而不是旧四卡主 baseline：
  - `world_size=3`
  - per-GPU batch size `32`
  - global batch size `96`
  - `epochs=80`
  - `lr=0.001`
  - `milestones=20,40,60`
  - `gamma=0.2`
  - `seed=20260425`
  - model config: `dim=64, stage_depths=1,1,1,1,1, head_dim=32, window_size=8, input_resolution=128`

训练：

- Command key:
  - `CUDA_VISIBLE_DEVICES=1,2,3 conda run -n quant torchrun --standalone --nproc_per_node=3 -m SCRN_BRECQ_app.scrn_repro.cli.train_scrn ...`
- Dataset:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_train_10750`
- Run path:
  - `SCRN_BRECQ_app/scrn_repro/runs/train/20260507_195614_paper5_energy_filtered_10750_ddp3_seed20260425`
- Git commit recorded by run:
  - `d2ad387cbe7f9cd200ef074bc2de08d02534bfb7`
- Metrics:
  - `best_loss=3.8867502100987448`
  - `best_epoch=63`
  - `last_loss=3.9057623196969784`
  - `last_epoch=80`

旧单样本 eval 口径：

- Checkpoint:
  - `SCRN_BRECQ_app/scrn_repro/runs/train/20260507_195614_paper5_energy_filtered_10750_ddp3_seed20260425/checkpoints/best.pth`
- Eval inputs:
  - clean: `SCRN-main/test_data/clear.npy`
  - input: `SCRN-main/test_data/noise_and_miss.npy`
- Eval run:
  - `SCRN_BRECQ_app/scrn_repro/runs/test/20260507_214045_paper5_energy_filtered_10750_ddp3_seed20260425_best_eval_gt_colorbar`
- Metrics:
  - `before_snr_db=3.969324203252889`
  - `before_ssim=0.6052755957782698`
  - `after_snr_db=10.842029016359723`
  - `after_ssim=0.8614298903567633`

对比口径：

- `old10750_main`:
  - train run: `SCRN_BRECQ_app/scrn_repro/runs/train/20260425_192916_four_gpu_train_quant_10750_0`
  - `world_size=4`, global batch `128`
  - `best_loss=1.3390747353301518`
  - old single eval: `after_snr_db=11.78722661219287`, `after_ssim=0.8699862043155245`
- `paper5_unfiltered`:
  - train run: `SCRN_BRECQ_app/scrn_repro/runs/train/20260507_170001_paper5_10750_ddp3_seed20260425`
  - `world_size=3`, global batch `96`
  - `best_loss=0.028283805948116685`
  - old single eval: `after_snr_db=8.286237604245681`, `after_ssim=0.7869134500690693`
- `paper5_energy_filtered`:
  - train run: `SCRN_BRECQ_app/scrn_repro/runs/train/20260507_195614_paper5_energy_filtered_10750_ddp3_seed20260425`
  - `world_size=3`, global batch `96`
  - `best_loss=3.8867502100987448`
  - old single eval: `after_snr_db=10.842029016359723`, `after_ssim=0.8614298903567633`

结论：

- Energy-filtered FP32 单样本 eval 明显优于首版 unfiltered paper5：
  - SNR: `10.8420` vs `8.2862` dB
  - SSIM: `0.8614` vs `0.7869`
- 仍略低于旧主 baseline：
  - SNR gap: `-0.9452` dB
  - SSIM gap: `-0.0086`
- 训练 loss 不能直接按数值和 unfiltered paper5 比较：unfiltered train set 曾有大量近零 patch，loss 被近空白样本显著压低；energy-filtered 的 loss 更接近有效地震 patch 的优化难度。
- 后续应使用 478 multi-eval，对 `old10750_main`、`paper5_unfiltered`、`paper5_energy_filtered` 在 legacy / paper5 / energy-filtered test sets 上统一比较，单样本 eval 只保留历史连续性。

## 2026-05-07 FP32 three-model eval on energy-filtered 478 test set

目的：

- 单样本 eval 中 `paper5_energy_filtered` 看起来明显优于 `paper5_unfiltered`。
- 本次用新的 `scrn_paper5_energy_filtered_test_478` 做固定退化网格评估，确认该趋势是否能在 478-patch test set 上成立。

评估设置：

- Models:
  - `old10750_main`: `SCRN_BRECQ_app/scrn_repro/runs/train/20260425_192916_four_gpu_train_quant_10750_0/checkpoints/best.pth`
  - `paper5_unfiltered`: `SCRN_BRECQ_app/scrn_repro/runs/train/20260507_170001_paper5_10750_ddp3_seed20260425/checkpoints/best.pth`
  - `paper5_energy_filtered`: `SCRN_BRECQ_app/scrn_repro/runs/train/20260507_195614_paper5_energy_filtered_10750_ddp3_seed20260425/checkpoints/best.pth`
- Test set:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_test_478`
- Conditions:
  - SNR settings: `-2,-1,1,5,10`
  - missing rates: `0.02,0.08,0.18,0.28,0.38`
  - `25` conditions per patch
- Rows:
  - `3 * 478 * 25 = 35850`
- Eval run:
  - `SCRN_BRECQ_app/scrn_repro/runs/test_multi/20260507_215143_fp32_three_model_energy_filtered_test478_seed20260507`

Overall metrics on `paper5_energy_filtered_478`:

| model | rows | output SNR mean | output SNR median | output SSIM mean | output SSIM median | SNR gain mean |
|---|---:|---:|---:|---:|---:|---:|
| `old10750_main` | 11950 | 6.4933 | 6.2893 | 0.8183 | 0.8333 | 5.4355 |
| `paper5_unfiltered` | 11950 | 7.9441 | 7.8208 | 0.8675 | 0.8924 | 6.8863 |
| `paper5_energy_filtered` | 11950 | 6.7422 | 6.6473 | 0.8197 | 0.8441 | 5.6844 |

Pairwise deltas:

| comparison | SNR delta mean | SNR delta median | SSIM delta mean | SSIM delta median |
|---|---:|---:|---:|---:|
| `paper5_unfiltered - old10750_main` | +1.4508 | +1.7236 | +0.0492 | +0.0450 |
| `paper5_energy_filtered - old10750_main` | +0.2489 | +0.5539 | +0.0014 | -0.0038 |
| `paper5_energy_filtered - paper5_unfiltered` | -1.2019 | -1.1296 | -0.0478 | -0.0360 |

By-source notes:

- `paper5_energy_filtered` is strong on `Anisotropic` and `Kerry3D`:
  - `Anisotropic` output SNR mean `9.0333`, SSIM mean `0.9354`
  - `Kerry3D` output SNR mean `8.3129`, SSIM mean `0.8843`
- But it is weak on `Shots0001`, which dominates the test set:
  - `Shots0001` rows: `9675 / 11950`
  - `paper5_energy_filtered` output SNR mean `6.2332`
  - `old10750_main` output SNR mean `6.3752`
  - `paper5_unfiltered` output SNR mean `7.4809`

Interpretation:

- 单样本 eval 的改善没有直接推广到 478-patch fixed-grid eval。
- 在对应的 energy-filtered test set 上，`paper5_energy_filtered` 只小幅超过旧主 baseline，且 SSIM median 仍略低于旧主 baseline。
- `paper5_unfiltered` 在该测试集上整体最好，主要因为它在占比最大的 `Shots0001` source 上优势明显。
- 当前不能把 `paper5_energy_filtered` 直接定为新的 FP32 主 baseline；它更像是修复数据污染后的一个候选，需要进一步看完整 3-model x 3-testset 交叉评估。
- 后续建议：
  - 扩展正式 multi-eval preset，避免继续用 one-off script。
  - 跑 `old10750_main / paper5_unfiltered / paper5_energy_filtered` x `legacy478 / paper5_478 / paper5_energy_filtered_478`。
  - 重点检查 `Shots0001` source 的 train/test 分布和 energy filtering 是否改变了有效样本结构。

## 2026-05-07 FP32 3-model x 3-testset 478 eval plan

目的：

- 单独的 `paper5_energy_filtered_478` eval 已显示单样本结论不可靠。
- 本次补齐完整 FP32 交叉评估，判断三个 FP32 checkpoint 在三个 478 test set 上的稳定性。
- 本轮不改正式 CLI、不重训、不跑 BRECQ，只产出 ignored eval artifacts 和日志。

Eval matrix:

- Models:
  - `old10750_main`: `SCRN_BRECQ_app/scrn_repro/runs/train/20260425_192916_four_gpu_train_quant_10750_0/checkpoints/best.pth`
  - `paper5_unfiltered`: `SCRN_BRECQ_app/scrn_repro/runs/train/20260507_170001_paper5_10750_ddp3_seed20260425/checkpoints/best.pth`
  - `paper5_energy_filtered`: `SCRN_BRECQ_app/scrn_repro/runs/train/20260507_195614_paper5_energy_filtered_10750_ddp3_seed20260425/checkpoints/best.pth`
- Test sets:
  - `legacy478`: `SCRN_BRECQ_app/scrn_repro/datasets/scrn_quant_test_478_legacy_logic`
  - `paper5_478`: `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_test_478`
  - `paper5_energy_filtered_478`: `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_test_478`
- Conditions:
  - SNR settings: `-2,-1,1,5,10`
  - missing rates: `0.02,0.08,0.18,0.28,0.38`
  - seed: `20260507`
  - batch size: `64`
  - CUDA device index: `1`
- Expected rows:
  - `3 * 3 * 478 * 25 = 107550`

Planned output:

- `SCRN_BRECQ_app/scrn_repro/runs/test_multi/<timestamp>_fp32_three_model_three_testset_grid478_seed20260507`
- Files:
  - `config.json`
  - `per_sample_metrics.jsonl`
  - `metrics.json`
  - `summary.md`

Decision criteria:

- Report overall mean / median SNR and SSIM for each model-testset pair.
- Report by-source metrics, with special attention to `Shots0001`.
- Report pairwise deltas:
  - `paper5_unfiltered - old10750_main`
  - `paper5_energy_filtered - old10750_main`
  - `paper5_energy_filtered - paper5_unfiltered`
- Decide whether `paper5_energy_filtered` is still a plausible FP32 candidate before BRECQ, or whether the next step should be dataset energy diagnostics and an energy-balanced training protocol.

## 2026-05-07 FP32 3-model x 3-testset 478 eval results

Run:

- `SCRN_BRECQ_app/scrn_repro/runs/test_multi/20260507_224936_fp32_three_model_three_testset_grid478_seed20260507`
- Rows:
  - `107550`
- Conditions:
  - `478` clean patches x `25` degradation conditions per model-testset pair

Overall metrics:

| model | testset | output SNR mean | output SNR median | output SSIM mean | output SSIM median |
|---|---|---:|---:|---:|---:|
| `old10750_main` | `legacy478` | 5.6730 | 5.4099 | 0.7527 | 0.7519 |
| `paper5_unfiltered` | `legacy478` | 4.7196 | 3.8869 | 0.6787 | 0.6592 |
| `paper5_energy_filtered` | `legacy478` | 4.1157 | 3.3877 | 0.6377 | 0.6340 |
| `old10750_main` | `paper5_478` | -6.5491 | 5.1644 | 0.8096 | 0.7965 |
| `paper5_unfiltered` | `paper5_478` | -3.0017 | 6.5355 | 0.8821 | 0.8976 |
| `paper5_energy_filtered` | `paper5_478` | -8.3037 | 5.4958 | 0.7783 | 0.8002 |
| `old10750_main` | `paper5_energy_filtered_478` | 6.4933 | 6.2893 | 0.8183 | 0.8333 |
| `paper5_unfiltered` | `paper5_energy_filtered_478` | 7.9441 | 7.8208 | 0.8675 | 0.8924 |
| `paper5_energy_filtered` | `paper5_energy_filtered_478` | 6.7422 | 6.6473 | 0.8197 | 0.8441 |

Pairwise deltas:

| comparison | testset | SNR delta mean | SNR delta median | SSIM delta mean | SSIM delta median |
|---|---|---:|---:|---:|---:|
| `paper5_unfiltered - old10750_main` | `legacy478` | -0.9534 | -1.2126 | -0.0740 | -0.0802 |
| `paper5_unfiltered - old10750_main` | `paper5_478` | +3.5473 | +2.9373 | +0.0725 | +0.0703 |
| `paper5_unfiltered - old10750_main` | `paper5_energy_filtered_478` | +1.4508 | +1.7236 | +0.0492 | +0.0450 |
| `paper5_energy_filtered - old10750_main` | `legacy478` | -1.5573 | -1.7892 | -0.1150 | -0.1347 |
| `paper5_energy_filtered - old10750_main` | `paper5_478` | -1.7547 | +0.0548 | -0.0312 | -0.0183 |
| `paper5_energy_filtered - old10750_main` | `paper5_energy_filtered_478` | +0.2489 | +0.5539 | +0.0014 | -0.0038 |
| `paper5_energy_filtered - paper5_unfiltered` | `legacy478` | -0.6039 | -0.6489 | -0.0410 | -0.0507 |
| `paper5_energy_filtered - paper5_unfiltered` | `paper5_478` | -5.3020 | -1.6413 | -0.1037 | -0.0626 |
| `paper5_energy_filtered - paper5_unfiltered` | `paper5_energy_filtered_478` | -1.2019 | -1.1296 | -0.0478 | -0.0360 |

Shots0001 notes:

- `Shots0001` dominates all three 478 test sets: `9675 / 11950` rows per model-testset pair.
- On `legacy478`, `old10750_main` remains best on Shots0001:
  - `old10750_main`: SNR mean / median `5.3624 / 5.1471`
  - `paper5_unfiltered`: `4.0268 / 3.5733`
  - `paper5_energy_filtered`: `3.4266 / 3.0502`
- On `paper5_478`, `paper5_unfiltered` is best on Shots0001 by median and SSIM:
  - `old10750_main`: SNR median `4.5728`, SSIM median `0.7927`
  - `paper5_unfiltered`: SNR median `5.8244`, SSIM median `0.8820`
  - `paper5_energy_filtered`: SNR median `4.5561`, SSIM median `0.7582`
- On `paper5_energy_filtered_478`, `paper5_unfiltered` is again best on Shots0001:
  - `old10750_main`: SNR mean / median `6.3752 / 6.1474`
  - `paper5_unfiltered`: `7.4809 / 7.4101`
  - `paper5_energy_filtered`: `6.2332 / 6.1405`

Conclusion:

- `paper5_energy_filtered` is not a strong FP32 main baseline candidate for BRECQ yet.
- It is never the best model in the 3-testset overall comparison.
- `old10750_main` is still best on `legacy478`, while `paper5_unfiltered` is best on both paper-style test sets.
- The very negative SNR means on `paper5_478` reinforce that mean SNR is unstable when near-zero clean patches exist; median SNR and SSIM are more interpretable there.
- Next step should be dataset energy diagnostics and an energy-balanced training protocol, not immediate W4A8 on `paper5_energy_filtered`.

## 2026-05-07 Train dataset energy diagnostics

目的：

- 解释为什么 source count 配额保持一致时，FP32 结果仍会显著变化。
- 对比三个 10750 train sets 的 patch energy 分布：
  - `legacy10750_0`: `SCRN_BRECQ_app/scrn_repro/datasets/scrn_quant_10750_0_patches`
  - `paper5_unfiltered`: `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_train_10750`
  - `paper5_energy_filtered`: `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_train_10750`
- 使用 `std² sum` 作为普通 `MSELoss(sum)` 下的粗略训练影响 proxy。
  - 这不是精确梯度归因，但能反映高幅值 patch 对 loss 的潜在支配能力。

Report:

- `SCRN_BRECQ_app/scrn_repro/runs/dataset_diagnostics/20260507_232819_train_energy_diagnostics_10750`
- Files:
  - `summary.md`
  - `energy_diagnostics.json`

Overall:

| dataset | count | std <= 1e-6 | std <= 1e-3 | std median | std q90 | dominant std² source | dominant std² share |
|---|---:|---:|---:|---:|---:|---|---:|
| `legacy10750_0` | 10750 | 0 | 0 | 0.002043 | 0.007039 | `Kerry3D` | 0.9493 |
| `paper5_unfiltered` | 10750 | 5715 | 8400 | 0.000000 | 0.004135 | `Anisotropic_FD_Model` | 0.4976 |
| `paper5_energy_filtered` | 10750 | 0 | 0 | 0.004090 | 0.014933 | `Kerry3D` | 0.9510 |

Per-source observations:

- `legacy10750_0`:
  - `Shots0001_0200` count share is `0.5456`, but std² share is only `0.0091`.
  - `Kerry3D` count share is `0.0447`, but std² share is `0.9493`.
- `paper5_unfiltered`:
  - It contains many near-zero patches:
    - `std <= 1e-6`: `5715 / 10750`
    - `std <= 1e-3`: `8400 / 10750`
  - `Kerry3D` is entirely near-zero in this split:
    - `480 / 480` have `std <= 1e-3`
    - std² share is effectively `0.0`
  - Training influence is spread mainly across:
    - `Anisotropic_FD_Model`: std² share `0.4976`
    - `Shots0001_0200`: `0.1850`
    - `7m_shots_0201`: `0.1753`
    - `1997_2.5D_shots`: `0.1420`
- `paper5_energy_filtered`:
  - It removes all `std <= 1e-3` patches.
  - `Kerry3D` becomes extremely high energy:
    - median std `0.150184`
    - std² share `0.9510`
  - `Shots0001_0200` still has count share `0.5456`, but std² share only `0.0108`.

Interpretation:

- Matching source patch counts is not enough under `MSELoss(sum)`.
- The effective training objective can be dominated by a small high-energy source, even when that source has few patches.
- However, `legacy10750_0` also has strong `Kerry3D` std² dominance and still performs best on `legacy478`, so std² dominance alone does not fully explain model ranking.
- The current evidence points to a combination of:
  - source/test protocol matching,
  - selected spatial regions,
  - near-zero patch distribution,
  - and energy-scale imbalance.

Next recommendation:

- Do not directly use `paper5_energy_filtered` for BRECQ.
- First design an energy-balanced paper-style train set:
  - keep hard low-energy rejection,
  - keep source count quotas,
  - add per-source energy-bin selection or per-source std² share constraints,
  - verify that no single source dominates the MSE-scale proxy before retraining.

## 2026-05-08 Per-patch absmax normalized paper5 datasets

目的：

- 新增一组实验性数据协议，用来测试“每个 clean patch 自己归一化”是否能缓解不同 source / patch 之间的幅值尺度差异。
- 不覆盖已有数据集：
  - `scrn_paper5_*`
  - `scrn_paper5_energy_filtered_*`
- 本轮只生成 clean train / calibration / test 数据集，不重训 FP32，不进入 BRECQ。

实现：

- 新增：
  - `SCRN_BRECQ_app/scrn_brecq/data/per_patch_normalization.py`
  - `SCRN_BRECQ_app/scrn_brecq/cli/prepare_per_patch_normalized_datasets.py`
  - `SCRN_BRECQ_app/scrn_brecq/tests/test_per_patch_normalization.py`
- 归一化公式：
  - `scale = max(abs(patch))`
  - `patch_norm = patch / scale if scale > 1e-12 else patch`
  - `patch_restored = patch_norm * normalization_scale`
- 每个 sample manifest 记录：
  - `input_dataset_dir`
  - `input_file`
  - `input_sha256`
  - `output_sha256`
  - `normalization_method = per_patch_absmax`
  - `normalization_scale`
  - `normalization_eps = 1e-12`
  - `zero_or_tiny_scale`
  - `restoration_formula`
- calibration 不重新抽样：
  - 读取原 cali manifest 的 `train_file`
  - 从对应 normalized train 目录复制同一个 normalized train patch
  - 保持 cali 数量和 source 配额不变

输出目录：

- `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_perpatch_absmax_train_10750`
- `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_perpatch_absmax_cali_1024_stratified`
- `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_perpatch_absmax_test_478`
- `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_train_10750`
- `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`

生成统计：

| dataset | count | tiny scale count | scale median | scale mean |
|---|---:|---:|---:|---:|
| `paper5_perpatch_absmax_train_10750` | 10750 | 5400 | `2.7119e-15` | `0.017224` |
| `paper5_perpatch_absmax_cali_1024` | 1024 | 504 | `7.0329e-09` | `0.018736` |
| `paper5_perpatch_absmax_test_478` | 478 | 18 | `0.038314` | `0.069184` |
| `paper5_energy_filtered_perpatch_absmax_train_10750` | 10750 | 0 | `0.040546` | `0.101754` |
| `paper5_energy_filtered_perpatch_absmax_cali_1024` | 1024 | 0 | `0.040998` | `0.105510` |
| `paper5_energy_filtered_perpatch_absmax_test_478` | 478 | 0 | `0.047740` | `0.085327` |

验证：

- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_per_patch_normalization -v`
- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/data/per_patch_normalization.py SCRN_BRECQ_app/scrn_brecq/cli/prepare_per_patch_normalized_datasets.py`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.prepare_per_patch_normalized_datasets --help`
- 生成后独立检查：
  - 六个输出目录 `.npy` 数量分别为 `10750 / 1024 / 478 / 10750 / 1024 / 478`
  - 所有非 tiny patch 的 `max(abs(patch_norm)) ~= 1`
  - 随机抽样反归一化最大误差不超过 `2.9803e-08`

解释：

- `paper5_unfiltered` 派生集保留了 near-zero patch 被放大的风险，因此 manifest / README 中显式记录 `normalization_scale` 和 `zero_or_tiny_scale`。
- `paper5_energy_filtered` 派生集没有 tiny-scale patch，更适合判断 per-patch absmax normalization 本身是否有帮助。
- 该协议是实验性派生数据协议，不替代原始 `paper5` / `paper5_energy_filtered` 数据集；后续必须先做 FP32 训练和 478 fixed-grid eval，再决定是否进入 W4A8。

## 2026-05-08 Paper5 per-patch absmax FP32 training

本次只训练 FP32 SCRN，不进入 BRECQ / W4A8。

Dataset:

- `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_perpatch_absmax_train_10750`
- count: `10750`
- tiny scale patch count: `5400`
- source counts:
  - `1997_2.5D_shots`: `300`
  - `7m_shots_0201`: `3355`
  - `Anisotropic_FD_Model`: `750`
  - `Kerry3D`: `480`
  - `Shots0001_0200`: `5865`

Run:

- `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_150851_paper5_perpatch_absmax_10750_ddp3_seed20260425`
- command used 3-GPU DDP with physical GPUs `1,2,3`:
  - `CUDA_VISIBLE_DEVICES=1,2,3`
  - `torchrun --standalone --nproc_per_node=3`
- note:
  - plain sandbox `nvidia-smi` could not access the driver
  - non-sandbox GPU check saw CUDA correctly
  - training was launched with GPU-accessible execution
- config:
  - `epochs=80`
  - `batch_size=32` per GPU
  - global batch `96`
  - `lr=0.001`
  - `milestones=20,40,60`
  - `gamma=0.2`
  - `seed=20260425`
  - `num_workers=2`
  - model config `dim=64, stage_depths=1,1,1,1,1, head_dim=32, window_size=8, input_resolution=128`
- git commit recorded in run config:
  - `247c1ee9522f515ef94335d74012fa5f3236a1a0`

Training metrics:

- best epoch: `75`
- best loss: `11.12054773739406`
- last epoch: `80`
- last loss: `11.920819751563526`
- checkpoints:
  - `best.pth`
  - `latest.pth`
  - `epoch_010.pth` through `epoch_080.pth` every 10 epochs

Single-sample historical eval:

- run:
  - `SCRN_BRECQ_app/scrn_repro/runs/test/20260508_164151_paper5_perpatch_absmax_10750_ddp3_seed20260425_best_eval_gt_colorbar`
- checkpoint:
  - train run `best.pth`
- input pair:
  - `SCRN-main/test_data/clear.npy`
  - `SCRN-main/test_data/noise_and_miss.npy`
- metrics:
  - before SNR: `3.9693 dB`
  - after SNR: `13.3167 dB`
  - before SSIM: `0.6053`
  - after SSIM: `0.9130`

Interpretation:

- Single-sample result is strong and exceeds the previous single-sample paper5 / energy-filtered runs.
- This does not decide the FP32 baseline yet because this test is only one historical sample.
- Next required comparison is still the 478 fixed-grid eval after the second normalized model, `paper5_energy_filtered_perpatch_absmax`, is trained.

## 2026-05-08 Paper5 energy-filtered per-patch absmax FP32 training

本次只训练第二个 normalized FP32 SCRN，不进入 BRECQ / W4A8。

Dataset:

- `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_train_10750`
- count: `10750`
- tiny scale patch count: `0`
- source counts:
  - `1997_2.5D_shots`: `300`
  - `7m_shots_0201`: `3355`
  - `Anisotropic_FD_Model`: `750`
  - `Kerry3D`: `480`
  - `Shots0001_0200`: `5865`

Run:

- `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_164907_paper5_energy_filtered_perpatch_absmax_10750_ddp3_seed20260425`
- command used 3-GPU DDP with physical GPUs `1,2,3`:
  - `CUDA_VISIBLE_DEVICES=1,2,3`
  - `torchrun --standalone --nproc_per_node=3`
- config:
  - `epochs=80`
  - `batch_size=32` per GPU
  - global batch `96`
  - `lr=0.001`
  - `milestones=20,40,60`
  - `gamma=0.2`
  - `seed=20260425`
  - `num_workers=2`
  - model config `dim=64, stage_depths=1,1,1,1,1, head_dim=32, window_size=8, input_resolution=128`
- git commit recorded in run config:
  - `c96ebc81444c9066f4035fd35e258c1fa940513d`

Training metrics:

- best epoch: `63`
- best loss: `22.153653881379537`
- last epoch: `80`
- last loss: `22.433070646865026`
- checkpoints:
  - `best.pth`
  - `latest.pth`
  - `epoch_010.pth` through `epoch_080.pth` every 10 epochs

Single-sample historical eval:

- run:
  - `SCRN_BRECQ_app/scrn_repro/runs/test/20260508_183043_paper5_energy_filtered_perpatch_absmax_10750_ddp3_seed20260425_best_eval_gt_colorbar`
- checkpoint:
  - train run `best.pth`
- input pair:
  - `SCRN-main/test_data/clear.npy`
  - `SCRN-main/test_data/noise_and_miss.npy`
- metrics:
  - before SNR: `3.9693 dB`
  - after SNR: `13.5520 dB`
  - before SSIM: `0.6053`
  - after SSIM: `0.9273`

Interpretation:

- Single-sample result is slightly stronger than `paper5_perpatch_absmax`:
  - SNR: `13.5520 dB` vs `13.3167 dB`
  - SSIM: `0.9273` vs `0.9130`
- Training loss is higher than `paper5_perpatch_absmax`:
  - best loss `22.1537` vs `11.1205`
  - this is expected to be harder to compare directly because the energy-filtered normalized dataset removes near-zero patches.
- Both normalized FP32 checkpoints are now available; next comparison should be a 478 fixed-grid eval across normalized and existing test sets.

## 2026-05-08 FP32 5-model x 5-testset fixed-grid 478 eval

本次只评估 FP32 SCRN，不运行 BRECQ / W4A8。

Run:

- `SCRN_BRECQ_app/scrn_repro/runs/test_multi/20260508_184559_fp32_five_model_five_testset_grid478_seed20260507`
- models:
  - `old10750_main`
  - `paper5_unfiltered`
  - `paper5_energy_filtered`
  - `paper5_perpatch_absmax`
  - `paper5_energy_filtered_perpatch_absmax`
- testsets:
  - `legacy478`
  - `paper5_478`
  - `paper5_energy_filtered_478`
  - `paper5_perpatch_absmax_478`
  - `paper5_energy_filtered_perpatch_absmax_478`
- degradation grid:
  - SNR settings: `-2,-1,1,5,10`
  - missing rates: `0.02,0.08,0.18,0.28,0.38`
  - seed: `20260507`
  - batch size: `64`
  - CUDA device index: `1`

Validation:

- expected rows: `298750`
- actual rows: `298750`
- condition count: `25`
- model/testset buckets: `25`
- each model/testset bucket rows: `11950`
- condition buckets: `625`
- each condition bucket rows: `478`
- manifest warnings: none

Native-pair overall results:

| model | matching testset | SNR mean | SNR median | SSIM mean | SSIM median | SNR gain mean |
|---|---|---:|---:|---:|---:|---:|
| `old10750_main` | `legacy478` | 5.6730 | 5.4099 | 0.7527 | 0.7519 | 4.6241 |
| `paper5_unfiltered` | `paper5_478` | -3.0017 | 6.5355 | 0.8821 | 0.8976 | -3.9993 |
| `paper5_energy_filtered` | `paper5_energy_filtered_478` | 6.7422 | 6.6473 | 0.8197 | 0.8441 | 5.6844 |
| `paper5_perpatch_absmax` | `paper5_perpatch_absmax_478` | 11.1449 | 16.1541 | 0.9345 | 0.9833 | 10.1063 |
| `paper5_energy_filtered_perpatch_absmax` | `paper5_energy_filtered_perpatch_absmax_478` | 16.7960 | 17.1248 | 0.9615 | 0.9792 | 15.7317 |

Cross-space result pattern:

- On raw-amplitude testsets (`legacy478`, `paper5_478`, `paper5_energy_filtered_478`), the normalized checkpoints do not transfer well:
  - `paper5_perpatch_absmax` raw-testset average SNR: `-2.7625 dB`
  - `paper5_energy_filtered_perpatch_absmax` raw-testset average SNR: `-2.9064 dB`
- On normalized testsets, normalized checkpoints dominate:
  - `paper5_perpatch_absmax` normalized-testset average SNR / SSIM: `13.0527 dB / 0.9407`
  - `paper5_energy_filtered_perpatch_absmax` normalized-testset average SNR / SSIM: `14.2395 dB / 0.9503`
- Therefore per-patch absmax normalization changes the amplitude space. It should be treated as a separate evaluation/training protocol, not as a checkpoint that can be directly compared on raw-amplitude testsets.

Important pairwise deltas:

| comparison | testset | SNR delta mean | SSIM delta mean |
|---|---|---:|---:|
| `paper5_energy_filtered_perpatch_absmax - paper5_perpatch_absmax` | `paper5_perpatch_absmax_478` | 0.5381 | 0.0046 |
| `paper5_energy_filtered_perpatch_absmax - paper5_perpatch_absmax` | `paper5_energy_filtered_perpatch_absmax_478` | 1.8354 | 0.0147 |
| `paper5_energy_filtered_perpatch_absmax - paper5_energy_filtered` | `paper5_energy_filtered_perpatch_absmax_478` | 8.5603 | 0.1077 |
| `paper5_perpatch_absmax - paper5_unfiltered` | `paper5_perpatch_absmax_478` | 7.5414 | 0.1090 |

Native-pair by-source results:

| source | `old10750_main` on `legacy478` | `paper5_energy_filtered` on `paper5_energy_filtered_478` | `paper5_perpatch_absmax` on `paper5_perpatch_absmax_478` | `paper5_energy_filtered_perpatch_absmax` on matching normalized test |
|---|---:|---:|---:|---:|
| Anisotropic SNR / SSIM | 6.8024 / 0.8215 | 9.0333 / 0.9354 | 17.4844 / 0.9654 | 20.8429 / 0.9955 |
| Kerry3D SNR / SSIM | 7.8915 / 0.7636 | 8.3129 / 0.8843 | 7.4658 / 0.9515 | 8.6027 / 0.9550 |
| Shots0001 SNR / SSIM | 5.3624 / 0.7389 | 6.2332 / 0.7946 | 10.0684 / 0.9279 | 16.3505 / 0.9551 |

Interpretation:

- `paper5_energy_filtered_perpatch_absmax` is the strongest normalized-space FP32 checkpoint.
- It is also the cleanest normalized candidate because its source dataset has `0` tiny-scale patches.
- For raw-amplitude evaluation, keep `old10750_main`, `paper5_unfiltered`, and `paper5_energy_filtered` as the comparable group.
- For normalized evaluation and possible normalized W4A8 experiments, the next candidate should be `paper5_energy_filtered_perpatch_absmax` with matching normalized calibration/test data.
- A raw checkpoint and a normalized checkpoint should not be ranked by a single mixed 5-testset average because they solve different amplitude-space protocols.

## 2026-05-08 Paper5 energy-filtered per-patch absmax no-decay FP32 training

本次只做 FP32 learning-rate ablation，不运行 BRECQ / W4A8。

Goal:

- Reproduce the latest normalized FP32 setup.
- Keep all settings fixed except learning-rate decay.
- Test whether the previous `20/40/60` decay made the later epochs under-train.

Dataset:

- `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_train_10750`
- count: `10750`
- tiny scale patch count: `0`

Run:

- `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_194718_paper5_energy_filtered_perpatch_absmax_10750_ddp3_seed20260425_nodecay_lr1e-3`
- changed variable:
  - old: `--milestones 20,40,60 --gamma 0.2`
  - new: `--milestones "" --gamma 0.2`
- effective LR:
  - `0.001` for all 80 epochs
- unchanged config:
  - 3-GPU DDP on physical GPUs `1,2,3`
  - global batch `96`
  - `epochs=80`
  - `batch_size=32` per GPU
  - `seed=20260425`
  - model config `dim=64, stage_depths=1,1,1,1,1, head_dim=32, window_size=8, input_resolution=128`
- run config git commit:
  - `a95fd213be8bf6feff417e41f68c42b6b1e897ec`

Training metrics:

| run | best epoch | best loss | last loss |
|---|---:|---:|---:|
| previous decayed LR | 63 | 22.153653881379537 | 22.433070646865026 |
| no decay LR | 79 | 18.80715600649516 | 19.151402472030547 |

Single-sample historical eval:

- run:
  - `SCRN_BRECQ_app/scrn_repro/runs/test/20260508_212331_paper5_energy_filtered_perpatch_absmax_10750_ddp3_seed20260425_nodecay_lr1e-3_best_eval_gt_colorbar`
- metrics:
  - before SNR / SSIM: `3.9693 dB / 0.6053`
  - after SNR / SSIM: `13.8807 dB / 0.9324`
- previous decayed checkpoint:
  - after SNR / SSIM: `13.5520 dB / 0.9273`

Matching normalized 478 eval:

- run:
  - `SCRN_BRECQ_app/scrn_repro/runs/test_multi/20260508_212429_fp32_nodecay_matching_normalized_grid478_seed20260507`
- testset:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`
- degradation grid:
  - SNR settings `-2,-1,1,5,10`
  - missing rates `0.02,0.08,0.18,0.28,0.38`
  - seed `20260507`
- validation:
  - rows: `11950`
  - condition count: `25`
  - manifest warnings: none

Overall 478 comparison:

| checkpoint | SNR mean | SNR median | SSIM mean | SSIM median | SNR gain mean |
|---|---:|---:|---:|---:|---:|
| previous decayed LR | 16.7960 | 17.1248 | 0.9615 | 0.9792 | 15.7317 |
| no decay LR | 17.8346 | 18.1752 | 0.9644 | 0.9788 | 16.7703 |

By-source no-decay 478 result:

| source | count | SNR mean | SSIM mean |
|---|---:|---:|---:|
| Anisotropic | 1875 | 22.0641 | 0.9928 |
| Kerry3D | 400 | 9.6066 | 0.9506 |
| Shots0001 | 9675 | 17.3551 | 0.9594 |

Interpretation:

- No-decay LR is clearly better for this normalized energy-filtered FP32 setup.
- Training loss improves substantially without late divergence.
- Single-sample and matching 478 eval also improve, so this is not just overfitting to the training loss.
- The no-decay checkpoint should replace the decayed checkpoint as the preferred normalized FP32 candidate.
- For normalized W4A8, use this no-decay FP32 checkpoint with the matching normalized calibration/test data.

## 2026-05-08 Paper5 energy-filtered per-patch absmax LR 0.005 four-GPU FP32 training

本次只做更激进的 FP32 learning-rate ablation，不运行 BRECQ / W4A8。

Goal:

- Test whether the normalized energy-filtered per-patch absmax setup can benefit from a much larger no-decay LR.
- This run is not a clean single-variable comparison against `lr=0.001` because it also changes world size/global batch.

Dataset:

- `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_train_10750`
- count: `10750`
- tiny scale patch count: `0`

Run:

- `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_213810_paper5_energy_filtered_perpatch_absmax_10750_ddp4_seed20260425_nodecay_lr5e-3`
- changed variables versus the previous preferred normalized FP32 run:
  - LR: `0.001 -> 0.005`
  - world size: `3 -> 4`
  - global batch: `96 -> 128`
- effective LR:
  - `0.005` for all 80 epochs
- run config git commit:
  - `ef43d7b81b27fe3fb8e893690741c1de508edc95`

Training metrics:

| run | best epoch | best loss | last loss |
|---|---:|---:|---:|
| `lr=0.001`, DDP3, no decay | 79 | 18.80715600649516 | 19.151402472030547 |
| `lr=0.005`, DDP4, no decay | 72 | 1674639.4090401786 | 780769248405156864 |

Training behavior:

- No NaN or inf was observed.
- The loss exploded immediately:
  - epoch 1: `443209635604.276245`
  - epoch 2: `3607585441617.143066`
- The run later decreased to million-scale loss, but never recovered to a usable range.
- Large late-epoch instability remained:
  - epoch 79: `2210775479834130432`
  - epoch 80: `780769248405156864`

Single-sample historical eval:

- run:
  - `SCRN_BRECQ_app/scrn_repro/runs/test/20260508_222944_paper5_energy_filtered_perpatch_absmax_10750_ddp4_seed20260425_nodecay_lr5e-3_best_eval_gt_colorbar`
- metrics:
  - before SNR / SSIM: `3.9693 dB / 0.6053`
  - after SNR / SSIM: `-35.8768 dB / 0.0791`
- previous preferred `lr=0.001` checkpoint:
  - after SNR / SSIM: `13.8807 dB / 0.9324`

Matching normalized 478 eval:

- run:
  - `SCRN_BRECQ_app/scrn_repro/runs/test_multi/20260508_223048_fp32_lr5e-3_matching_normalized_grid478_seed20260507`
- testset:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`
- validation:
  - rows: `11950`
  - condition count: `25`
  - manifest warnings: none

Overall 478 comparison:

| checkpoint | SNR mean | SNR median | SSIM mean | SSIM median | SNR gain mean |
|---|---:|---:|---:|---:|---:|
| `lr=0.001`, DDP3, no decay | 17.8346 | 18.1752 | 0.9644 | 0.9788 | 16.7703 |
| `lr=0.005`, DDP4, no decay | -38.8719 | -38.1403 | 0.1705 | 0.1384 | -39.9362 |

By-source `lr=0.005` 478 result:

| source | count | SNR mean | SSIM mean |
|---|---:|---:|---:|
| Anisotropic | 1875 | -40.6888 | 0.1006 |
| Kerry3D | 400 | -40.6976 | 0.1685 |
| Shots0001 | 9675 | -38.4443 | 0.1842 |

Interpretation:

- `lr=0.005` is too aggressive for this setup.
- The best checkpoint is unusable despite finite losses.
- This run must not replace the `lr=0.001`, DDP3, no-decay checkpoint as the normalized FP32 candidate.
- The preferred normalized W4A8 starting point remains:
  - `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_194718_paper5_energy_filtered_perpatch_absmax_10750_ddp3_seed20260425_nodecay_lr1e-3/checkpoints/best.pth`
- If testing larger LR further, try a smaller step than `0.005`; this run shows that jumping directly to `0.005` destabilizes optimization badly.

## 2026-05-09 Paper5 energy-filtered per-patch absmax LR 0.002 four-GPU FP32 training

本次只做 FP32 learning-rate ablation，不运行 BRECQ / W4A8。

Goal:

- Test whether a moderate no-decay LR above `0.001` can improve the normalized energy-filtered per-patch absmax FP32 candidate.
- This run is not a clean single-variable comparison against `lr=0.001` because it also changes world size/global batch.

Dataset:

- `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_train_10750`
- count: `10750`
- tiny scale patch count: `0`

Run:

- `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_231616_paper5_energy_filtered_perpatch_absmax_10750_ddp4_seed20260425_nodecay_lr2e-3`
- changed variables versus the preferred normalized FP32 run:
  - LR: `0.001 -> 0.002`
  - world size: `3 -> 4`
  - global batch: `96 -> 128`
- effective LR:
  - `0.002` for all 80 epochs
- run config git commit:
  - `415702b4d87d196c6b4eb6ba3f18d528a94d4d64`

Training metrics:

| run | best epoch | best loss | last loss |
|---|---:|---:|---:|
| `lr=0.001`, DDP3, no decay | 79 | 18.80715600649516 | 19.151402472030547 |
| `lr=0.002`, DDP4, no decay | 80 | 24.6274932878358 | 24.6274932878358 |
| `lr=0.005`, DDP4, no decay | 72 | 1674639.4090401786 | 780769248405156864 |

Training behavior:

- No NaN or inf was observed.
- Unlike `lr=0.005`, this run was numerically stable.
- However, it converged more slowly and to a worse loss than `lr=0.001`.
- The best checkpoint was the final epoch, so the run might still be slowly descending, but it is far behind the `lr=0.001` baseline at the same 80-epoch budget.

Single-sample historical eval:

- run:
  - `SCRN_BRECQ_app/scrn_repro/runs/test/20260509_000403_paper5_energy_filtered_perpatch_absmax_10750_ddp4_seed20260425_nodecay_lr2e-3_best_eval_gt_colorbar`
- metrics:
  - before SNR / SSIM: `3.9693 dB / 0.6053`
  - after SNR / SSIM: `13.5487 dB / 0.9226`
- previous preferred `lr=0.001` checkpoint:
  - after SNR / SSIM: `13.8807 dB / 0.9324`

Matching normalized 478 eval:

- run:
  - `SCRN_BRECQ_app/scrn_repro/runs/test_multi/20260509_000456_fp32_lr2e-3_matching_normalized_grid478_seed20260507`
- testset:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`
- validation:
  - rows: `11950`
  - condition count: `25`
  - manifest warnings: none

Overall 478 comparison:

| checkpoint | SNR mean | SNR median | SSIM mean | SSIM median | SNR gain mean |
|---|---:|---:|---:|---:|---:|
| `lr=0.001`, DDP3, no decay | 17.8346 | 18.1752 | 0.9644 | 0.9788 | 16.7703 |
| `lr=0.002`, DDP4, no decay | 16.0660 | 16.3905 | 0.9546 | 0.9738 | 15.0017 |
| `lr=0.005`, DDP4, no decay | -38.8719 | -38.1403 | 0.1705 | 0.1384 | -39.9362 |

By-source `lr=0.002` 478 result:

| source | count | SNR mean | SSIM mean |
|---|---:|---:|---:|
| Anisotropic | 1875 | 19.6571 | 0.9922 |
| Kerry3D | 400 | 8.9998 | 0.9550 |
| Shots0001 | 9675 | 15.6622 | 0.9473 |

Interpretation:

- `lr=0.002` is stable but worse than `lr=0.001` for the 80-epoch four-GPU run.
- It should not replace the `lr=0.001`, DDP3, no-decay checkpoint as the normalized FP32 candidate.
- The preferred normalized W4A8 starting point remains:
  - `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_194718_paper5_energy_filtered_perpatch_absmax_10750_ddp3_seed20260425_nodecay_lr1e-3/checkpoints/best.pth`
- If further LR tuning is needed, the next informative experiment should isolate global batch from LR, because both `lr=0.002` and `lr=0.005` used DDP4/global batch `128` while the preferred baseline used DDP3/global batch `96`.

## 2026-05-09 Paper5 energy-filtered per-patch absmax LR 0.0015 four-GPU FP32 training

本次只做 FP32 learning-rate ablation，不运行 BRECQ / W4A8。

Goal:

- Test an intermediate no-decay LR between the successful `0.001` run and the weaker `0.002` run.
- This is still not a clean single-variable comparison against `lr=0.001`, because it also uses DDP4/global batch `128` instead of DDP3/global batch `96`.

Dataset:

- `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_train_10750`
- count: `10750`
- tiny scale patch count: `0`

Run:

- `SCRN_BRECQ_app/scrn_repro/runs/train/20260509_001355_paper5_energy_filtered_perpatch_absmax_10750_ddp4_seed20260425_nodecay_lr1p5e-3`
- changed variables versus the preferred normalized FP32 run:
  - LR: `0.001 -> 0.0015`
  - world size: `3 -> 4`
  - global batch: `96 -> 128`
- effective LR:
  - `0.0015` for all 80 epochs
- run config git commit:
  - `8edd1f57e36bd55a7853cc5c9cf9736f1b9b9b02`

Training metrics:

| run | best epoch | best loss | last loss |
|---|---:|---:|---:|
| `lr=0.001`, DDP3, no decay | 79 | 18.80715600649516 | 19.151402472030547 |
| `lr=0.0015`, DDP4, no decay | 79 | 20.458939271313803 | 20.465225462402618 |
| `lr=0.002`, DDP4, no decay | 80 | 24.6274932878358 | 24.6274932878358 |
| `lr=0.005`, DDP4, no decay | 72 | 1674639.4090401786 | 780769248405156864 |

Training behavior:

- No NaN or inf was observed.
- `lr=0.0015` is clearly more stable and better optimized than `lr=0.002` and `lr=0.005`.
- It still does not match the `lr=0.001`, DDP3, no-decay loss after the same 80-epoch budget.

Single-sample historical eval:

- run:
  - `SCRN_BRECQ_app/scrn_repro/runs/test/20260509_010027_paper5_energy_filtered_perpatch_absmax_10750_ddp4_seed20260425_nodecay_lr1p5e-3_best_eval_gt_colorbar`
- metrics:
  - before SNR / SSIM: `3.9693 dB / 0.6053`
  - after SNR / SSIM: `13.7548 dB / 0.9374`
- previous preferred `lr=0.001` checkpoint:
  - after SNR / SSIM: `13.8807 dB / 0.9324`

Matching normalized 478 eval:

- run:
  - `SCRN_BRECQ_app/scrn_repro/runs/test_multi/20260509_010150_fp32_lr1p5e-3_matching_normalized_grid478_seed20260507`
- testset:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`
- validation:
  - rows: `11950`
  - condition count: `25`
  - per-condition count: `478`
  - manifest warnings: none

Overall 478 comparison:

| checkpoint | SNR mean | SNR median | SSIM mean | SSIM median | SNR gain mean |
|---|---:|---:|---:|---:|---:|
| `lr=0.001`, DDP3, no decay | 17.8346 | 18.1752 | 0.9644 | 0.9788 | 16.7703 |
| `lr=0.0015`, DDP4, no decay | 16.9180 | 17.3165 | 0.9566 | 0.9728 | 15.8536 |
| `lr=0.002`, DDP4, no decay | 16.0660 | 16.3905 | 0.9546 | 0.9738 | 15.0017 |
| `lr=0.005`, DDP4, no decay | -38.8719 | -38.1403 | 0.1705 | 0.1384 | -39.9362 |

By-source `lr=0.0015` 478 result:

| source | count | SNR mean | SSIM mean |
|---|---:|---:|---:|
| Anisotropic | 1875 | 21.0776 | 0.9897 |
| Kerry3D | 400 | 9.5860 | 0.9526 |
| Shots0001 | 9675 | 16.4150 | 0.9504 |

Interpretation:

- `lr=0.0015` is a useful midpoint: it improves over `lr=0.002` but still underperforms `lr=0.001` on the matching 478 benchmark.
- It should not replace the `lr=0.001`, DDP3, no-decay checkpoint as the normalized FP32 candidate.
- The preferred normalized W4A8 starting point remains:
  - `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_194718_paper5_energy_filtered_perpatch_absmax_10750_ddp3_seed20260425_nodecay_lr1e-3/checkpoints/best.pth`
- The current evidence suggests increasing LR under DDP4/global batch `128` is not beneficial for this 80-epoch budget.

## 2026-05-09 Default SCRN dataset and FP32 checkpoint

This section records the project default after the dataset rebuild, normalization, FP32 retraining, 5x5 multi-eval, and LR sweep.

Default dataset protocol:

- Name:
  - `paper5_energy_filtered_perpatch_absmax`
- Train:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_train_10750`
- Calibration:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- Test:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`

Default data interpretation:

- This is the default dataset family for future SCRN FP32 and W4A8 work.
- The protocol combines:
  - paper-style five-source data construction,
  - hard filtering of invalid / near-zero low-energy patches,
  - per-patch absmax normalization,
  - matching normalized train / calibration / test amplitude space.
- Do not mix this checkpoint with raw-amplitude calibration/test data when judging model or quantization quality.
- Older datasets remain useful only as historical baselines or diagnostic comparisons:
  - `scrn_quant_10750_0_*`
  - `scrn_paper5_*`
  - `scrn_paper5_energy_filtered_*`
  - `scrn_paper5_perpatch_absmax_*`

Default FP32 checkpoint:

- `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_194718_paper5_energy_filtered_perpatch_absmax_10750_ddp3_seed20260425_nodecay_lr1e-3/checkpoints/best.pth`

Default FP32 training configuration:

- dataset:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_train_10750`
- epochs: `80`
- seed: `20260425`
- per-GPU batch size: `32`
- world size: `3`
- global batch: `96`
- LR: `0.001`
- LR schedule: no decay, `--milestones ""`
- weight decay: `0.0`
- model:
  - `dim=64`
  - `stage_depths=1,1,1,1,1`
  - `head_dim=32`
  - `window_size=8`
  - `input_resolution=128`

Default FP32 evidence:

| metric | value |
|---|---:|
| best epoch | 79 |
| best loss | 18.80715600649516 |
| last loss | 19.151402472030547 |
| single-sample after SNR / SSIM | 13.8807 / 0.9324 |
| matching normalized 478 SNR mean / SSIM mean | 17.8346 / 0.9644 |
| matching normalized 478 SNR median / SSIM median | 18.1752 / 0.9788 |
| matching normalized 478 SNR gain mean | 16.7703 |
| Shots0001 SNR mean / SSIM mean | 17.3551 / 0.9594 |

Decision:

- Use the above dataset family and checkpoint as the default starting point for future normalized SCRN experiments.
- Use the same calibration/test family for any W4A8 BRECQ experiment that starts from this checkpoint.
- The LR sweep did not find a better replacement:
  - `lr=0.0015`, DDP4/global batch `128`: worse than default on matching 478.
  - `lr=0.002`, DDP4/global batch `128`: worse than default on matching 478.
  - `lr=0.005`, DDP4/global batch `128`: unstable / unusable.

## 2026-05-09 Quantized SCRN fixed-grid evaluator

Added a quantized checkpoint evaluator for the normalized 478x25 test protocol.

New files:

- `SCRN_BRECQ_app/scrn_brecq/cli/evaluate_quantized_scrn_grid.py`
- `SCRN_BRECQ_app/scrn_brecq/tests/test_evaluate_quantized_scrn_grid.py`

Default evaluation grid:

- SNR settings: `-2,-1,1,5,10`
- missing rates: `0.02,0.08,0.18,0.28,0.38`
- seed: `20260507`
- default dataset:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`

Outputs written per run:

- `config.json`
- `metrics.json`
- `summary.md`
- `per_sample_metrics.jsonl`

Metrics:

- per-sample rows include FP32, pre-reconstruction quantized, and post-reconstruction quantized SNR/SSIM fields when a pre-reconstruction checkpoint is provided.
- delta fields include `quant_pre_minus_fp32_*`, `quant_post_minus_fp32_*`, and `quant_post_minus_pre_*`.
- grouped summaries are emitted for:
  - overall
  - source
  - SNR setting
  - missing rate
  - SNR/missing-rate condition

Validation:

- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_evaluate_quantized_scrn_grid -v`
  - result: 6 tests passed
- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/cli/evaluate_quantized_scrn_grid.py`
  - result: passed
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_grid --help`
  - result: help text rendered successfully
- smoke run on GPU `1`:
  - run: `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E007_normalized_w4a32_baseline/smoke/20260509_144714_smoke_quantized_grid2`
  - rows: `2`
  - patches: `2`
  - conditions: `1`
  - files confirmed: `config.json`, `metrics.json`, `summary.md`, `per_sample_metrics.jsonl`

## 2026-05-09 Normalized W4A32 single-GPU baseline

Rebuilt the W4A32 weight-only BRECQ baseline on the default normalized protocol.

Run:

- Quantization run:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1`
- Post-reconstruction checkpoint:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1/checkpoints/quantized_scrn_brecq.pth`
- Pre-reconstruction checkpoint:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1/checkpoints/quantized_scrn_brecq_pre_recon.pth`
- Source FP32 checkpoint:
  - `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_194718_paper5_energy_filtered_perpatch_absmax_10750_ddp3_seed20260425_nodecay_lr1e-3/checkpoints/best.pth`
- Calibration dataset:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- Test dataset:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`
- GPU: physical GPU `1`
- BRECQ settings:
  - `num_samples=1024`
  - `batch_size=16`
  - `iters_w=20000`
  - `n_bits_w=4`
  - `n_bits_a=8`
  - `act_quant=false` via `--no-act-quant`
- Single-sample CLI sanity output:
  - `fp32_snr=13.8808`
  - `pre_w_snr=13.4675`
  - `post_w_snr=13.8918`
  - `post_recon_ssim=0.9290`
  - `reconstruction_seconds=2597.64`
  - `elapsed_seconds=2643.75`

Checkpoint verification:

- Verification JSON:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1/verification.json`
- `passed=true`
- final quant state:
  - `weight_quant=true`
  - `act_quant=false`
- quant config:
  - `n_bits_w=4`
  - `n_bits_a=8`
  - `channel_wise=true`
  - `scale_method=mse`
- layer summary:
  - quant modules: `52`
  - weight bit counts: `4bit=50`, `8bit=2`
  - `level_offender_count=0`
- activation summary:
  - activation quant modules: `52`
  - activation delta count: `0`

Normalized 478x25 grid eval:

- Eval run:
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E007_normalized_w4a32_baseline/eval/20260509_153415_normalized_w4a32_single_gpu1_grid478_seed20260507`
- Rows:
  - `11950 = 478 patches * 25 conditions`
- Grid:
  - SNR settings: `-2,-1,1,5,10`
  - missing rates: `0.02,0.08,0.18,0.28,0.38`
  - seed: `20260507`
- Eval runtime:
  - elapsed seconds: `179.26`
  - FP32 inference seconds: `26.70`
  - pre-recon quant inference seconds: `69.00`
  - post-recon quant inference seconds: `27.52`

Overall grid metrics:

| path | SNR mean | SNR median | SSIM mean | SSIM median |
|---|---:|---:|---:|---:|
| FP32 | 17.8329 | 18.1742 | 0.964330 | 0.978794 |
| W4A32 pre-recon | 16.6142 | 17.1775 | 0.935462 | 0.949638 |
| W4A32 post-recon | 17.7856 | 18.1128 | 0.964137 | 0.978461 |

Overall deltas:

| delta | SNR mean | SNR median | SSIM mean | SSIM median |
|---|---:|---:|---:|---:|
| pre - FP32 | -1.2186 | -0.8585 | -0.028868 | -0.023025 |
| post - FP32 | -0.0473 | -0.0370 | -0.000193 | 0.000015 |
| post - pre | 1.1713 | 0.8055 | 0.028675 | 0.022634 |

By-source post-recon metrics:

| source | rows | FP32 SNR mean | W4A32 SNR mean | delta mean | FP32 SSIM mean | W4A32 SSIM mean | delta SSIM mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Anisotropic | 1875 | 22.0595 | 21.9718 | -0.0877 | 0.992762 | 0.992832 | 0.000070 |
| Kerry3D | 400 | 9.6085 | 9.6082 | -0.0003 | 0.950559 | 0.949887 | -0.000672 |
| Shots0001 | 9675 | 17.3538 | 17.3124 | -0.0414 | 0.959390 | 0.959165 | -0.000225 |

Decision:

- The normalized W4A32 post-reconstruction checkpoint is authentic and effectively matches FP32 on the normalized 478x25 protocol.
- The remaining overall post-FP32 gap is small enough to proceed to W4A8 activation-init and activation quantization diagnostics from this checkpoint.
- The pre-reconstruction gap confirms BRECQ weight reconstruction is still materially useful under the normalized protocol.

## 2026-05-09 Normalized W4A32 four-GPU baseline

Repeated the normalized W4A32 weight-only BRECQ baseline with distributed four-GPU reconstruction. The only intended experiment change versus the single-GPU baseline was enabling `torchrun --nproc_per_node=4 --distributed --gpus 0,1,2,3`.

Execution notes:

- The first sandboxed `torchrun --standalone` attempt failed before model work with local TCP rendezvous blocked by sandbox networking:
  - error class: `RendezvousConnectionError`
  - relevant message: `Operation not permitted`
- The same command was rerun outside the sandbox with unchanged experiment parameters.
- During the run, GPU `0` also had an external `swinir` process using about `5.6 GiB`; the quantization run still completed without OOM.

Run:

- Quantization run:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_160247_normalized_w4a32_1024cali_w20000_dist4_bsz16_global64`
- Post-reconstruction checkpoint:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_160247_normalized_w4a32_1024cali_w20000_dist4_bsz16_global64/checkpoints/quantized_scrn_brecq.pth`
- Pre-reconstruction checkpoint:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_160247_normalized_w4a32_1024cali_w20000_dist4_bsz16_global64/checkpoints/quantized_scrn_brecq_pre_recon.pth`
- Source FP32 checkpoint:
  - `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_194718_paper5_energy_filtered_perpatch_absmax_10750_ddp3_seed20260425_nodecay_lr1e-3/checkpoints/best.pth`
- Calibration dataset:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- Test dataset:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`
- Distributed settings:
  - physical GPUs: `0,1,2,3`
  - `world_size=4`
  - local rank batch size: `16`
  - effective reconstruction global batch: `64`
- BRECQ settings:
  - `num_samples=1024`
  - `iters_w=20000`
  - `n_bits_w=4`
  - `n_bits_a=8`
  - `act_quant=false` via `--no-act-quant`
- Single-sample CLI sanity output:
  - `fp32_snr=13.8808`
  - `pre_w_snr=13.4675`
  - `post_w_snr=13.8607`
  - `post_recon_ssim=0.9276`
  - `reconstruction_seconds=4046.79`
  - `elapsed_seconds=4097.86`

Checkpoint verification:

- Verification JSON:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_160247_normalized_w4a32_1024cali_w20000_dist4_bsz16_global64/verification.json`
- `passed=true`
- final quant state:
  - `weight_quant=true`
  - `act_quant=false`
- config:
  - distributed enabled: `true`
  - `world_size=4`
  - `n_bits_w=4`
  - `n_bits_a=8`
  - `channel_wise=true`
  - `scale_method=mse`
- layer summary:
  - quant modules: `52`
  - weight bit counts: `4bit=50`, `8bit=2`
  - `level_offender_count=0`
- activation summary:
  - activation quant modules: `52`
  - activation delta count: `0`

Normalized 478x25 grid eval:

- Eval run:
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E008_normalized_w4a32_dist4_baseline/eval/20260509_171210_normalized_w4a32_dist4_bsz16_global64_grid478_seed20260507`
- Rows:
  - `11950 = 478 patches * 25 conditions`
- Grid:
  - SNR settings: `-2,-1,1,5,10`
  - missing rates: `0.02,0.08,0.18,0.28,0.38`
  - seed: `20260507`
- Eval runtime:
  - elapsed seconds: `226.81`
  - FP32 inference seconds: `26.96`
  - pre-recon quant inference seconds: `95.66`
  - post-recon quant inference seconds: `28.09`

Overall grid metrics:

| path | SNR mean | SNR median | SSIM mean | SSIM median |
|---|---:|---:|---:|---:|
| FP32 | 17.8329 | 18.1742 | 0.964330 | 0.978794 |
| W4A32 pre-recon | 16.6142 | 17.1775 | 0.935462 | 0.949638 |
| W4A32 post-recon dist4 | 17.7120 | 18.0740 | 0.964234 | 0.978921 |

Overall deltas:

| delta | SNR mean | SNR median | SSIM mean | SSIM median |
|---|---:|---:|---:|---:|
| pre - FP32 | -1.2186 | -0.8585 | -0.028868 | -0.023025 |
| post - FP32 | -0.1209 | -0.0752 | -0.000096 | -0.000088 |
| post - pre | 1.0977 | 0.7782 | 0.028772 | 0.022477 |

By-source post-recon metrics:

| source | rows | FP32 SNR mean | W4A32 dist4 SNR mean | delta mean | FP32 SSIM mean | W4A32 dist4 SSIM mean | delta SSIM mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Anisotropic | 1875 | 22.0595 | 21.8008 | -0.2586 | 0.992762 | 0.992702 | -0.000060 |
| Kerry3D | 400 | 9.6085 | 9.5971 | -0.0113 | 0.950559 | 0.950481 | -0.000078 |
| Shots0001 | 9675 | 17.3538 | 17.2551 | -0.0987 | 0.959390 | 0.959286 | -0.000104 |

Comparison with the single-GPU W4A32 baseline:

| checkpoint | SNR mean | SNR median | SSIM mean | SSIM median | post - FP32 SNR mean |
|---|---:|---:|---:|---:|---:|
| single GPU | 17.7856 | 18.1128 | 0.964137 | 0.978461 | -0.0473 |
| dist4 global64 | 17.7120 | 18.0740 | 0.964234 | 0.978921 | -0.1209 |

Decision:

- The four-GPU W4A32 checkpoint is authentic W-only quantization and remains close to FP32 on the normalized 478x25 grid.
- This dist4 run is slightly worse in SNR than the single-GPU baseline by `0.0736 dB` mean and `0.0387 dB` median, while SSIM is comparable.
- This run was also slower than the single-GPU run in wall-clock reconstruction time, likely affected by GPU `0` contention and distributed overhead.
- Keep the single-GPU normalized W4A32 checkpoint as the preferred W4A8 starting point for now; keep this dist4 checkpoint as a valid distributed comparison baseline, not the default replacement.

## 2026-05-09 Normalized W4A32 four-GPU global128 probe

Ran a second distributed W4A32 weight-only reconstruction on the normalized protocol with local batch `32`, effective global batch `128`. The goal was to test whether the E008 dist4/global64 gap to E007 single-GPU was mainly an effective-batch issue.

Execution notes:

- The first sandboxed `torchrun --standalone` attempt failed before model work with the same local TCP rendezvous restriction seen in E008:
  - error class: `RendezvousConnectionError`
  - relevant message: `Operation not permitted`
- The command was rerun outside the sandbox with unchanged experiment parameters.
- GPU `0` had an external `swinir` process using about `5.6 GiB` during the first part of reconstruction; it exited before the run finished.
- The run completed without OOM, but reconstruction timing should still be treated as affected by early GPU `0` contention and distributed overhead.

Run:

- Quantization run:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_182611_normalized_w4a32_1024cali_w20000_dist4_bsz32_global128`
- Post-reconstruction checkpoint:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_182611_normalized_w4a32_1024cali_w20000_dist4_bsz32_global128/checkpoints/quantized_scrn_brecq.pth`
- Pre-reconstruction checkpoint:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_182611_normalized_w4a32_1024cali_w20000_dist4_bsz32_global128/checkpoints/quantized_scrn_brecq_pre_recon.pth`
- Source FP32 checkpoint:
  - `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_194718_paper5_energy_filtered_perpatch_absmax_10750_ddp3_seed20260425_nodecay_lr1e-3/checkpoints/best.pth`
- Calibration dataset:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- Test dataset:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`
- Distributed settings:
  - physical GPUs: `0,1,2,3`
  - `world_size=4`
  - local rank batch size: `32`
  - effective reconstruction global batch: `128`
- BRECQ settings:
  - `num_samples=1024`
  - `iters_w=20000`
  - `n_bits_w=4`
  - `n_bits_a=8`
  - `act_quant=false` via `--no-act-quant`
- Single-sample CLI sanity output:
  - `fp32_snr=13.8808`
  - `pre_w_snr=13.4675`
  - `post_w_snr=13.8346`
  - `post_recon_ssim=0.9280`
  - `reconstruction_seconds=6920.35`
  - `elapsed_seconds=7076.65`

Checkpoint verification:

- Verification JSON:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_182611_normalized_w4a32_1024cali_w20000_dist4_bsz32_global128/verification.json`
- `passed=true`
- final quant state:
  - `weight_quant=true`
  - `act_quant=false`
- config:
  - distributed enabled: `true`
  - `world_size=4`
  - local batch size: `32`
  - `n_bits_w=4`
  - `n_bits_a=8`
  - `channel_wise=true`
  - `scale_method=mse`
- layer summary:
  - quant modules: `52`
  - weight bit counts: `4bit=50`, `8bit=2`
  - `level_offender_count=0`
- activation summary:
  - activation quant modules: `52`
  - activation delta count: `0`

Normalized 478x25 grid eval:

- Eval run:
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E009_normalized_w4a32_dist4_global128_probe/eval/20260509_202444_normalized_w4a32_dist4_bsz32_global128_grid478_seed20260507`
- Rows:
  - `11950 = 478 patches * 25 conditions`
- Grid:
  - SNR settings: `-2,-1,1,5,10`
  - missing rates: `0.02,0.08,0.18,0.28,0.38`
  - seed: `20260507`
- Eval runtime:
  - elapsed seconds: `198.69`
  - FP32 inference seconds: `26.76`
  - pre-recon quant inference seconds: `82.52`
  - post-recon quant inference seconds: `27.77`

Overall grid metrics:

| path | SNR mean | SNR median | SSIM mean | SSIM median |
|---|---:|---:|---:|---:|
| FP32 | 17.8329 | 18.1742 | 0.964330 | 0.978794 |
| W4A32 pre-recon | 16.6142 | 17.1775 | 0.935462 | 0.949638 |
| W4A32 post-recon dist4 global128 | 17.7370 | 18.0836 | 0.963943 | 0.978616 |

Overall deltas:

| delta | SNR mean | SNR median | SSIM mean | SSIM median |
|---|---:|---:|---:|---:|
| pre - FP32 | -1.2186 | -0.8585 | -0.028868 | -0.024085 |
| post - FP32 | -0.0959 | -0.0589 | -0.000387 | -0.000446 |
| post - pre | 1.1228 | 0.7935 | 0.028481 | 0.023789 |

By-source post-recon metrics:

| source | rows | FP32 SNR mean | W4A32 global128 SNR mean | delta mean | FP32 SSIM mean | W4A32 global128 SSIM mean | delta SSIM mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Anisotropic | 1875 | 22.0595 | 21.9149 | -0.1446 | 0.992762 | 0.992198 | -0.000564 |
| Kerry3D | 400 | 9.6085 | 9.6534 | 0.0449 | 0.950559 | 0.950640 | 0.000080 |
| Shots0001 | 9675 | 17.3538 | 17.2616 | -0.0922 | 0.959390 | 0.959018 | -0.000372 |

Comparison with normalized W4A32 baselines:

| checkpoint | SNR mean | SNR median | SSIM mean | SSIM median | post - FP32 SNR mean | reconstruction seconds |
|---|---:|---:|---:|---:|---:|---:|
| E007 single GPU | 17.7856 | 18.1128 | 0.964137 | 0.978461 | -0.0473 | 2597.64 |
| E008 dist4 global64 | 17.7120 | 18.0740 | 0.964234 | 0.978921 | -0.1209 | 4046.79 |
| E009 dist4 global128 | 17.7370 | 18.0836 | 0.963943 | 0.978616 | -0.0959 | 6920.35 |

Decision:

- E009 dist4/global128 improves over E008 dist4/global64 by `+0.0250 dB` mean SNR and `+0.0096 dB` median SNR, so effective batch size explains part of the E008 gap.
- E009 still remains below E007 single-GPU by `0.0486 dB` mean SNR and `0.0292 dB` median SNR, and it is much slower in wall-clock reconstruction.
- Do not replace E007 as the preferred W4A8 starting checkpoint. Keep E009 as diagnostic evidence that larger distributed batch partially helps SNR but does not solve the distributed quality/runtime tradeoff under the normalized protocol.

## 2026-05-09 Normalized W4A32 single-GPU representative visual comparisons

Generated representative visual comparisons for the current default W4A32 baseline, E007 single-GPU. No reconstruction or grid metrics were rerun; the script selected rows from the existing E007 full 478x25 grid metrics and regenerated only the chosen degraded inputs and predictions for visualization.

Run:

- Visualization run:
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E010_normalized_w4a32_single_gpu_visuals/20260509_204855_representative_figures_source_x_condition`
- Checkpoint:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1/checkpoints/quantized_scrn_brecq.pth`
- Pre-recon checkpoint:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1/checkpoints/quantized_scrn_brecq_pre_recon.pth`
- Source metric rows:
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E007_normalized_w4a32_baseline/eval/20260509_153415_normalized_w4a32_single_gpu1_grid478_seed20260507/per_sample_metrics.jsonl`
- Output files:
  - `config.json`
  - `selected_samples.json`
  - `summary.md`
  - `figures/*.png`

Selection strategy:

- Sources: `Anisotropic`, `Kerry3D`, `Shots0001`
- Conditions:
  - `low_snr_high_missing`: SNR `-2`, missing rate `0.38`
  - `mid_snr_mid_missing`: SNR `1`, missing rate `0.18`
  - `high_snr_low_missing`: SNR `10`, missing rate `0.02`
- For each source and condition group, selected the row whose `quant_post_minus_fp32_snr_db` was closest to that group median.
- This produced `9` figures, each with panels:
  - Ground Truth
  - Input
  - FP32
  - W4A32 pre-recon
  - W4A32 post-recon

Generated figures:

| # | source | condition | patch | input SNR | FP32 SNR | pre SNR | post SNR | post-FP32 | figure |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | Anisotropic | low_snr_high_missing | `test_000044.npy` | -2.95 | 18.29 | 17.38 | 18.19 | -0.100 | `01_Anisotropic_low_snr_high_missing_test_000044.png` |
| 2 | Anisotropic | mid_snr_mid_missing | `test_000013.npy` | 0.06 | 23.32 | 21.77 | 23.21 | -0.114 | `02_Anisotropic_mid_snr_mid_missing_test_000013.png` |
| 3 | Anisotropic | high_snr_low_missing | `test_000025.npy` | 9.49 | 28.75 | 24.63 | 28.65 | -0.103 | `03_Anisotropic_high_snr_low_missing_test_000025.png` |
| 4 | Kerry3D | low_snr_high_missing | `test_000081.npy` | -2.96 | 6.05 | 5.92 | 6.04 | -0.008 | `04_Kerry3D_low_snr_high_missing_test_000081.png` |
| 5 | Kerry3D | mid_snr_mid_missing | `test_000084.npy` | 0.08 | 11.49 | 10.83 | 11.47 | -0.020 | `05_Kerry3D_mid_snr_mid_missing_test_000084.png` |
| 6 | Kerry3D | high_snr_low_missing | `test_000077.npy` | 10.00 | 10.63 | 11.27 | 10.64 | 0.008 | `06_Kerry3D_high_snr_low_missing_test_000077.png` |
| 7 | Shots0001 | low_snr_high_missing | `test_000374.npy` | -3.01 | 12.81 | 12.11 | 12.78 | -0.037 | `07_Shots0001_low_snr_high_missing_test_000374.png` |
| 8 | Shots0001 | mid_snr_mid_missing | `test_000395.npy` | 0.22 | 17.08 | 16.50 | 17.04 | -0.033 | `08_Shots0001_mid_snr_mid_missing_test_000395.png` |
| 9 | Shots0001 | high_snr_low_missing | `test_000349.npy` | 9.84 | 20.85 | 18.69 | 20.80 | -0.046 | `09_Shots0001_high_snr_low_missing_test_000349.png` |

Verification:

- `find .../figures -name '*.png' | wc -l` returned `9`.
- `file .../figures/*.png` reported all figures as PNG images with size `3600 x 720`.
- The run was generated on CPU because only 9 selected samples were needed; this did not alter any W4A32 checkpoint or evaluation metric.

## 2026-05-09 Normalized W4A32 seismic visual comparisons

Generated a second visualization set using the `seismic` colormap requested for signed wave amplitudes. This run keeps the same 9 representative normalized samples from E010 and adds the usual default single sample from `SCRN-main/test_data/clear.npy` and `SCRN-main/test_data/noise_and_miss.npy`.

Run:

- Visualization run:
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E011_normalized_w4a32_seismic_visuals/20260509_205659_seismic_representative_plus_default_single`
- Checkpoint:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1/checkpoints/quantized_scrn_brecq.pth`
- Pre-recon checkpoint:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1/checkpoints/quantized_scrn_brecq_pre_recon.pth`
- Output files:
  - `config.json`
  - `selected_samples.json`
  - `summary.md`
  - `figures/*.png`

Display settings:

- Colormap: `seismic`
- Color scale: symmetric per figure over all panels, centered at `0`
- Panels:
  - Ground Truth
  - Input
  - FP32
  - W4A32 pre-recon
  - W4A32 post-recon

Generated figures:

| # | source | condition | amplitude | patch | input SNR | FP32 SNR | pre SNR | post SNR | post-FP32 | figure |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | Anisotropic | low_snr_high_missing | normalized per-patch absmax | `test_000044.npy` | -2.95 | 18.29 | 17.38 | 18.19 | -0.100 | `01_Anisotropic_low_snr_high_missing_test_000044_seismic.png` |
| 2 | Anisotropic | mid_snr_mid_missing | normalized per-patch absmax | `test_000013.npy` | 0.06 | 23.32 | 21.77 | 23.21 | -0.114 | `02_Anisotropic_mid_snr_mid_missing_test_000013_seismic.png` |
| 3 | Anisotropic | high_snr_low_missing | normalized per-patch absmax | `test_000025.npy` | 9.49 | 28.75 | 24.63 | 28.65 | -0.103 | `03_Anisotropic_high_snr_low_missing_test_000025_seismic.png` |
| 4 | Kerry3D | low_snr_high_missing | normalized per-patch absmax | `test_000081.npy` | -2.96 | 6.05 | 5.92 | 6.04 | -0.008 | `04_Kerry3D_low_snr_high_missing_test_000081_seismic.png` |
| 5 | Kerry3D | mid_snr_mid_missing | normalized per-patch absmax | `test_000084.npy` | 0.08 | 11.49 | 10.83 | 11.47 | -0.020 | `05_Kerry3D_mid_snr_mid_missing_test_000084_seismic.png` |
| 6 | Kerry3D | high_snr_low_missing | normalized per-patch absmax | `test_000077.npy` | 10.00 | 10.63 | 11.27 | 10.64 | 0.008 | `06_Kerry3D_high_snr_low_missing_test_000077_seismic.png` |
| 7 | Shots0001 | low_snr_high_missing | normalized per-patch absmax | `test_000374.npy` | -3.01 | 12.81 | 12.11 | 12.78 | -0.037 | `07_Shots0001_low_snr_high_missing_test_000374_seismic.png` |
| 8 | Shots0001 | mid_snr_mid_missing | normalized per-patch absmax | `test_000395.npy` | 0.22 | 17.08 | 16.50 | 17.04 | -0.033 | `08_Shots0001_mid_snr_mid_missing_test_000395_seismic.png` |
| 9 | Shots0001 | high_snr_low_missing | normalized per-patch absmax | `test_000349.npy` | 9.84 | 20.85 | 18.69 | 20.80 | -0.046 | `09_Shots0001_high_snr_low_missing_test_000349_seismic.png` |
| 10 | `SCRN-main/test_data` | default_single_sample | raw default SCRN sample | `clear.npy` | 3.97 | 13.88 | 13.47 | 13.89 | 0.011 | `10_default_single_sample_seismic_raw_amplitude.png` |

Verification:

- `find .../figures -name '*.png' | wc -l` returned `10`.
- `file .../figures/*.png` reported all figures as PNG images with size `3600 x 720`.
- All new generated artifacts are under `SCRN_BRECQ_app`; no project-external helper script or output is retained.

## 2026-05-09 Normalized W4A32 seismic visual comparisons with denormalized display

Updated the seismic visualization helper so normalized representative panels are restored before display using the dataset manifest scale:

- Script:
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E011_normalized_w4a32_seismic_visuals/generate_seismic_visuals.py`
- Restoration rule:
  - `display = normalized * normalization_scale`
- Scale source:
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478/manifest.json`
  - field: `samples[].normalization_scale`
- Metric note:
  - SNR/SSIM labels stay in the normalized protocol metric space.
  - Only the displayed image panels are restored to raw amplitude scale.
- Direct-run fix:
  - The helper now inserts the repo root into `sys.path`, so it can run directly by file path from the repo root.

Run:

- Visualization run:
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E012_normalized_w4a32_seismic_denormalized_visuals/20260509_211000_seismic_denormalized_representative_plus_default_single`
- Output files:
  - `config.json`
  - `selected_samples.json`
  - `summary.md`
  - `figures/*.png`

Display settings:

- Colormap: `seismic`
- Color scale: symmetric per figure over all panels, centered at `0`
- Panels: Ground Truth, Input, FP32, W4A32 pre-recon, W4A32 post-recon
- Normalized representative display: restored from per-patch absmax normalization
- Default single sample display: unchanged raw default SCRN amplitude

Generated figures and scales:

| # | source | condition | patch | normalization scale | figure |
|---:|---|---|---|---:|---|
| 1 | Anisotropic | low_snr_high_missing | `test_000044.npy` | 0.260053 | `01_Anisotropic_low_snr_high_missing_test_000044_seismic_denormalized.png` |
| 2 | Anisotropic | mid_snr_mid_missing | `test_000013.npy` | 0.0957499 | `02_Anisotropic_mid_snr_mid_missing_test_000013_seismic_denormalized.png` |
| 3 | Anisotropic | high_snr_low_missing | `test_000025.npy` | 0.133937 | `03_Anisotropic_high_snr_low_missing_test_000025_seismic_denormalized.png` |
| 4 | Kerry3D | low_snr_high_missing | `test_000081.npy` | 0.623027 | `04_Kerry3D_low_snr_high_missing_test_000081_seismic_denormalized.png` |
| 5 | Kerry3D | mid_snr_mid_missing | `test_000084.npy` | 0.817539 | `05_Kerry3D_mid_snr_mid_missing_test_000084_seismic_denormalized.png` |
| 6 | Kerry3D | high_snr_low_missing | `test_000077.npy` | 0.571108 | `06_Kerry3D_high_snr_low_missing_test_000077_seismic_denormalized.png` |
| 7 | Shots0001 | low_snr_high_missing | `test_000374.npy` | 0.056497 | `07_Shots0001_low_snr_high_missing_test_000374_seismic_denormalized.png` |
| 8 | Shots0001 | mid_snr_mid_missing | `test_000395.npy` | 0.0439193 | `08_Shots0001_mid_snr_mid_missing_test_000395_seismic_denormalized.png` |
| 9 | Shots0001 | high_snr_low_missing | `test_000349.npy` | 0.0698473 | `09_Shots0001_high_snr_low_missing_test_000349_seismic_denormalized.png` |
| 10 | `SCRN-main/test_data` | default_single_sample | `clear.npy` | n/a | `10_default_single_sample_seismic_raw_amplitude.png` |

Verification:

- `conda run -n quant python -m py_compile .../generate_seismic_visuals.py` passed.
- `conda run -n quant python .../generate_seismic_visuals.py` completed with `figure_count=10`.
- `find .../figures -name '*.png' | wc -l` returned `10`.
- `file .../figures/*.png` reported all figures as PNG images with size `3600 x 720`.

## 2026-05-09 NE000-NE006 归一化协议激活量化路线图

开启新的激活量化实验序列，统一使用 `NE00X` 前缀；其中 `NE` 表示新的归一化数据协议：

- 协议：
  - `paper5_energy_filtered_perpatch_absmax`
- FP32 checkpoint：
  - `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_194718_paper5_energy_filtered_perpatch_absmax_10750_ddp3_seed20260425_nodecay_lr1e-3/checkpoints/best.pth`
- 校准集：
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- 测试集：
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`
- 默认 W4A8 起点 checkpoint：
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1/checkpoints/quantized_scrn_brecq.pth`
- 默认评估口径：
  - normalized `478 x 25` grid
  - SNR settings：`-2,-1,1,5,10`
  - missing rates：`0.02,0.08,0.18,0.28,0.38`
  - seed：`20260507`
- GPU 原则：
  - 涉及模型推理或重建时优先使用 GPU
  - 单卡默认优先级：`1 -> 2 -> 3 -> 0`
  - 如需偏离默认 GPU 选择，必须在日志中记录原因

计划序列：

| 实验 | 目的 | 默认动作 | 预期输出 | 决策点 |
|---|---|---|---|---|
| NE000 | 在正式诊断前，重建新协议下的 W4A8 激活量化 baseline。 | 从 E007 W4A32 出发，运行 tensor-wise A8 init 和 `iters_a=5000` activation reconstruction，再在 normalized `478 x 25` grid 上评估 pre/post activation reconstruction。 | W4A8 pre-act-recon 与 final checkpoint、验证摘要、grid 指标、by-source 指标。 | 建立新的 W4A8 baseline，判断 A8 相对 E007 W4A32 和 FP32 的掉点幅度。 |
| NE001 | 在新的 W4A8 checkpoint 上复现实验诊断。 | 使用 normalized calibration set，对 NE000 pre-act-recon 和 final checkpoint 跑 diagnostics。 | quantizer 数量、`non_positive_delta_count`、activation delta 统计、fake-quant error 统计、Conv2d/Linear 分组摘要。 | 确认负 scale / 非正 scale 是否仍然不存在，并定位 activation fake-quant 误差集中位置。 |
| NE002 | 做合法状态和 checkpoint sanity sweep。 | 检查 activation init 后和 activation reconstruction 后的 quantizer 合法性、checkpoint reload 一致性和 final quant state。 | verification JSON，以及 pre/final W4A8 状态完整性表。 | 排除 invalid delta、reload 状态错误或 output quantizer 泄漏导致的 W4A8 失败。 |
| NE003 | 固定评估口径并检查单样本敏感性。 | 在不改 checkpoint 的前提下，对比默认单样本、代表样本、小 subset 和完整 normalized `478 x 25` grid。 | 证明 full-grid 指标是决策口径；必要时可重新生成代表图。 | 防止单样本或小 subset 偶然结果主导结论。 |
| NE004 | 做新 baseline 上的 activation quantizer 敏感性分析。 | 评估关闭不同 activation quantizer 分组的效果，重点看 Conv2d vs Linear/transformer 以及 stage/role 分组。 | 按分组和 source 统计的 SNR/SSIM 恢复表。 | 检查旧结论是否仍成立：Conv2d activation quantization 是否仍是 W4A8 gap 主因。 |
| NE005 | 在新协议下重新检查 range、clipping 和 outlier 控制。 | 在 NE004 明确目标分组后，测试 tensor-wise max、percentile、MSE-grid 等 range 方法。 | range-method 对比表，以及 clipping/range 是否修复主误差的诊断结果。 | 判断 tensor-wise range 调整在 normalized 数据下是否仍基本无效。 |
| NE006 | 做结构化 activation granularity 搜索。 | 复测旧 E006 强候选：all Conv2d per-channel、selective split/merge/stage-output Conv2d per-channel、selective group-wise `g4`、stage5 sanity checks。 | 候选策略表，包含 full-grid 指标、by-source 指标、checkpoint 路径和部署备注。 | 选择下一阶段用于更深 activation reconstruction、mixed precision 或 selective FP32 的 W4A8 候选。 |

NE00X 执行和记录规则：

- 每个 NE 实验必须记录命令、run directory、checkpoint 路径、数据协议、GPU 选择、验证结果和 normalized `478 x 25` grid 指标。
- 修改 `SCRN_BRECQ_app/scrn_brecq/` 下代码或文档时，必须更新本日志。
- 涉及 activation quantization 时，必须同步更新 `SCRN_BRECQ_app/scrn_brecq/ACTIVATION_QUANTIZATION_LOG.md`。
- 每个完成的实验或代码/日志变更后都要 commit。
- 不 push，除非明确要求。
- 不修改 `SCRN-main/` 或 `BRECQ-main/`。
- 不在 `SCRN_BRECQ_app/` 之外写入产物。

## 2026-05-09 NE000 归一化 W4A8 激活量化重建 baseline

完成新的 `NE000` baseline：在 `paper5_energy_filtered_perpatch_absmax` 协议下，从当前默认 E007 单卡 W4A32 checkpoint 出发，运行 tensor-wise W4A8 activation initialization 和 `iters_a=5000` activation reconstruction，并在 normalized `478 x 25` grid 上评估 pre-act-recon 与 final W4A8。

预检：

- Branch：`main`
- Worktree：开始时 clean，本地领先 `origin/main` 18 个提交。
- Repo root：`/home/data1/hanwen/project/Project/SCRN_Quant`
- GPU：GPU 1/2/3 空闲，GPU 0 有外部 `swinir` 进程占用约 `5.6 GiB`。
- 本次使用物理 GPU `1`。

输入：

- W4A32 起点：
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1/checkpoints/quantized_scrn_brecq.pth`
- 校准集：
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- 测试集：
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`

实现修复：

- 首次正式运行在 activation reconstruction 阶段失败，错误为：
  - `RuntimeError: Activation quantizer delta is not learnable. Construct QuantModel with act_quant_params['leaf_param']=True.`
- 原因：
  - `activation_only_quantize_scrn.py` 从 E007 W4A32 checkpoint 构建模型时直接使用 checkpoint 中的 `quant_config.act_quant=false`。
  - A8 init 可以写入 delta tensor，但 activation reconstruction 需要 learnable activation delta。
- 修复：
  - 新增 `build_activation_only_checkpoint_config()`，只构造一个 activation-only checkpoint view，将 `act_quant`、`n_bits_a`、`scale_method` 与当前 activation-only 配置对齐。
  - 源 W4A32 checkpoint 不被修改。
  - 增加单测确认该 helper 会启用 activation quantization，并且不会改写源 checkpoint 的 `quant_config.act_quant`。
- 验证：
  - `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_only_quantize_scrn -v` 通过，18 tests OK。
  - `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/cli/activation_only_quantize_scrn.py` 通过。

NE000 quant run：

- Run dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_0_normalized_w4a8_activation_reconstruction/quant/20260509_213701_normalized_w4a8_tensor_a5000_1024cali_single_gpu1`
- Pre-act-recon checkpoint：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_0_normalized_w4a8_activation_reconstruction/quant/20260509_213701_normalized_w4a8_tensor_a5000_1024cali_single_gpu1/checkpoints/quantized_scrn_brecq_pre_act_recon.pth`
- Final checkpoint：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_0_normalized_w4a8_activation_reconstruction/quant/20260509_213701_normalized_w4a8_tensor_a5000_1024cali_single_gpu1/checkpoints/quantized_scrn_brecq.pth`
- 核心配置：
  - `n_bits_w=4`
  - `n_bits_a=8`
  - `activation_granularity=tensor`
  - `activation_range_method=none`
  - `num_samples=1024`
  - `batch_size=16`
  - `init_batch_size=64`
  - `iters_a=5000`
  - `activation_lr=0.0004`
  - `lp_norm=2.4`
- 时间：
  - activation initialization：`24.8221 s`
  - activation reconstruction：`1335.2904 s`，约 `22.2548 min`
  - elapsed：`1363.2839 s`，约 `22.7214 min`

单样本 sanity：

| metric | value |
|---|---:|
| post-weight W4A32 SNR / SSIM | `13.8918 / 0.929029` |
| W4A8 pre-act SNR / SSIM | `13.8772 / 0.927889` |
| W4A8 final SNR / SSIM | `13.8640 / 0.928557` |
| act init SNR delta | `-0.0146 dB` |
| act recon SNR gain | `-0.0131 dB` |

Activation quantizer summary：

- `quant_modules=52`
- `activation_quantizers=52`
- `activation_delta_count=52`
- `activation_zero_point_count=52`
- `initialized_activation_quantizers=52`
- `learnable_activation_delta_count=52`
- `disabled_activation_quantizers=1`
- `delta_min=0.0040127067`
- `delta_max=0.0526451916`
- `zero_point_min=-0.0`
- `zero_point_max=158.0`
- `non_positive_delta_count=0`
- `non_positive_delta_elements=0`

Checkpoint verification：

| checkpoint | passed | final quant state | weight bits | level offenders | activation delta count | learnable activation deltas |
|---|---|---|---|---:|---:|---:|
| pre-act-recon | `true` | `weight_quant=true, act_quant=true` | `4bit=50, 8bit=2` | `0` | `52` | `52` |
| final | `true` | `weight_quant=true, act_quant=true` | `4bit=50, 8bit=2` | `0` | `52` | `52` |

Grid eval：

- Eval run dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_0_normalized_w4a8_activation_reconstruction/eval/20260509_221644_normalized_w4a8_tensor_a5000_grid478_seed20260507`
- 产物：
  - `config.json`
  - `metrics.json`
  - `summary.md`
  - `per_sample_metrics.jsonl`
- 行数：
  - `11950 = 478 x 25`

Overall normalized `478 x 25` grid：

| model | SNR mean | SNR median | SSIM mean | SSIM median |
|---|---:|---:|---:|---:|
| FP32 | `17.8329` | `18.1742` | `0.964330` | `0.978794` |
| E007 W4A32 final | `17.7856` | `18.1128` | `0.964137` | `0.978461` |
| NE000 W4A8 pre-act | `17.3727` | `17.7855` | `0.962674` | `0.977030` |
| NE000 W4A8 final | `17.4495` | `17.8777` | `0.962868` | `0.977292` |

主要 gap：

- NE000 W4A8 final 相对 FP32：
  - mean SNR：`-0.3834 dB`
  - mean SSIM：`-0.001462`
- NE000 W4A8 final 相对 E007 W4A32 final：
  - mean SNR：`-0.3361 dB`
  - mean SSIM：`-0.001269`
- Activation reconstruction 相对 pre-act：
  - mean SNR：`+0.0768 dB`
  - mean SSIM：`+0.000195`

By-source：

| source | rows | FP32 SNR mean | W4A8 pre SNR mean | W4A8 final SNR mean | final-FP32 SNR mean | W4A8 final SSIM mean |
|---|---:|---:|---:|---:|---:|---:|
| Anisotropic | `1875` | `22.0595` | `21.2895` | `21.2873` | `-0.7722` | `0.991631` |
| Kerry3D | `400` | `9.6085` | `9.5817` | `9.5704` | `-0.0381` | `0.948695` |
| Shots0001 | `9675` | `17.3538` | `16.9358` | `17.0315` | `-0.3223` | `0.957880` |

判断：

- 新协议下 tensor-wise W4A8 不再出现旧 raw 协议中明显崩坏的 full-grid 级别结果，但相对 E007 W4A32 仍有稳定掉点。
- 最大 by-source 掉点集中在 Anisotropic，其次是 Shots0001；Kerry3D 基本接近 FP32。
- Activation reconstruction 在 full grid 上有小幅正收益，但不足以消除 W4A8 与 W4A32 的 gap。
- 下一步进入 NE001 diagnostics，重点确认 activation delta 合法性、Conv2d/Linear fake-quant error 分布和 source/stage/role 差异。

## 2026-05-09 NE000_1 packed deployment equivalence 计划

在正式进入 NE001 diagnostics 前，增加 `NE000_1`，用于验证 NE000 及其 W4A32 对照是否能从恢复型 PyTorch checkpoint 转换为具体部署数值后保持等价。

范围边界：

- `NE000_1` 仍属于 NE000 baseline 真实性验证，不进入 activation diagnostics。
- 本实验只验证部署 artifact 的导出、恢复和 full-grid 指标对齐。
- activation int8 level 分布、fake-quant error、Conv2d/Linear/stage/role 诊断全部留给 NE001。
- 当前 `.pth` 是恢复型 checkpoint，包含 FP32 权重、AdaRound alpha、delta/zero_point 和 activation quantizer 状态；`NE000_1` 要验证 packed 整数权重和恢复后的 activation qparams 是否能复现 checkpoint 指标。

实验拆分：

| 子实验 | 输入 checkpoint | 量化状态 | 目的 |
|---|---|---|---|
| NE000_1a | `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1/checkpoints/quantized_scrn_brecq.pth` | W4A32，`weight_quant=true, act_quant=false` | 验证当前默认 E007 单卡 W4A32 weight-recon checkpoint 的 packed 权重导出和恢复是否仍对齐。 |
| NE000_1b | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_0_normalized_w4a8_activation_reconstruction/quant/20260509_213701_normalized_w4a8_tensor_a5000_1024cali_single_gpu1/checkpoints/quantized_scrn_brecq.pth` | W4A8，`weight_quant=true, act_quant=true` | 验证 NE000 W4A8 final checkpoint 在 packed 权重整数化并恢复 activation qparams 后是否仍能复现 NE000 final 指标。 |

关键解释：

- NE000 W4A8 不会生成一个新的独立 W4A32 weight-recon checkpoint。
- NE000 W4A8 的 W4 权重部分继承自 E007 单卡 W4A32；W4A8 中关闭 activation quantization 得到的 W4A32 行为只是同一权重的 sanity/reference，不是新的权重重建实验。
- 因此 NE000_1 必须分别验证：
  - E007 W4A32 checkpoint 的部署等价性；
  - NE000 W4A8 checkpoint 的部署等价性。

计划动作：

1. 对 E007 W4A32 final checkpoint 运行 `export_quantized_scrn.py`，输出 packed artifact：
   - `weights.bin`
   - `aux_fp32.bin`
   - `manifest.json`
   - `summary.json`
2. 对 NE000 W4A8 final checkpoint 运行 `export_quantized_scrn.py`，输出对应 packed artifact。
3. 使用 packed restore 路径恢复两个 artifact：
   - packed 权重从 `weights.bin` 读取整数值并反量化写回模型权重；
   - weight fake-quant 路径关闭，因为权重已经代表部署量化值；
   - W4A8 的 activation `delta/zero_point` 从 `aux_fp32.bin` 恢复，推理时保持 activation fake quant。
4. 在 normalized `478 x 25` grid 上分别评估：
   - E007 W4A32 checkpoint final vs W4A32 packed-restored；
   - NE000 W4A8 checkpoint final vs W4A8 packed-restored。
5. 记录 artifact size：
   - 原 `.pth` 文件大小；
   - `weights.bin`；
   - `aux_fp32.bin`；
   - raw deployment payload；
   - total export size；
   - estimated compression ratio。

必要输出：

- `NE000_1a` W4A32 packed export dir。
- `NE000_1a` W4A32 packed grid eval dir。
- `NE000_1b` W4A8 packed export dir。
- `NE000_1b` W4A8 packed grid eval dir。
- 每个 artifact 的 restore summary：
  - restored quantized layers；
  - restored non-quantized tensors；
  - restored activation quantizers；
  - final quant state。
- 每个 eval 的 row count 必须为 `11950`。
- 对齐指标：
  - checkpoint SNR/SSIM mean/median；
  - packed-restored SNR/SSIM mean/median；
  - packed minus checkpoint 的 mean/median SNR/SSIM delta；
  - prediction diff 的 MSE / mean abs / max abs。

验收标准：

- W4A32 packed-restored 与 E007 W4A32 checkpoint final 在 full-grid 上应高度对齐。
- W4A8 packed-restored 与 NE000 W4A8 checkpoint final 在 full-grid 上应高度对齐。
- 如果 W4A32 对齐但 W4A8 不对齐，优先怀疑 activation qparams restore 或 packed W4A8 eval 状态处理。
- 如果两者都不对齐，优先检查 packed weight integer export/restore 链路。
- 如果两者都对齐，则可认为 NE000 的好结果不只存在于恢复型 `.pth` fake-quant checkpoint 中，可以进入 NE001 diagnostics。

## 2026-05-09 NE000_1 packed grid evaluator

为 NE000_1 增加 packed deployment full-grid 等价评估入口。

新增：

- CLI：
  - `SCRN_BRECQ_app/scrn_brecq/cli/evaluate_packed_scrn_grid.py`
- Unit tests：
  - `SCRN_BRECQ_app/scrn_brecq/tests/test_evaluate_packed_scrn_grid.py`

功能：

- 加载 packed deployment artifact：
  - `manifest.json`
  - `weights.bin`
  - `aux_fp32.bin`
- 同时加载 reference quantized checkpoint。
- 在 fixed normalized grid 上对齐：
  - FP32 reference path；
  - checkpoint final quant path；
  - packed-restored path。
- packed-restored 推理语义：
  - `weight_quant=false`，因为权重已由 packed integer payload 恢复为部署量化权重；
  - `act_quant` 从 packed manifest final state / quant config 继承；
  - W4A8 仍保持 activation fake quant。

默认 grid：

- SNR settings：`-2,-1,1,5,10`
- Missing rates：`0.02,0.08,0.18,0.28,0.38`
- Seed：`20260507`

输出：

- `config.json`
- `metrics.json`
- `summary.md`
- `per_sample_metrics.jsonl`

逐样本字段：

- `fp32_snr_db / fp32_ssim`
- `checkpoint_snr_db / checkpoint_ssim`
- `packed_snr_db / packed_ssim`
- `packed_minus_checkpoint_snr_db / packed_minus_checkpoint_ssim`
- `packed_vs_checkpoint_mse`
- `packed_vs_checkpoint_mean_abs_diff`
- `packed_vs_checkpoint_max_abs_diff`

聚合维度：

- overall
- by source
- by SNR setting
- by missing rate
- by condition

验证：

- 先运行新增测试，确认缺少 CLI 时失败：
  - `ModuleNotFoundError: No module named 'SCRN_BRECQ_app.scrn_brecq.cli.evaluate_packed_scrn_grid'`
- 实现后通过：
  - `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_evaluate_packed_scrn_grid -v`
  - 5 tests OK
- 相关 packed 测试通过：
  - `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_evaluate_packed_scrn_grid SCRN_BRECQ_app.scrn_brecq.tests.test_evaluate_packed_scrn SCRN_BRECQ_app.scrn_brecq.tests.test_packed_deployment -v`
  - 13 tests OK
- CLI 检查通过：
- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/cli/evaluate_packed_scrn_grid.py`
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_packed_scrn_grid --help`

## 2026-05-09 NE000_1 packed deployment equivalence 结果

完成 NE000_1a / NE000_1b packed deployment equivalence。该实验只验证 packed artifact 导出、恢复和 full-grid 指标对齐，不进入 activation diagnostics。

预检：

- Branch：`main`
- Worktree：开始时 clean，本地领先 `origin/main` 20 个提交。
- Repo root：`/home/data1/hanwen/project/Project/SCRN_Quant`
- GPU：0/1/2/3 均空闲，仅 Xorg 占用约 `4 MiB`。
- 本次 full-grid eval 使用物理 GPU `1`，通过 `CUDA_VISIBLE_DEVICES=1` 暴露为进程内 `cuda:0`。

### NE000_1a W4A32 packed export / restore / grid eval

输入 checkpoint：

- `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1/checkpoints/quantized_scrn_brecq.pth`

Packed export：

- Export dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_1_packed_deployment_equivalence/w4a32_packed/e007_normalized_w4a32_single_gpu1_final`
- Artifact sizes：

| item | bytes | MiB |
|---|---:|---:|
| source `.pth` | `5,320,020` | `5.0736` |
| `weights.bin` | `213,632` | `0.2037` |
| `aux_fp32.bin` | `40,860` | `0.0390` |
| `manifest.json` | `94,254` | `0.0899` |
| `summary.json` | `2,695` | `0.0026` |
| raw deployment payload | `254,492` | `0.2427` |
| total export files | `348,746` | `0.3326` |

Export summary：

- `estimated_model_compression_ratio=6.765430740455495`
- `quantized_layer_count=52`
- `activation_quantizer_count=0`
- `final_quant_state={"weight_quant": true, "act_quant": false}`

Packed grid eval：

- Eval run dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_1_packed_deployment_equivalence/eval/20260509_225448_ne000_1a_w4a32_packed_grid478_seed20260507`
- Row count：
  - `11950`
- Restore summary：
  - `restored_quantized_layers=52`
  - `restored_non_quantized_tensors=70`
  - `restored_activation_quantizers=0`
  - `final_quant_state={"weight_quant": true, "act_quant": false}`

Overall：

| metric | checkpoint | packed-restored | packed-checkpoint |
|---|---:|---:|---:|
| SNR mean | `17.785581719` | `17.785577215` | `-0.000004504` |
| SNR median | `18.112756971` | `18.112657055` | `-0.000000972` |
| SSIM mean | `0.964137084` | `0.964137092` | `0.000000008` |
| SSIM median | `0.978461333` | `0.978461489` | `-0.000000009` |

Prediction diff：

- `packed_vs_checkpoint_mse_mean=3.9482866637798976e-10`
- `packed_vs_checkpoint_mean_abs_diff_mean=1.2906794042757771e-05`
- `packed_vs_checkpoint_max_abs_diff_mean=0.0002254785246958091`
- `packed_vs_checkpoint_max_abs_diff_max=0.0005853455513715744`

By-source：

| source | rows | checkpoint SNR mean | packed SNR mean | packed-checkpoint SNR mean | checkpoint SSIM mean | packed SSIM mean |
|---|---:|---:|---:|---:|---:|---:|
| Anisotropic | `1875` | `21.971754726` | `21.971745046` | `-0.000009680` | `0.992832075` | `0.992832056` |
| Kerry3D | `400` | `9.608162795` | `9.608165681` | `0.000002886` | `0.949887469` | `0.949887462` |
| Shots0001 | `9675` | `17.312392384` | `17.312388578` | `-0.000003806` | `0.959165170` | `0.959165184` |

### NE000_1b W4A8 packed export / restore / grid eval

输入 checkpoint：

- `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_0_normalized_w4a8_activation_reconstruction/quant/20260509_213701_normalized_w4a8_tensor_a5000_1024cali_single_gpu1/checkpoints/quantized_scrn_brecq.pth`

Packed export：

- Export dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_1_packed_deployment_equivalence/w4a8_packed/ne000_normalized_w4a8_tensor_a5000_final`
- Artifact sizes：

| item | bytes | MiB |
|---|---:|---:|
| source `.pth` | `5,359,500` | `5.1112` |
| `weights.bin` | `213,632` | `0.2037` |
| `aux_fp32.bin` | `41,276` | `0.0394` |
| `manifest.json` | `121,554` | `0.1159` |
| `summary.json` | `2,709` | `0.0026` |
| raw deployment payload | `254,908` | `0.2431` |
| total export files | `376,462` | `0.3590` |

Export summary：

- `estimated_model_compression_ratio=6.765430740455495`
- `quantized_layer_count=52`
- `activation_quantizer_count=52`
- `final_quant_state={"weight_quant": true, "act_quant": true}`

Packed grid eval：

- Eval run dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_1_packed_deployment_equivalence/eval/20260509_225927_ne000_1b_w4a8_packed_grid478_seed20260507`
- Row count：
  - `11950`
- Restore summary：
  - `restored_quantized_layers=52`
  - `restored_non_quantized_tensors=70`
  - `restored_activation_quantizers=52`
  - `final_quant_state={"weight_quant": true, "act_quant": true}`

Overall：

| metric | checkpoint | packed-restored | packed-checkpoint |
|---|---:|---:|---:|
| SNR mean | `17.449507172` | `17.449533809` | `0.000026637` |
| SNR median | `17.877688616` | `17.880496749` | `0.000000000` |
| SSIM mean | `0.962868418` | `0.962868814` | `0.000000396` |
| SSIM median | `0.977292375` | `0.977292636` | `0.000000000` |

Prediction diff：

- `packed_vs_checkpoint_mse_mean=3.7573586346050463e-07`
- `packed_vs_checkpoint_mean_abs_diff_mean=0.00012680872421293757`
- `packed_vs_checkpoint_max_abs_diff_mean=0.0025956151830850787`
- `packed_vs_checkpoint_max_abs_diff_max=0.018667370080947876`

By-source：

| source | rows | checkpoint SNR mean | packed SNR mean | packed-checkpoint SNR mean | checkpoint SSIM mean | packed SSIM mean |
|---|---:|---:|---:|---:|---:|---:|
| Anisotropic | `1875` | `21.287276634` | `21.287250178` | `-0.000026456` | `0.991630723` | `0.991630800` |
| Kerry3D | `400` | `9.570384048` | `9.570566086` | `0.000182038` | `0.948695016` | `0.948697731` |
| Shots0001 | `9675` | `17.031505261` | `17.031535762` | `0.000030501` | `0.957880308` | `0.957880670` |

判断：

- W4A32 packed-restored 与 E007 W4A32 checkpoint final 高度对齐，mean SNR delta 约 `-4.5e-06 dB`。
- W4A8 packed-restored 与 NE000 W4A8 checkpoint final 高度对齐，mean SNR delta 约 `+2.66e-05 dB`。
- 两者均远小于 `0.01 dB` 验收阈值。
- NE000 的 W4A8 好结果不仅存在于恢复型 `.pth` fake-quant checkpoint 中；packed 整数权重导出、activation qparams 恢复和部署视角 PyTorch restore 链路均通过 full-grid 等价验证。
- 可以进入 NE001 diagnostics。

## 2026-05-09 NE000 目录排序重命名

为让 NE000 baseline 目录在文件浏览器和 shell 排序中位于 `NE000_1_packed_deployment_equivalence` 之前，将 NE000 原始目录重命名：

- 旧目录：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_normalized_w4a8_activation_reconstruction`
- 新目录：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_0_normalized_w4a8_activation_reconstruction`

同步处理：

- 已对 `SCRN_BRECQ_app/` 下的文本型记录文件执行路径替换，包括 `.md`、`.json`、`.jsonl`、`.csv`、`.txt`。
- 已同步更新本日志和 `ACTIVATION_QUANTIZATION_LOG.md` 中的 NE000 checkpoint / eval 路径。
- 二进制 `.pth` checkpoint 内部历史 `quant_config.run_root` 不做原地改写；该字段不参与模型加载、验证或 grid evaluation。

校验：

- `find SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization -maxdepth 1 -type d -name 'NE000*' -printf '%f\n' | sort`
  - `NE000_0_normalized_w4a8_activation_reconstruction`
  - `NE000_1_packed_deployment_equivalence`

## 2026-05-09 NE000_2 归一化 W4A4 激活量化探针

完成 `NE000_2_normalized_w4a4_activation_reconstruction_probe`：在不进入 NE001 的前提下，从当前默认 E007 单卡 W4A32 checkpoint 出发，只把 activation bitwidth 从 A8 降到 A4，测试 normalized 协议下 tensor-wise W4A4 是否仍可用。

预检：

- Branch：`main`
- Worktree：clean
- Repo root：`/home/data1/hanwen/project/Project/SCRN_Quant`
- GPU：`0/1/2/3` 仅 Xorg 低占用；本次使用物理 GPU `1`

输入和执行说明：

- 原始 E007 W4A32 checkpoint：
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1/checkpoints/quantized_scrn_brecq.pth`
- A4 seed metadata checkpoint：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_2_normalized_w4a4_activation_reconstruction_probe/inputs/e007_w4a32_nbitsa4_metadata_seed.pth`
- 说明：
  - `activation_only_quantize_scrn.py` 当前 CLI 没有 `--n-bits-a` 参数，且 E007 checkpoint 的 `quant_config.n_bits_a=8` 会覆盖 `--config`。
  - 因此本次生成仅修改 `quant_config.n_bits_a: 8 -> 4` 的 seed metadata checkpoint；权重张量保持 E007 不变。
  - 这是为了执行 W4A4 探针，不是新的 W4 weight reconstruction。

Quant run：

- Run dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_2_normalized_w4a4_activation_reconstruction_probe/quant/20260509_232313_normalized_w4a4_tensor_a5000_1024cali_single_gpu1`
- Pre-act checkpoint：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_2_normalized_w4a4_activation_reconstruction_probe/quant/20260509_232313_normalized_w4a4_tensor_a5000_1024cali_single_gpu1/checkpoints/quantized_scrn_brecq_pre_act_recon.pth`
- Final checkpoint：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_2_normalized_w4a4_activation_reconstruction_probe/quant/20260509_232313_normalized_w4a4_tensor_a5000_1024cali_single_gpu1/checkpoints/quantized_scrn_brecq.pth`
- 核心配置：
  - `n_bits_w=4`
  - `n_bits_a=4`
  - `activation_granularity=tensor`
  - `activation_range_method=none`
  - `num_samples=1024`
  - `batch_size=16`
  - `init_batch_size=64`
  - `iters_a=5000`
  - `activation_lr=0.0004`
  - `lp_norm=2.4`

运行时间和 single-sample sanity：

- Activation initialization：`24.8694 s`
- Activation reconstruction：`1328.2162 s` / `22.1369 min`
- Total elapsed：`1356.1941 s` / `22.6032 min`
- `post_weight_snr=13.8918`
- `pre_act_snr=13.1723`
- `act_init_delta=-0.7195`
- `quant_post_act_recon_snr_db=13.2697`
- `quant_act_recon_snr_gain_db=0.0974`
- `non_positive_delta_count=0`

Checkpoint verification：

- Pre-act：
  - `passed=true`
  - `final_quant_state={"weight_quant": true, "act_quant": true}`
  - `n_bits_a=4`
  - `activation_delta_count=52`
  - `activation_zero_point_count=52`
  - `weight_bit_counts={"4": 50, "8": 2}`
  - `level_offender_count=0`
- Final：
  - `passed=true`
  - `final_quant_state={"weight_quant": true, "act_quant": true}`
  - `n_bits_a=4`
  - `activation_delta_count=52`
  - `activation_zero_point_count=52`
  - `weight_bit_counts={"4": 50, "8": 2}`
  - `level_offender_count=0`
- 额外 delta 检查：
  - Pre：`52` 个 activation delta，`non_positive=0`，`min=0.0088924468`，`max=0.5482574105`
  - Final：`52` 个 activation delta，`non_positive=0`，`min=0.0088924468`，`max=0.4546860754`

Grid eval：

- Eval run dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_2_normalized_w4a4_activation_reconstruction_probe/eval/20260509_234948_normalized_w4a4_tensor_a5000_grid478_seed20260507`
- Row count：`11950`
- Eval elapsed：`238.4515 s`
- Test protocol：
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`
  - SNR settings：`-2,-1,1,5,10`
  - missing rates：`0.02,0.08,0.18,0.28,0.38`
  - seed：`20260507`

Overall：

| model | SNR mean | SNR median | SSIM mean | SSIM median |
|---|---:|---:|---:|---:|
| FP32 | `17.832885` | `18.174198` | `0.964330` | `0.978794` |
| E007 W4A32 final | `17.785582` | `18.112757` | `0.964137` | `0.978461` |
| NE000 W4A8 final | `17.449507` | `17.877689` | `0.962868` | `0.977292` |
| NE000_2 W4A4 pre-act | `11.172733` | `11.157722` | `0.941492` | `0.955772` |
| NE000_2 W4A4 final | `12.914963` | `13.118019` | `0.939563` | `0.954078` |

W4A4 final gaps：

- vs FP32：
  - SNR mean / median：`-4.917922 / -4.152035`
  - SSIM mean / median：`-0.024767 / -0.025209`
- vs E007 W4A32：
  - SNR mean / median：`-4.870618 / -4.994738`
  - SSIM mean / median：`-0.024574 / -0.024384`
- vs NE000 W4A8：
  - SNR mean / median：`-4.534544 / -4.759669`
  - SSIM mean / median：`-0.023305 / -0.023215`
- Activation reconstruction gain：
  - SNR mean / median：`+1.742231 / +1.861192`
  - SSIM mean / median：`-0.001929 / -0.002785`

By-source：

| source | rows | FP32 SNR mean | W4A4 pre SNR mean | W4A4 final SNR mean | final-FP32 SNR mean | W4A4 final SSIM mean |
|---|---:|---:|---:|---:|---:|---:|
| Anisotropic | `1875` | `22.059463` | `10.712972` | `13.280217` | `-8.779246` | `0.966844` |
| Kerry3D | `400` | `9.608488` | `9.108445` | `9.033177` | `-0.575312` | `0.929975` |
| Shots0001 | `9675` | `17.353808` | `11.347179` | `13.004665` | `-4.349143` | `0.934673` |

判断：

- W4A4 checkpoint 状态合法，确认是 `n_bits_a=4` 且 52 个 activation quantizer 均已保存和恢复。
- W4A4 不像旧 raw 协议 W4A8 那样完全崩坏，但相对 W4A8 掉点约 `4.53 dB` mean SNR，不应视为当前可部署候选。
- Activation reconstruction 对 W4A4 有明显 SNR 正收益（`+1.74 dB` mean），但 SSIM 略降，说明 A4 的 reconstruction 目标和视觉结构指标存在张力。
- 暂不做 W4A4 packed export；NE001 主线仍应先诊断 NE000 W4A8，W4A4 作为后续 range/granularity/mixed precision 的强压力对照保留。

## 2026-05-10 NE000_1c W4A4 packed deployment 等价验证

完成 `NE000_1c`：将 `NE000_2` W4A4 final checkpoint 导出为 packed deployment artifact，并在 normalized `478 x 25` grid 上验证 packed-restored 推理与原 W4A4 checkpoint final 是否等价。

Packed export：

- 输入 checkpoint：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_2_normalized_w4a4_activation_reconstruction_probe/quant/20260509_232313_normalized_w4a4_tensor_a5000_1024cali_single_gpu1/checkpoints/quantized_scrn_brecq.pth`
- Export dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_1_packed_deployment_equivalence/w4a4_packed/ne000_2_normalized_w4a4_tensor_a5000_final`
- Manifest / summary：
  - `final_quant_state={"weight_quant": true, "act_quant": true}`
  - `quant_config.n_bits_a=4`
  - `quantized_layer_count=52`
  - `activation_quantizer_count=52`
  - `estimated_model_compression_ratio=6.765430740455495`

Artifact sizes：

| item | bytes | MiB |
|---|---:|---:|
| source `.pth` | `5359500` | `5.1112` |
| `weights.bin` | `213632` | `0.2037` |
| `aux_fp32.bin` | `41276` | `0.0394` |
| `manifest.json` | `121590` | `0.1160` |
| `summary.json` | `2708` | `0.0026` |
| raw deployment payload | `254908` | `0.2431` |
| export files including `summary.json` | `379206` | `0.3616` |

Packed grid eval：

- Eval run dir：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_1_packed_deployment_equivalence/eval/20260510_001507_ne000_1c_w4a4_packed_grid478_seed20260507`
- Row count：`11950`
- Eval elapsed：`237.7457 s`
- Restore summary：
  - `restored_quantized_layers=52`
  - `restored_non_quantized_tensors=70`
  - `restored_activation_quantizers=52`
  - `final_quant_state={"weight_quant": true, "act_quant": true}`

Overall equivalence：

| metric | checkpoint | packed-restored | packed-checkpoint |
|---|---:|---:|---:|
| SNR mean | `12.914963390` | `12.915036109` | `0.000072719` |
| SNR median | `13.118019299` | `13.120819498` | `0.000000000` |
| SSIM mean | `0.939563064` | `0.939564812` | `0.000001748` |
| SSIM median | `0.954077528` | `0.954055301` | `0.000000000` |

Prediction diff：

- `packed_vs_checkpoint_mse_mean=3.893126945986073e-06`
- `packed_vs_checkpoint_mean_abs_diff_mean=0.0003266090435713198`
- `packed_vs_checkpoint_max_abs_diff_mean=0.009579205599552113`
- `packed_vs_checkpoint_max_abs_diff_max=0.07725390791893005`

By-source：

| source | rows | checkpoint SNR mean | packed SNR mean | packed-checkpoint SNR mean | checkpoint SSIM mean | packed SSIM mean |
|---|---:|---:|---:|---:|---:|---:|
| Anisotropic | `1875` | `13.280216794` | `13.280417211` | `0.000200416` | `0.966843563` | `0.966847291` |
| Kerry3D | `400` | `9.033176792` | `9.033380279` | `0.000203487` | `0.929975485` | `0.929969113` |
| Shots0001 | `9675` | `13.004665148` | `13.004707713` | `0.000042565` | `0.934672531` | `0.934674231` |

后续核心对比对象：

| object | SNR mean | SNR median | SSIM mean | SSIM median | role |
|---|---:|---:|---:|---:|---|
| FP32 | `17.832885` | `18.174198` | `0.964330` | `0.978794` | normalized upper baseline |
| E007 W4A32 final | `17.785582` | `18.112757` | `0.964137` | `0.978461` | default W4 weight-only baseline |
| NE000 W4A8 final | `17.449507` | `17.877689` | `0.962868` | `0.977292` | current usable activation quantization baseline |
| NE000_2 W4A4 final | `12.914963` | `13.118019` | `0.939563` | `0.954078` | A4 pressure reference, not deployment candidate |

判断：

- W4A4 packed-restored 与 NE000_2 W4A4 checkpoint final 高度对齐，mean SNR delta 约 `7.27e-05 dB`，远小于 `0.01 dB` 阈值。
- W4A4 是合法、可导出、可恢复的 A4 activation 压力对照，但当前 full-grid 指标明显低于 W4A8，不作为部署候选。
- W4A8 是当前最重要的可用 activation quantization baseline；NE001 主线仍应优先诊断 NE000 W4A8。

## 2026-05-10 四个核心结果与重建前后结论汇总

本节补充记录 NE000 系列到目前为止最重要的横向结论。这里的“四个核心结果”指：

1. normalized FP32 baseline；
2. E007 W4A32 final，当前默认 weight-only baseline；
3. NE000 W4A8 final，当前最重要的可用 activation quantization baseline；
4. NE000_2 W4A4 final，合法可部署恢复的 A4 压力对照。

### 四个核心结果的最终指标

| object | SNR mean | SNR median | SSIM mean | SSIM median | FP32 SNR gap mean |
|---|---:|---:|---:|---:|---:|
| FP32 | `17.832885` | `18.174198` | `0.964330` | `0.978794` | `0.000000` |
| E007 W4A32 final | `17.785582` | `18.112757` | `0.964137` | `0.978461` | `-0.047304` |
| NE000 W4A8 final | `17.449507` | `17.877689` | `0.962868` | `0.977292` | `-0.383378` |
| NE000_2 W4A4 final | `12.914963` | `13.118019` | `0.939563` | `0.954078` | `-4.917922` |

跨对象差距：

| comparison | SNR mean delta | SNR median delta | SSIM mean delta |
|---|---:|---:|---:|
| E007 W4A32 final - FP32 | `-0.047304` | `-0.061441` | `-0.000193` |
| NE000 W4A8 final - E007 W4A32 final | `-0.336075` | `-0.235068` | `-0.001269` |
| NE000_2 W4A4 final - NE000 W4A8 final | `-4.534544` | `-4.759669` | `-0.023305` |
| NE000_2 W4A4 final - E007 W4A32 final | `-4.870618` | `-4.994738` | `-0.024574` |

### 三个重建实验的 pre/post 改变量

这三个模型都涉及重建，pre/post 改变量是判断该阶段是否真正有效的关键指标。

| model | pre SNR mean | post SNR mean | SNR gain | pre SSIM mean | post SSIM mean | SSIM gain | post-FP32 SNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| E007 W4A32 weight recon | `16.614250` | `17.785582` | `+1.171332` | `0.935462` | `0.964137` | `+0.028675` | `-0.047304` |
| NE000 W4A8 act recon | `17.372734` | `17.449507` | `+0.076773` | `0.962674` | `0.962868` | `+0.000195` | `-0.383378` |
| NE000_2 W4A4 act recon | `11.172733` | `12.914963` | `+1.742231` | `0.941492` | `0.939563` | `-0.001929` | `-4.917922` |

Median 视角：

| model | pre SNR median | post SNR median | SNR gain median | pre SSIM median | post SSIM median | SSIM gain median |
|---|---:|---:|---:|---:|---:|---:|
| E007 W4A32 weight recon | `17.177490` | `18.112757` | `+0.805456` | `0.949638` | `0.978461` | `+0.023764` |
| NE000 W4A8 act recon | `17.785509` | `17.877689` | `+0.006535` | `0.977030` | `0.977292` | `+0.000069` |
| NE000_2 W4A4 act recon | `11.157722` | `13.118019` | `+1.861192` | `0.955772` | `0.954078` | `-0.002785` |

结论：

- E007 W4A32 的 weight reconstruction 是强有效步骤：mean SNR `+1.17 dB`，mean SSIM `+0.0287`，最终距离 FP32 只剩 `0.047 dB`，因此它是后续所有 activation quantization 的稳固起点。
- NE000 W4A8 的 activation reconstruction 是小幅精修：A8 init 本身已经接近 final，mean SNR 只再提升 `0.0768 dB`，说明新 normalized 协议下 A8 的主要改善来自数据/幅值协议和合法量化状态，而不是 activation reconstruction 大幅救回。
- NE000_2 W4A4 的 activation reconstruction 对 SNR 有强正收益：mean SNR `+1.742 dB`，但 SSIM 反而下降 `0.00193`，说明当前 reconstruction 目标对 A4 会优先优化能量误差，不一定同步改善结构相似性。
- A4 final 虽然合法且可恢复，但仍比 W4A8 final 低 `4.53 dB` mean SNR；A4 后续需要 range/granularity/mixed precision 的专门路线，不能直接当部署候选。

### 按 source 看重建收益

E007 W4A32：

| source | rows | pre SNR | post SNR | SNR gain | post-FP32 SNR | pre SSIM | post SSIM | SSIM gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Anisotropic | `1875` | `20.003475` | `21.971755` | `+1.968280` | `-0.087708` | `0.981766` | `0.992832` | `+0.011066` |
| Kerry3D | `400` | `9.437826` | `9.608163` | `+0.170337` | `-0.000326` | `0.950352` | `0.949887` | `-0.000465` |
| Shots0001 | `9675` | `16.254123` | `17.312392` | `+1.058270` | `-0.041416` | `0.925873` | `0.959165` | `+0.033292` |

NE000 W4A8：

| source | rows | pre SNR | post SNR | SNR gain | post-FP32 SNR | pre SSIM | post SSIM | SSIM gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Anisotropic | `1875` | `21.289510` | `21.287277` | `-0.002234` | `-0.772186` | `0.991605` | `0.991631` | `+0.000026` |
| Kerry3D | `400` | `9.581705` | `9.570384` | `-0.011321` | `-0.038104` | `0.948793` | `0.948695` | `-0.000098` |
| Shots0001 | `9675` | `16.935779` | `17.031505` | `+0.095726` | `-0.322303` | `0.957641` | `0.957880` | `+0.000239` |

NE000_2 W4A4：

| source | rows | pre SNR | post SNR | SNR gain | post-FP32 SNR | pre SSIM | post SSIM | SSIM gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Anisotropic | `1875` | `10.712972` | `13.280217` | `+2.567245` | `-8.779246` | `0.969940` | `0.966844` | `-0.003096` |
| Kerry3D | `400` | `9.108445` | `9.033177` | `-0.075269` | `-0.575312` | `0.939263` | `0.929975` | `-0.009288` |
| Shots0001 | `9675` | `11.347179` | `13.004665` | `+1.657486` | `-4.349143` | `0.936071` | `0.934673` | `-0.001399` |

Source 结论：

- `Anisotropic` 是 activation bitwidth 压力最强的 source：W4A8 final 已有 `-0.772 dB` gap，W4A4 final 扩大到 `-8.779 dB`。后续 NE001/NE004 应重点检查它对应样本上的 activation range、Conv2d fake-quant error 和 stage 输出误差。
- `Kerry3D` 对 activation bitwidth 最不敏感：W4A8 gap 只有 `-0.038 dB`，W4A4 gap 也只有 `-0.575 dB`，说明 A4 崩坏不是所有 source 的统一现象。
- `Shots0001` 是总体结论的主导 source，因为样本数 `9675/11950`；W4A8 在这里仍有 `-0.322 dB` gap，W4A4 为 `-4.349 dB`。
- W4A8 activation reconstruction 对 Anisotropic/Kerry3D 的 SNR 没有正收益，主要收益来自 Shots0001；因此 NE001 不应只看 overall gain。
- W4A4 activation reconstruction 对 Anisotropic 和 Shots0001 的 SNR 有大幅恢复，但 Kerry3D 轻微变差，并且三个 source 的 SSIM gain 都为负，说明 A4 的重建目标存在 source-dependent tradeoff。

### packed deployment 等价链路

| packed object | checkpoint SNR mean | packed SNR mean | packed-checkpoint SNR mean | pred MSE mean | max abs diff mean |
|---|---:|---:|---:|---:|---:|
| E007 W4A32 packed | `17.785581719` | `17.785577215` | `-0.000004504` | `3.94828666378e-10` | `0.000225479` |
| NE000 W4A8 packed | `17.449507172` | `17.449533809` | `0.000026637` | `3.75735863461e-07` | `0.002595615` |
| NE000_2 W4A4 packed | `12.914963390` | `12.915036109` | `0.000072719` | `3.89312694599e-06` | `0.009579206` |

Deployment 结论：

- W4A32、W4A8、W4A4 三个量化模型都已经通过 packed export / restore / full-grid equivalence；这些结果不只是 `.pth` fake-quant checkpoint 内的偶然状态。
- packed-vs-checkpoint 差异随 activation bitwidth 变低而增大，但三者 mean SNR delta 都远低于 `0.01 dB` 验收阈值。
- 后续比较精度时可以优先引用 checkpoint final 的 grid 指标；讨论部署链路时引用 packed equivalence 证明。

### 模型大小与部署存储对比

Checkpoint / estimated packed size：

| object | checkpoint MiB | estimated packed MiB | estimated compression | quantized weight MiB | weight compression | act quantizers |
|---|---:|---:|---:|---:|---:|---:|
| E007 W4A32 | `5.0736` | `0.2427` | `6.7654x` | `0.2037` | `7.9784x` | `0` |
| NE000 W4A8 | `5.1112` | `0.2427` | `6.7654x` | `0.2037` | `7.9784x` | `52` |
| NE000_2 W4A4 | `5.1112` | `0.2427` | `6.7654x` | `0.2037` | `7.9784x` | `52` |

Packed export files：

| object | `weights.bin` MiB | `aux_fp32.bin` MiB | raw payload MiB | `manifest.json` MiB | `summary.json` MiB | total export MiB |
|---|---:|---:|---:|---:|---:|---:|
| E007 W4A32 | `0.2037` | `0.0390` | `0.2427` | `0.0899` | `0.0026` | `0.3352` |
| NE000 W4A8 | `0.2037` | `0.0394` | `0.2431` | `0.1159` | `0.0026` | `0.3616` |
| NE000_2 W4A4 | `0.2037` | `0.0394` | `0.2431` | `0.1160` | `0.0026` | `0.3616` |

Parameter / layer accounting：

| object | base params | quantized weight params | non-quantized params | 4bit layers | 8bit layers |
|---|---:|---:|---:|---:|---:|
| E007 W4A32 | `430437` | `426112` | `4325` | `50` | `2` |
| NE000 W4A8 | `430437` | `426112` | `4325` | `50` | `2` |
| NE000_2 W4A4 | `430437` | `426112` | `4325` | `50` | `2` |

Size 结论：

- 三个量化模型的权重存储完全一致：`weights.bin=0.2037 MiB`，因为它们共享同一套 W4 weight reconstruction，且 head/tail 保持 8bit。
- W4A8/W4A4 的 estimated packed model size 与 W4A32 基本相同，都是 `0.2427 MiB`；当前 packed 估算主要统计权重和必要 FP32 aux，不把 activation bitwidth 带来的运行时激活张量存储作为模型文件大小收益。
- W4A8/W4A4 比 W4A32 多出 activation qparams，因此 checkpoint 从 `5.0736 MiB` 增至 `5.1112 MiB`，`aux_fp32.bin` 从 `0.0390 MiB` 增至 `0.0394 MiB`。
- W4A8 与 W4A4 的模型文件大小几乎相同，因为两者都保存 52 个 activation `delta/zero_point` 浮点参数；activation bitwidth 影响推理 fake quant 网格和精度，不显著改变当前 packed artifact 的文件体积。
- 实际 export 文件总大小大于 raw payload，是因为 `manifest.json` 和 `summary.json` 记录了层级元数据、量化状态和可复现实验信息；部署时真正的二进制 payload 主要是 `weights.bin + aux_fp32.bin`。

### 当前研究判断和后续优先级

- 默认对比表应固定为 FP32 / E007 W4A32 / NE000 W4A8 / NE000_2 W4A4。
- 后续 NE001 diagnostics 的主对象仍是 NE000 W4A8，因为它是当前最接近可用部署的 activation quantization 结果。
- NE000_2 W4A4 应作为压力对照进入诊断视野，但不应把主线改成 A4 优化；A4 的主要价值是放大 activation quantization 热点。
- 下一步诊断必须同时报告 pre/post reconstruction，而不是只报告 final。特别是 W4A8 的 final gain 很小，单看 final 容易误判 activation reconstruction 的作用。
- 如果 NE001 发现 W4A8 和 W4A4 的误差热点高度重合，NE004/NE006 的 Conv2d sensitivity 和结构化 granularity 搜索可能同时解释 A8 gap 和 A4 崩坏。
- 如果 W4A4 的热点与 W4A8 不重合，则 A4 应另开 A4-specific range/granularity/mixed precision 路线，不能用 W4A8 的诊断结论直接外推。

## 2026-05-10 NE001-NE006 W4A4 主线实验计划

在正式开始 NE001 前，重新记录后续实验主线。NE000 和 NE000_1/2 已经确认：W4A8 在 normalized 协议下表现很好，W4A4 合法、可恢复、可 packed 等价，但仍有约 `4.53 dB` mean SNR 的 A4 activation gap。因此后续主对象从“修 W4A8”调整为“以 W4A8 为成功参照，重点攻 W4A4”。

本节是对上一节“当前研究判断和后续优先级”的更新：旧文字保留为历史判断，本节之后执行 NE001-NE006 时以 W4A4 主线为准。

### 固定实验对象

| object | role | 使用方式 |
|---|---|---|
| FP32 | 上限基准 | 不做 activation quantization 实验，只作为 full-grid 指标、逐样本输出和可视化上限参考。 |
| E007 W4A32 final | 权重量化起点 | 后续 A4/A8 activation quantization 都从该 checkpoint 出发；同时作为 weight-only gap 参考。 |
| NE000 W4A8 final | 成功参照与护栏 | 不作为主要优化对象；用于判断 W4A4 热点是否和 A8 一致，并验证新策略不能明显破坏 A8。 |
| NE000_2 W4A4 final | 主研究对象 | 后续诊断、敏感性、range、granularity 和 mixed precision 优先围绕它展开。 |

### 全局执行口径

- 数据协议固定为 `paper5_energy_filtered_perpatch_absmax`。
- 正式判断固定使用 normalized `478 x 25` grid：SNR settings `-2,-1,1,5,10`，missing rates `0.02,0.08,0.18,0.28,0.38`，seed `20260507`。
- 每个涉及 reconstruction 的实验必须报告 pre/post 改变量，不能只报告 final。
- 单样本和代表图只作为解释材料，不能替代 full-grid mean/median、by-source、by-condition 指标。
- 运行资源优先使用 GPU；单卡默认优先级仍为 `1 -> 2 -> 3 -> 0`，可并行 sweep 时优先做 job-level 多 GPU 并行。
- 所有新产物只写入 `SCRN_BRECQ_app/` 下；涉及 `scrn_brecq/` 代码或文档时同步更新两份日志并提交。

### 后续实验安排

| 编号 | 实验对象 | 实验内容 | 实验方法 | 实验目标 | 主要输出 / 决策 |
|---|---|---|---|---|---|
| NE001 | 主：NE000_2 W4A4 pre/final；参照：NE000 W4A8 pre/final、E007 W4A32、FP32 | W4A4-centered activation diagnostics。 | 对 W4A4 pre-act 与 final checkpoint 跑 activation quantizer 诊断，统计 activation qparams、delta/zero_point 合法性、fake-quant error、effective levels、module type、stage、role、source 和 condition 分布；同时抽 W4A8 做同口径对照。 | 找到 W4A4 主要误差热点，判断是否和 W4A8 热点重合，明确后续该优先修 Conv2d、Linear/transformer、stage output 还是特定 source/condition。 | quantizer 清单、热点表、by-source/stage/role error ranking、W4A4 vs W4A8 热点重合度；决定 NE004 的 sensitivity 分组。 |
| NE002 | 主：NE000_2 W4A4；参照：NE000 W4A8、E007 W4A32 packed | 合法状态和 checkpoint / deployment sanity。 | 复核 `verify_quantized_scrn`、packed manifest、summary、reload 后 final quant state、activation bitwidth、52 个 activation quantizer、`non_positive_delta_count=0`、checkpoint-vs-packed 等价；必要时补充 state toggle sanity。 | 排除“结果看起来好/坏是因为未真正启用 activation quantization、reload 状态错误、packed restore 不一致或 bitwidth 配置错位”。 | W4A4/A8 legality 表；若发现状态问题，先修状态再继续；若无问题，确认 NE003-NE006 都可使用当前 checkpoint。 |
| NE003 | 四个核心对象 | 固定评估和代表图口径。 | 复核 full-grid 聚合、by-source、by-SNR、by-missing-rate、by-condition；选择固定代表样本，使用 seismic colormap 和反归一化/幅值一致显示，展示 FP32 / W4A32 / W4A8 / W4A4 以及误差图。 | 让后续所有改进都有统一可视化和数值解释口径，避免单样本偶然性或显示尺度误导。 | 代表样本清单、图像输出目录、四对象同图对比、W4A4 最差/中位/最好样本案例；后续报告固定复用。 |
| NE004 | 主：W4A4；参照：W4A8 | Activation quantizer 敏感性。 | 基于 NE001 热点做分组关闭/保留实验，优先按 Conv2d vs Linear/transformer、split_proj、merge_proj、stage_output_conv、stage1-5、source-sensitive groups 分组；先跑小规模 scout，再对关键组跑 full-grid。 | 判断 W4A4 的 `4.53 dB` gap 由哪些 activation quantizer 组主导，并验证旧协议“Conv2d 多层累积误差”在 A4 下是否仍成立。 | 分组恢复表、group ranking、source-specific recovery；决定 NE005 是否值得做 range，NE006 应优先处理哪些结构。 |
| NE005 | 主：W4A4 的 NE004 目标分组 | Range、clipping、outlier 和 calibration 代表性检查。 | 在不改变 granularity 的前提下，测试 tensor-wise max、percentile、MSE-grid、source/condition-aware calibration、目标分组 range variants；只对 NE004 指出的关键组做完整验证。 | 判断 A4 主要问题是否能通过 range / clipping / calibration 修复，还是必须进入结构化粒度或 mixed precision。 | range variant 对比表；若 tensor-wise range 有效，进入更精细 range；若无效，收束到 NE006 粒度/混精度。 |
| NE006 | 主：W4A4；护栏：W4A8 | 结构化 activation granularity / mixed precision 搜索。 | 从旧 E006 强候选开始：all Conv2d per-channel、split_proj + merge_proj + stage_output_conv per-channel、对应 group-wise `g4`、stage_output_conv `g4`；在 A4 下必要时加入 selective A8 fallback 或 selective FP32。 | 尝试把 W4A4 从压力对照推进到可用候选；同时确认策略不会明显破坏 W4A8。 | W4A4 候选策略表、pre/post 指标、packed 可行性判断；若出现接近 W4A8 的 A4 候选，再做 packed export/equivalence。 |

### 阶段性判断规则

- 如果 NE001 显示 W4A4 和 W4A8 热点高度重合，后续优先复用旧 E006 的结构化 Conv2d 粒度路线，只是把目标 bitwidth 改成 A4。
- 如果 W4A4 热点明显不同于 W4A8，后续不要把 W4A8 的好结果外推到 A4；NE004-NE006 必须以 A4-specific 分组和 source-specific 结论为准。
- 如果 W4A4 的 SSIM 持续随 reconstruction 或 range 改善而下降，需要单独记录 SNR/SSIM tradeoff，并考虑 reconstruction loss 或代表样本可视化，而不是只追求 SNR。
- W4A8 目前不需要主动优化，但每个最终候选策略都应至少用 W4A8 做护栏验证，防止为了 A4 引入会破坏 A8 的工程默认配置。

## 2026-05-10 NE001 full-reference activation diagnostics

完成 `NE001`：以 `NE000_2 W4A4 pre/final` 为主诊断对象，同时纳入 `NE000 W4A8 pre/final`、`E007 W4A32 pre/final` 和 FP32 上限指标，建立完整参考链条。

### 运行口径

- Branch：`main`
- 开始时 worktree：clean，本地领先 `origin/main` 28 个提交。
- 数据协议：`paper5_energy_filtered_perpatch_absmax`
- Calibration：
  - `SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- Diagnostics 参数：
  - `num_samples=1024`
  - `batch_size=16`
  - `num_workers=0`
  - `seed=1005`
- 输出根目录：
  - `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE001_w4a4_centered_activation_diagnostics/diagnostics`

GPU 说明：

- 预检时 GPU `0/1/2/3` 基本空闲，默认优先选择 GPU `1`。
- 首次按计划使用 GPU `1` 跑 `W4A4 pre-act` diagnostics 时，sandbox 外 CUDA 可用，但现有 `diagnose_activation_quantization` 会把 1024 个 calibration samples 整批送入一次 forward；`batch_size=16` 只影响 DataLoader，不影响 diagnostics forward。
- GPU run 在 full-batch activation diagnostics 中 OOM：
  - 错误点：`_per_channel_absmax_stats`
  - 现象：尝试额外分配约 `4.00 GiB`，进程已占约 `12.38 GiB`
- 为保持“不改代码、不降 num_samples”的计划约束，本次 6 个 diagnostics 全部改用 CPU fallback 完成，并在 run name 中显式标注 `cpu_fallback`。

### Diagnostics run directories

| object | run dir |
|---|---|
| W4A4 pre-act | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE001_w4a4_centered_activation_diagnostics/diagnostics/20260510_015710_ne001a_w4a4_pre_act_diagnostics_1024cali_cpu_fallback` |
| W4A4 final | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE001_w4a4_centered_activation_diagnostics/diagnostics/20260510_020106_ne001b_w4a4_final_diagnostics_1024cali_cpu_fallback` |
| W4A8 pre-act | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE001_w4a4_centered_activation_diagnostics/diagnostics/20260510_020454_ne001c_w4a8_pre_act_reference_diagnostics_1024cali_cpu_fallback` |
| W4A8 final | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE001_w4a4_centered_activation_diagnostics/diagnostics/20260510_020847_ne001d_w4a8_final_reference_diagnostics_1024cali_cpu_fallback` |
| W4A32 pre-recon | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE001_w4a4_centered_activation_diagnostics/diagnostics/20260510_021257_ne001e_w4a32_pre_recon_reference_diagnostics_1024cali_cpu_fallback` |
| W4A32 final | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE001_w4a4_centered_activation_diagnostics/diagnostics/20260510_021648_ne001f_w4a32_final_reference_diagnostics_1024cali_cpu_fallback` |

每个 run 均生成：

- `config.json`
- `summary.json`
- `summary.md`
- `quantizers.csv`
- `activation_stats.jsonl`
- `offender_layers.json`

### 八个对象的 fixed-grid 指标

FP32 checkpoint：

- `SCRN_BRECQ_app/scrn_repro/runs/train/20260508_194718_paper5_energy_filtered_perpatch_absmax_10750_ddp3_seed20260425_nodecay_lr1e-3/checkpoints/best.pth`

| object | SNR mean | SNR median | SSIM mean | SSIM median | post/pre gain SNR mean |
|---|---:|---:|---:|---:|---:|
| FP32 | `17.832885` | `18.174198` | `0.964330` | `0.978794` | N/A |
| E007 W4A32 pre-recon | `16.614250` | `17.177490` | `0.935462` | `0.949638` | N/A |
| E007 W4A32 final | `17.785582` | `18.112757` | `0.964137` | `0.978461` | `+1.171332` |
| NE000 W4A8 pre-act | `17.372734` | `17.785509` | `0.962674` | `0.977030` | N/A |
| NE000 W4A8 final | `17.449507` | `17.877689` | `0.962868` | `0.977292` | `+0.076773` |
| NE000_2 W4A4 pre-act | `11.172733` | `11.157722` | `0.941492` | `0.955772` | N/A |
| NE000_2 W4A4 final | `12.914963` | `13.118019` | `0.939563` | `0.954078` | `+1.742231` |

### Legality / diagnostics summary

| object | final state | act bits | delta/init | non-positive delta | delta min/max | fake-quant MSE mean/max | effective levels min/max |
|---|---|---:|---:|---:|---:|---:|---:|
| W4A4 pre | `W=true A=true` | `4/8` | `52/52` | `0` | `0.00889245 / 0.548257` | `0.00612955 / 0.027745` | `13 / 246` |
| W4A4 final | `W=true A=true` | `4/8` | `52/52` | `0` | `0.00889245 / 0.454686` | `0.00468151 / 0.0464743` | `14 / 235` |
| W4A8 pre | `W=true A=true` | `8` | `52/52` | `0` | `0.00416813 / 0.0480133` | `5.65246e-05 / 0.000505416` | `168 / 256` |
| W4A8 final | `W=true A=true` | `8` | `52/52` | `0` | `0.00401271 / 0.0526452` | `4.43473e-05 / 0.000241724` | `153 / 256` |
| W4A32 pre | `W=true A=false` | `8` | `0/0` | `0` | `None / None` | `None / None` | `None / None` |
| W4A32 final | `W=true A=false` | `8` | `0/0` | `0` | `None / None` | `None / None` | `None / None` |

结论：

- W4A4/W4A8 checkpoint 都是真正启用 activation quantization 的状态：`act_quant=true`，52 个 activation quantizer 均初始化，`non_positive_delta_count=0`。
- W4A32 pre/final 是合法 weight-only 对照：`act_quant=false`，activation delta count 为 `0`，fake-quant MSE 为 `None` 符合预期。
- A4 的 fake-quant MSE 比 A8 高约两个数量级；A4 final 的 mean MSE 比 A4 pre 降低，但 max MSE 从 `0.027745` 升到 `0.0464743`，说明 reconstruction 改善总体误差时把最坏层压力集中到少数层。
- A4 final effective levels 最低只有 `14`，A8 final 最低为 `153`，A4 的核心问题是有效量化级别严重不足，而不是非法 scale。

### Module type 对比

| object | Conv2d MSE mean/max | Conv2d relative MSE mean/max | Linear MSE mean/max | Linear relative MSE mean/max |
|---|---:|---:|---:|---:|
| W4A4 pre | `0.00688291 / 0.0222065` | `0.0351968 / 0.0674264` | `0.00492416 / 0.027745` | `0.0216285 / 0.0524481` |
| W4A4 final | `0.00448758 / 0.0136385` | `0.0273810 / 0.0546570` | `0.00499179 / 0.0464743` | `0.0211767 / 0.0470783` |
| W4A8 pre | `7.25379e-05 / 0.000505416` | `0.000491862 / 0.00160065` | `3.09033e-05 / 0.000137540` | `0.000302101 / 0.000998137` |
| W4A8 final | `5.08725e-05 / 0.000138683` | `0.000451410 / 0.00165058` | `3.39071e-05 / 0.000241724` | `0.000212343 / 0.000597379` |

判断：

- W4A4 reconstruction 明显降低 Conv2d mean/max MSE，但 Linear max MSE 升高，主要来自 attention qkv。
- 旧协议里 “Conv2d 是主瓶颈” 不能直接完整外推到 W4A4。A4 下需要同时看：
  - Conv2d / stage output / fusion 的 relative error；
  - attention qkv 的 absolute MSE；
  - stage5 transformer 的 low effective-level 和 relative error。

### W4A4 final 热点

W4A4 final absolute MSE top layers：

| layer | type | role | stage | fake-quant MSE |
|---|---|---|---|---:|
| `model.stage3.0.block.trans_branch.attn.qkv` | Linear | attention_qkv | stage3 | `0.0464743` |
| `model.stage2.0.block.trans_branch.attn.qkv` | Linear | attention_qkv | stage2 | `0.0226427` |
| `model.stage1.0.block.conv_branch.3` | Conv2d | conv | stage1 | `0.0136385` |
| `model.stage2.0.block.conv_branch.3` | Conv2d | conv | stage2 | `0.0114776` |
| `model.stage4.0.block.trans_branch.attn.qkv` | Linear | attention_qkv | stage4 | `0.0109560` |
| `model.stage3.0.block.conv_branch.0` | Conv2d | conv | stage3 | `0.0107701` |
| `model.stage3.0.block.conv_branch.3` | Conv2d | conv | stage3 | `0.0099101` |
| `model.stage1.0.block.conv_branch.0` | Conv2d | conv | stage1 | `0.00983864` |

W4A4 final relative MSE top layers：

| layer | type | role | stage | relative MSE |
|---|---|---|---|---:|
| `model.stage1.1` | Conv2d | stage_output_conv | stage1 | `0.0546570` |
| `model.stage5.0.block.trans_branch.attn.proj` | Linear | attention_proj | stage5 | `0.0470783` |
| `model.stage5.0.block.trans_branch.mlp.2` | Linear | mlp | stage5 | `0.0442115` |
| `model.stage2.1` | Conv2d | stage_output_conv | stage2 | `0.0439959` |
| `model.stage1.0.block.merge_proj` | Conv2d | merge_proj | stage1 | `0.0431563` |
| `model.stage4.1` | Conv2d | stage_output_conv | stage4 | `0.0419215` |
| `model.stage1.0.block.trans_branch.mlp.2` | Linear | mlp | stage1 | `0.0399722` |
| `model.stage2.0.block.merge_proj` | Conv2d | merge_proj | stage2 | `0.0388929` |

W4A4 final role summary，按 relative MSE mean 排序：

| role | count | MSE mean | relative MSE mean | effective levels min |
|---|---:|---:|---:|---:|
| merge_proj | `5` | `0.00134185` | `0.0388603` | `16` |
| tail | `1` | `0.000599201` | `0.0355458` | `16` |
| stage_output_conv | `5` | `0.000991273` | `0.0342028` | `16` |
| split_proj | `5` | `0.000734917` | `0.0284331` | `16` |
| attention_proj | `5` | `0.000465102` | `0.0240871` | `14` |
| mlp | `10` | `0.000415627` | `0.0232609` | `16` |
| conv | `15` | `0.00851042` | `0.0221592` | `16` |
| attention_qkv | `5` | `0.0186708` | `0.0140978` | `16` |
| head | `1` | `6.80622e-06` | `0.000778224` | `235` |

### W4A4 vs W4A8 热点重合

- W4A4 final 与 W4A8 final 的 absolute MSE top10 有 `7/10` 层重合。
- 重合层：
  - `model.stage1.0.block.conv_branch.3`
  - `model.stage2.0.block.conv_branch.3`
  - `model.stage2.0.block.trans_branch.attn.qkv`
  - `model.stage3.0.block.conv_branch.0`
  - `model.stage3.0.block.trans_branch.attn.qkv`
  - `model.stage4.0.block.conv_branch.3`
  - `model.stage4.0.block.trans_branch.attn.qkv`

判断：

- W4A4 和 W4A8 的热点结构高度重合，但 A4 把这些热点放大到部署不可接受的量级。
- A4 的 absolute MSE 首要热点是 `attention_qkv` 和部分 CNN conv；relative MSE 首要热点是 `stage_output_conv`、`merge_proj`、`stage5 attention_proj/mlp`。
- NE004 不应只复刻旧 E004 的 all Conv2d 关闭实验；应该把 W4A4 分组 sensitivity 设计成：
  1. `attention_qkv`；
  2. CNN `conv_branch`；
  3. `stage_output_conv`；
  4. `split_proj + merge_proj`；
  5. stage5 transformer sanity；
  6. all Conv2d 与 all Linear/transformer 作为全局对照。

### NE001 结论

- W4A4 低指标不是 activation delta 非法、checkpoint reload 错误或没有真正启用 activation quantization；它是真实 A4 bitwidth 压力。
- W4A4 activation reconstruction 对 full-grid SNR 有明显收益，但 diagnostics 显示它更像“整体误差再分配”：Conv2d 误差下降，attention qkv 最坏层压力上升。
- W4A8 当前足够好，不需要把 NE001-NE006 主线用于继续修 W4A8；W4A8 应作为成功参照和护栏。
- 下一步进入 NE004 前，建议先做一个轻量 NE002/NE003 收尾：
  - NE002：确认 full-reference legality / state 表已经满足可继续条件；
  - NE003：固定代表样本图和 full-grid 可视化口径；
  - 然后 NE004 直接以 W4A4 分组 sensitivity 为主。

## 2026-05-11 NE001 结果解释与旧 E 系列对照

本节把 NE001 的诊断结果整理成人类可读结论，重点说明各指标的含义、NE001 与旧 E001-E006 的一致/不一致之处，以及对后续实验的直接指导。

### 一句话结论

NE001 的核心结论是：

> `W4A4` 低指标不是因为 checkpoint 状态错误、activation delta 非法、没有真正启用 activation quantization，或者 packed/reload 失败；它是真实的 A4 activation bitwidth 压力。`W4A8` 的热点结构与 `W4A4` 高度重合，但 A8 有足够量化级别，因此误差仍可接受；A4 把同一批敏感结构的误差放大到不可部署的量级。

后续主线应从“继续修 W4A8”转为：

> 以 W4A8 为成功参照和护栏，重点定位并优化 W4A4 的 A4-specific activation bottleneck。

### NE001 中各类指标分别说明什么

| 指标 | 读法 | NE001 观察 | 说明 |
|---|---|---|---|
| full-grid SNR / SSIM | 最终任务质量，决定模型是否真的可用。 | W4A32 final `17.7856 dB`，W4A8 final `17.4495 dB`，W4A4 final `12.9150 dB`。 | W4A32 几乎贴近 FP32；W4A8 可用；W4A4 有约 `4.53 dB` A4 gap，是主攻对象。 |
| pre/final gain | 重建阶段是否有效。 | W4A4 act recon `+1.7422 dB`，W4A8 act recon `+0.0768 dB`，W4A32 weight recon `+1.1713 dB`。 | W4A4 reconstruction 有明显 SNR 收益，但仍不足；W4A8 reconstruction 只是微调。 |
| `activation_quantizers` | 模型中被包装的 activation quantizer 数量。 | A4/A8/W4A32 都统计到 `52` 个 QuantModule。 | 结构插入位置一致，便于横向对比。 |
| `activation_delta_count` / `initialized_activation_quantizers` | activation quantizer 是否真正初始化。 | W4A4/W4A8 为 `52/52`；W4A32 为 `0/0`。 | W4A4/W4A8 确实启用了 activation quantization；W4A32 正确保持 weight-only。 |
| `non_positive_delta_count` | 是否存在非法 scale。 | A4/A8 全部为 `0`。 | 当前问题不是旧 E001 中的负 delta bug。 |
| fake-quant MSE mean/max | 局部 activation 被量化前后的绝对误差。 | W4A4 final `0.00468 / 0.04647`，W4A8 final `4.43e-05 / 0.000242`。 | A4 局部误差比 A8 高约两个数量级。 |
| effective int levels | 实际使用了多少整数格点。 | W4A4 final min `14`，W4A8 final min `153`。 | A4 的核心瓶颈是表达级别太少；A8 仍有足够离散级别。 |
| absolute MSE top layers | 哪些层制造最大绝对误差。 | W4A4 final top 是 `stage3/stage2/stage4 attention_qkv` 和部分 CNN conv。 | A4 下 attention qkv 不能忽略。 |
| relative MSE top layers | 哪些层相对自身信号最脆弱。 | W4A4 final top 是 `stage_output_conv`、`merge_proj`、`stage5 attention_proj/mlp`。 | A4 的弱点不只在大误差层，也在结构边界和融合层。 |
| top10 热点重合 | A4/A8 是否是同一种结构问题。 | W4A4 final 与 W4A8 final absolute MSE top10 重合 `7/10`。 | 热点位置相似，但 A4 误差量级远大于 A8。 |

### 为什么 W4A4 final SNR 提升但 SSIM 下降

W4A4 activation reconstruction 让 mean SNR 从 `11.1727` 提升到 `12.9150`，但 mean SSIM 从 `0.941492` 降到 `0.939563`。

这说明：

- 当前 reconstruction 目标更接近局部 MSE / 能量误差优化。
- SNR 受能量误差影响较大，因此可以明显提高。
- SSIM 更关注结构相似性；A4 下如果 reconstruction 把误差重新分配到结构敏感位置，SSIM 可能下降。
- 后续 A4 优化不能只看 SNR；NE003 代表图和 by-source SSIM 必须保留。

### 与旧 E 系列的一致之处

| 旧 E 系列结论 | NE001 是否支持 | 说明 |
|---|---|---|
| activation quantization 需要固定多样本评估，单样本不可靠。 | 支持。 | NE001 继续使用 normalized `478 x 25` full-grid 作为最终质量口径，diagnostics 只解释局部机制。 |
| 负 delta 是必须修复的合法性 bug，但不是全部问题。 | 支持。 | NE001 中 A4/A8 都没有负 delta，但 A4 仍明显掉点。 |
| Conv2d activation quantization 是重要误差来源。 | 部分支持。 | W4A4 Conv2d relative MSE 仍高，Conv branch 多层进入 top list。 |
| split/merge/stage_output 是结构化粒度的重要候选。 | 支持且加强。 | W4A4 relative MSE top 明确包含 `stage_output_conv`、`merge_proj`、`split_proj`。 |
| stage5 独立策略需要谨慎。 | 支持。 | W4A4 relative MSE 中 stage5 attention_proj/mlp 很突出，但旧 E006c 发现 stage5 独立细粒度可能有害，后续需要作为 sanity 而不是直接默认优化。 |

### 与旧 E 系列不完全一致的地方

| 旧 E 系列结论 | NE001 新观察 | 是否矛盾 | 当前解释 |
|---|---|---|---|
| 旧 raw 协议下 W4A8 一做 activation quantization 就严重崩坏。 | normalized 协议下 W4A8 final 只比 W4A32 低约 `0.336 dB`。 | 不是直接矛盾。 | 数据协议、幅值归一化、FP32 起点、eval 口径都变了；旧数值不能迁移。normalized per-patch absmax 极大缓解了 A8 activation range 压力。 |
| 旧 E004d 认为 W4A8 A8 init 崩坏主因是 Conv2d，Linear/transformer 不是主因。 | NE001 中 W4A4 final absolute MSE top 是 attention_qkv，Linear max MSE 高于 Conv2d。 | 不完全一致，但不是直接冲突。 | 旧 E004d 是 W4A8 raw 协议的 full-grid sensitivity；NE001 是 W4A4 normalized 协议的局部 diagnostics。bitwidth、数据协议、指标语义都不同。 |
| 旧 E004/E006 主线从 transformer 转向 Conv2d。 | NE001 显示 A4 下 transformer attention qkv/proj 重新变得重要。 | 是 A4-specific 新信号。 | A8 有 153+ effective levels 时 transformer 误差可接受；A4 只有约 14-16 levels 时 transformer 和融合边界会重新成为瓶颈。 |
| 旧 E005 tensor-wise range/clipping 基本无效。 | NE001 显示 A4 effective levels 很低，似乎 range 可能重要。 | 不能直接推翻旧结论。 | 低 effective levels 说明 A4 表达力不足，但不等于 tensor-wise clipping 能修复。必须先做 NE004 sensitivity，再对目标组做 NE005 range。 |
| 旧 E006c 最强候选是 selective split/merge/stage_output Conv2d 粒度。 | NE001 的 relative MSE 明确指向 stage_output/merge/split。 | 一致。 | 这条旧结论在新协议下仍有机制支持，值得优先复测，但目标应从 W4A8 扩展到 W4A4。 |

### 关键区别：diagnostics 不是 sensitivity

NE001 的 fake-quant MSE / relative MSE 是局部指标，它回答：

- 哪些层本地量化误差大？
- 哪些层相对自身信号最脆弱？
- A4 和 A8 的局部误差结构是否相似？

它不能直接回答：

- 关闭某组 activation quantizer 后 full-grid SNR 能恢复多少？
- 某层局部误差大是否一定导致最终输出差？
- 某个结构是否应该保留 FP32、改 per-channel，还是改 range？

因此 NE001 只能给 NE004 排序和分组优先级，不能替代 NE004 sensitivity。

### 对 NE004 的直接指导

NE004 不应只复刻旧 E004 的 `all Conv2d` / `all Linear` 二分实验。新 W4A4 主线需要同时覆盖 absolute MSE 和 relative MSE 两类热点。

建议 NE004 第一轮分组：

| 优先级 | group | 来源 | 要回答的问题 |
|---:|---|---|---|
| 1 | `attention_qkv` | W4A4 absolute MSE top | qkv 的巨大局部误差是否真的主导 full-grid A4 gap？ |
| 2 | `cnn conv_branch` | W4A4 absolute MSE top + 旧 E004 Conv2d 结论 | CNN conv 累积误差是否仍是 A4 的主要 SNR 损失来源？ |
| 3 | `stage_output_conv` | W4A4 relative MSE top + 旧 E006c 强候选 | stage 输出卷积是否解释 A4 结构性失真？ |
| 4 | `split_proj + merge_proj` | W4A4 relative MSE top + 旧 E006c 强候选 | 融合投影是否是 A4 下最值得做细粒度的结构？ |
| 5 | `stage5 attention_proj/mlp` | W4A4 relative MSE top + 旧 stage5 风险 | stage5 transformer 是否是 A4-specific 风险点，还是旧 E006c 中“独立 stage5 有害”的延续？ |
| 6 | `all Conv2d` / `all Linear` | 旧 E004 对照 | 用全局组验证 A4 是否仍满足“Conv2d 主导”假设。 |

NE004 判读规则：

- 如果 `attention_qkv` 关闭后 full-grid 恢复很小，说明 attention qkv 的 high absolute MSE 可能被后续结构吸收，主线回到 Conv2d/fusion/stage_output。
- 如果 `attention_qkv` 关闭后恢复很大，W4A4 需要 A4-specific transformer 策略，不能只沿用旧 E006 Conv2d selective granularity。
- 如果 `stage_output_conv` 或 `split+merge` 恢复大，旧 E006c 的 selective granularity 候选应直接迁移到 W4A4。
- 如果 `all Conv2d` 仍远强于细分组，说明 A4 可能存在更广泛的 Conv2d 累积误差，NE006 要考虑 all Conv2d per-channel 作为上限。

### 对 NE005 / NE006 的影响

NE005：

- 不应马上大范围做 percentile / MSE-grid range sweep。
- 只应在 NE004 证明敏感的组上做 range/clipping。
- 需要同时看 SNR 和 SSIM，因为 W4A4 reconstruction 已经表现出 SNR/SSIM tradeoff。

NE006：

- 旧 E006c 的 `split_proj + merge_proj + stage_output_conv` selective per-channel / g4 仍是优先候选。
- 但 W4A4 还应加入 `attention_qkv` 或 transformer qkv/proj 的 A4-specific 候选：
  - selective A8 fallback；
  - qkv-only higher precision；
  - qkv per-channel / group-wise；
  - qkv 保持 FP32 作为上限反事实。
- W4A8 不再作为主要优化对象，但每个 W4A4 策略都应至少验证不会明显破坏 W4A8。

### 当前对“冲突”的最终判断

NE001 和旧 E 系列不是简单互相否定，而是说明：

1. 旧 E 系列回答的是 raw-amplitude 旧协议下的 W4A8 崩坏机制。
2. NE001 回答的是 normalized 新协议下 W4A4 的 A4-specific 压力机制。
3. 旧结论中 “Conv2d / split / merge / stage_output 很重要” 仍然成立。
4. 旧结论中 “Linear/transformer 不是主因” 不能直接外推到 W4A4。
5. 后续实验必须以 full-grid sensitivity 验证 NE001 的局部诊断，不能只凭 diagnostics 排序决定最终策略。

## 2026-05-11 NE001 统计项横向对比与指标用途

本节补充 NE001 中此前没有展开的统计项。NE001 实际记录了大量 activation distribution / quantization error / structure grouping 统计，但不同统计项回答的问题不同，不能全部按同一权重解释。

### 指标分层

| 指标组 | 包含字段 | 回答的问题 | 后续主要服务 |
|---|---|---|---|
| 任务质量 | SNR、SSIM、pre/final gain | 模型最终是否可用，重建是否真的提升输出质量。 | 所有实验的最终验收。 |
| 合法性 | quant state、delta count、initialized count、non-positive delta | checkpoint 是否真的启用 activation quantization，scale 是否非法。 | NE002 sanity / state verification。 |
| 量化误差 | fake-quant MSE、relative MSE | 哪些层量化前后局部误差大，哪些层相对自身信号最脆弱。 | NE004 sensitivity 分组。 |
| 有效级别 | effective int levels | activation 实际利用了多少整数格点。 | 判断 A4 是否是 bitwidth 表达力瓶颈。 |
| 分布离群 | min/max/std、p99、p99.9、p99.99、absmax、absmax/p99 | range 是否被长尾或离群值撑大。 | NE005 range / clipping。 |
| 通道不均衡 | per-channel absmax ratio | 同一层内通道幅值差异是否大，tensor-wise scale 是否过粗。 | NE006 per-channel / group-wise granularity。 |
| 结构分组 | module type、stage、branch、role summary | 误差是否集中在 Conv2d、Linear、stage output、fusion、attention 等结构。 | NE004 分组、NE006 selective strategy。 |

因此，NE001 的正确读法不是“谁的数最大就直接修谁”，而是：

1. 先用 full-grid SNR/SSIM 判断现象是否真实重要；
2. 用 legality 排除状态错误；
3. 用 fake-quant MSE / relative MSE 形成 NE004 sensitivity 候选；
4. 用 outlier 和 per-channel ratio 分别决定 NE005 / NE006 是否值得做。

### 六个 diagnostics 对象的横向可比性

| 对象 | 可比较字段 | 不应比较字段 | 原因 |
|---|---|---|---|
| W4A4 pre/final | 全部 activation diagnostics 字段。 | N/A | 主对象，`act_quant=true` 且 52 个 activation quantizer 均初始化。 |
| W4A8 pre/final | 全部 activation diagnostics 字段。 | N/A | 成功参照，`act_quant=true` 且 52 个 activation quantizer 均初始化。 |
| W4A32 pre/final | activation distribution、结构分组、最终 SNR/SSIM。 | fake-quant MSE、effective levels、delta stats。 | W4A32 是 weight-only，`act_quant=false`，没有 activation delta。 |
| FP32 | 最终 SNR/SSIM、可视化上限。 | activation quantizer diagnostics。 | FP32 不是 quantized checkpoint，不存在 activation quantizer。 |

### 六对象统计摘要

| object | absmax/p99 max | top outlier layer | per-channel ratio max | top channel-imbalance layer | fake-quant MSE mean/max | relative MSE max | levels min/max |
|---|---:|---|---:|---|---:|---:|---:|
| W4A4 pre | `9.83285` | `model.stage5.1` | `2.68474` | `model.stage1.0.block.conv_branch.6` | `0.00612955 / 0.027745` | `0.0674264` | `13 / 246` |
| W4A4 final | `9.83285` | `model.stage5.1` | `2.68474` | `model.stage1.0.block.conv_branch.6` | `0.00468151 / 0.0464743` | `0.0546570` | `14 / 235` |
| W4A8 pre | `9.83285` | `model.stage5.1` | `2.68474` | `model.stage1.0.block.conv_branch.6` | `5.65246e-05 / 0.000505416` | `0.00160065` | `168 / 256` |
| W4A8 final | `9.83285` | `model.stage5.1` | `2.68474` | `model.stage1.0.block.conv_branch.6` | `4.43473e-05 / 0.000241724` | `0.00165058` | `153 / 256` |
| W4A32 pre | `9.08709` | `model.stage5.1` | `2.62299` | `model.stage1.0.block.conv_branch.6` | `None / None` | `None` | `None / None` |
| W4A32 final | `9.83285` | `model.stage5.1` | `2.68474` | `model.stage1.0.block.conv_branch.6` | `None / None` | `None` | `None / None` |

读法：

- `absmax/p99 max` 和 `per-channel ratio max` 在 W4A4/W4A8/W4A32 final 中基本一致，说明这些分布形态主要来自模型结构和输入分布，不是 A4 特有状态错误。
- W4A4 与 W4A8 的分布离群位置相同，但 fake-quant MSE 和 effective levels 差异巨大；这说明 A4 的主要问题不是“看到了不同分布”，而是同样分布在 A4 网格下表达不够。
- `model.stage5.1` 是最强 outlier 层，`model.stage1.0.block.conv_branch.6` 是最强 per-channel imbalance 层。它们不是最终 SNR 结论本身，但分别是 NE005 range 和 NE006 granularity 的重点候选。

### W4A4 pre -> final：reconstruction 改善了什么，又恶化了什么

| role | pre MSE mean | final MSE mean | MSE delta | pre relative MSE | final relative MSE | relative delta | final levels min |
|---|---:|---:|---:|---:|---:|---:|---:|
| attention_proj | `0.000525719` | `0.000465102` | `-0.0000606` | `0.0266923` | `0.0240871` | `-0.002605` | `14` |
| attention_qkv | `0.0184496` | `0.0186708` | `+0.000221` | `0.0165272` | `0.0140978` | `-0.002429` | `16` |
| conv | `0.0135093` | `0.00851042` | `-0.004999` | `0.0364380` | `0.0221592` | `-0.014279` | `16` |
| merge_proj | `0.00165389` | `0.00134185` | `-0.000312` | `0.0464129` | `0.0388603` | `-0.007553` | `16` |
| mlp | `0.000360641` | `0.000415627` | `+0.0000550` | `0.0216472` | `0.0232609` | `+0.001614` | `16` |
| split_proj | `0.000727764` | `0.000734917` | `+0.00000715` | `0.0277123` | `0.0284331` | `+0.000721` | `16` |
| stage_output_conv | `0.00101980` | `0.000991273` | `-0.0000285` | `0.0345558` | `0.0342028` | `-0.000353` | `16` |

读法：

- W4A4 reconstruction 最大的正收益来自 `conv`：MSE mean 从 `0.0135093` 降到 `0.00851042`，relative MSE mean 从 `0.0364380` 降到 `0.0221592`。
- `merge_proj`、`attention_proj` 也有改善。
- `attention_qkv` 的 relative MSE 下降，但 absolute MSE 略升，并且 final 的 worst layer 变成 `stage3 attention_qkv`。这解释了为什么日志中说 reconstruction 像是“整体误差再分配”。
- `mlp` 和 `split_proj` 略有变差。虽然幅度不大，但它们提示后续不能只看 mean SNR，需要保留 role-level 表。

### W4A4 final vs W4A8 final：同结构下 A4 放大了多少误差

| role | A4 MSE mean | A8 MSE mean | A4/A8 MSE ratio | A4 relative mean | A8 relative mean | A4/A8 relative ratio |
|---|---:|---:|---:|---:|---:|---:|
| attention_proj | `0.000465102` | `3.72275e-06` | `124.9x` | `0.0240871` | `0.000202129` | `119.2x` |
| attention_qkv | `0.0186708` | `0.000122226` | `152.8x` | `0.0140978` | `0.000105402` | `133.8x` |
| conv | `0.00851042` | `0.0000895306` | `95.1x` | `0.0221592` | `0.000236041` | `93.9x` |
| merge_proj | `0.00134185` | `0.0000241400` | `55.6x` | `0.0388603` | `0.000666999` | `58.3x` |
| mlp | `0.000415627` | `0.00000483982` | `85.9x` | `0.0232609` | `0.000270922` | `85.9x` |
| split_proj | `0.000734917` | `0.0000107998` | `68.0x` | `0.0284331` | `0.000418082` | `68.0x` |
| stage_output_conv | `0.000991273` | `0.0000192701` | `51.4x` | `0.0342028` | `0.000855874` | `40.0x` |
| tail | `0.000599201` | `0.00000710523` | `84.3x` | `0.0355458` | `0.000421496` | `84.3x` |

读法：

- A4 并不是只在某一个结构上比 A8 差，而是几乎所有 role 都被放大几十到一百多倍。
- `attention_qkv` 的 A4/A8 MSE ratio 最高，说明 A4 的 transformer qkv 压力是新协议下必须单独验证的点。
- `stage_output_conv` 和 `merge_proj` 的 relative MSE 很高，即使 MSE 绝对值不如 qkv，也可能对结构输出质量更敏感。

### W4A4 final 的 stage 视角

| stage | count | MSE mean/max | relative MSE mean/max | absmax/p99 mean/max | per-channel ratio mean/max | levels min |
|---|---:|---:|---:|---:|---:|---:|
| head | `1` | `6.806e-06 / 6.806e-06` | `0.000778 / 0.000778` | `6.5907 / 6.5907` | `2.2708 / 2.2708` | `235` |
| stage1 | `10` | `0.004106 / 0.013639` | `0.027554 / 0.054657` | `5.3266 / 6.7599` | `1.6807 / 2.6847` | `16` |
| stage2 | `10` | `0.005423 / 0.022643` | `0.026152 / 0.043996` | `4.3956 / 6.7357` | `1.4086 / 2.0229` | `16` |
| stage3 | `10` | `0.008114 / 0.046474` | `0.024563 / 0.035186` | `4.4163 / 5.9881` | `1.4648 / 1.9141` | `16` |
| stage4 | `10` | `0.004042 / 0.010956` | `0.023266 / 0.041922` | `5.4073 / 7.7720` | `1.5144 / 2.0534` | `16` |
| stage5 | `10` | `0.002597 / 0.006698` | `0.024806 / 0.047078` | `6.1928 / 9.8329` | `1.6723 / 2.5266` | `14` |
| tail | `1` | `0.000599 / 0.000599` | `0.035546 / 0.035546` | `1.9711 / 1.9711` | `1.0000 / 1.0000` | `16` |

读法：

- stage3 的 absolute MSE 最大，主要由 `stage3 attention_qkv` 拉高。
- stage1 的 relative MSE max 最大，对应 `stage1.1 stage_output_conv`。
- stage5 的 outlier 和 effective levels 最差，包含 `stage5.1`、`stage5 attention_proj/mlp` 等风险点。
- 因此 “stage5 有信号” 和旧 E006c “stage5 独立细粒度有害” 不矛盾；它说明 stage5 需要作为 sanity check，而不是默认单独优化。

### outlier 与 per-channel imbalance 的具体候选

W4A4 final top per-channel imbalance：

| rank | layer | type | role | stage | ratio |
|---:|---|---|---|---|---:|
| 1 | `model.stage1.0.block.conv_branch.6` | Conv2d | conv | stage1 | `2.68474` |
| 2 | `model.stage5.1` | Conv2d | stage_output_conv | stage5 | `2.52657` |
| 3 | `model.stage5.0.block.conv_branch.6` | Conv2d | conv | stage5 | `2.50420` |
| 4 | `model.stage1.0.block.split_proj` | Conv2d | split_proj | stage1 | `2.41581` |
| 5 | `model.head` | Conv2d | head | head | `2.27078` |
| 6 | `model.stage5.0.block.merge_proj` | Conv2d | merge_proj | stage5 | `2.15788` |

W4A4 final top outlier：

| rank | layer | type | role | stage | absmax/p99 |
|---:|---|---|---|---|---:|
| 1 | `model.stage5.1` | Conv2d | stage_output_conv | stage5 | `9.83285` |
| 2 | `model.stage5.0.block.conv_branch.6` | Conv2d | conv | stage5 | `8.37609` |
| 3 | `model.stage4.0.block.conv_branch.3` | Conv2d | conv | stage4 | `7.77198` |
| 4 | `model.stage5.0.block.conv_branch.3` | Conv2d | conv | stage5 | `7.33216` |
| 5 | `model.stage4.0.block.conv_branch.6` | Conv2d | conv | stage4 | `7.15543` |
| 6 | `model.stage5.0.block.merge_proj` | Conv2d | merge_proj | stage5 | `7.12966` |

读法：

- per-channel imbalance 指向 NE006：这些层更可能从 per-channel / group-wise activation granularity 中获益。
- outlier 指向 NE005：这些层更值得做 percentile / MSE-grid / structured clipping。
- 两张表高度偏向 Conv2d、stage output、split/merge，而不是 qkv；这说明 qkv 的问题更像 bitwidth/absolute MSE，Conv/fusion/stage output 的问题更像 range/granularity。

### 这次完整对比对后续实验的具体影响

1. NE004 sensitivity 必须先验证 diagnostics 排名是否会转化为 full-grid SNR 恢复。
2. NE004 不能只做旧 E004 的 all Conv2d / all Linear，应加入：
   - `attention_qkv`
   - `cnn conv_branch`
   - `stage_output_conv`
   - `split_proj + merge_proj`
   - `stage5 attention_proj/mlp`
3. NE005 range/clipping 不应全模型乱扫，应优先对 top outlier 中的 stage5/stage4 Conv2d、stage_output、merge_proj 做。
4. NE006 granularity 应优先覆盖 per-channel imbalance top layers 所属结构：
   - conv_branch.6
   - stage_output_conv
   - split_proj / merge_proj
5. qkv 的 high absolute MSE 不一定能被 Conv2d per-channel 修复，因此 NE006 需要保留 qkv-only A8 fallback / higher precision / group-wise 的反事实候选。
6. W4A8 只作为护栏：若某个 W4A4 策略让 W4A8 明显变差，则不能作为通用默认策略。

### 结论更新

NE001 的完整统计并没有推翻前面的简化结论，而是把它细化成三条并行线索：

1. **bitwidth 线索**：A4 effective levels 只有约 `14-16`，导致所有结构的误差相对 A8 放大几十到一百多倍。
2. **range 线索**：stage5/stage4 Conv2d、stage_output、merge_proj 有明显 outlier，后续 NE005 只应优先处理这些目标组。
3. **granularity 线索**：conv_branch.6、stage_output_conv、split/merge 有明显 per-channel imbalance，后续 NE006 的 selective per-channel / g4 应从这些结构开始。

因此，进入下一部分实验前的优先级是：先做 NE004 W4A4 分组 sensitivity，验证这些局部统计是否真的对应 full-grid 恢复；再决定 NE005 range 和 NE006 granularity 的具体展开顺序。

## 2026-05-11 NE002 与 NE003 新计划记录

本节把 NE002 / NE003 的新语义固定下来，避免机械复刻旧 E002 / E003。旧 E002 的核心是修复 activation `delta` 负数；旧 E003 的核心是建立 128-sample 多样本评估口径。当前 NE 系列已经换成 normalized 协议，且 W4A8 已经可用、W4A4 成为主对象，因此 NE002 / NE003 应作为进入 NE004 前的状态与解释口径收口，而不是优化实验。

### NE002：合法状态、checkpoint reload 与 packed deployment sanity

实验目标：

- 排除 W4A4 / W4A8 结果由 checkpoint 状态错误、activation bitwidth 配错、未真正启用 activation quantization、reload 状态丢失或 packed restore 不一致造成的可能。
- 确认 NE004-NE006 可以把差异归因到 activation quantization 策略本身，而不是工程状态问题。
- 将 W4A4 低指标正式定性为真实 A4 activation bitwidth 压力，而不是合法性 bug。

实验对象：

| 类别 | 对象 | 用途 |
|---|---|---|
| 主对象 | `NE000_2 W4A4 pre_act_recon`、`NE000_2 W4A4 final` | 检查 A4 activation quantization 在重建前后的状态是否合法。 |
| 成功参照 | `NE000 W4A8 pre_act_recon`、`NE000 W4A8 final` | 确认 A8 好结果不是因为 activation quantization 没有真正启用。 |
| weight-only 参照 | `E007 W4A32 pre_recon`、`E007 W4A32 final` | 确认 W4A32 正确保持 `act_quant=false`，只作为 weight-only 对照。 |
| 部署参照 | W4A32 packed、W4A8 packed、W4A4 packed | 确认 packed 权重整数化和恢复后仍与 fake-quant checkpoint 对齐。 |
| 上限参照 | FP32 checkpoint | 只记录 full-grid 指标，不做 quantizer sanity。 |

实验方法：

1. 对 6 个 quantized checkpoint 统一运行 `verify_quantized_scrn`，输出 verification JSON。
2. 检查每个 checkpoint 的 `final_quant_state`、weight bit counts、activation bitwidth、activation quantizer count、initialized activation quantizer count、`non_positive_delta_count`、`level_offender_count`。
3. 对 W4A32 / W4A8 / W4A4 packed artifact 检查 `manifest.json` 和 `summary.json`，确认 `weights.bin`、`aux_fp32.bin`、activation qparams、bitwidth 和 final quant state 完整。
4. 复核 packed-vs-checkpoint full-grid equivalence：W4A32、W4A8、W4A4 的 packed restored mean SNR delta 应保持在 `< 0.01 dB` 量级。
5. 如现有结果仍不能充分证明 state toggle 正确，再补一个小规模 state toggle sanity：同一 checkpoint 分别强制 `act_quant=false/true`，确认输出会按预期切换。

验收标准：

| 对象 | 预期状态 |
|---|---|
| W4A4 pre/final | `weight_quant=true`、`act_quant=true`、activation bitwidth `4`、52 个 activation quantizer 初始化、`non_positive_delta_count=0`。 |
| W4A8 pre/final | `weight_quant=true`、`act_quant=true`、activation bitwidth `8`、52 个 activation quantizer 初始化、`non_positive_delta_count=0`。 |
| W4A32 pre/final | `weight_quant=true`、`act_quant=false`、无 activation delta，作为 weight-only 参照。 |
| packed W4A32/W4A8/W4A4 | packed restored 与原 checkpoint full-grid 指标对齐，mean SNR delta 绝对值 `< 0.01 dB`。 |

NE002 不做：

- 不做新的 activation reconstruction。
- 不做 range / clipping。
- 不做 granularity / mixed precision。
- 不以单样本 SNR 判断结果好坏。

NE002 完成后的判断：

- 若所有 sanity 均通过，进入 NE003 和 NE004。
- 若发现 bitwidth、quant state、reload 或 packed restore 问题，先修复状态问题并重新验证，不进入 NE004。
- 若只发现 packed 等价失败但 fake-quant checkpoint 合法，则 NE004 可继续使用 fake-quant checkpoint，但部署候选必须暂停。

### NE003：固定 full-grid 解释口径和代表图口径

实验目标：

- 固定后续 W4A4 优化实验的数值和可视化解释口径。
- 避免后续再次被单样本、显示尺度、反归一化方式或 source/condition 偶然性误导。
- 为 NE004-NE006 每个候选策略提供统一的可读对比模板。

实验对象：

| 对象 | 角色 |
|---|---|
| FP32 | normalized 协议上限。 |
| E007 W4A32 pre/final | weight-only 重建前后参照。 |
| NE000 W4A8 pre/final | 可用 activation quantization 成功参照。 |
| NE000_2 W4A4 pre/final | W4A4 主对象，重点看 activation reconstruction 前后差异。 |

实验方法：

1. 复核已有 normalized `478 x 25` full-grid 指标，统一记录 overall、by-source、by-SNR、by-missing-rate、by-condition。
2. 固定一组代表样本：
   - W4A4 final 最差样本；
   - W4A4 final 中位样本；
   - W4A4 final 较好样本；
   - W4A4 reconstruction 提升最大样本；
   - W4A4 reconstruction 变差或 SSIM 下降样本；
   - 默认单样本 sanity 图，仅作为历史可视化参照，不作为正式指标。
3. 使用一致显示规范：
   - seismic colormap；
   - normalized 样本使用反归一化或统一幅值尺度；
   - 同一图内 FP32 / W4A32 / W4A8 / W4A4 使用一致 `vmin/vmax`；
   - 同时输出误差图，误差图单独固定尺度。
4. 每个代表样本记录 source、patch index、SNR setting、missing rate、condition index、FP32/W4A32/W4A8/W4A4 SNR/SSIM。

验收标准：

- 代表样本清单可复现，记录 sample id / source / condition。
- 图像输出全部位于 `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE003_*` 下。
- 记录 W4A4 pre -> final 的 SNR/SSIM 变化，特别保留“W4A4 SNR 提升但 SSIM 下降”的案例。
- 后续 NE004-NE006 的每个候选策略都复用同一 full-grid 和代表图口径。

NE003 不做：

- 不改变 checkpoint。
- 不改变量化参数。
- 不做 sensitivity 或 range / granularity 搜索。

### 与旧 E002 / E003 的区别

| 阶段 | 旧 E 系列语义 | NE 系列新语义 |
|---|---|---|
| E002 / NE002 | 修复 activation `delta` 负数，并验证正 scale clamp 是否恢复 W4A8。 | 复核 W4A4/W4A8/W4A32 的 quant state、bitwidth、reload 和 packed deployment 等价。 |
| E003 / NE003 | 建立 128-sample eval，证明单样本 SNR 不可靠。 | 固定 normalized `478 x 25` full-grid 解释口径、代表样本和 seismic 反归一化可视化规范。 |

当前建议顺序：

1. 先执行 NE002，正式锁定 checkpoint 和 packed deployment 的合法性。
2. 再执行 NE003，固定代表图和 full-grid 解释口径。
3. 然后进入 NE004，以 W4A4 分组 sensitivity 验证 NE001 的局部诊断是否真的能转化为 full-grid 恢复。

## 2026-05-11 NE002 checkpoint / deployment sanity 结果

完成 NE002 合法状态与部署等价 sanity。NE002 不改变 checkpoint、不重新量化、不做优化，只确认当前 W4A4 / W4A8 / W4A32 结果是否可以作为后续 NE003 / NE004 的可靠输入。

执行环境：

- branch：`main`
- repo root：`/home/data1/hanwen/project/Project/SCRN_Quant`
- 初始 worktree：clean
- GPU：物理 GPU `1`
- 备注：普通沙箱内 `torch.cuda.is_available() == False`，实际 GPU 命令按同参数在沙箱外执行；GPU 2 当时有外部 `swinir` 进程，未使用。

### NE002 verification 产物

输出目录：

- `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE002_checkpoint_deployment_sanity/verification`

| object | verification JSON | passed | weight_quant | act_quant | n_bits_a | weight bits | level offenders | activation delta / initialized |
|---|---|---:|---:|---:|---:|---|---:|---|
| W4A4 pre | `ne002_w4a4_pre_verification.json` | true | true | true | 4 | `{"4":50,"8":2}` | 0 | `52 / 52` |
| W4A4 final | `ne002_w4a4_final_verification.json` | true | true | true | 4 | `{"4":50,"8":2}` | 0 | `52 / 52` |
| W4A8 pre | `ne002_w4a8_pre_verification.json` | true | true | true | 8 | `{"4":50,"8":2}` | 0 | `52 / 52` |
| W4A8 final | `ne002_w4a8_final_verification.json` | true | true | true | 8 | `{"4":50,"8":2}` | 0 | `52 / 52` |
| W4A32 pre | `ne002_w4a32_pre_verification.json` | true | true | false | 8 | `{"4":50,"8":2}` | 0 | `0 / 0` |
| W4A32 final | `ne002_w4a32_final_verification.json` | true | true | false | 8 | `{"4":50,"8":2}` | 0 | `0 / 0` |

非正 activation delta 复核：

- `verify_quantized_scrn` 当前报告 activation delta 是否恢复，但不单独输出 `non_positive_delta_count` 字段。
- 复用 NE001 同 checkpoint diagnostics summary 交叉确认：
  - W4A4 pre：`non_positive_delta_count=0`
  - W4A4 final：`non_positive_delta_count=0`
  - W4A8 pre：`non_positive_delta_count=0`
  - W4A8 final：`non_positive_delta_count=0`
  - W4A32 pre/final：`act_quant=false`，activation delta 为 `0/0`，不适用。

判断：

- W4A4/W4A8 checkpoint 都是真正的 W4 activation + A4/A8 activation fake-quant checkpoint。
- W4A32 pre/final 都是合法 W4A32 weight-only checkpoint。
- 6 个 checkpoint 均无 weight level offender。

### NE002 state toggle smoke

输出目录：

- `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE002_checkpoint_deployment_sanity/state_toggle`

设置：

- eval dataset：`SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`
- `num_eval_samples=16`
- `batch_size=16`
- `seed=20260507`
- GPU：物理 GPU `1`
- 说明：这只是 activation state toggle sanity，不作为正式 full-grid 指标。

| run | mode | selected quantizers | samples | FP32 SNR mean | quant SNR mean | quant SSIM mean | quant-FP32 SNR |
|---|---|---:|---:|---:|---:|---:|---:|
| `20260511_125156_ne002_w4a4_final_all_on` | all_on | 51 | 16 | 16.4840 | 12.0236 | 0.930772 | -4.4604 |
| `20260511_125203_ne002_w4a4_final_all_off` | all_off | 51 | 16 | 16.4840 | 16.4281 | 0.953357 | -0.0559 |
| `20260511_125210_ne002_w4a8_final_all_on` | all_on | 51 | 16 | 16.4840 | 16.1103 | 0.951832 | -0.3737 |
| `20260511_125216_ne002_w4a8_final_all_off` | all_off | 51 | 16 | 16.4840 | 16.4281 | 0.953357 | -0.0559 |

判断：

- `all_on` / `all_off` 都选中 51 个候选 activation quantizer，符合默认排除 output quantizer 的预期。
- W4A4 与 W4A8 的 `all_on` 和 `all_off` 输出明显不同，说明 activation quantizer 开关确实生效。
- W4A4 `all_on` 明显低于 W4A8 `all_on`，且二者 `all_off` 回到相同的 W4A32-like 行为；这支持 W4A4 问题来自 activation A4，而不是 weight checkpoint 或 eval 流程。

### NE002 packed deployment equivalence 复核

已复核的 packed export 目录：

- W4A32：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_1_packed_deployment_equivalence/w4a32_packed/e007_normalized_w4a32_single_gpu1_final`
- W4A8：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_1_packed_deployment_equivalence/w4a8_packed/ne000_normalized_w4a8_tensor_a5000_final`
- W4A4：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_1_packed_deployment_equivalence/w4a4_packed/ne000_2_normalized_w4a4_tensor_a5000_final`

三组 packed artifact 均包含：

- `weights.bin`
- `aux_fp32.bin`
- `manifest.json`
- `summary.json`

| object | weights.bin | aux_fp32.bin | manifest.json | raw payload | total export | quantized layers | activation quantizers |
|---|---:|---:|---:|---:|---:|---:|---:|
| W4A32 packed | 213632 | 40860 | 94254 | 254492 | 348746 | 52 | 0 |
| W4A8 packed | 213632 | 41276 | 121554 | 254908 | 376462 | 52 | 52 |
| W4A4 packed | 213632 | 41276 | 121590 | 254908 | 376498 | 52 | 52 |

Packed full-grid equivalence：

| object | eval rows | packed runtime state | checkpoint SNR mean | packed SNR mean | packed-checkpoint SNR | checkpoint SSIM mean | packed SSIM mean | pred MSE mean |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| W4A32 | 11950 | `weight_quant=false, act_quant=false` | 17.785582 | 17.785577 | -0.000005 | 0.964137 | 0.964137 | 3.95e-10 |
| W4A8 | 11950 | `weight_quant=false, act_quant=true` | 17.449507 | 17.449534 | +0.000027 | 0.962868 | 0.962869 | 3.76e-07 |
| W4A4 | 11950 | `weight_quant=false, act_quant=true` | 12.914963 | 12.915036 | +0.000073 | 0.939563 | 0.939565 | 3.89e-06 |

判断：

- 三个 packed restored 模型的 mean SNR delta 绝对值都远小于 `0.01 dB`，通过部署等价阈值。
- W4A8 / W4A4 packed restore 使用 `weight_quant=false, act_quant=true`，语义正确：权重已经恢复为部署量化值，activation fake quant 仍开启。
- W4A4 packed 部署等价通过，因此 W4A4 可作为合法 A4 压力对照继续进入后续实验。

### NE002 最终结论

- W4A4 低指标不是 checkpoint reload 错误、bitwidth 配错、activation quantizer 未启用、负 scale 或 packed restore 不一致导致的。
- W4A8 好结果也不是因为 activation quantization 没有真正开启；W4A8 final 是合法 W4A8 activation quantization baseline。
- E007 W4A32 pre/final 是合法 weight-only 参照。
- W4A32 / W4A8 / W4A4 packed deployment 均与对应 fake-quant checkpoint full-grid 对齐。
- NE002 sanity 通过，可以进入 NE003 固定代表图口径，并随后进入 NE004 W4A4 分组 sensitivity。

## 2026-05-11 NE003 fixed grid visual protocol 结果

完成 NE003 固定评估与代表图口径。本轮不改变 checkpoint、不重新量化、不做 sensitivity；新增一个可复用可视化 CLI，用已有 normalized `478 x 25` full-grid metrics 和 6 个量化 checkpoint 生成统一 FP32 / W4A32 / W4A8 / W4A4 pre/final 对比图。

新增代码：

- `SCRN_BRECQ_app/scrn_brecq/cli/visualize_quantized_scrn_grid.py`
- `SCRN_BRECQ_app/scrn_brecq/tests/test_visualize_quantized_scrn_grid.py`

验证：

- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_visualize_quantized_scrn_grid -v`：通过，5 tests。
- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/cli/visualize_quantized_scrn_grid.py`：通过。
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.visualize_quantized_scrn_grid --help`：通过。

运行信息：

- run dir：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE003_fixed_grid_visual_protocol/20260511_140941_ne003_w4a4_w4a8_w4a32_seismic_denorm_seed20260507`
- GPU：物理 GPU `1`。普通沙箱内 CUDA 不可用，实际生成命令按同参数在沙箱外执行。
- figure count：`16`
- normalized representative count：`15`
- default single sanity：`1`
- elapsed：`90.89s`

输出文件：

- `config.json`
- `metrics_summary.json`
- `selected_samples.json`
- `summary.md`
- `figures/*.png`

NE003 显示协议：

- normalized full-grid 样本显示时使用 manifest 中 `normalization_scale` 反归一化：`display = normalized * normalization_scale`。
- full-grid SNR / SSIM 仍保留 normalized per-patch absmax 口径，不因显示反归一化而改变。
- colormap 固定为 `seismic`。
- 每张 normalized 图为 3 行：
  - row 1：Clean、Input、FP32、W4A32 final、W4A8 final、W4A4 final；
  - row 2：W4A32 pre/final、W4A8 pre/final、W4A4 pre/final；
  - row 3：Input / FP32 / W4A32 final / W4A8 final / W4A4 final error map。
- prediction panels 使用同一张图内统一的 zero-centered symmetric scale；error panels 使用单独的 zero-centered symmetric scale。
- `SCRN-main/test_data` 默认单样本只作为历史 sanity 图，不进入 normalized full-grid 结论。

### NE003 full-grid 指标快照

| object | SNR mean | SNR median | SSIM mean | SSIM median |
|---|---:|---:|---:|---:|
| FP32 | 17.832885 | 18.174198 | 0.964330 | 0.978794 |
| E007 W4A32 pre | 16.614250 | 17.177490 | 0.935462 | 0.949638 |
| E007 W4A32 final | 17.785582 | 18.112757 | 0.964137 | 0.978461 |
| NE000 W4A8 pre | 17.372734 | 17.785509 | 0.962674 | 0.977030 |
| NE000 W4A8 final | 17.449507 | 17.877689 | 0.962868 | 0.977292 |
| NE000_2 W4A4 pre | 11.172733 | 11.157722 | 0.941492 | 0.955772 |
| NE000_2 W4A4 final | 12.914963 | 13.118019 | 0.939563 | 0.954078 |

重建前后变化：

| object | SNR mean delta | SNR median delta | SSIM mean delta | SSIM median delta |
|---|---:|---:|---:|---:|
| E007 W4A32 final - pre | +1.171332 | +0.805456 | +0.028675 | +0.023764 |
| NE000 W4A8 final - pre | +0.076773 | +0.006535 | +0.000195 | +0.000069 |
| NE000_2 W4A4 final - pre | +1.742231 | +1.861192 | -0.001929 | -0.002785 |

解释：

- W4A32 weight reconstruction 是明确有效的：SNR 和 SSIM 都显著提升，final 基本贴近 FP32。
- W4A8 activation reconstruction 的 full-grid 收益很小，因为 A8 init 已经接近可用；它仍是合法且可部署等价通过的 activation quantization baseline。
- W4A4 activation reconstruction 对 SNR 有明显恢复，但 SSIM 平均略降。这说明 A4 优化主要在均方/能量误差口径上恢复，结构相似性不一定同步变好；后续 NE004 不能只看 SNR，也要保留 SSIM 和代表图 error map。

### NE003 代表样本清单

| # | selection | source | patch | condition | scale | W4A4 final SNR | W4A4 SNR gain |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | E012 low missing continuity | Anisotropic | `test_000044.npy` | 4 | 0.260053 | 13.516 | +3.153 |
| 2 | E012 mid continuity | Anisotropic | `test_000013.npy` | 12 | 0.095750 | 14.195 | +3.404 |
| 3 | E012 high continuity | Anisotropic | `test_000025.npy` | 20 | 0.133937 | 16.486 | +3.086 |
| 4 | E012 low continuity | Kerry3D | `test_000081.npy` | 4 | 0.623027 | 5.825 | +0.076 |
| 5 | E012 mid continuity | Kerry3D | `test_000084.npy` | 12 | 0.817539 | 10.817 | +0.181 |
| 6 | E012 high continuity | Kerry3D | `test_000077.npy` | 20 | 0.571108 | 7.928 | -0.254 |
| 7 | E012 low continuity | Shots0001 | `test_000374.npy` | 4 | 0.056497 | 10.554 | +2.087 |
| 8 | E012 mid continuity | Shots0001 | `test_000395.npy` | 12 | 0.043919 | 13.752 | +2.063 |
| 9 | E012 high continuity | Shots0001 | `test_000349.npy` | 20 | 0.069847 | 16.642 | +1.028 |
| 10 | W4A4 worst final SNR | Anisotropic | `test_000034.npy` | 18 | 0.123766 | -1.504 | -0.153 |
| 11 | W4A4 best final SNR | Shots0001 | `test_000334.npy` | 20 | 0.055231 | 18.571 | +1.803 |
| 12 | W4A4 median final SNR | Shots0001 | `test_000372.npy` | 8 | 0.048749 | 13.118 | +2.394 |
| 13 | W4A4 max reconstruction gain | Anisotropic | `test_000037.npy` | 22 | 0.095750 | 16.447 | +3.912 |
| 14 | W4A4 worst reconstruction change | Kerry3D | `test_000091.npy` | 20 | 0.748988 | 13.302 | -1.363 |
| 15 | W4A4 max SSIM drop | Kerry3D | `test_000086.npy` | 22 | 0.746568 | 12.714 | -1.107 |
| 16 | default single sanity | `SCRN-main/test_data` | `clear.npy` | -1 | n/a | 13.270 | +0.097 |

后续使用规则：

- NE004-NE006 每个候选策略都必须先报告 normalized `478 x 25` full-grid 指标，再用 NE003 同一批代表样本输出可视化。
- 重点观察 W4A4：
  - Anisotropic 上 W4A4 reconstruction 往往能带来较大 SNR 恢复；
  - Kerry3D 存在 reconstruction 变差与 SSIM drop 代表样本，是后续 sensitivity / range / granularity 优先排查对象；
  - Shots0001 多数样本有中等恢复，是判断策略是否泛化的重要 source。
- NE003 不改变任何实验结论本身；它固定的是“后续怎么比较”和“图怎么看”的协议。

## 2026-05-11 NE002 / NE003 整理版与 NE004-NE006 后续计划

本节把 NE002 和 NE003 的结果整理成后续实验可以直接引用的结论，并明确 NE004、NE005、NE006 分别要做什么、怎么做、判断什么。后续主对象按当前讨论调整为 `NE000_2 W4A4`；`NE000 W4A8` 作为成功参照，`E007 W4A32` 和 FP32 作为上下限参照。

### NE002 / NE003 已经确认的事实

| 类别 | 结论 | 对后续实验的意义 |
|---|---|---|
| checkpoint legality | W4A4 pre/final、W4A8 pre/final、W4A32 pre/final 全部 verification passed。 | 后续不再优先怀疑 checkpoint reload、bitwidth、quant state 错误。 |
| activation 状态 | W4A4/W4A8 均为 `weight_quant=true, act_quant=true`，activation delta `52/52` 初始化；W4A32 为 `act_quant=false`。 | W4A4 低指标是真实 A4 activation pressure，不是“没开/开错 activation quant”。 |
| state toggle | W4A4/W4A8 `all_on` 与 `all_off` 输出明显不同；二者 `all_off` 回到 W4A32-like 行为。 | activation quantizer 开关有效，NE004 可以做分组 sensitivity。 |
| packed deployment | W4A32/W4A8/W4A4 packed restored 与 fake-quant checkpoint full-grid mean SNR delta 均远小于 `0.01 dB`。 | 当前结果不仅存在于恢复型 `.pth` 中，也能经 packed artifact 恢复对齐。 |
| fixed grid | NE003 固定 normalized `478 x 25` 为正式评估口径。 | 后续所有策略必须用同一 grid 判断，不用默认单样本做结论。 |
| fixed visuals | NE003 固定 15 个 normalized 代表样本 + 1 个默认单样本 sanity，使用 `seismic` 和 manifest 反归一化显示。 | 后续候选策略要复用同一批图，避免图像选择偏差。 |

核心对比指标：

| object | SNR mean / median | SSIM mean / median | 角色 |
|---|---:|---:|---|
| FP32 | 17.832885 / 18.174198 | 0.964330 / 0.978794 | normalized 上限。 |
| E007 W4A32 final | 17.785582 / 18.112757 | 0.964137 / 0.978461 | weight-only 主参照，几乎贴近 FP32。 |
| NE000 W4A8 final | 17.449507 / 17.877689 | 0.962868 / 0.977292 | A8 成功参照，优化空间小。 |
| NE000_2 W4A4 final | 12.914963 / 13.118019 | 0.939563 / 0.954078 | 后续主优化对象。 |

重建前后变化：

| object | SNR mean delta | SSIM mean delta | 解释 |
|---|---:|---:|---|
| E007 W4A32 final - pre | +1.171332 | +0.028675 | weight reconstruction 明确有效。 |
| NE000 W4A8 final - pre | +0.076773 | +0.000195 | A8 init 已经很好，activation reconstruction 只是小幅微调。 |
| NE000_2 W4A4 final - pre | +1.742231 | -0.001929 | A4 reconstruction 恢复 SNR，但 SSIM 略降；后续不能只追 SNR。 |

当前最重要的可读结论：

- W4A8 已经足够好，后续不应把主资源放在继续提高 A8；它主要用于判断 W4A4 的异常是否是 A4 专属。
- W4A4 是合法、可恢复、可 packed 对齐的 A4 activation checkpoint，但与 W4A32 / FP32 仍有约 `4.87 dB` mean SNR gap。
- W4A4 reconstruction 的行为有 source 差异：Anisotropic 和 Shots0001 多数样本 SNR 恢复明显，Kerry3D 出现 reconstruction 变差和 SSIM drop，是后续优先关注对象。
- 后续优化目标不是“让单个样本好看”，而是在 normalized `478 x 25` grid 上缩小 W4A4 final 与 W4A32 final 的 gap，同时不能牺牲 SSIM 和 by-source 稳定性。

### NE004：W4A4 activation quantizer 分组 sensitivity

实验目标：

- 找出 W4A4 full-grid gap 主要来自哪些 activation quantizer 组。
- 验证旧 E004 的“Conv2d activation 多层累积误差主导”在新 normalized 数据 + A4 bitwidth 下是否仍成立。
- 将 NE001 的局部统计热点转化为 full-grid 行为证据，为 NE005/NE006 缩小搜索空间。

实验对象：

| 对象 | 用途 |
|---|---|
| `NE000_2 W4A4 final` | 主 sensitivity 对象。 |
| `NE000_2 W4A4 pre` | 可选对照，用于判断 activation reconstruction 是否改变热点。 |
| `NE000 W4A8 final` | 成功参照；同样分组关闭时应表现为小幅变化。 |
| `E007 W4A32 final` / FP32 | 固定上限参照，不做 activation sensitivity。 |

实验方法：

1. 使用现有 activation sensitivity 工具或补一个最小选择文件接口，固定 normalized `478 x 25` grid。
2. 对 W4A4 final 运行：
   - `all_on`：完整 W4A4 final；
   - `all_off`：activation 全关，得到 W4A32-like 参照；
   - `conv2d_off`：只关闭 Conv2d activation quantizers；
   - `linear_off`：只关闭 Linear / transformer activation quantizers；
   - `stage_output_conv_off`；
   - `split_proj_off`；
   - `merge_proj_off`；
   - `split_proj + merge_proj + stage_output_conv_off`；
   - stage-wise off：stage1、stage2、stage3、stage4、stage5；
   - role-wise off：attention projection、FFN / MLP、input/output quantizer，如工具可表达则纳入。
3. 对 W4A8 final 跑一个精简同构 sanity：`all_on`、`all_off`、`conv2d_off`、`split/merge/stage_output_conv_off`。
4. 每个 run 记录 overall、by-source、by-condition、相对 W4A4 all_on 的 SNR/SSIM gain。
5. 对最有价值的 2-3 个 sensitivity 结果复用 NE003 代表样本生成图。

验收标准：

- 每个正式 sensitivity run 行数为 `11950`。
- 明确给出“关闭哪一组最能恢复 W4A4”，以及该恢复是否集中在 Anisotropic / Kerry3D / Shots0001。
- 如果关闭某组能显著提升 W4A4 SNR 且 SSIM 不下降，NE006 优先把该组改为 finer granularity。
- 如果关闭某组只提升单一 source 或导致 SSIM 明显下降，后续不能直接作为部署策略，只作为诊断证据。

预期产物：

- `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE004_w4a4_activation_sensitivity/...`
- sensitivity 汇总表；
- top group 的 NE003 同口径代表图；
- 两份日志中的最终判断：A4 主瓶颈是 Conv2d、stage、role，还是少数组合。

### NE005：W4A4 activation range / clipping / calibration 口径实验

实验目标：

- 判断 W4A4 剩余 gap 是否主要来自 tensor-wise activation range 估计过宽、离群值、clip 策略或 calibration 分布。
- 复核旧 E005 “range / clipping 对 W4A8 帮助有限”的结论在 A4 上是否仍成立；A4 bitwidth 更低，旧结论不能直接照搬。

实验对象：

| 对象 | 用途 |
|---|---|
| `E007 W4A32 final` | 所有 W4A4 range 变体的共同起点。 |
| `NE000_2 W4A4 final` | baseline：tensor-wise A4、range method `none`、a5000。 |
| `NE000 W4A8 final` | 判断 range 变化是否只对 A4 敏感。 |

实验方法：

1. 先检查 `activation_only_quantize_scrn` 当前支持的 `activation_range_method` 和 clipping 参数；如不支持需要的策略，单独做小补丁并测试。
2. 保持核心变量不变：
   - `n_bits_w=4`
   - `n_bits_a=4`
   - `activation_granularity=tensor`
   - `num_samples=1024`
   - `batch_size=16`
   - `init_batch_size=64`
   - `iters_a=5000`
   - `activation_lr=0.0004`
   - `lp_norm=2.4`
3. 只改 range / clipping / calibration 相关变量，候选包括：
   - percentile clipping：如 p99、p99.5、p99.9、p99.99；
   - MSE grid range init；
   - absmax / no-clipping baseline 复跑；
   - 按 NE001 / NE004 热点只对敏感 group 试 clipping；
   - 可选：校准集 source balance 或 per-source calibration 对照。
4. 每个候选都跑 pre-act 和 final full-grid eval，记录重建前后变化。
5. 对有希望的候选复用 NE003 代表图，尤其检查 Kerry3D worst recon change 和 max SSIM drop。

验收标准：

- 候选策略必须在 normalized `478 x 25` 上超过 NE000_2 W4A4 final，而不是只提升单样本。
- 主要比较：
  - W4A4 final mean/median SNR 是否提高；
  - SSIM mean/median 是否不再下降；
  - by-source 是否改善 Kerry3D 且不明显伤害 Anisotropic / Shots0001；
  - activation delta、bitwidth、packed restore 是否仍合法。
- 若 range / clipping 全部收益有限，则明确排除“简单调 range 能解决 A4”的路径，进入 NE006 granularity。

预期产物：

- `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping/...`
- range variant 表；
- 每个 variant 的 pre/final 指标与相对 NE000_2 的 gain；
- 是否值得做 packed export 的判断。

### NE006：W4A4 structured granularity / selective fine granularity

实验目标：

- 在 NE004 定位出的敏感组上测试 finer activation granularity，寻找 W4A4 可部署候选。
- 复核旧 E006 的核心发现：selective fine granularity 可能优于 all Conv2d fine granularity；但 NE006 必须以新 normalized 数据和 A4 为准。

实验对象：

| 对象 | 用途 |
|---|---|
| `E007 W4A32 final` | 所有 W4A4 granularity 变体共同起点。 |
| `NE000_2 W4A4 final` | tensor-wise A4 baseline。 |
| `NE000 W4A8 final` | 成功参照，不作为主要优化目标。 |

实验方法：

1. 保持 NE000_2 的 reconstruction 设置不变，只改 activation granularity / 选择组。
2. 至少测试以下候选：
   - all Conv2d activation per-channel；
   - all Conv2d activation group-wise g4；
   - `split_proj + merge_proj + stage_output_conv` per-channel；
   - `split_proj + merge_proj + stage_output_conv` g4；
   - `stage_output_conv` g4；
   - stage5 单独 finer granularity 或 stage5 保持 tensor-wise 的对照；
   - Linear / transformer sanity：只改 Linear 或只关 Linear，确认是否不是主路径。
3. 每个候选都从 E007 W4A32 final 出发重跑 activation init + activation reconstruction，随后做 normalized full-grid eval。
4. 对 top candidates 生成 NE003 同口径代表图。
5. 对最终候选做 verification 和 packed deployment equivalence，确认不是 fake-quant-only 结果。

验收标准：

- 目标是 W4A4 final 明显缩小与 W4A32 final 的 gap，同时保持 SSIM 不恶化。
- 优先级：
  - 若 per-channel 提升最大但部署成本高，记录为上限参考；
  - 若 g4 接近 per-channel 且 packed 等价通过，优先作为部署候选；
  - 若 all Conv2d 不如 selective，保留旧 E006 的机制链条；
  - 若 stage5 finer granularity 仍有害，应明确记录为禁用方向。
- 最终至少给出：
  - W4A4 tensor-wise baseline；
  - 最强质量候选；
  - 最强部署友好候选；
  - 与 W4A8 / W4A32 / FP32 的 gap 表。

预期产物：

- `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_structured_granularity/...`
- granularity strategy 表；
- top candidates 的 NE003 代表图；
- verification / packed equivalence；
- 是否进入后续 mixed precision 或 selective FP32 的决策。

### NE004-NE006 执行顺序建议

1. 先做 NE004：它决定“该改哪里”，避免 NE005/NE006 盲扫。
2. 再做 NE005：如果 A4 是 range / clipping 问题，成本最低；若无效，尽早排除。
3. 最后做 NE006：用 NE004 定位和 NE005 排除结果指导 selective granularity，寻找真正候选。

阶段性成功标准：

- 短期：W4A4 final mean SNR 从 `12.914963` 明显提高，且 SSIM 不再继续下降。
- 中期：找到一个 W4A4 selective granularity 策略，full-grid 指标稳定优于 tensor-wise W4A4，并通过 packed equivalence。
- 长期：形成一条清晰机制结论，说明 normalized 数据下 A8 为什么成功、A4 为什么仍有 gap、哪些结构性策略能恢复 A4。

## 2026-05-11 NE004 grid sensitivity evaluator 实现记录

为 NE004 正式实验新增 normalized fixed-grid activation sensitivity 工具。原因是旧 `evaluate_activation_sensitivity.py` 仍使用旧 multi-sample degradation 口径，不能作为 NE003 固定的 normalized `478 x 25` 正式结论。

代码变更：

- 新增 `SCRN_BRECQ_app/scrn_brecq/cli/evaluate_activation_sensitivity_grid.py`
  - 复用 `evaluate_quantized_scrn_grid.py` 的 fixed grid、manifest source map、SNR/SSIM 和 aggregation。
  - 支持 `all_on`、`all_off`、`disable_group`、`enable_group` 等现有 sensitivity mode。
  - 输出 `config.json`、`metrics.json`、`summary.md`、`per_sample_metrics.jsonl`、`selected_quantizers.csv`。
- 扩展 `SCRN_BRECQ_app/scrn_brecq/quant/activation_sensitivity.py`
  - 新增 plural OR selector：`stages`、`branches`、`roles`、`module_types`。
  - 保留原有 singular selector，跨字段仍为 AND，同一 plural 字段内部为 OR。
- 更新测试：
  - `SCRN_BRECQ_app/scrn_brecq/tests/test_activation_sensitivity.py`
  - `SCRN_BRECQ_app/scrn_brecq/tests/test_evaluate_activation_sensitivity_grid.py`

验证：

- TDD RED 已确认：
  - plural selector 缺失时报 `unexpected keyword argument 'roles'`；
  - grid CLI 缺失时报 `ModuleNotFoundError`。
- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_sensitivity -v`：7 tests 通过。
- `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_evaluate_activation_sensitivity_grid -v`：5 tests 通过。
- `conda run -n quant python -m py_compile SCRN_BRECQ_app/scrn_brecq/cli/evaluate_activation_sensitivity_grid.py`：通过。
- `conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_activation_sensitivity_grid --help`：通过。

Smoke：

- 命令对象：`NE000_2 W4A4 final`
- mode：`disable_group --module-types Conv2d`
- subset：`num_eval_samples=2`、`snr_settings=1`、`missing_rates=0.08`
- GPU：物理 GPU `1`；普通沙箱内 CUDA 不可用，按同参数升级执行。
- run dir：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE004_w4a4_activation_sensitivity/smoke/20260511_190920_ne004_smoke_w4a4_conv2d_off_2patch_1cond`
- selected quantizers：`31`
- rows：`2`
- quant SNR mean / median：`19.1097 / 19.1097`
- quant SSIM mean：`0.973638`

下一步：

- 以该 grid sensitivity CLI 跑 NE004 W4A4 full-grid run matrix。
- 再跑 W4A8 sanity subset。
- 汇总各组相对 W4A4 all_on 的 full-grid gain，判断 NE005/NE006 优先方向。

## 2026-05-11 NE004 W4A4 activation sensitivity 结果记录

NE004 使用新 `evaluate_activation_sensitivity_grid.py` 完成 normalized `478 x 25 = 11950` full-grid sensitivity。执行 GPU 为物理 GPU `1`；普通沙箱内 `torch.cuda.is_available()` 为 false，正式矩阵按完全相同参数升级执行。GPU 2 当时仍有外部 `swinir` 进程，未使用。

验收：

- W4A4 full matrix：17 个 run 全部完成。
- W4A8 sanity subset：4 个 run 全部完成。
- 每个 run 都生成 `per_sample_metrics.jsonl`、`metrics.json`、`config.json`、`summary.md`、`selected_quantizers.csv`。
- 每个 full-grid run 的 row count 都是 `11950`。
- selected quantizer count 与计划一致：all on/off `51`，Conv2d `31`，Linear `20`，split/merge/stage-output `15`，stage-output `5`，单 stage `10`。

输出根目录：

- `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE004_w4a4_activation_sensitivity/eval`

关键 run dirs：

- W4A4 all_on：`20260511_191550_ne004a_w4a4_all_on_grid478_seed20260507`
- W4A4 all_off：`20260511_191826_ne004b_w4a4_all_off_grid478_seed20260507`
- W4A4 Conv2d off：`20260511_192011_ne004c_w4a4_conv2d_off_grid478_seed20260507`
- W4A4 Linear off：`20260511_192222_ne004d_w4a4_linear_off_grid478_seed20260507`
- W4A4 split/merge/stage-output off：`20260511_193421_ne004i_w4a4_split_merge_stage_output_off_grid478_seed20260507`
- W4A4 stage5 off：`20260511_195435_ne004m5_w4a4_stage5_off_grid478_seed20260507`
- W4A8 all_on：`20260511_200058_ne004n_w4a8_all_on_grid478_seed20260507`
- W4A8 all_off：`20260511_200344_ne004o_w4a8_all_off_grid478_seed20260507`
- W4A8 Conv2d off：`20260511_200535_ne004p_w4a8_conv2d_off_grid478_seed20260507`
- W4A8 split/merge/stage-output off：`20260511_200753_ne004q_w4a8_split_merge_stage_output_off_grid478_seed20260507`

W4A4 overall 结果，相对 gain 均以 W4A4 all_on 为基准：

| run | selected | SNR mean | SNR median | SSIM mean | gain |
|---|---:|---:|---:|---:|---:|
| all_on | 51 | 12.9150 | 13.1180 | 0.939563 | +0.0000 |
| all_off | 51 | 17.7856 | 18.1128 | 0.964137 | +4.8706 |
| Conv2d off | 31 | 17.1117 | 17.6774 | 0.960019 | +4.1968 |
| Linear off | 20 | 13.2336 | 13.4659 | 0.940903 | +0.3187 |
| cnn branch off | 15 | 13.4094 | 13.6834 | 0.943983 | +0.4944 |
| fusion branch off | 10 | 13.5998 | 13.8398 | 0.949349 | +0.6848 |
| transformer branch off | 20 | 13.2336 | 13.4659 | 0.940903 | +0.3187 |
| stage_output_conv off | 5 | 13.8209 | 14.0836 | 0.945819 | +0.9060 |
| split_proj+merge_proj+stage_output_conv off | 15 | 14.8177 | 15.0830 | 0.955504 | +1.9027 |
| conv role off | 15 | 13.4094 | 13.6834 | 0.943983 | +0.4944 |
| attention_qkv+attention_proj off | 10 | 12.9862 | 13.1711 | 0.940272 | +0.0712 |
| mlp off | 10 | 13.1516 | 13.3905 | 0.940516 | +0.2366 |
| stage1 off | 10 | 13.7005 | 13.9380 | 0.944190 | +0.7855 |
| stage2 off | 10 | 12.8254 | 13.0275 | 0.940593 | -0.0895 |
| stage3 off | 10 | 13.0420 | 13.2555 | 0.940532 | +0.1270 |
| stage4 off | 10 | 13.7307 | 14.0081 | 0.943316 | +0.8157 |
| stage5 off | 10 | 14.1329 | 14.4166 | 0.951537 | +1.2179 |

W4A8 sanity subset，相对 gain 均以 W4A8 all_on 为基准：

| run | selected | SNR mean | SNR median | SSIM mean | gain |
|---|---:|---:|---:|---:|---:|
| W4A8 all_on | 51 | 17.4495 | 17.8777 | 0.962868 | +0.0000 |
| W4A8 all_off | 51 | 17.7856 | 18.1128 | 0.964137 | +0.3361 |
| W4A8 Conv2d off | 31 | 17.7775 | 18.1059 | 0.964084 | +0.3279 |
| W4A8 split/merge/stage-output off | 15 | 17.6917 | 18.0479 | 0.963734 | +0.2422 |

By-source 重点结果：

| run | Anisotropic mean / gain | Kerry3D mean / gain | Shots0001 mean / gain |
|---|---:|---:|---:|
| W4A4 all_on | 13.2802 / +0.0000 | 9.0332 / +0.0000 | 13.0047 / +0.0000 |
| W4A4 all_off | 21.9718 / +8.6915 | 9.6082 / +0.5750 | 17.3124 / +4.3077 |
| W4A4 Conv2d off | 20.4664 / +7.1862 | 9.5707 / +0.5375 | 16.7734 / +3.7687 |
| W4A4 split/merge/stage-output off | 16.3668 / +3.0866 | 9.4215 / +0.3883 | 14.7406 / +1.7359 |
| W4A4 stage_output_conv off | 14.8241 / +1.5439 | 9.2018 / +0.1686 | 13.8175 / +0.8128 |
| W4A4 stage5 off | 15.1255 / +1.8453 | 9.3795 / +0.3463 | 14.1370 / +1.1324 |

结论：

- W4A4 的 activation gap 是真实且主要由 activation quantization 造成的：`all_off` 回到 E007 W4A32 final，说明权重量化状态没有混入异常。
- Conv2d activation 是主导因素：关闭 31 个 Conv2d quantizer 追回 `+4.1968 dB`，接近 all_off 的 `+4.8706 dB`；关闭 20 个 Linear/transformer quantizer 只有 `+0.3187 dB`。
- 结构化小集合仍然有效：`split_proj + merge_proj + stage_output_conv` 只覆盖 15 个 quantizer，却追回 `+1.9027 dB`，是目前最强 selective group。
- `stage_output_conv` 单独 5 个 quantizer 追回 `+0.9060 dB`，stage5 单独关闭追回 `+1.2179 dB`；stage1/stage4 也有中等收益，stage2 单独关闭略负。
- Attention 和 MLP 不是主线：attention group 仅 `+0.0712 dB`，MLP 仅 `+0.2366 dB`。
- W4A8 的 gap 很小，但方向一致：W4A8 Conv2d off 追回 `+0.3279 dB`，几乎等于 all_off 的 `+0.3361 dB`。这说明 A8 成功不是因为 activation quantization 未开启，而是 normalized 数据和 A8 bitwidth 下 Conv2d activation 误差已经很小。

与旧 E 系列的关系：

- 与旧 E004 一致：主因仍是 Conv2d activation，不是 Linear/transformer，不是单个 attention/MLP 组。
- 与旧 E006C 一致：`split_proj + merge_proj + stage_output_conv` 仍是比普通 branch/stage 更有信息量的结构化候选。
- 与旧 E 系列不同的是：normalized 数据下 W4A8 不再崩坏，A4 才重新暴露出强 activation sensitivity。因此 NE 系列的主对象应继续以 W4A4 为中心，W4A8 作为成功参照。
- `stage5 off` 在 sensitivity 中有收益，但这不等价于“stage5 finer granularity 一定有益”。旧 E006C 曾观察 stage5 独立细粒度可能有害；NE006 需要把 stage5 作为高风险组单独验证，而不能直接把 disable 结果外推为 granularity 策略。

下一步决策：

- NE005 仍可按计划做 range / clipping sanity，但预期它只能解释一部分问题；NE004 已显示主要恢复来自 Conv2d 结构组，而不是全局开关。
- NE006 应优先测试：
  - all Conv2d per-channel / g4，作为质量上限和部署上限参考；
  - `split_proj + merge_proj + stage_output_conv` per-channel / g4，作为最强 selective 候选；
  - `stage_output_conv` g4；
  - stage5 单独 finer granularity 与 stage5 保持 tensor-wise 对照。
- 后续正式比较仍必须使用 normalized `478 x 25` grid，不能回退到旧 multi-sample sensitivity 口径。

## 2026-05-11 NE004 全量结果二次整理

本节把 NE004 所有正式 run 按统一解释口径重新整理，避免只看少数关键结果。NE004 的读法是：`all_on` 是量化模型原始状态；`all_off` 是同一个 W4A4/W4A8 checkpoint 关闭 activation quantizer 后的 weight-only 参照；`disable_group` 表示只关闭指定 activation quantizer 组，其相对 `all_on` 的 gain 表示该组 activation quantization 对当前损失的贡献上限。注意 sensitivity disable 不是最终部署策略，不能直接等同于 finer granularity 一定有效。

W4A4 overall 全量表：

| group | selected | SNR mean | SNR median | SSIM mean | SSIM median | SNR gain | SSIM gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_on | 51 | 12.9150 | 13.1180 | 0.939563 | 0.954078 | +0.0000 | +0.000000 |
| all_off | 51 | 17.7856 | 18.1128 | 0.964137 | 0.978461 | +4.8706 | +0.024574 |
| Conv2d off | 31 | 17.1117 | 17.6774 | 0.960019 | 0.974592 | +4.1968 | +0.020456 |
| Linear off | 20 | 13.2336 | 13.4659 | 0.940903 | 0.954589 | +0.3187 | +0.001340 |
| cnn branch off | 15 | 13.4094 | 13.6834 | 0.943983 | 0.958733 | +0.4944 | +0.004420 |
| fusion branch off | 10 | 13.5998 | 13.8398 | 0.949349 | 0.964306 | +0.6848 | +0.009786 |
| transformer branch off | 20 | 13.2336 | 13.4659 | 0.940903 | 0.954589 | +0.3187 | +0.001340 |
| stage_output_conv off | 5 | 13.8209 | 14.0836 | 0.945819 | 0.960161 | +0.9060 | +0.006256 |
| split+merge+stage_output off | 15 | 14.8177 | 15.0830 | 0.955504 | 0.970028 | +1.9027 | +0.015941 |
| conv role off | 15 | 13.4094 | 13.6834 | 0.943983 | 0.958733 | +0.4944 | +0.004420 |
| attention off | 10 | 12.9862 | 13.1711 | 0.940272 | 0.954257 | +0.0712 | +0.000709 |
| mlp off | 10 | 13.1516 | 13.3905 | 0.940516 | 0.955035 | +0.2366 | +0.000953 |
| stage1 off | 10 | 13.7005 | 13.9380 | 0.944190 | 0.958810 | +0.7855 | +0.004627 |
| stage2 off | 10 | 12.8254 | 13.0275 | 0.940593 | 0.954710 | -0.0895 | +0.001030 |
| stage3 off | 10 | 13.0420 | 13.2555 | 0.940532 | 0.954761 | +0.1270 | +0.000969 |
| stage4 off | 10 | 13.7307 | 14.0081 | 0.943316 | 0.957741 | +0.8157 | +0.003753 |
| stage5 off | 10 | 14.1329 | 14.4166 | 0.951537 | 0.966062 | +1.2179 | +0.011974 |

W4A4 gain 排名和占 all_off gap 比例：

| group | SNR gain | all_off gap share |
|---|---:|---:|
| all_off | +4.8706 | 100.0% |
| Conv2d off | +4.1968 | 86.2% |
| split+merge+stage_output off | +1.9027 | 39.1% |
| stage5 off | +1.2179 | 25.0% |
| stage_output_conv off | +0.9060 | 18.6% |
| stage4 off | +0.8157 | 16.7% |
| stage1 off | +0.7855 | 16.1% |
| fusion branch off | +0.6848 | 14.1% |
| conv role off | +0.4944 | 10.2% |
| cnn branch off | +0.4944 | 10.2% |
| transformer branch off | +0.3187 | 6.5% |
| Linear off | +0.3187 | 6.5% |
| mlp off | +0.2366 | 4.9% |
| stage3 off | +0.1270 | 2.6% |
| attention off | +0.0712 | 1.5% |
| stage2 off | -0.0895 | -1.8% |

W4A4 by-source 全量表，单元格为 `SNR mean / gain`：

| group | Anisotropic | Kerry3D | Shots0001 |
|---|---:|---:|---:|
| all_on | 13.2802 / +0.0000 | 9.0332 / +0.0000 | 13.0047 / +0.0000 |
| all_off | 21.9718 / +8.6915 | 9.6082 / +0.5750 | 17.3124 / +4.3077 |
| Conv2d off | 20.4664 / +7.1862 | 9.5707 / +0.5375 | 16.7734 / +3.7687 |
| Linear off | 13.7033 / +0.4231 | 9.1653 / +0.1321 | 13.3108 / +0.3062 |
| cnn branch off | 13.9821 / +0.7019 | 9.1507 / +0.1175 | 13.4744 / +0.4698 |
| fusion branch off | 14.3097 / +1.0295 | 9.2607 / +0.2275 | 13.6416 / +0.6369 |
| transformer branch off | 13.7033 / +0.4231 | 9.1653 / +0.1321 | 13.3108 / +0.3062 |
| stage_output_conv off | 14.8241 / +1.5439 | 9.2018 / +0.1686 | 13.8175 / +0.8128 |
| split+merge+stage_output off | 16.3668 / +3.0866 | 9.4215 / +0.3883 | 14.7406 / +1.7359 |
| conv role off | 13.9821 / +0.7019 | 9.1507 / +0.1175 | 13.4744 / +0.4698 |
| attention off | 13.3643 / +0.0841 | 9.1376 / +0.1044 | 13.0720 / +0.0673 |
| mlp off | 13.6062 / +0.3260 | 9.0553 / +0.0221 | 13.2328 / +0.2282 |
| stage1 off | 14.5999 / +1.3197 | 9.2133 / +0.1801 | 13.7117 / +0.7071 |
| stage2 off | 13.0996 / -0.1806 | 9.0359 / +0.0027 | 12.9290 / -0.0757 |
| stage3 off | 13.3865 / +0.1062 | 9.0349 / +0.0018 | 13.1409 / +0.1362 |
| stage4 off | 14.6582 / +1.3780 | 9.1442 / +0.1110 | 13.7406 / +0.7359 |
| stage5 off | 15.1255 / +1.8453 | 9.3795 / +0.3463 | 14.1370 / +1.1324 |

W4A8 sanity 全量表：

| group | selected | SNR mean | SNR median | SSIM mean | SSIM median | SNR gain |
|---|---:|---:|---:|---:|---:|---:|
| W4A8 all_on | 51 | 17.4495 | 17.8777 | 0.962868 | 0.977292 | +0.0000 |
| W4A8 all_off | 51 | 17.7856 | 18.1128 | 0.964137 | 0.978461 | +0.3361 |
| W4A8 Conv2d off | 31 | 17.7775 | 18.1059 | 0.964084 | 0.978429 | +0.3279 |
| W4A8 split+merge+stage_output off | 15 | 17.6917 | 18.0479 | 0.963734 | 0.978095 | +0.2422 |

按输入 SNR setting 的关键趋势，括号为相对 W4A4 all_on 的 gain：

| group | -2 dB | -1 dB | 1 dB | 5 dB | 10 dB |
|---|---:|---:|---:|---:|---:|
| all_on | 11.6505 / +0.0000 | 12.0574 / +0.0000 | 12.7252 / +0.0000 | 13.6971 / +0.0000 | 14.4447 / +0.0000 |
| Conv2d off | 14.9635 / +3.3130 | 15.5722 / +3.5148 | 16.6447 / +3.9195 | 18.4045 / +4.7074 | 19.9737 / +5.5291 |
| split+merge+stage_output off | 13.1685 / +1.5180 | 13.6601 / +1.6026 | 14.5119 / +1.7867 | 15.8227 / +2.1256 | 16.9252 / +2.4806 |
| stage5 off | 12.6297 / +0.9793 | 13.1003 / +1.0428 | 13.8837 / +1.1585 | 15.0560 / +1.3589 | 15.9948 / +1.5502 |
| all_off | 15.3934 / +3.7430 | 16.0500 / +3.9926 | 17.2188 / +4.4936 | 19.2096 / +5.5125 | 21.0561 / +6.6115 |

按 missing rate 的关键趋势，括号为相对 W4A4 all_on 的 gain：

| group | 0.02 | 0.08 | 0.18 | 0.28 | 0.38 |
|---|---:|---:|---:|---:|---:|
| all_on | 13.9245 / +0.0000 | 13.6206 / +0.0000 | 13.0601 / +0.0000 | 12.4028 / +0.0000 | 11.5668 / +0.0000 |
| Conv2d off | 18.1624 / +4.2378 | 17.8429 / +4.2223 | 17.2784 / +4.2183 | 16.5907 / +4.1879 | 15.6843 / +4.1175 |
| split+merge+stage_output off | 15.8166 / +1.8920 | 15.5298 / +1.9092 | 14.9793 / +1.9192 | 14.3164 / +1.9136 | 13.4464 / +1.8796 |
| stage5 off | 15.2414 / +1.3169 | 14.8965 / +1.2760 | 14.2799 / +1.2198 | 13.5663 / +1.1635 | 12.6803 / +1.1135 |
| all_off | 18.8276 / +4.9030 | 18.4873 / +4.8667 | 17.9192 / +4.8590 | 17.2641 / +4.8613 | 16.4298 / +4.8630 |

二次分析：

- 模块类型层面，Conv2d off 追回 all_off gap 的 `86.2%`，而 Linear/transformer 只有 `6.5%`。这比单看 overall 数值更清楚地说明 A4 激活误差不是平均分布在所有 quantizer 上，而是高度集中在 Conv2d activation。
- 结构角色层面，`split_proj + merge_proj + stage_output_conv` 追回 `39.1%` 的 all_off gap，是最强的部署可操作小集合。它同时提升 Anisotropic、Kerry3D、Shots0001，没有只对单一 source 有效。
- source 层面，Anisotropic 对 A4 activation 最敏感：all_off 可追回 `+8.6915 dB`，Conv2d off 可追回 `+7.1862 dB`；Shots0001 次之；Kerry3D 的绝对 SNR 低且 activation-off gain 小，后续优化不应只用 Kerry3D 判断策略成败。
- 输入 SNR setting 层面，Conv2d / selective 组在高输入 SNR 条件下 gain 更大，说明 activation quantization 更限制“本来模型能恢复得更好”的样本；这类样本最适合用来观察视觉差异。
- missing rate 层面，Conv2d 和 split/merge/stage-output 的 gain 在 `0.02` 到 `0.38` 基本稳定，说明 NE004 结论不是由某个缺失率条件偶然驱动。
- stage 层面，stage5 off 的收益最高，但 stage2 off 轻微负收益。该现象提示 activation sensitivity 存在跨层补偿，不能把 disable_group 结果机械外推成“该 stage finer granularity 一定更好”。
- W4A8 sanity 与 W4A4 方向一致但幅度小：W4A8 Conv2d off 追回 `0.3279 / 0.3361 = 97.6%` 的 A8 activation gap。A8 成功不是状态错误，而是 normalized 数据 + 8bit activation 已经把 Conv2d 误差压到较小。

对后续实验的直接约束：

- NE005 range/clipping 可以做，但如果收益不接近 `+1 dB`，应尽快收束，不要在全局 range 参数上长时间扫。
- NE006 的主线应以 Conv2d granularity 为中心，优先顺序为：
  1. `split_proj + merge_proj + stage_output_conv` g4 / per-channel；
  2. all Conv2d g4 / per-channel 上限参考；
  3. `stage_output_conv` g4；
  4. stage5 独立策略作为风险对照；
  5. Linear / attention / MLP 只保留 sanity，不应作为主优化方向。
- 后续报告 W4A4 改进时，必须同时报告 overall、by-source、by-input-SNR、by-missing-rate，避免一个策略只改善单一条件却被误判为通用提升。

## 2026-05-11 关于最终输出 activation quantizer 禁用的说明

当前 W+A 量化实现中，`act_quant=true` 表示中间层激活量化整体开启；但模型最后一个 `QuantModule` 的 activation quantizer 会通过 `disable_network_output_quantization()` 被禁用。也就是说，结构上每个 `QuantModule` 都有 activation quantizer 状态，W4A4/W4A8 checkpoint 中可以看到 52 个 activation quantizer 的 `delta/zero_point`；但实际前向部署状态下，最后网络输出层的 activation quantizer 被 bypass。因此更准确的表述是：中间激活参与量化，最终预测输出不再额外量化。

这样设计的原因：

- SCRN 的最终输出已经是恢复后的单通道地震 patch，不会继续作为下一层输入。对最终输出再做一次 4bit/8bit fake quantization，主要影响的是保存/评估结果本身，而不是后续低精度计算链路。
- SNR/SSIM 和可视化比较都直接基于最终预测 patch。如果最终输出也被量化，指标会混入一层额外的输出网格截断误差，难以区分中间激活量化误差和最终结果格式误差。
- BRECQ/PTQ 的核心目标是让量化网络内部计算尽量接近 FP32 路径；保留最终输出为浮点，更利于和 FP32、W4A32、W4A8/W4A4 之间做公平质量比较。

潜在优点：

- 评估更公平：SNR/SSIM 主要反映中间层权重与激活量化带来的模型误差，而不是最终输出被强制压到低 bit 网格后的显示/存储误差。
- 视觉结果更稳定：地震 patch 输出通常用于连续幅值图像展示，保留浮点输出可以避免最后一步人为带来的条带化或离散化伪影。
- 与当前 packed deployment 验证口径一致：packed 恢复后的模型保持内部 fake-quant 行为，并将最终输出作为浮点预测参与等价性验证。

潜在缺点或注意事项：

- 如果未来目标硬件要求最终输出也必须以 INT4/INT8 张量形式落盘或传给后处理算子，那么当前评估会略偏乐观，需要单独增加“output quantized”部署口径。
- checkpoint 中仍保存最后一个 activation quantizer 的状态，容易让人误以为 52 个激活量化器全部实际生效。后续报告应明确区分“结构上存在 52 个 quantizer”和“实际启用约 51 个，中间层启用、最终输出禁用”。
- 对比不同实验时必须保持该策略一致。若某个实验打开最终输出量化，而另一个实验关闭，就不能直接比较 SNR/SSIM。

当前结论：最终输出 activation quantizer 禁用不是状态错误，而是有意的评估和部署口径选择。除非后续明确研究“最终输出也低 bit 存储/传输”的部署场景，否则 W4A4/W4A8 主线应继续保持中间激活量化、最终输出不量化。

## 2026-05-12 NE005 W4A4 range / clipping sanity 结果

目标：验证 normalized 新协议下，`NE000_2 W4A4` 的 A4 activation gap 是否能通过 tensor-wise range / clipping 低成本修复。NE005 不改 granularity、不做 sensitivity、不做 packed，只比较 activation range 初始化策略及其后续 activation reconstruction 的 full-grid 结果。

执行环境与固定设置：

- 代码改动：无。
- GPU：物理 GPU `1`，执行前 GPU 1 空闲；GPU 2 有外部 python 进程占用约 `436 MiB`，未使用。
- 起点 checkpoint：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_2_normalized_w4a4_activation_reconstruction_probe/inputs/e007_w4a32_nbitsa4_metadata_seed.pth`
- Calibration：`SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- Eval：`SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`
- 统一参数：`n_bits_w=4`，`n_bits_a=4`，`activation_granularity=tensor`，`num_samples=1024`，`batch_size=16`，`init_batch_size=64`，`iters_a=5000`，`activation_lr=0.0004`，`lp_norm=2.4`。
- 正式评估口径：normalized `478 x 25` grid，`11950` rows，seed `20260507`。
- 对照：NE000_2 W4A4 final 为 `12.9150 / 13.1180` SNR mean/median，SSIM mean `0.939563`；E007 W4A32 final 为 `17.7856 / 18.1128` SNR mean/median，SSIM mean `0.964137`。

Run 目录：

| id | range method / selector | selected range quantizers | quant run | eval run |
|---|---|---:|---|---|
| NE005a | max / all activation candidates | 51 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/quant/20260511_205121_ne005a_w4a4_range_max_tensor_a5000_1024cali_gpu1` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/eval/20260512_001617_ne005a_w4a4_range_max_grid478_seed20260507` |
| NE005b | percentile 99.9 / all activation candidates | 51 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/quant/20260511_211446_ne005b_w4a4_percentile_p999_tensor_a5000_1024cali_gpu1` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/eval/20260512_002037_ne005b_w4a4_percentile_p999_grid478_seed20260507` |
| NE005c | percentile 99.99 / all activation candidates | 51 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/quant/20260511_213752_ne005c_w4a4_percentile_p9999_tensor_a5000_1024cali_gpu1` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/eval/20260512_002456_ne005c_w4a4_percentile_p9999_grid478_seed20260507` |
| NE005d | mse_grid / all activation candidates | 51 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/quant/20260511_220200_ne005d_w4a4_mse_grid_tensor_a5000_1024cali_gpu1` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/eval/20260512_002916_ne005d_w4a4_mse_grid_grid478_seed20260507` |
| NE005e | mse_grid / Conv2d only | 31 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/quant/20260511_224146_ne005e_w4a4_mse_grid_conv2d_only_a5000_1024cali_gpu1` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/eval/20260512_003336_ne005e_w4a4_mse_grid_conv2d_only_grid478_seed20260507` |
| NE005f | mse_grid / split_proj + merge_proj + stage_output_conv | 15 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/quant/20260511_230441_ne005f_w4a4_mse_grid_split_merge_stage_output_a5000_1024cali_gpu1` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/eval/20260512_003756_ne005f_w4a4_mse_grid_split_merge_stage_output_grid478_seed20260507` |
| NE005g | percentile 99.9 / split_proj + merge_proj + stage_output_conv | 15 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/quant/20260511_232726_ne005g_w4a4_percentile_p999_split_merge_stage_output_a5000_1024cali_gpu1` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE005_w4a4_range_clipping_sanity/eval/20260512_004215_ne005g_w4a4_percentile_p999_split_merge_stage_output_grid478_seed20260507` |

合法性验证：

- 7 个 final checkpoint 均 `passed=true`。
- 7 个 final checkpoint 均为 `weight_quant=true, act_quant=true`。
- activation bitwidth 均为 `4`，`activation_delta_count=52`，`initialized_activation_quantizers=52`。
- weight bit counts 均保持 `4bit=50, 8bit=2`。
- `non_positive_delta_count=0`，`level_offender_count=0`。
- 7 个 grid eval 均生成 `11950` rows，并包含 `metrics.json`、`config.json`、`summary.md`、`per_sample_metrics.jsonl`。

总体结果，正式数值来自 normalized `478 x 25` grid：

| variant | selected | rows | pre SNR/SSIM mean | final SNR/SSIM mean | final SNR/SSIM median | recon gain SNR/SSIM | final vs NE000_2 W4A4 | remaining SNR gap to E007 W4A32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NE005a max | 51 | 11950 | 7.8501 / 0.748581 | 12.9330 / 0.938922 | 13.1377 / 0.953237 | +5.0829 / +0.190341 | +0.0180 / -0.000641 | -4.8526 |
| NE005b p99.9 | 51 | 11950 | 10.5794 / 0.925247 | 12.2617 / 0.936723 | 12.4995 / 0.950616 | +1.6824 / +0.011476 | -0.6532 / -0.002840 | -5.5238 |
| NE005c p99.99 | 51 | 11950 | 11.8503 / 0.892493 | 12.8394 / 0.937749 | 13.0574 / 0.951726 | +0.9892 / +0.045256 | -0.0755 / -0.001814 | -4.9462 |
| NE005d mse_grid all | 51 | 11950 | 12.1405 / 0.900725 | 12.8898 / 0.939433 | 13.0843 / 0.954021 | +0.7492 / +0.038708 | -0.0252 / -0.000130 | -4.8958 |
| NE005e mse_grid Conv2d | 31 | 11950 | 12.0340 / 0.901563 | 12.8932 / 0.939358 | 13.0856 / 0.953876 | +0.8592 / +0.037795 | -0.0217 / -0.000205 | -4.8923 |
| NE005f mse_grid selective | 15 | 11950 | 11.0522 / 0.909328 | 12.8956 / 0.939338 | 13.0950 / 0.953766 | +1.8434 / +0.030010 | -0.0194 / -0.000225 | -4.8900 |
| NE005g p99.9 selective | 15 | 11950 | 11.1966 / 0.931744 | 12.8730 / 0.939889 | 13.0629 / 0.954297 | +1.6764 / +0.008145 | -0.0420 / +0.000326 | -4.9126 |

关键解释：

- 最佳 SNR 变体是 NE005a `max`，但相对 NE000_2 W4A4 final 只有 `+0.0180 dB`，远低于 NE005 预设的 `+0.5 dB` 最低继续投入阈值。
- 最佳 SSIM mean 变体是 NE005g selective p99.9，为 `0.939889`，只比 NE000_2 W4A4 final 高 `+0.000326`，幅度也不足以构成有效改进。
- p99.9 全局 clipping 明显有害：SNR mean 从 `12.9150` 掉到 `12.2617`，说明 A4 不是简单靠强 percentile clipping 就能修复。
- p99.99 轻 clipping 也低于 baseline：`12.8394`，说明 clipping 强度调轻后仍没有收益。
- mse_grid 能明显改变 pre-act-recon：全量 mse_grid 的 pre SNR mean 达到 `12.1405`，比 NE000_2 原始 pre SNR mean `11.1727` 高约 `+0.9678 dB`；但经过 activation reconstruction 后 final 反而略低于 NE000_2 final。这说明 range 初始化可以改变起点，但当前 activation reconstruction 会把不同 range 初始化收敛到接近且不优于原 baseline 的区域。
- Conv2d-only 和 selective mse_grid 没有带来收益：NE005e `12.8932`，NE005f `12.8956`，均低于 NE000_2 final。这点与 NE004 “Conv2d 是误差主因”不冲突；NE004 说明关掉这些 activation quantizer 能恢复大量 gap，但 NE005 说明仅在 tensor-wise A4 下重选 range 不能恢复该 gap。

重点 source 对比，单元格为 `final SNR mean / 相对 NE000_2 W4A4 gain`：

| variant | Anisotropic | Kerry3D | Shots0001 | min source gain | max source gain |
|---|---:|---:|---:|---:|---:|
| NE005a max | 13.3243 / +0.0441 | 9.0339 / +0.0008 | 13.0184 / +0.0137 | +0.0008 | +0.0441 |
| NE005b p99.9 | 12.6868 / -0.5934 | 8.8181 / -0.2150 | 12.3217 / -0.6829 | -0.6829 | -0.2150 |
| NE005c p99.99 | 13.2607 / -0.0195 | 8.9967 / -0.0365 | 12.9166 / -0.0880 | -0.0880 | -0.0195 |
| NE005d mse_grid all | 13.2573 / -0.0229 | 9.0292 / -0.0040 | 12.9781 / -0.0265 | -0.0265 | -0.0040 |
| NE005e mse_grid Conv2d | 13.2633 / -0.0169 | 9.0363 / +0.0032 | 12.9810 / -0.0237 | -0.0237 | +0.0032 |
| NE005f mse_grid selective | 13.2668 / -0.0134 | 9.0315 / -0.0017 | 12.9834 / -0.0213 | -0.0213 | -0.0017 |
| NE005g p99.9 selective | 13.2485 / -0.0317 | 9.0577 / +0.0246 | 12.9580 / -0.0467 | -0.0467 | +0.0246 |

最佳 SNR 变体 NE005a 的 by-input-SNR 稳定性：

| input SNR setting | NE005a final SNR | gain vs NE000_2 W4A4 |
|---:|---:|---:|
| -2 | 11.6748 | +0.0243 |
| -1 | 12.0769 | +0.0194 |
| 1 | 12.7424 | +0.0172 |
| 5 | 13.7116 | +0.0145 |
| 10 | 14.4594 | +0.0147 |

最佳 SNR 变体 NE005a 的 by-missing-rate 稳定性：

| missing rate | NE005a final SNR | gain vs NE000_2 W4A4 |
|---:|---:|---:|
| 0.02 | 13.9624 | +0.0378 |
| 0.08 | 13.6566 | +0.0360 |
| 0.18 | 13.0809 | +0.0207 |
| 0.28 | 12.4077 | +0.0049 |
| 0.38 | 11.5574 | -0.0094 |

NE005 结论：

- range / clipping 路线收益有限，应收束。最佳 SNR 改进只有 `+0.0180 dB`，不满足 `+0.5 dB` 的继续投入标准，更不接近 `+1.0 dB` 的强信号标准。
- NE004 的主因判断仍成立：A4 gap 主要来自 Conv2d activation，尤其 selective 小集合；但 NE005 排除了“在 tensor-wise A4 下只调 range 就能修复”的低成本路径。
- 后续应优先进入 NE006 granularity：先做 `split_proj + merge_proj + stage_output_conv` 的 g4 / per-channel，再做 all Conv2d g4 / per-channel 上限参考，stage_output_conv g4 与 stage5 独立作为对照。
- 如果未来需要组合策略，range 只适合作为 granularity 之后的小幅辅助项，而不是主线。

### NE005 完整结果补充与高可读结论

如何读 NE005 指标：

- `pre SNR/SSIM`：只做 activation range 初始化、尚未做 activation reconstruction 的 W4A4 结果。它反映 range 初始化质量，但不能单独代表最终模型。
- `final SNR/SSIM`：activation reconstruction 后的正式 W4A4 结果，后续决策只看这个字段。
- `recon gain`：final 相对 pre 的变化。gain 大不代表最终好，只说明重建从差起点追回更多。
- `final vs NE000_2 W4A4`：本实验最重要字段，表示 range/clipping 是否真的超过原始 W4A4 baseline。
- `gap to E007 W4A32`：距离 weight-only 上限还差多少。NE005 的所有变体都仍有约 `4.85 dB` 以上 SNR gap。

含 NE000_2 baseline 的总表：

| variant | range | selected | pre SNR mean/median | final SNR mean/median | final SSIM mean/median | recon gain SNR/SSIM | final vs NE000_2 | gap to W4A32 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NE000_2 baseline | none baseline | 51 | 11.1727 / 11.1577 | 12.9150 / 13.1180 | 0.939563 / 0.954078 | +1.7422 / -0.001929 | +0.0000 / +0.000000 | -4.8706 / -0.024574 |
| NE005a max | max all | 51 | 7.8501 / 8.2980 | 12.9330 / 13.1377 | 0.938922 / 0.953237 | +5.0829 / +0.190341 | +0.0180 / -0.000641 | -4.8526 / -0.025215 |
| NE005b p99.9 | p99.9 all | 51 | 10.5794 / 10.7291 | 12.2617 / 12.4995 | 0.936723 / 0.950616 | +1.6824 / +0.011476 | -0.6532 / -0.002840 | -5.5238 / -0.027414 |
| NE005c p99.99 | p99.99 all | 51 | 11.8503 / 12.3437 | 12.8394 / 13.0574 | 0.937749 / 0.951726 | +0.9892 / +0.045256 | -0.0755 / -0.001814 | -4.9462 / -0.026388 |
| NE005d mse all | mse all | 51 | 12.1405 / 12.6383 | 12.8898 / 13.0843 | 0.939433 / 0.954021 | +0.7492 / +0.038708 | -0.0252 / -0.000130 | -4.8958 / -0.024704 |
| NE005e mse Conv2d | mse Conv2d | 31 | 12.0340 / 12.4996 | 12.8932 / 13.0856 | 0.939358 / 0.953876 | +0.8592 / +0.037795 | -0.0217 / -0.000205 | -4.8923 / -0.024779 |
| NE005f mse selective | mse split+merge+stage_output | 15 | 11.0522 / 11.2294 | 12.8956 / 13.0950 | 0.939338 / 0.953766 | +1.8434 / +0.030010 | -0.0194 / -0.000225 | -4.8900 / -0.024799 |
| NE005g p99.9 selective | p99.9 split+merge+stage_output | 15 | 11.1966 / 11.2490 | 12.8730 / 13.0629 | 0.939889 / 0.954297 | +1.6764 / +0.008145 | -0.0420 / +0.000326 | -4.9126 / -0.024248 |

运行时间与 single-sample sanity。该表只用于解释运行成本和单样本误导风险，正式判断仍以上表 full-grid 为准：

| variant | init s | recon s | elapsed min | single pre SNR/SSIM | single final SNR/SSIM | single recon gain |
|---|---:|---:|---:|---:|---:|---:|
| NE005a max | 25.2 | 1369.4 | 23.32 | 10.2431 / 0.792973 | 13.2465 / 0.902703 | +3.0033 / +0.109730 |
| NE005b p99.9 | 25.5 | 1351.8 | 23.04 | 12.3065 / 0.845850 | 13.1131 / 0.882638 | +0.8066 / +0.036788 |
| NE005c p99.99 | 25.2 | 1414.9 | 24.05 | 12.6385 / 0.877608 | 13.2652 / 0.904508 | +0.6267 / +0.026900 |
| NE005d mse all | 25.4 | 2352.1 | 39.69 | 12.7619 / 0.884633 | 13.2842 / 0.900688 | +0.5223 / +0.016055 |
| NE005e mse Conv2d | 25.6 | 1340.6 | 22.84 | 12.7369 / 0.885862 | 13.2878 / 0.900895 | +0.5509 / +0.015034 |
| NE005f mse selective | 25.3 | 1331.2 | 22.67 | 12.8111 / 0.883907 | 13.2766 / 0.902313 | +0.4655 / +0.018405 |
| NE005g p99.9 selective | 25.4 | 1328.5 | 22.62 | 12.5781 / 0.858694 | 13.2749 / 0.903449 | +0.6967 / +0.044754 |

single-sample 与 full-grid 的差异很重要：单样本上 NE005e/NE005d/NE005f 看起来都比 NE005a 好，但 full-grid 上只有 NE005a 在 SNR mean 上略微超过 baseline，而且幅度只有 `+0.0180 dB`。因此 NE 系列后续仍必须坚持 `478 x 25` grid，不能用单样本选择策略。

by-source 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`。当前 normalized test manifest 只有 `Anisotropic`、`Kerry3D`、`Shots0001` 三个 source：

| variant | Anisotropic | Kerry3D | Shots0001 |
|---|---:|---:|---:|
| NE000_2 baseline | 13.28 / +0.00 | 9.03 / +0.00 | 13.00 / +0.00 |
| NE005a max | 13.32 / +0.04 | 9.03 / +0.00 | 13.02 / +0.01 |
| NE005b p99.9 | 12.69 / -0.59 | 8.82 / -0.22 | 12.32 / -0.68 |
| NE005c p99.99 | 13.26 / -0.02 | 9.00 / -0.04 | 12.92 / -0.09 |
| NE005d mse all | 13.26 / -0.02 | 9.03 / -0.00 | 12.98 / -0.03 |
| NE005e mse Conv2d | 13.26 / -0.02 | 9.04 / +0.00 | 12.98 / -0.02 |
| NE005f mse selective | 13.27 / -0.01 | 9.03 / -0.00 | 12.98 / -0.02 |
| NE005g p99.9 selective | 13.25 / -0.03 | 9.06 / +0.02 | 12.96 / -0.05 |

by-source 结论：没有任何变体在三个 source 上形成稳定有效提升。NE005a 的提升主要是 Anisotropic 上 `+0.04 dB` 的极小改动；NE005g 的 SSIM mean 略高，但 SNR 在 Anisotropic 和 Shots0001 都下降。range/clipping 没有解决 NE004 暴露的 source-sensitive activation gap。

by-input-SNR 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | -2 dB | -1 dB | 1 dB | 5 dB | 10 dB |
|---|---:|---:|---:|---:|---:|
| NE000_2 baseline | 11.65 / +0.00 | 12.06 / +0.00 | 12.73 / +0.00 | 13.70 / +0.00 | 14.44 / +0.00 |
| NE005a max | 11.67 / +0.02 | 12.08 / +0.02 | 12.74 / +0.02 | 13.71 / +0.01 | 14.46 / +0.01 |
| NE005b p99.9 | 11.12 / -0.53 | 11.50 / -0.56 | 12.11 / -0.61 | 12.98 / -0.72 | 13.61 / -0.84 |
| NE005c p99.99 | 11.60 / -0.06 | 12.00 / -0.06 | 12.66 / -0.06 | 13.61 / -0.08 | 14.33 / -0.11 |
| NE005d mse all | 11.64 / -0.01 | 12.04 / -0.02 | 12.70 / -0.02 | 13.66 / -0.03 | 14.41 / -0.04 |
| NE005e mse Conv2d | 11.64 / -0.01 | 12.04 / -0.02 | 12.70 / -0.02 | 13.67 / -0.03 | 14.41 / -0.03 |
| NE005f mse selective | 11.64 / -0.01 | 12.04 / -0.01 | 12.71 / -0.02 | 13.67 / -0.03 | 14.42 / -0.03 |
| NE005g p99.9 selective | 11.63 / -0.02 | 12.02 / -0.03 | 12.69 / -0.04 | 13.64 / -0.05 | 14.39 / -0.06 |

by-input-SNR 结论：NE005a 的微小收益在各 SNR setting 上都只有 `+0.01` 到 `+0.02 dB`，没有实质意义；p99.9 全局 clipping 对高输入 SNR 样本伤害更大，`10 dB` 条件下下降 `-0.84 dB`，说明强 clipping 会破坏本来恢复潜力更高的样本。

by-missing-rate 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | 0.02 | 0.08 | 0.18 | 0.28 | 0.38 |
|---|---:|---:|---:|---:|---:|
| NE000_2 baseline | 13.92 / +0.00 | 13.62 / +0.00 | 13.06 / +0.00 | 12.40 / +0.00 | 11.57 / +0.00 |
| NE005a max | 13.96 / +0.04 | 13.66 / +0.04 | 13.08 / +0.02 | 12.41 / +0.00 | 11.56 / -0.01 |
| NE005b p99.9 | 12.99 / -0.93 | 12.80 / -0.82 | 12.41 / -0.65 | 11.90 / -0.50 | 11.21 / -0.36 |
| NE005c p99.99 | 13.82 / -0.11 | 13.53 / -0.10 | 12.98 / -0.08 | 12.34 / -0.06 | 11.53 / -0.04 |
| NE005d mse all | 13.92 / -0.01 | 13.61 / -0.01 | 13.04 / -0.02 | 12.37 / -0.04 | 11.52 / -0.05 |
| NE005e mse Conv2d | 13.92 / -0.01 | 13.61 / -0.01 | 13.04 / -0.02 | 12.37 / -0.03 | 11.53 / -0.04 |
| NE005f mse selective | 13.92 / -0.01 | 13.61 / -0.01 | 13.04 / -0.02 | 12.37 / -0.03 | 11.53 / -0.04 |
| NE005g p99.9 selective | 13.90 / -0.02 | 13.59 / -0.03 | 13.02 / -0.04 | 12.35 / -0.05 | 11.51 / -0.06 |

by-missing-rate 结论：NE005a 的微小收益主要集中在低 missing rate 条件，`0.38` 时已经变成 `-0.01 dB`；p99.9 全局 clipping 对低 missing rate 伤害最大，`0.02` 时下降 `-0.93 dB`。这再次说明 clipping 不是稳定修复策略。

按可读性整理的最终判断：

1. NE005 没有找到有效 range/clipping 修复。最佳 SNR 只提升 `+0.0180 dB`，最佳 SSIM 只提升 `+0.000326`，都属于实验噪声级别。
2. range 初始化会改变 pre-act 结果，但 activation reconstruction 后 final 几乎回到同一水平，说明当前瓶颈不是简单的 tensor-wise range 选得不够好。
3. 强 clipping 明确有害，尤其伤害高输入 SNR 和低 missing rate 条件，这些本来是模型恢复空间较大的样本。
4. mse_grid 成本更高，尤其全量 mse_grid 用时 `39.69 min`，但 final SNR 仍低于 baseline；没有继续扩大 mse_grid 搜索的价值。
5. NE004 与 NE005 合起来给出的机制判断是：A4 gap 的主因确实在 Conv2d activation，但不是通过 tensor-wise range/clipping 可以解决，而需要改变 activation quantization granularity 或 bit allocation。
6. 下一步 NE006 应直接做 granularity：优先 `split_proj + merge_proj + stage_output_conv` 的 g4 / per-channel，再做 all Conv2d 上限参考；range/clipping 暂时只保留为 granularity 后的辅助组合，不再作为主线。

## 2026-05-12 NE006 W4A4 activation granularity 计划

NE006 的核心问题：在 normalized 新协议下，`NE000_2 W4A4` 的主要 gap 是否来自 Conv2d activation tensor-wise A4 粒度过粗；如果是，能否找到比 all Conv2d per-channel 更可部署的 selective granularity 策略。

为什么现在应该做 granularity：

- NE004 sensitivity 已经证明 W4A4 all_off 可从 `12.9150 dB` 回到接近 W4A32 的 `17.7856 dB`，activation gap 很真实。
- NE004 中 Conv2d off 追回 all_off gap 的 `86.2%`，Linear / transformer 只有 `6.5%`，主因集中在 Conv2d activation。
- NE004 中 `split_proj + merge_proj + stage_output_conv` off 可追回 `39.1%` 的 all_off gap，是最强的部署可操作小集合。
- NE005 排除了 tensor-wise range / clipping：最佳 range 只提升 `+0.0180 dB`，强 clipping 有害，mse_grid 成本高但 final 不优。
- 旧 E006C 在旧协议下也显示 selective Conv2d granularity 比 all Conv2d per-channel 更有潜力；NE006 要在新协议和 A4 主对象上重新验证这一点。

实验主对象：

- 主线对象：`NE000_2 W4A4 final` 的 A4 activation 量化策略。
- 起点：沿用 E007 W4A32 single-GPU best checkpoint / A4 metadata seed，保持与 NE000_2、NE005 相同的 W4 weight reconstruction 起点。
- 评估：normalized `478 x 25` grid，seed `20260507`，正式结论只看 full-grid mean/median SNR、SSIM、by-source、by-input-SNR、by-missing-rate。
- W4A8：只做 sanity subset，不作为优化主线，因为 NE000 W4A8 final 已接近 W4A32。

NE006 分层实验矩阵：

| priority | experiment | granularity | selected group | 目的 | 预期判断 |
|---:|---|---|---|---|---|
| 1 | NE006a | group-wise g4 | `split_proj + merge_proj + stage_output_conv` | 最重要部署候选，验证 selective 小集合能否低成本追回 gap | 若 `>= +1 dB`，进入代表图和 packed 候选 |
| 2 | NE006b | per-channel | `split_proj + merge_proj + stage_output_conv` | selective 上限，复验旧 E006C 的关键结论 | 若显著强于 g4，再考虑 g2/g4/g8 sweep |
| 3 | NE006c | group-wise g4 | all Conv2d | 部署友好的 Conv2d 上限参考 | 判断 selective 是否接近 all Conv2d |
| 4 | NE006d | per-channel | all Conv2d | 理论上限参考 | 如果这个也弱，说明 granularity 不是主瓶颈 |
| 5 | NE006e | group-wise g4 | `stage_output_conv` | 拆解 selective 组合，测试 stage output 单独贡献 | 若接近 NE006a，说明 stage output 是核心 |
| 6 | NE006f | group-wise g4 | `split_proj + merge_proj` | 拆解 fusion 投影贡献 | 判断 split/merge 是否需要与 stage output 绑定 |
| 7 | NE006g | group-wise g4 | stage5 | 风险对照；NE004 stage5 off 强，但旧 E006C 中 stage5 独立细粒度可能有害 | 只作为风险验证，不默认主线 |
| 8 | NE006h | per-channel | stage5 | stage5 上限风险对照 | 若有害，明确禁止 stage5 独立策略 |
| 9 | NE006i | group-wise g4 | Linear / transformer sanity | 排除项 | 只确认 Linear/transformer 不是主方向 |
| 10 | NE006j | W4A8 sanity subset | all Conv2d 与 selective 小集合 | 检查 A8 与 A4 方向是否一致 | 不投入大规模 A8 优化 |

建议先执行的最小闭环：

1. `split_proj + merge_proj + stage_output_conv g4`
2. `split_proj + merge_proj + stage_output_conv per-channel`
3. `all Conv2d g4`
4. `all Conv2d per-channel`

这 4 个实验可以直接回答 NE006 的主问题：selective 细粒度是否有效、g4 是否足够、selective 是否接近 all Conv2d、per-channel 上限有多高。只有这 4 个有明确收益后，再拆 `stage_output_conv`、`split+merge`、`stage5`。

固定实验设置建议：

- `n_bits_w=4`，`n_bits_a=4`。
- 保持 `num_samples=1024`、`batch_size=16`、`init_batch_size=64`、`iters_a=5000`、`activation_lr=0.0004`、`lp_norm=2.4`，与 NE000_2 / NE005 对齐。
- 执行计划修正：细粒度 activation delta shape 需要通过 `activation_range_method=mse_grid` 写入，因此 NE006 实际使用 `mse_grid`；解释上仍把主变量限定为 granularity，不把 NE005 的 tensor-wise range/clipping 作为主线。
- 使用物理 GPU 优先级 `1 -> 2 -> 3 -> 0`；单卡默认 GPU 1。
- 输出目录建议：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/quant` 和 `.../eval`。

正式验收与解释标准：

- 每个 final checkpoint 必须通过 `verify_quantized_scrn`：`passed=true`，`weight_quant=true`，`act_quant=true`，activation bitwidth `4`，`activation_delta_count=52`，`non_positive_delta_count=0`，`level_offender_count=0`。
- 每个 grid eval 必须为 `11950` rows。
- 与 NE000_2 W4A4 final 对比：
  - `< +0.5 dB`：不作为主线；
  - `+0.5 ~ +1.0 dB`：辅助价值，但不足以成为核心策略；
  - `>= +1.0 dB`：强信号，进入代表图、packed equivalence 和后续组合优化；
  - `>= +2.0 dB`：优先级提高，可考虑直接作为 W4A4 主候选。
- 同时报告：
  - overall mean/median SNR、SSIM；
  - by-source，重点 `Anisotropic`、`Kerry3D`、`Shots0001`；
  - by-input-SNR，确认高输入 SNR 是否追回更多；
  - by-missing-rate，确认策略不是只改善单一 missing 条件；
  - pre-act 与 final，判断 activation reconstruction 是否放大或抵消 granularity 收益。

NE006 的关键决策树：

- 如果 `selective g4` 已经接近 `selective per-channel`，优先保留 `g4`，因为部署更友好。
- 如果 `selective per-channel` 强而 `selective g4` 弱，再做 g2/g4/g8 小 sweep。
- 如果 `all Conv2d` 明显强于 selective，说明当前 selective 小集合还不够，需要回到 NE004 的 group 分解继续扩集合。
- 如果 `all Conv2d per-channel` 仍提升有限，说明 A4 问题不只是 granularity，应考虑 mixed precision 或 bit allocation。
- 如果 `stage5` 独立策略有害，则后续禁止把 stage5 单独作为 fine granularity 默认策略，只能作为组合中谨慎纳入。

预期结论方向：

- 最可能成为 W4A4 主候选的是 `split_proj + merge_proj + stage_output_conv g4` 或其 per-channel 版本。
- `all Conv2d per-channel` 主要是上限，不一定适合部署。
- `stage5` 需要单独验证风险，不应因为 NE004 stage5 off gain 高就直接作为优化策略。
- W4A8 只做 sanity，不应挤占 W4A4 主线资源。

## 2026-05-12 NE006 W4A4 activation granularity 核心结果

本轮完成 NE006 的 4 个最小闭环实验，目标是验证 W4A4 A4 activation gap 是否能通过 Conv2d selective granularity 修复。实际执行时按实现要求使用 `--activation-range-method mse_grid` 生成 per-channel / group-wise activation delta shape；解释上仍归类为 granularity 实验，不把 NE005 的 tensor-wise range/clipping 路线重新打开。

固定设置：

- 起点：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_2_normalized_w4a4_activation_reconstruction_probe/inputs/e007_w4a32_nbitsa4_metadata_seed.pth`
- calibration：`SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- eval：normalized `478 x 25` grid，seed `20260507`
- activation reconstruction：`num_samples=1024`、`batch_size=16`、`init_batch_size=64`、`iters_a=5000`、`activation_lr=0.0004`、`lp_norm=2.4`
- GPU：NE006a 用 GPU 1；NE006b 用 GPU 3；NE006c 用 GPU 0；NE006d 因 GPU 2 持续有外部进程，转用 GPU 3。NE006c 运行期间 GPU 0 出现外部 `swinir` 进程约 `868 MiB`，未 OOM，实验参数未改。

产物目录与合法性：

| variant | quant dir | eval dir | selected | verification |
|---|---|---|---:|---|
| NE006a selective g4 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/quant/20260512_012630_ne006a_w4a4_g4_split_merge_stage_output_a5000_1024cali_gpu1` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/eval/20260512_113739_ne006a_w4a4_g4_split_merge_stage_output_grid478_seed20260507` | 15 | passed |
| NE006b selective per-channel | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/quant/20260512_012630_ne006b_w4a4_pc_split_merge_stage_output_a5000_1024cali_gpu3` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/eval/20260512_113739_ne006b_w4a4_pc_split_merge_stage_output_grid478_seed20260507` | 15 | passed |
| NE006c all Conv2d g4 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/quant/20260512_012629_ne006c_w4a4_g4_all_conv2d_a5000_1024cali_gpu0` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/eval/20260512_113739_ne006c_w4a4_g4_all_conv2d_grid478_seed20260507` | 31 | passed |
| NE006d all Conv2d per-channel | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/quant/20260512_102116_ne006d_w4a4_pc_all_conv2d_a5000_1024cali_gpu3` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/eval/20260512_114157_ne006d_w4a4_pc_all_conv2d_grid478_seed20260507` | 31 | passed |

四个 final checkpoint 均为合法 W4A4：`weight_quant=true`、`act_quant=true`、`n_bits_a=4`、`activation_delta_count=52`、`initialized_activation_quantizers=52`、`non_positive_delta_count=0`、`level_offender_count=0`、weight bit counts 为 `4bit=50 / 8bit=2`。四个 grid eval 均为 `11950` rows。

overall full-grid 结果：

| variant | granularity | selected | pre SNR mean/median | final SNR mean/median | final SSIM mean/median | recon gain SNR/SSIM | vs NE000_2 W4A4 | gap to E007 W4A32 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NE000_2 baseline | tensor | 51 | 11.1727 / 11.1577 | 12.9150 / 13.1180 | 0.939563 / 0.954078 | +1.7422 / -0.001929 | +0.0000 / +0.000000 | -4.8706 / -0.024574 |
| NE006a selective g4 | g4 | 15 | 11.6854 / 11.7630 | 13.2824 / 13.4767 | 0.943762 / 0.958305 | +1.5970 / +0.011829 | +0.3674 / +0.004199 | -4.5032 / -0.020375 |
| NE006b selective per-channel | per-channel | 15 | 11.7721 / 11.8035 | 13.2731 / 13.4541 | 0.944148 / 0.958580 | +1.5010 / +0.004267 | +0.3582 / +0.004585 | -4.5125 / -0.019989 |
| NE006c all Conv2d g4 | g4 | 31 | 12.6877 / 12.8746 | 13.7231 / 13.9766 | 0.945327 / 0.960126 | +1.0354 / +0.014151 | +0.8082 / +0.005764 | -4.0624 / -0.018810 |
| NE006d all Conv2d per-channel | per-channel | 31 | 12.5608 / 12.6432 | 13.6752 / 13.9117 | 0.945889 / 0.960663 | +1.1144 / +0.002959 | +0.7603 / +0.006326 | -4.1104 / -0.018248 |

single-sample sanity 只用于确认运行，没有用于策略优劣判断：

| variant | init s | recon s | elapsed min | single pre SNR/SSIM | single final SNR/SSIM | single recon gain |
|---|---:|---:|---:|---:|---:|---:|
| NE006a selective g4 | 26.3 | 1367.4 | 23.39 | 13.2416 / 0.893237 | 13.3777 / 0.907454 | +0.1361 / +0.014217 |
| NE006b selective per-channel | 29.1 | 1366.9 | 23.43 | 13.3452 / 0.901292 | 13.4085 / 0.904635 | +0.0633 / +0.003342 |
| NE006c all Conv2d g4 | 26.9 | 1968.7 | 33.41 | 13.0798 / 0.897449 | 13.4598 / 0.909815 | +0.3800 / +0.012366 |
| NE006d all Conv2d per-channel | 29.4 | 1314.0 | 22.44 | 13.3368 / 0.908258 | 13.4437 / 0.910167 | +0.1070 / +0.001909 |

by-source 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | Anisotropic | Kerry3D | Shots0001 |
|---|---:|---:|---:|
| NE000_2 baseline | 13.2802 / +0.00 | 9.0332 / +0.00 | 13.0047 / +0.00 |
| NE006a selective g4 | 13.8332 / +0.55 | 9.1187 / +0.09 | 13.3477 / +0.34 |
| NE006b selective per-channel | 13.7811 / +0.50 | 9.1485 / +0.12 | 13.3452 / +0.34 |
| NE006c all Conv2d g4 | 14.6337 / +1.35 | 9.1944 / +0.16 | 13.7339 / +0.73 |
| NE006d all Conv2d per-channel | 14.5110 / +1.23 | 9.2155 / +0.18 | 13.6976 / +0.69 |

by-input-SNR 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | -2 dB | -1 dB | 1 dB | 5 dB | 10 dB |
|---|---:|---:|---:|---:|---:|
| NE000_2 baseline | 11.6505 / +0.00 | 12.0574 / +0.00 | 12.7252 / +0.00 | 13.6971 / +0.00 | 14.4447 / +0.00 |
| NE006a selective g4 | 11.9312 / +0.28 | 12.3545 / +0.30 | 13.0656 / +0.34 | 14.1120 / +0.41 | 14.9485 / +0.50 |
| NE006b selective per-channel | 11.9029 / +0.25 | 12.3312 / +0.27 | 13.0518 / +0.33 | 14.1140 / +0.42 | 14.9656 / +0.52 |
| NE006c all Conv2d g4 | 12.3396 / +0.69 | 12.7811 / +0.72 | 13.5151 / +0.79 | 14.5793 / +0.88 | 15.4006 / +0.96 |
| NE006d all Conv2d per-channel | 12.2798 / +0.63 | 12.7257 / +0.67 | 13.4628 / +0.74 | 14.5364 / +0.84 | 15.3714 / +0.93 |

by-missing-rate 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | 0.02 | 0.08 | 0.18 | 0.28 | 0.38 |
|---|---:|---:|---:|---:|---:|
| NE000_2 baseline | 13.9245 / +0.00 | 13.6206 / +0.00 | 13.0601 / +0.00 | 12.4028 / +0.00 | 11.5668 / +0.00 |
| NE006a selective g4 | 14.3411 / +0.42 | 14.0199 / +0.40 | 13.4314 / +0.37 | 12.7460 / +0.34 | 11.8734 / +0.31 |
| NE006b selective per-channel | 14.3468 / +0.42 | 14.0236 / +0.40 | 13.4200 / +0.36 | 12.7241 / +0.32 | 11.8510 / +0.28 |
| NE006c all Conv2d g4 | 14.8362 / +0.91 | 14.5018 / +0.88 | 13.8812 / +0.82 | 13.1596 / +0.76 | 12.2368 / +0.67 |
| NE006d all Conv2d per-channel | 14.8009 / +0.88 | 14.4599 / +0.84 | 13.8326 / +0.77 | 13.1000 / +0.70 | 12.1826 / +0.62 |

人类可读结论：

1. NE006 证明 granularity 比 NE005 的 range/clipping 更有效。NE005 最好只有 `+0.018 dB`，NE006 selective 可到 `+0.36 dB`，all Conv2d 可到 `+0.76 ~ +0.81 dB`。
2. 但这 4 个核心实验都没有达到 `+1 dB` 强信号阈值，也没有接近 W4A32。即使最好的 NE006c，距离 E007 W4A32 final 仍有 `-4.0624 dB` mean SNR gap。
3. `split_proj + merge_proj + stage_output_conv` selective 组合有稳定正收益，但幅度不足：g4 `+0.3674 dB`，per-channel `+0.3582 dB`。这说明旧 E006C 的 selective 结论在新 normalized A4 场景下不完全复现，当前 selective 小集合不是足够强的 A4 主策略。
4. g4 不弱于 per-channel。本轮两个对照里，selective g4 略高于 selective per-channel，all Conv2d g4 也高于 all Conv2d per-channel。对 W4A4 来说，更细的 per-channel 并没有自动带来更高 full-grid SNR，后续不应默认把 per-channel 当上限最优。
5. all Conv2d 明显强于 selective，说明 NE004 里 Conv2d 主导的判断成立，但当前 selective 集合只覆盖了一部分关键误差。需要继续拆分或扩展 Conv2d 组，而不是直接把 selective g4 标为候选。
6. by-source 显示 Anisotropic 获益最大，all Conv2d g4 达到 `+1.35 dB`；Kerry3D 只有 `+0.16 ~ +0.18 dB`。W4A4 的残余 gap 仍有强 source 依赖，后续不能只看 overall。
7. by-input-SNR 显示输入越干净收益越大：all Conv2d g4 从 `-2 dB` 的 `+0.69 dB` 增至 `10 dB` 的 `+0.96 dB`，说明细粒度主要恢复高质量输入场景中的 activation 表达能力。
8. by-missing-rate 显示低 missing rate 收益更大，高 missing rate 收益下降：all Conv2d g4 从 `0.02` 的 `+0.91 dB` 降至 `0.38` 的 `+0.67 dB`。细粒度不是只改善单一条件，但在困难缺失条件下仍不足。
9. activation reconstruction 仍然重要。四个变体 final-pre mean SNR gain 都为正，`+1.04 ~ +1.60 dB`；但 selective 组 pre-act 不低，final 仍弱，说明仅在 15 个 Conv2d quantizer 上细粒度化不能充分释放 reconstruction。

后续决策：

- 暂不做 NE006 packed export，因为最佳 all Conv2d g4 只有 `+0.8082 dB`，未达到 `>= +1 dB` 主候选阈值。
- 继续 NE006e-h 拆分实验是有必要的，但优先级应调整为“寻找缺失的 Conv2d 关键组”而不是验证原 selective 小集合。
- 下一批建议：
  1. `stage_output_conv g4` 和 `split_proj + merge_proj g4` 拆分，判断 selective 小集合内谁贡献主要收益。
  2. `conv role g4`、`fusion branch g4`、`cnn branch g4`，从 NE004 的 Conv2d 主导结果里寻找比当前 selective 更接近 all Conv2d 的较小集合。
  3. `stage4/stage5 g4` 风险对照，验证旧 E006 中 stage5 独立细粒度有害的现象在 A4 normalized 协议下是否仍存在。
  4. 若仍没有 `>= +1 dB`，转向 mixed precision / selective A8，而不是继续扩大 tensor-wise range/clipping。

## 2026-05-12 NE006 后续展开计划修订

NE006 核心四组实验之后，后续路线需要从“验证旧 selective 策略”调整为“寻找 all Conv2d g4 收益中的关键缺失组”。当前证据是：

- 旧重点组合 `split_proj + merge_proj + stage_output_conv` 有稳定收益，但只有 `+0.36 dB` 左右，不足以成为 W4A4 主策略。
- `all Conv2d g4` 达到 `+0.8082 dB`，明显强于 selective，但仍未达到 `+1 dB` 强信号阈值。
- `g4` 不弱于 per-channel，甚至在 selective 和 all Conv2d 两组对照里都略高，因此下一批优先做 g4 拆分，不优先做 per-channel sweep。
- NE005 已排除 tensor-wise range/clipping 主线，后续不再回到 range/clipping，除非与 granularity 或 mixed precision 组合验证。

下一批 NE006 建议按 g4 拆分执行：

| priority | experiment | selector | 目的 |
|---:|---|---|---|
| 1 | NE006e | `stage_output_conv g4` | 判断原 selective 小集合里 stage output 是否贡献主要收益 |
| 2 | NE006f | `split_proj + merge_proj g4` | 判断 fusion 投影是否需要与 stage output 绑定 |
| 3 | NE006g | `conv role g4` | 覆盖普通 Conv2d role，寻找当前 selective 漏掉的收益 |
| 4 | NE006h | `fusion branch g4` | 从 branch 维度验证 fusion 是否比 role-based 更集中 |
| 5 | NE006i | `cnn branch g4` | 检查早期 CNN branch 是否贡献大 |
| 6 | NE006j | `stage1/2/3/4/5 Conv2d g4` | 找 stage 层级热点，特别复核 stage5 风险 |
| 7 | NE006k | `all Conv2d except stage5 g4` | 如果 stage5 独立细粒度不稳定，验证排除 stage5 后是否更稳 |

判断标准：

- 如果某个小集合达到 `+0.7 ~ +0.8 dB`，接近 all Conv2d g4，则优先围绕该集合做 g2/g4/g8 或少量 per-channel 对照。
- 如果只有 all Conv2d 有收益，小集合都明显弱，说明 A4 activation 误差更分散，后续应转向 mixed precision / selective A8，而不是继续找单一小集合。
- 如果任何拆分组合达到 `>= +1 dB`，再进入代表图、packed equivalence 和部署候选评估。
- 如果 stage5 独立策略有害，应避免把 stage5 单独作为 fine granularity 默认策略，只在组合里谨慎纳入或显式排除。

当前不建议立即做：

- 不做 packed export：最佳结果未达到 `>= +1 dB` 主候选阈值。
- 不做 per-channel 大 sweep：当前 per-channel 未优于 g4。
- 不做 range/clipping 扩展：NE005 已显示收益接近噪声或有害。

下一步实际执行应优先启动 NE006e-h 的 g4 拆分；如果这些拆分仍不能逼近 all Conv2d g4，则进入 mixed precision / selective A8 方向。

## 2026-05-12 NE006e-h W4A4 g4 granularity 拆分结果

本批完成 4 个 g4 拆分实验，用于定位 `all Conv2d g4` 的 `+0.8082 dB` 收益来自哪些 Conv2d 子组。本批不改代码，不做 packed，不做 per-channel sweep。

固定设置：

- 起点：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_2_normalized_w4a4_activation_reconstruction_probe/inputs/e007_w4a32_nbitsa4_metadata_seed.pth`
- calibration：`SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- eval：normalized `478 x 25` grid，seed `20260507`
- activation reconstruction：`num_samples=1024`、`batch_size=16`、`init_batch_size=64`、`iters_a=5000`、`activation_lr=0.0004`、`lp_norm=2.4`
- granularity：`group_wise g4`
- range shape 写入：`activation_range_method=mse_grid`、`range_loss_p=2.4`、`range_max_values_per_layer=500000`
- GPU：NE006e 用 GPU 1；NE006f 用 GPU 2；NE006g 用 GPU 3；NE006h 用 GPU 0。沙箱内 CUDA 不可用，按完全相同参数提升权限执行。

产物目录与合法性：

| variant | selector | selected | quant dir | eval dir | verification |
|---|---|---:|---|---|---|
| NE006e stage_output_conv g4 | `role=stage_output_conv,module_type=Conv2d` | 5 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/quant/20260512_120537_ne006e_w4a4_g4_stage_output_conv_a5000_1024cali_gpu1` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/eval/20260512_123059_ne006e_w4a4_g4_stage_output_conv_grid478_seed20260507` | passed |
| NE006f split+merge g4 | `split_proj + merge_proj` | 10 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/quant/20260512_120537_ne006f_w4a4_g4_split_merge_a5000_1024cali_gpu2` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/eval/20260512_123059_ne006f_w4a4_g4_split_merge_grid478_seed20260507` | passed |
| NE006g conv role g4 | `role=conv,module_type=Conv2d` | 15 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/quant/20260512_120536_ne006g_w4a4_g4_conv_role_a5000_1024cali_gpu3` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/eval/20260512_123059_ne006g_w4a4_g4_conv_role_grid478_seed20260507` | passed |
| NE006h fusion branch g4 | `branch=fusion,module_type=Conv2d` | 10 | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/quant/20260512_120535_ne006h_w4a4_g4_fusion_branch_a5000_1024cali_gpu0` | `SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/eval/20260512_123055_ne006h_w4a4_g4_fusion_branch_grid478_seed20260507` | passed |

四个 final checkpoint 均为合法 W4A4：`weight_quant=true`、`act_quant=true`、`n_bits_a=4`、`activation_delta_count=52`、`initialized_activation_quantizers=52`、`non_positive_delta_count=0`、`level_offender_count=0`、weight bit counts 为 `4bit=50 / 8bit=2`。四个 grid eval 均为 `11950` rows。

overall full-grid 结果：

| variant | selected | pre SNR mean/median | final SNR mean/median | final SSIM mean/median | recon gain SNR/SSIM | vs NE000_2 W4A4 | gap to all Conv2d g4 | gap to E007 W4A32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NE000_2 baseline | 51 | 11.1727 / 11.1577 | 12.9150 / 13.1180 | 0.939563 / 0.954078 | +1.7422 / -0.001929 | +0.0000 / +0.000000 | -0.8082 / -0.005764 | -4.8706 / -0.024574 |
| NE006e stage_output_conv g4 | 5 | 11.5374 / 11.5735 | 13.2582 / 13.4803 | 0.940681 / 0.955035 | +1.7208 / +0.002893 | +0.3433 / +0.001118 | -0.4649 / -0.004646 | -4.5274 / -0.023456 |
| NE006f split+merge g4 | 10 | 11.2482 / 11.2813 | 12.9314 / 13.1217 | 0.942282 / 0.956952 | +1.6832 / +0.006632 | +0.0164 / +0.002719 | -0.7917 / -0.003045 | -4.8542 / -0.021855 |
| NE006g conv role g4 | 15 | 11.6154 / 11.6726 | 13.2241 / 13.4930 | 0.941117 / 0.956052 | +1.6087 / +0.001176 | +0.3091 / +0.001554 | -0.4990 / -0.004210 | -4.5615 / -0.023020 |
| NE006h fusion branch g4 | 10 | 11.2482 / 11.2813 | 12.9314 / 13.1217 | 0.942282 / 0.956952 | +1.6832 / +0.006632 | +0.0164 / +0.002719 | -0.7917 / -0.003045 | -4.8542 / -0.021855 |

single-sample sanity 只用于确认运行，不用于策略优劣判断：

| variant | init s | recon s | elapsed min | single pre SNR/SSIM | single final SNR/SSIM | single recon gain |
|---|---:|---:|---:|---:|---:|---:|
| NE006e stage_output_conv g4 | 25.3 | 1327.7 | 22.63 | 13.2265 / 0.907595 | 13.2017 / 0.899682 | -0.0248 / -0.007913 |
| NE006f split+merge g4 | 25.6 | 1328.0 | 22.64 | 13.1519 / 0.893781 | 13.3712 / 0.907287 | +0.2193 / +0.013506 |
| NE006g conv role g4 | 25.3 | 1327.2 | 22.62 | 13.1746 / 0.905096 | 13.2919 / 0.904346 | +0.1173 / -0.000750 |
| NE006h fusion branch g4 | 25.5 | 1324.2 | 22.57 | 13.1519 / 0.893781 | 13.3712 / 0.907287 | +0.2193 / +0.013506 |

by-source 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | Anisotropic | Kerry3D | Shots0001 |
|---|---:|---:|---:|
| NE000_2 baseline | 13.2802 / +0.00 | 9.0332 / +0.00 | 13.0047 / +0.00 |
| NE006e stage_output_conv g4 | 13.8765 / +0.60 | 9.0623 / +0.03 | 13.3119 / +0.31 |
| NE006f split+merge g4 | 13.2709 / -0.01 | 9.0911 / +0.06 | 13.0244 / +0.02 |
| NE006g conv role g4 | 13.7999 / +0.52 | 9.1086 / +0.08 | 13.2827 / +0.28 |
| NE006h fusion branch g4 | 13.2709 / -0.01 | 9.0911 / +0.06 | 13.0244 / +0.02 |

by-input-SNR 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | -2 dB | -1 dB | 1 dB | 5 dB | 10 dB |
|---|---:|---:|---:|---:|---:|
| NE000_2 baseline | 11.6505 / +0.00 | 12.0574 / +0.00 | 12.7252 / +0.00 | 13.6971 / +0.00 | 14.4447 / +0.00 |
| NE006e stage_output_conv g4 | 11.9284 / +0.28 | 12.3515 / +0.29 | 13.0549 / +0.33 | 14.0807 / +0.38 | 14.8756 / +0.43 |
| NE006f split+merge g4 | 11.6522 / +0.00 | 12.0586 / +0.00 | 12.7332 / +0.01 | 13.7145 / +0.02 | 14.4985 / +0.05 |
| NE006g conv role g4 | 11.9770 / +0.33 | 12.3801 / +0.32 | 13.0499 / +0.32 | 14.0011 / +0.30 | 14.7125 / +0.27 |
| NE006h fusion branch g4 | 11.6522 / +0.00 | 12.0586 / +0.00 | 12.7332 / +0.01 | 13.7145 / +0.02 | 14.4985 / +0.05 |

by-missing-rate 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | 0.02 | 0.08 | 0.18 | 0.28 | 0.38 |
|---|---:|---:|---:|---:|---:|
| NE000_2 baseline | 13.9245 / +0.00 | 13.6206 / +0.00 | 13.0601 / +0.00 | 12.4028 / +0.00 | 11.5668 / +0.00 |
| NE006e stage_output_conv g4 | 14.2427 / +0.32 | 13.9623 / +0.34 | 13.4224 / +0.36 | 12.7624 / +0.36 | 11.9012 / +0.33 |
| NE006f split+merge g4 | 13.9721 / +0.05 | 13.6466 / +0.03 | 13.0696 / +0.01 | 12.4036 / +0.00 | 11.5650 / -0.00 |
| NE006g conv role g4 | 14.2861 / +0.36 | 13.9599 / +0.34 | 13.3725 / +0.31 | 12.6849 / +0.28 | 11.8172 / +0.25 |
| NE006h fusion branch g4 | 13.9721 / +0.05 | 13.6466 / +0.03 | 13.0696 / +0.01 | 12.4036 / +0.00 | 11.5650 / -0.00 |

关键结构发现：

- NE006f 和 NE006h 选中的 quantizer 完全相同，都是 5 个 stage 的 `split_proj + merge_proj` 共 10 个 Conv2d quantizer；因此 full-grid 结果完全一致。当前代码结构中 `branch=fusion,module_type=Conv2d` 等价于 `split_proj + merge_proj`。
- NE006e 选中 5 个 stage output Conv2d：`model.stage1.1` 到 `model.stage5.1`。
- NE006g 选中 15 个 `conv_branch` Conv2d。
- NE006c all Conv2d g4 选中 31 个 Conv2d，比 `stage_output_conv + split/merge + conv role` 的 30 个多出 `model.head`。因此 all Conv2d 的额外收益可能来自 head，或来自 head 与其它 Conv2d 组的组合效应。

人类可读结论：

1. stage output 是原 selective 组合里的主要贡献。NE006e 单独达到 `+0.3433 dB`，而 NE006a 的 `split_proj + merge_proj + stage_output_conv` 是 `+0.3674 dB`；加入 split/merge 只多 `+0.024 dB` 左右。
2. split/merge 基本不是有效修复方向。NE006f/NE006h 只有 `+0.0164 dB`，接近噪声级；这与旧 E006C 中 split/merge 重要的结论不一致，是 NE 系列与旧 E 系列的关键差异之一。
3. conv role 有独立正收益但也不够强。NE006g 为 `+0.3091 dB`，略弱于 stage output；它对低输入 SNR 条件收益更强，对高输入 SNR 条件收益反而下降。
4. all Conv2d g4 的 `+0.8082 dB` 不是由单一小组贡献。stage output 和 conv role 各有约 `+0.3 dB`，split/merge 近乎无效；剩余差距可能来自 head 或组合效应。
5. 当前没有任何拆分组合接近 `+0.7 ~ +0.8 dB`，因此还不能把某个小集合作为 W4A4 主候选，也不应做 packed export。
6. full-grid 与 single-sample 再次不一致：NE006e 单样本 final 低于 pre，但 full-grid 是本批最强；后续仍必须以 `478 x 25` 为准。

后续决策：

- 继续 NE006 是必要的，但下一批应从“角色拆分”转向“head 与组合效应”：
  1. `head g4`：验证 all Conv2d 中唯一未被本批覆盖的 `model.head` 是否有显著贡献。
  2. `stage_output_conv + conv role g4`：验证两个有效小组组合后是否接近 all Conv2d。
  3. `all Conv2d except split/merge g4`：验证排除无效 split/merge 后是否保留 all Conv2d 收益。
  4. `all Conv2d except head g4`：验证 head 是否是 all Conv2d 收益的重要来源。
- stage1-5 拆分仍有价值，但优先级低于 head / 组合效应。若组合实验仍不能解释 all Conv2d 收益，再进入 stage-level。
- 不进入 packed export；不做 per-channel sweep；不回到 range/clipping。

## 2026-05-12 NE006e-h 后续影响与下一批计划

NE006e-h 对后续路线的影响很明确：新 normalized W4A4 场景下，不应继续沿旧 E006C 的 `split_proj + merge_proj + stage_output_conv` selective 策略直接加码。旧 selective 组合的收益主要来自 `stage_output_conv`，而不是 `split_proj + merge_proj`。

当前关键判断：

- `split_proj + merge_proj` 从旧实验里的重要对象降级为近似无效对象。NE006f/NE006h full-grid 只有 `+0.0164 dB`，接近噪声级。
- `stage_output_conv` 是当前最有效的小集合，5 个 quantizer 带来 `+0.3433 dB`，基本解释了旧 selective 组合的大部分收益。
- `conv role` 是另一个独立有效组，15 个 quantizer 带来 `+0.3091 dB`，说明 all Conv2d 收益分散在多个 Conv2d 子组。
- `all Conv2d g4` 仍有 `+0.8082 dB`，明显强于任何单独拆分组；这说明剩余收益可能来自 `model.head`、head 与其它 Conv2d 的组合效应，或多组 activation 误差的非线性叠加。
- full-grid 与 single-sample 再次冲突，后续仍必须只用 normalized `478 x 25` grid 做策略判断。

下一批 NE006 建议只做 4 个 head / 组合 / 排除实验：

| priority | experiment | selector | 目的 | 预期解释 |
|---:|---|---|---|---|
| 1 | NE006i | `head g4` | 验证 all Conv2d 中唯一未拆出的 `model.head` 是否关键 | 如果有明显收益，head 需要纳入后续策略 |
| 2 | NE006j | `stage_output_conv + conv role g4` | 组合两个已知有效组 | 如果接近 all Conv2d，说明 split/merge 可排除 |
| 3 | NE006k | `all Conv2d except split/merge g4` | 从 all Conv2d 中移除近似无效 split/merge | 如果接近 all Conv2d，证明 split/merge 不该进主策略 |
| 4 | NE006l | `all Conv2d except head g4` | 从 all Conv2d 中移除 head | 判断 head 是否解释 all Conv2d 的剩余优势 |

执行设置应继续沿用 NE006 固定口径：

- 起点：`e007_w4a32_nbitsa4_metadata_seed.pth`
- `n_bits_w=4`、`n_bits_a=4`
- `activation_granularity=group_wise`、`activation_group_size=4`
- `activation_range_method=mse_grid`，用于写入 g4 activation delta shape
- `iters_a=5000`、`activation_lr=0.0004`、`lp_norm=2.4`
- calibration：normalized 1024 stratified calibration
- eval：normalized `478 x 25` grid，seed `20260507`

判断标准：

- 若 `all Conv2d except split/merge` 接近 NE006c all Conv2d g4，后续策略应明确排除 split/merge。
- 若 `head g4` 或 `all Conv2d except head` 显著改变结果，后续需要围绕 head 做组合验证。
- 若 `stage_output_conv + conv role` 仍明显低于 all Conv2d，说明存在 head 参与效应或多组非线性组合效应。
- 若这 4 个实验仍没有 `>= +1 dB` 的策略，NE006 granularity 单线应降级，下一阶段转向 `W4A4 + selective A8` 或 mixed precision。

当前不建议：

- 不做 packed export：没有达到主候选阈值。
- 不做 per-channel sweep：g4 当前不弱于 per-channel。
- 不回到 range/clipping：NE005 已排除主线价值。
- 不用单样本做任何策略筛选。

## 2026-05-12 NE006i-l W4A4 g4 head / combination 实验结果

本批实验按上一节计划完成 4 个 head / 组合 / 排除实验，目标是解释 NE006c `all Conv2d g4` 的 `+0.8082 dB` 收益是否来自 `head`、`stage_output_conv + conv role` 组合，或是否可以排除 `split_proj + merge_proj`。

固定口径：

- 起点：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_2_normalized_w4a4_activation_reconstruction_probe/inputs/e007_w4a32_nbitsa4_metadata_seed.pth`
- calibration：`SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- eval：`SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`
- eval grid：`478 x 25 = 11950` rows，seed `20260507`
- activation：W4A4，`group_wise`，`group_size=4`，`activation_range_method=mse_grid`
- reconstruction：`iters_a=5000`，`activation_lr=0.0004`，`lp_norm=2.4`

run 目录：

| id | quant run dir | eval run dir | GPU | selector | selected |
|---|---|---|---:|---|---:|
| NE006i | `.../quant/20260512_131527_ne006i_w4a4_g4_head_a5000_1024cali_gpu1` | `.../eval/20260512_134445_ne006i_w4a4_g4_head_grid478_seed20260507` | 1 | `role=head,module_type=Conv2d` | 1 |
| NE006j | `.../quant/20260512_131527_ne006j_w4a4_g4_stage_output_conv_role_a5000_1024cali_gpu2` | `.../eval/20260512_134445_ne006j_w4a4_g4_stage_output_conv_role_grid478_seed20260507` | 2 | `stage_output_conv + conv` | 20 |
| NE006k | `.../quant/20260512_131526_ne006k_w4a4_g4_all_conv2d_except_split_merge_a5000_1024cali_gpu3` | `.../eval/20260512_134444_ne006k_w4a4_g4_all_conv2d_except_split_merge_grid478_seed20260507` | 3 | `all Conv2d except split_proj/merge_proj` | 21 |
| NE006l | `.../quant/20260512_131526_ne006l_w4a4_g4_all_conv2d_except_head_a5000_1024cali_gpu0` | `.../eval/20260512_134444_ne006l_w4a4_g4_all_conv2d_except_head_grid478_seed20260507` | 0 | `all Conv2d except head` | 30 |

验证结果：4 个 final checkpoint 均 `passed=true`，`weight_quant=true`，`act_quant=true`，`n_bits_a=4`，`activation_delta_count=52`，`initialized_activation_quantizers=52`，`level_offender_count=0`。activation-only metrics 中 `non_positive_delta_count=0`，说明本批没有合法性问题。

overall full-grid 结果：

| variant | selected | pre SNR mean/median | final SNR mean/median | final SSIM mean/median | recon gain SNR/SSIM | vs NE000_2 W4A4 | gap to NE006c all Conv2d g4 | gap to E007 W4A32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NE000_2 baseline | 51 | 11.1727 / 11.1577 | 12.9150 / 13.1180 | 0.939563 / 0.954078 | +1.7422 / -0.001929 | +0.0000 / +0.000000 | -0.8082 / -0.005764 | -4.8706 / -0.024574 |
| NE006c all Conv2d g4 | 31 | 12.6877 / 12.8746 | 13.7231 / 13.9766 | 0.945327 / 0.960126 | +1.0354 / +0.003835 | +0.8082 / +0.005764 | +0.0000 / +0.000000 | -4.0624 / -0.018810 |
| NE006i head g4 | 1 | 11.1606 / 11.1444 | 12.9031 / 13.1007 | 0.939624 / 0.954139 | +1.7425 / -0.002001 | -0.0119 / +0.000061 | -0.8200 / -0.005703 | -4.8825 / -0.024513 |
| NE006j stage_output+conv g4 | 20 | 12.1962 / 12.3257 | 13.6186 / 13.8908 | 0.942366 / 0.956958 | +1.4224 / +0.005869 | +0.7036 / +0.002803 | -0.1045 / -0.002961 | -4.1670 / -0.021771 |
| NE006k all Conv2d except split/merge g4 | 21 | 12.1841 / 12.3104 | 13.6097 / 13.8824 | 0.942403 / 0.957057 | +1.4256 / +0.005919 | +0.6948 / +0.002840 | -0.1134 / -0.002924 | -4.1759 / -0.021734 |
| NE006l all Conv2d except head g4 | 30 | 12.6993 / 12.8885 | 13.7326 / 13.9838 | 0.945303 / 0.960038 | +1.0334 / +0.003626 | +0.8177 / +0.005740 | +0.0095 / -0.000024 | -4.0529 / -0.018834 |

single-sample sanity 只用于确认运行，不作为策略判断：

| variant | init s | recon s | elapsed min | single pre SNR/SSIM | single final SNR/SSIM | single gain |
|---|---:|---:|---:|---:|---:|---:|
| NE006i head g4 | 25.0 | 1320.9 | 22.51 | 13.1970 / 0.901847 | 13.2355 / 0.897892 | +0.0384 / -0.003955 |
| NE006j stage_output+conv g4 | 25.7 | 1340.7 | 22.86 | 13.1508 / 0.906505 | 13.3789 / 0.906634 | +0.2281 / +0.000128 |
| NE006k all Conv2d except split/merge g4 | 25.7 | 1321.4 | 22.51 | 13.1808 / 0.904265 | 13.3432 / 0.904612 | +0.1624 / +0.000347 |
| NE006l all Conv2d except head g4 | 26.1 | 1325.5 | 22.61 | 13.1257 / 0.896068 | 13.4422 / 0.906453 | +0.3165 / +0.010384 |

by-source 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | Anisotropic | Kerry3D | Shots0001 |
|---|---:|---:|---:|
| NE000_2 baseline | 13.2802 / +0.00 | 9.0332 / +0.00 | 13.0047 / +0.00 |
| NE006i head g4 | 13.2608 / -0.02 | 9.0372 / +0.00 | 12.9936 / -0.01 |
| NE006j stage_output+conv g4 | 14.4953 / +1.22 | 9.1336 / +0.10 | 13.6341 / +0.63 |
| NE006k all Conv2d except split/merge g4 | 14.4754 / +1.20 | 9.1433 / +0.11 | 13.6266 / +0.62 |
| NE006l all Conv2d except head g4 | 14.6524 / +1.37 | 9.1988 / +0.17 | 13.7419 / +0.74 |

by-input-SNR 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | -2 dB | -1 dB | 1 dB | 5 dB | 10 dB |
|---|---:|---:|---:|---:|---:|
| NE006i head g4 | 11.6400 / -0.01 | 12.0429 / -0.01 | 12.7113 / -0.01 | 13.6870 / -0.01 | 14.4342 / -0.01 |
| NE006j stage_output+conv g4 | 12.2814 / +0.63 | 12.7096 / +0.65 | 13.4247 / +0.70 | 14.4523 / +0.76 | 15.2251 / +0.78 |
| NE006k all Conv2d except split/merge g4 | 12.2715 / +0.62 | 12.7029 / +0.65 | 13.4131 / +0.69 | 14.4424 / +0.75 | 15.2187 / +0.77 |
| NE006l all Conv2d except head g4 | 12.3480 / +0.70 | 12.7912 / +0.73 | 13.5235 / +0.80 | 14.5910 / +0.89 | 15.4095 / +0.96 |

by-missing-rate 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | 0.02 | 0.08 | 0.18 | 0.28 | 0.38 |
|---|---:|---:|---:|---:|---:|
| NE006i head g4 | 13.9038 / -0.02 | 13.6028 / -0.02 | 13.0506 / -0.01 | 12.3946 / -0.01 | 11.5636 / -0.00 |
| NE006j stage_output+conv g4 | 14.6629 / +0.74 | 14.3597 / +0.74 | 13.7830 / +0.72 | 13.0911 / +0.69 | 12.1965 / +0.63 |
| NE006k all Conv2d except split/merge g4 | 14.6463 / +0.72 | 14.3440 / +0.72 | 13.7752 / +0.72 | 13.0885 / +0.69 | 12.1945 / +0.63 |
| NE006l all Conv2d except head g4 | 14.8501 / +0.93 | 14.5173 / +0.90 | 13.8922 / +0.83 | 13.1633 / +0.76 | 12.2403 / +0.67 |

关键结论：

1. `head g4` 单独无效。NE006i final SNR mean `12.9031`，比 NE000_2 baseline 还低 `-0.0119 dB`，说明 `model.head` 不是独立收益来源。
2. `all Conv2d except head` 与 `all Conv2d g4` 基本等价，甚至 SNR mean 高 `+0.0095 dB`。因此 head 不但不是关键，作为 g4 细粒度对象还可以从主策略中排除。
3. `stage_output_conv + conv role` 解释了 all Conv2d 的大部分收益。NE006j 达到 `13.6186 dB`，只比 all Conv2d g4 低 `0.1045 dB`，且比 W4A4 baseline 高 `+0.7036 dB`。
4. `all Conv2d except split/merge` 与 NE006j 几乎一致：`13.6097 dB`，比 NE006j 低 `0.0089 dB`。这进一步证明 split/merge 不是新协议 W4A4 主收益来源。
5. `all Conv2d except head` 是当前最强 W4A4 g4 结果：`13.7326 / 13.9838`，相对 NE000_2 提升 `+0.8177 dB`，略高于 NE006c all Conv2d g4。
6. by-source 收益主要来自 Anisotropic 和 Shots0001，Kerry3D 提升始终较小。NE006l 在 Anisotropic 上 `+1.37 dB`，Shots0001 上 `+0.74 dB`，Kerry3D 只有 `+0.17 dB`。
7. by-input-SNR 趋势显示收益在高输入 SNR 条件下更大；NE006l 从 `-2 dB` 条件的 `+0.70 dB` 增至 `10 dB` 条件的 `+0.96 dB`。
8. by-missing-rate 趋势显示收益在低 missing rate 更大，missing rate 越高收益越小；NE006l 从 `0.02` 的 `+0.93 dB` 降至 `0.38` 的 `+0.67 dB`。

对后续实验的影响：

- NE006 granularity 仍有价值，但当前最佳只到 `+0.8177 dB`，没有达到 `>= +1.0 dB` 强候选阈值，更没有达到可以替代 W4A8 的程度。
- 后续不应再优先研究 `split_proj + merge_proj`，也不应把 `head` 纳入 W4A4 g4 主策略。
- 当前最合理的 W4A4 g4 候选是 `all Conv2d except head`，但部署前还需要更强证据；它可以作为后续 mixed precision / selective A8 的基础组合。
- 若继续 granularity，应转向 stage-level 拆分：优先验证 `stage_output_conv + conv role` 是否只需要某些 stage，或者 `all Conv2d except head` 是否可以进一步减少 selected count。
- 若目标是把 W4A4 推近 W4A32/W4A8，单纯 g4 granularity 可能不够，下一阶段应准备 `W4A4 + selective A8` 或 mixed precision，优先把残余误差最大的 Conv2d 子集升到 A8。

下一步建议：

1. 可选继续 NE006m-q：做 `stage1/2/3/4/5 Conv2d except head g4` 或 `stage_output_conv + conv role` 的 stage-level 拆分，目标是减少 selected count 并定位 residual gain。
2. 更直接的下一阶段是 NE007：以 `all Conv2d except head g4` / `stage_output_conv + conv role g4` 为基础，做 selective A8 或 mixed precision，对 residual gap 最高的 Conv2d 子组升 A8。
3. 暂不做 packed export。当前 W4A4 g4 最强结果仍比 E007 W4A32 低 `4.0529 dB`，packed equivalence 不会改变模型质量判断。

## 2026-05-12 NE006m-q W4A4 g4 stage-level granularity 实验结果

本批继续 NE006，只做 stage-level W4A4 g4 拆分，目标是解释 `NE006l all Conv2d except head g4` 的 `+0.8177 dB` 收益是否由单个 stage 主导，或来自跨 stage 叠加。

固定口径：

- 起点：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_2_normalized_w4a4_activation_reconstruction_probe/inputs/e007_w4a32_nbitsa4_metadata_seed.pth`
- calibration：`SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- eval：`SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`
- eval grid：`478 x 25 = 11950` rows，seed `20260507`
- activation：W4A4，`group_wise`，`group_size=4`，`activation_range_method=mse_grid`
- reconstruction：`iters_a=5000`，`activation_lr=0.0004`，`lp_norm=2.4`

执行备注：

- NE006m-p 使用 GPU `1,2,3,0` 并行完成；NE006q 最初在 GPU3 启动的 run `20260512_163124_ne006q_w4a4_g4_stage5_conv2d_a5000_1024cali_gpu3` 只产出 pre-act checkpoint，没有产出 final/config/metrics，因此不纳入结果。
- NE006q 使用完全相同参数在 GPU3 重跑，实际有效 run 为 `20260512_172203_ne006q_w4a4_g4_stage5_conv2d_a5000_1024cali_gpu3_rerun`。
- eval 阶段 GPU0 出现外部进程占用，正式 grid eval 使用 GPU `1,2,3` 轮转完成，未修改 batch、seed 或实验变量。

run 目录：

| id | quant run dir | eval run dir | selected |
|---|---|---|---:|
| NE006m | `.../quant/20260512_155253_ne006m_w4a4_g4_stage1_conv2d_a5000_1024cali_gpu1` | `.../eval/20260512_175147_ne006m_w4a4_g4_stage1_conv2d_grid478_seed20260507` | 6 |
| NE006n | `.../quant/20260512_155252_ne006n_w4a4_g4_stage2_conv2d_a5000_1024cali_gpu2` | `.../eval/20260512_175147_ne006n_w4a4_g4_stage2_conv2d_grid478_seed20260507` | 6 |
| NE006o | `.../quant/20260512_155252_ne006o_w4a4_g4_stage3_conv2d_a5000_1024cali_gpu3` | `.../eval/20260512_175146_ne006o_w4a4_g4_stage3_conv2d_grid478_seed20260507` | 6 |
| NE006p | `.../quant/20260512_155252_ne006p_w4a4_g4_stage4_conv2d_a5000_1024cali_gpu0` | `.../eval/20260512_181016_ne006p_w4a4_g4_stage4_conv2d_grid478_seed20260507` | 6 |
| NE006q | `.../quant/20260512_172203_ne006q_w4a4_g4_stage5_conv2d_a5000_1024cali_gpu3_rerun` | `.../eval/20260512_181016_ne006q_w4a4_g4_stage5_conv2d_grid478_seed20260507` | 6 |

验证结果：5 个 final checkpoint 均 `passed=true`，`weight_quant=true`，`act_quant=true`，`n_bits_a=4`，`activation_delta_count=52`，`initialized_activation_quantizers=52`，`level_offender_count=0`。activation-only metrics 中 `non_positive_delta_count=0`。每个 eval 的 `per_sample_metrics.jsonl` 都是 `11950` 行。

overall full-grid 结果：

| variant | selected | pre SNR mean/median | final SNR mean/median | final SSIM mean/median | recon gain SNR/SSIM | vs NE000_2 W4A4 | gap to NE006l | gap to E007 W4A32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NE000_2 baseline | 51 | 11.1727 / 11.1577 | 12.9150 / 13.1180 | 0.939563 / 0.954078 | +1.7422 / -0.001929 | +0.0000 / +0.000000 | -0.8177 / -0.005740 | -4.8706 / -0.024574 |
| NE006l all Conv2d except head | 30 | 12.6993 / 12.8885 | 13.7326 / 13.9838 | 0.945303 / 0.960038 | +1.0334 / +0.003626 | +0.8177 / +0.005740 | +0.0000 / +0.000000 | -4.0529 / -0.018834 |
| NE006m stage1 Conv2d g4 | 6 | 11.5111 / 11.5033 | 13.1610 / 13.3724 | 0.940107 / 0.954443 | +1.6499 / +0.000399 | +0.2460 / +0.000544 | -0.5716 / -0.005195 | -4.6246 / -0.024030 |
| NE006n stage2 Conv2d g4 | 6 | 11.0960 / 11.0894 | 12.8185 / 13.0006 | 0.939369 / 0.953903 | +1.7225 / -0.000123 | -0.0965 / -0.000194 | -0.9141 / -0.005934 | -4.9671 / -0.024768 |
| NE006o stage3 Conv2d g4 | 6 | 11.1805 / 11.1818 | 12.8957 / 13.0939 | 0.939752 / 0.954255 | +1.7152 / -0.001827 | -0.0193 / +0.000189 | -0.8369 / -0.005551 | -4.8899 / -0.024385 |
| NE006p stage4 Conv2d g4 | 6 | 11.6581 / 11.7158 | 13.1833 / 13.4054 | 0.940243 / 0.954534 | +1.5252 / +0.000364 | +0.2683 / +0.000680 | -0.5493 / -0.005060 | -4.6023 / -0.023895 |
| NE006q stage5 Conv2d g4 | 6 | 11.6677 / 11.7584 | 13.2033 / 13.4520 | 0.943476 / 0.958300 | +1.5356 / +0.007208 | +0.2883 / +0.003913 | -0.5294 / -0.001827 | -4.5823 / -0.020661 |

single-sample sanity 只用于确认运行，不作为策略判断：

| variant | single pre SNR | single final SNR |
|---|---:|---:|
| NE006m stage1 | 13.2654 | 13.3391 |
| NE006n stage2 | 13.1999 | 13.2849 |
| NE006o stage3 | 13.2277 | 13.2529 |
| NE006p stage4 | 13.1922 | 13.2534 |
| NE006q stage5 | 13.0585 | 13.2928 |

by-source 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | Anisotropic | Kerry3D | Shots0001 |
|---|---:|---:|---:|
| NE006m stage1 | 13.7274 / +0.4472 | 9.0688 / +0.0356 | 13.2204 / +0.2158 |
| NE006n stage2 | 13.1214 / -0.1589 | 9.0472 / +0.0140 | 12.9157 / -0.0889 |
| NE006o stage3 | 13.2403 / -0.0399 | 9.0576 / +0.0244 | 12.9876 / -0.0171 |
| NE006p stage4 | 13.7863 / +0.5061 | 9.0643 / +0.0311 | 13.2367 / +0.2321 |
| NE006q stage5 | 13.6798 / +0.3996 | 9.0883 / +0.0552 | 13.2811 / +0.2764 |

by-input-SNR 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | -2 dB | -1 dB | 1 dB | 5 dB | 10 dB |
|---|---:|---:|---:|---:|---:|
| NE006m stage1 | 11.8556 / +0.2052 | 12.2781 / +0.2206 | 12.9693 / +0.2441 | 13.9681 / +0.2711 | 14.7339 / +0.2893 |
| NE006n stage2 | 11.5924 / -0.0580 | 11.9858 / -0.0716 | 12.6333 / -0.0918 | 13.5742 / -0.1229 | 14.3067 / -0.1380 |
| NE006o stage3 | 11.6430 / -0.0075 | 12.0441 / -0.0134 | 12.7083 / -0.0169 | 13.6710 / -0.0261 | 14.4122 / -0.0324 |
| NE006p stage4 | 11.8864 / +0.2359 | 12.3045 / +0.2470 | 12.9879 / +0.2628 | 13.9860 / +0.2889 | 14.7518 / +0.3071 |
| NE006q stage5 | 11.8788 / +0.2283 | 12.3030 / +0.2456 | 13.0011 / +0.2759 | 14.0238 / +0.3267 | 14.8096 / +0.3650 |

by-missing-rate 结果，单元格为 `final SNR mean / 相对 NE000_2 gain`：

| variant | 0.02 | 0.08 | 0.18 | 0.28 | 0.38 |
|---|---:|---:|---:|---:|---:|
| NE006m stage1 | 14.2050 / +0.2805 | 13.8991 / +0.2785 | 13.3185 / +0.2584 | 12.6297 / +0.2269 | 11.7528 / +0.1860 |
| NE006n stage2 | 13.8302 / -0.0944 | 13.5176 / -0.1030 | 12.9522 / -0.1079 | 12.3042 / -0.0986 | 11.4883 / -0.0784 |
| NE006o stage3 | 13.9245 / -0.0000 | 13.6138 / -0.0068 | 13.0433 / -0.0168 | 12.3758 / -0.0270 | 11.5211 / -0.0456 |
| NE006p stage4 | 14.1806 / +0.2561 | 13.8856 / +0.2650 | 13.3393 / +0.2791 | 12.6788 / +0.2760 | 11.8323 / +0.2655 |
| NE006q stage5 | 14.2333 / +0.3088 | 13.9186 / +0.2980 | 13.3463 / +0.2862 | 12.6797 / +0.2769 | 11.8385 / +0.2717 |

人类可读结论：

1. 单个 stage 不能解释 `all Conv2d except head` 的收益。最强单 stage 是 NE006q stage5，仅 `+0.2883 dB`；NE006l 是 `+0.8177 dB`，说明收益来自跨 stage 叠加，而不是单 stage 主导。
2. stage5、stage4、stage1 是正收益 stage。排序为 stage5 `+0.2883 dB`，stage4 `+0.2683 dB`，stage1 `+0.2460 dB`。
3. stage2 和 stage3 不应作为 g4 主策略对象。stage2 相对 baseline 为 `-0.0965 dB`，stage3 为 `-0.0193 dB`，都没有正向 full-grid 价值。
4. by-source 显示 stage1/4/5 的主要收益来自 Anisotropic 与 Shots0001，Kerry3D 始终只有很小提升。stage4 在 Anisotropic 上最高 `+0.5061 dB`，stage5 在 Shots0001 上最高 `+0.2764 dB`。
5. by-input-SNR 显示 stage1/4/5 在高输入 SNR 条件收益更大，尤其 stage5 从 `-2 dB` 的 `+0.2283 dB` 增至 `10 dB` 的 `+0.3650 dB`。
6. by-missing-rate 显示 stage5/stage4 的收益相对稳定；stage1 随 missing rate 增大收益下降更明显。

对后续实验的影响：

- 继续做纯 g4 granularity 的边际价值下降。stage-level 已证明单 stage 不够强，组合收益来自多 stage 叠加。
- 若仍在 NE006 内收尾，唯一值得补的最小组合是 `stage1 + stage4 + stage5 Conv2d g4`，因为这三个 stage 是正收益来源，selected count 18，可能接近 NE006l 的 30 个 selected count 效果。
- 更建议进入 NE007：以 W4A4 为主对象做 selective A8 / mixed precision。优先候选为 stage5、stage4、stage1 的 Conv2d，或者以 NE006l `all Conv2d except head g4` 为基础，将 residual 最大的 stage/role 升 A8。
- 不建议继续研究 stage2/stage3 g4，也不建议回到 split/merge、head 或 range/clipping。

## 2026-05-12 NE006r W4A4 g4 stage1+stage4+stage5 组合实验结果

本批作为 NE006 的 granularity 收尾验证，目标是检查单 stage 正收益来源 `stage1 + stage4 + stage5 Conv2d g4` 是否能用更少 selected quantizer 接近或超过 `NE006l all Conv2d except head g4`。实验不改代码，不改变 A4 起点、calibration、eval grid 或重建超参。

固定口径：

- 起点：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE000_2_normalized_w4a4_activation_reconstruction_probe/inputs/e007_w4a32_nbitsa4_metadata_seed.pth`
- quant run：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/quant/20260512_194621_ne006r_w4a4_g4_stage145_conv2d_a5000_1024cali_gpu1_rerun2`
- eval run：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE006_w4a4_activation_granularity/eval/20260512_202217_ne006r_w4a4_g4_stage145_conv2d_grid478_seed20260507`
- selector：`[{"stage":"stage1","module_type":"Conv2d"},{"stage":"stage4","module_type":"Conv2d"},{"stage":"stage5","module_type":"Conv2d"}]`
- selected count：18
- activation：W4A4，`group_wise`，`group_size=4`，`activation_range_method=mse_grid`
- reconstruction：`iters_a=5000`，`activation_lr=0.0004`，`lp_norm=2.4`
- eval grid：`478 x 25 = 11950` rows，seed `20260507`

执行备注：

- 首次 run `20260512_191720_ne006r_w4a4_g4_stage145_conv2d_a5000_1024cali_gpu1` 只产出 `quantized_scrn_brecq_pre_act_recon.pth`，没有 final checkpoint/config/metrics，不纳入结果。
- 一次后台执行包装未成功拉起 conda 进程，无有效 run dir，不纳入结果。
- 有效 run 为 `_rerun2`，使用 GPU1；该 run 完整产出 pre/final checkpoint、`config.json`、`metrics.json` 和 `summary.md`。
- activation initialization 用时 `25.64 s`，activation reconstruction 用时 `1317.19 s`（约 `21.95 min`）。
- single-sample sanity：`post_weight_snr=13.8918`，`pre_act_snr=13.1765`，`act_init_delta=-0.7152`，`non_positive_delta_count=0`。该单样本只用于运行 sanity，不参与正式结论。

验证结果：

- `verification_final.json`：`passed=true`
- final quant state：`weight_quant=true`，`act_quant=true`
- `n_bits_a=4`，`n_bits_w=4`
- activation quantizer：`activation_delta_count=52`，`initialized_activation_quantizers=52`
- weight bit counts：`4bit=50`，`8bit=2`
- `level_offender_count=0`

overall full-grid 结果：

| variant | selected | pre SNR mean/median | final SNR mean/median | final SSIM mean/median | recon gain SNR/SSIM | vs NE000_2 W4A4 | vs NE006l | gap to E007 W4A32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NE000_2 W4A4 baseline | 51 | 11.1727 / 11.1577 | 12.9150 / 13.1180 | 0.939563 / 0.954078 | +1.7422 / -0.001929 | +0.0000 / +0.000000 | -0.8177 / -0.005740 | -4.8706 / -0.024574 |
| NE006l all Conv2d except head g4 | 30 | 12.6993 / 12.8885 | 13.7326 / 13.9838 | 0.945303 / 0.960038 | +1.0334 / +0.003626 | +0.8177 / +0.005740 | +0.0000 / +0.000000 | -4.0529 / -0.018834 |
| NE006r stage1+stage4+stage5 Conv2d g4 | 18 | 12.6441 / 12.8061 | 13.8098 / 14.0845 | 0.945084 / 0.959776 | +1.1656 / +0.011979 | +0.8948 / +0.005521 | +0.0771 / -0.000219 | -3.9758 / -0.019053 |

by-source 结果，单元格为 `final SNR mean / 相对 NE000_2 gain / 相对 NE006l gain`：

| source | NE006r result |
|---|---:|
| Anisotropic | 14.8681 / +1.5879 / +0.2158 |
| Kerry3D | 9.1589 / +0.1257 / -0.0399 |
| Shots0001 | 13.7969 / +0.7923 / +0.0551 |

by-input-SNR 结果，单元格为 `final SNR mean / 相对 NE000_2 gain / 相对 NE006l gain`：

| input SNR | NE006r result |
|---:|---:|
| -2 dB | 12.3851 / +0.7347 / +0.0371 |
| -1 dB | 12.8418 / +0.7844 / +0.0506 |
| 1 dB | 13.5927 / +0.8675 / +0.0692 |
| 5 dB | 14.6928 / +0.9957 / +0.1018 |
| 10 dB | 15.5363 / +1.0917 / +0.1268 |

by-missing-rate 结果，单元格为 `final SNR mean / 相对 NE000_2 gain / 相对 NE006l gain`：

| missing rate | NE006r result |
|---:|---:|
| 0.02 | 14.8449 / +0.9204 / -0.0052 |
| 0.08 | 14.5507 / +0.9302 / +0.0334 |
| 0.18 | 13.9845 / +0.9244 / +0.0922 |
| 0.28 | 13.2879 / +0.8851 / +0.1246 |
| 0.38 | 12.3808 / +0.8140 / +0.1405 |

人类可读结论：

1. NE006r 是当前最强且更紧凑的 W4A4 g4 granularity 候选。它只选 18 个 Conv2d activation quantizer，却比 NE006l 的 30 个 selected count 高 `+0.0771 dB` mean SNR。
2. stage1/4/5 的组合收益不是单 stage 简单弱信号，而是可叠加的主信号。此前单 stage 最强仅 `+0.2883 dB`，三者组合达到 `+0.8948 dB`。
3. head、split/merge、stage2、stage3 不再是 W4A4 g4 主线。NE006r 不包含这些组仍超过 all Conv2d except head，说明它们至少不是必要条件，部分还可能拖累。
4. NE006r 的主要收益来自 Anisotropic 与 Shots0001；Kerry3D 仍提升很小且略低于 NE006l，说明后续若追求全源稳定性，需要结合 selective A8/mixed precision，而不是只继续 g4。
5. by-input-SNR 显示高输入 SNR 条件收益更强，说明 g4 granularity 更像是在减少模型自身 activation 量化误差上限，而不是只在严重退化输入上做补救。
6. by-missing-rate 显示 NE006r 在高 missing rate 下相对 NE006l 反而优势更大；这让它比 NE006l 更适合作为后续 W4A4 主候选。

后续决策：

- 将 NE006r 作为当前 W4A4 g4 主 baseline，优先级高于 NE006l。
- NE006 的纯 g4 granularity 阶段可以收束；除非需要论文消融，不再继续 stage2/stage3、head、split/merge 或 range/clipping。
- 下一阶段建议进入 NE007 selective A8 / mixed precision，以 NE006r 为紧凑 g4 基线，优先比较：
  - `stage1+stage4+stage5 Conv2d` 升 A8；
  - `stage1+stage4+stage5 Conv2d g4 + residual group A8`；
  - `all Conv2d except head` 升 A8 作为上限参考。

## 2026-05-12 NE006 W4A4 activation granularity 阶段总览

NE006 目标是在 normalized 新协议下确认 W4A4 的 activation gap 是否能通过 activation granularity 缓解，并定位最值得作为后续 mixed precision / selective A8 基线的 Conv2d 子集。本阶段共形成 18 个正式 full-grid eval 结果（NE006a-r），全部使用 `478 x 25 = 11950` 固定 grid、seed `20260507`。复核结果显示，18 个正式 eval 行数均为 `11950`，对应 final checkpoint verification 均 `passed=true`。

固定背景：

- 主对象：`NE000_2 W4A4 final`，SNR mean/median `12.9150 / 13.1180`，SSIM mean/median `0.939563 / 0.954078`
- weight-only 上限参照：`E007 W4A32 final`，SNR mean/median `17.7856 / 18.1128`，SSIM mean/median `0.964137 / 0.978461`
- 起点：`e007_w4a32_nbitsa4_metadata_seed.pth`
- 所有 granularity 实验均保持 `n_bits_w=4`、`n_bits_a=4`、`iters_a=5000`、`activation_lr=0.0004`、`lp_norm=2.4`
- `mse_grid` 在 NE006 中主要承担细粒度 delta shape 初始化职责；实验解释以 granularity 为主，不把它重新解释为 range/clipping 路线

NE006 全量结果表：

| id | strategy | granularity | selected | final SNR mean/median | final SSIM mean/median | recon gain | vs W4A4 | gap to W4A32 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| NE006a | g4 selective split+merge+stage_output | g4 | 15 | 13.2824 / 13.4767 | 0.943762 / 0.958305 | +1.5970 | +0.3674 | -4.5032 |
| NE006b | per-channel selective split+merge+stage_output | per-channel | 15 | 13.2731 / 13.4541 | 0.944148 / 0.958580 | +1.5010 | +0.3582 | -4.5125 |
| NE006c | g4 all Conv2d | g4 | 31 | 13.7231 / 13.9766 | 0.945327 / 0.960126 | +1.0354 | +0.8082 | -4.0624 |
| NE006d | per-channel all Conv2d | per-channel | 31 | 13.6752 / 13.9117 | 0.945889 / 0.960663 | +1.1144 | +0.7603 | -4.1104 |
| NE006e | g4 stage_output_conv | g4 | 5 | 13.2582 / 13.4803 | 0.940681 / 0.955035 | +1.7208 | +0.3433 | -4.5274 |
| NE006f | g4 split+merge | g4 | 10 | 12.9314 / 13.1217 | 0.942282 / 0.956952 | +1.6832 | +0.0164 | -4.8542 |
| NE006g | g4 conv role | g4 | 15 | 13.2241 / 13.4930 | 0.941117 / 0.956052 | +1.6087 | +0.3091 | -4.5615 |
| NE006h | g4 fusion branch | g4 | 10 | 12.9314 / 13.1217 | 0.942282 / 0.956952 | +1.6832 | +0.0164 | -4.8542 |
| NE006i | g4 head | g4 | 1 | 12.9031 / 13.1007 | 0.939624 / 0.954139 | +1.7425 | -0.0119 | -4.8825 |
| NE006j | g4 stage_output+conv role | g4 | 20 | 13.6186 / 13.8908 | 0.942366 / 0.956958 | +1.4224 | +0.7036 | -4.1670 |
| NE006k | g4 all Conv2d except split/merge | g4 | 21 | 13.6097 / 13.8824 | 0.942403 / 0.957057 | +1.4256 | +0.6948 | -4.1759 |
| NE006l | g4 all Conv2d except head | g4 | 30 | 13.7326 / 13.9838 | 0.945303 / 0.960038 | +1.0334 | +0.8177 | -4.0529 |
| NE006m | g4 stage1 Conv2d | g4 | 6 | 13.1610 / 13.3724 | 0.940107 / 0.954443 | +1.6499 | +0.2460 | -4.6246 |
| NE006n | g4 stage2 Conv2d | g4 | 6 | 12.8185 / 13.0006 | 0.939369 / 0.953903 | +1.7225 | -0.0965 | -4.9671 |
| NE006o | g4 stage3 Conv2d | g4 | 6 | 12.8957 / 13.0939 | 0.939752 / 0.954255 | +1.7152 | -0.0193 | -4.8899 |
| NE006p | g4 stage4 Conv2d | g4 | 6 | 13.1833 / 13.4054 | 0.940243 / 0.954534 | +1.5252 | +0.2683 | -4.6023 |
| NE006q | g4 stage5 Conv2d | g4 | 6 | 13.2033 / 13.4520 | 0.943476 / 0.958300 | +1.5356 | +0.2883 | -4.5823 |
| NE006r | g4 stage1+stage4+stage5 Conv2d | g4 | 18 | 13.8098 / 14.0845 | 0.945084 / 0.959776 | +1.1656 | +0.8948 | -3.9758 |

按 final SNR mean 排序的关键层级：

1. `NE006r stage1+stage4+stage5 Conv2d g4`：`13.8098`，`+0.8948 dB`，selected 18
2. `NE006l all Conv2d except head g4`：`13.7326`，`+0.8177 dB`，selected 30
3. `NE006c all Conv2d g4`：`13.7231`，`+0.8082 dB`，selected 31
4. `NE006d all Conv2d per-channel`：`13.6752`，`+0.7603 dB`，selected 31
5. `NE006j stage_output+conv role g4`：`13.6186`，`+0.7036 dB`，selected 20
6. `NE006k all Conv2d except split/merge g4`：`13.6097`，`+0.6948 dB`，selected 21

阶段性结论：

1. Conv2d activation granularity 仍是 W4A4 的主要可优化方向，但收益上限有限。最佳 NE006r 追回 `+0.8948 dB`，但距离 W4A32 仍有 `-3.9758 dB` gap。
2. 新协议 W4A4 与旧 E006 结论不完全一致。旧协议中 `split_proj + merge_proj + stage_output_conv` 是强候选；新协议 W4A4 下 split/merge 基本无效，NE006f/NE006h 只有 `+0.0164 dB`。
3. g4 比 per-channel 更适合作为当前主线。all Conv2d g4 比 all Conv2d per-channel 高 `+0.0479 dB`，selective g4 也略高于 selective per-channel。per-channel 不再是必然上限。
4. `head` 应从 W4A4 granularity 主策略中排除。NE006i 单独低于 baseline，NE006l 去掉 head 后反而略高于 all Conv2d。
5. stage1/4/5 是正收益组合，stage2/3 是负收益或近零收益。NE006r 不包含 stage2/3 仍超过 NE006l，说明 stage2/3 不是必要条件，可能拖累整体策略。
6. 单 stage 不够强，但 stage1/4/5 有明显组合效应。最强单 stage 只有 stage5 的 `+0.2883 dB`，三者组合达到 `+0.8948 dB`。
7. by-source 结论保持一致：主要收益来自 Anisotropic 与 Shots0001，Kerry3D 提升很小。这说明 W4A4 的 residual gap 不是所有 source 均匀下降，后续策略需要检查 source stability。
8. NE006r 在高输入 SNR 和高 missing rate 条件下相对 NE006l 更有优势，因此比 NE006l 更适合作为后续 W4A4 主 baseline。

NE006 收束决策：

- 当前 W4A4 g4 主 baseline：`NE006r stage1+stage4+stage5 Conv2d g4`
- W4A4 g4 上限参考：`NE006l all Conv2d except head g4` 和 `NE006c all Conv2d g4`
- 不再优先投入：range/clipping、split/merge、head、stage2/stage3 单独 g4、更多 per-channel sweep
- 下一阶段：进入 NE007 selective A8 / mixed precision。建议以 NE006r 为紧凑 g4 基线，比较 `stage1+stage4+stage5 Conv2d` 升 A8，以及 `all Conv2d except head` 升 A8 作为上限参考。

## 2026-05-12 论文图件工作区初始化

为本科毕业论文第 4 章实验结果展示新增图件管理工作区：

- 根目录：`SCRN_BRECQ_app/scrn_brecq/paper_artifacts`
- 首个实验：`experiments/ch4_2_exp01_w4a32_visual_recovery`
- 目标小节：4.2.2 W4A32 权重量化结果分析
- 图件目标：管理 `Clean / Degraded input / FP32 / W4A32 pre-reconstruction / W4A32 final` 的 3x5 视觉恢复候选图

本次只完成目录骨架、实验说明、样本追踪规范和日志入口，不生成候选图。后续每个候选图版本必须写入 `manifest_vXXX.json`，记录 `testset_id`、`patch_index`、`patch_file`、`source`、`condition_index`、`snr_setting_db`、`missing_rate` 以及 FP32/W4A32 pre/final 指标，确保论文图中的每一行都能追溯到固定测试集中的具体样本。

详细图件日志见：`SCRN_BRECQ_app/scrn_brecq/paper_artifacts/ARTIFACTS_LOG.md`。

## 2026-05-12 论文 W4A32 视觉恢复图生成脚本

为论文 4.2.2 的 W4A32 权重量化结果展示新增候选图生成脚本：

- 脚本：`SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_2_exp01_w4a32_visual_recovery/scripts/make_w4a32_visual_recovery.py`
- 测试：`SCRN_BRECQ_app/scrn_brecq/paper_artifacts/tests/test_w4a32_visual_recovery_selection.py`
- 功能：从 E007 W4A32 full-grid `per_sample_metrics.jsonl` 自动选择代表样本，生成 3x5 候选图，并为每个版本写入 `manifest_vXXX.json`

本步骤只完成脚本与选择规则测试，不生成图片。测试命令 `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.paper_artifacts.tests.test_w4a32_visual_recovery_selection` 已通过。

## 2026-05-12 W4A32 图生成脚本路径执行修复

首次按文件路径运行 `make_w4a32_visual_recovery.py` 时出现 `ModuleNotFoundError: No module named 'SCRN_BRECQ_app'`。原因是文件路径执行时 Python 未自动加入仓库根目录。已在脚本启动阶段加入仓库根目录到 `sys.path`，并新增 `--help` 路径执行回归测试。

验证命令 `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.paper_artifacts.tests.test_w4a32_visual_recovery_selection` 通过，覆盖 4 个测试。

## 2026-05-12 W4A32 视觉恢复候选图 v001

使用 `make_w4a32_visual_recovery.py` 生成论文 4.2.2 的两套 3x5 候选图：

- `set_a_three_degradation_levels`：轻度 / 中度 / 重度退化各 1 行
- `set_b_three_medium_samples`：中等退化条件下 3 个不同 source 样本

输出位置：

- `SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_2_exp01_w4a32_visual_recovery/candidates/set_a_three_degradation_levels`
- `SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_2_exp01_w4a32_visual_recovery/candidates/set_b_three_medium_samples`

每套图均保存 `png`、`pdf`、`manifest_v001.json` 和 `selection_summary_v001.md`。详细选样结果和验证记录见 `SCRN_BRECQ_app/scrn_brecq/paper_artifacts/ARTIFACTS_LOG.md`。

## 2026-05-12 论文图件日志语言与 git 提交范围规范

根据新的图件管理原则，后续论文图件相关日志尽量使用中文记录；命令、文件名、指标字段和实验代号保持原始形式，避免与代码和 manifest 字段不一致。

git 提交范围进一步调整为只提交代码、说明和日志类文件：

- 提交说明和日志：README、`experiment_info.json`、`ARTIFACTS_LOG.md`、`DEVELOPMENT_LOG.md`
- 提交可复现代码：图件生成脚本和测试脚本
- 不提交候选图或最终图：`.png`、`.pdf`
- 不提交按版本生成的候选结果元数据：`manifest_vXXX.json`、`selection_summary_vXXX.md`
- 不提交 Python 缓存和临时文件

本次同步会把已跟踪的候选图、manifest 和 selection summary 从 git 索引中移除，但保留本地文件用于继续查看和挑选。

## 2026-05-12 478 张 clean patch 浏览图册

为论文第 4 章样本挑选新增 clean patch 浏览图册实验：

- 实验目录：`SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_common_exp01_testset_clean_patch_atlas`
- 脚本：`scripts/make_testset_clean_atlas.py`
- 测试：`SCRN_BRECQ_app/scrn_brecq/paper_artifacts/tests/test_clean_patch_atlas.py`
- 数据集：`scrn_paper5_energy_filtered_perpatch_absmax_test_478`
- 图册内容：478 张 clean normalized patch，每页 48 张，共 10 页

生成结果保留在本地 `candidates/clean_patch_atlas/`，包括 PDF、10 张 PNG、`selection_index_v001.csv` 和 `manifest_v001.json`。这些结果文件按 `.gitignore` 规则不提交；代码、测试、README、`experiment_info.json` 和日志提交。测试命令 `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.paper_artifacts.tests.test_clean_patch_atlas` 已通过。

## 2026-05-14 NE007 W4A4 selective A8 / mixed precision 统筹计划

本节在开始实现前记录 NE007 的整体统筹计划。NE007 不是单个孤立实验，而是 W4A4 activation quantization 在 NE006 收束后的下一阶段：先补齐 mixed precision 基础设施，再用少量锚点实验判断后续是否继续细分。

固定背景：

- 当前默认数据协议：`paper5_energy_filtered_perpatch_absmax`
- calibration：`SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_cali_1024_stratified`
- test：`SCRN_BRECQ_app/scrn_repro/datasets/scrn_paper5_energy_filtered_perpatch_absmax_test_478`
- 正式评估：`478 x 25 = 11950` fixed grid，seed `20260507`
- 当前 W4A4 g4 主 baseline：`NE006r stage1+stage4+stage5 Conv2d g4`
- NE006r final SNR mean/median：`13.8098 / 14.0845`
- NE006r selected count：`18`
- W4A32 上限参照：`E007 W4A32 final`，SNR mean/median `17.7856 / 18.1128`

NE007 主目标：

- 验证 `W4 + mostly A4 activation + selective A8 activation` 是否能显著缩小 NE006r 到 W4A32 的剩余 gap。
- 找到 A8 activation quantizer 数量、结构位置和 full-grid 质量收益之间的关系。
- 判断继续做 mixed precision 是否比继续扩展纯 g4 granularity 更有价值。

基础设施计划：

1. 新增 activation bitwidth override 能力，支持按 `stage`、`branch`、`role`、`module_type`、`index` 或 selector group 将指定 activation quantizer 升 A8。
2. checkpoint 的 `quant_config` 必须保存 mixed precision 配置；eval、grid eval 和 verify 重新加载 checkpoint 时必须恢复逐 quantizer bitwidth，不能退回统一 `n_bits_a`。
3. metrics / verification 需要记录 activation effective bit counts，并区分全局 `n_bits_a=4`、BRECQ 固有 first/last 8bit activation、NE007 主动 selector override A8。
4. 默认未配置 override 时，现有 NE000-NE006 行为保持不变。

计划新增配置命名：

- `SCRN_BRECQ_app/scrn_brecq/configs/activation_quantization/ne007a_stage145_conv2d_a8.json`
- `SCRN_BRECQ_app/scrn_brecq/configs/activation_quantization/ne007b_stage145_g4_stage23_conv2d_a8.json`
- `SCRN_BRECQ_app/scrn_brecq/configs/activation_quantization/ne007c_all_conv2d_except_head_a8.json`

计划 run root：

- quant：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE007_w4a4_mixed_precision/quant`
- eval：`SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/NE007_w4a4_mixed_precision/eval`

计划锚点实验：

| id | 目的 | 初始策略 | 解释 |
|---|---|---|---|
| NE007a | 验证 NE006r 正收益组合升 A8 是否继续增益 | `stage1+stage4+stage5 Conv2d -> A8` | 如果接近 NE007c，说明主收益集中在 NE006r 组合 |
| NE007b | 检查 residual Conv2d 是否只是 A4 太粗 | `NE006r g4 + stage2/stage3 Conv2d A8` | 如果改善 Kerry3D 或 overall，后续再拆 residual group |
| NE007c | selective A8 上限参考 | `all Conv2d except head -> A8` | 如果仍不强，说明单纯 activation bitwidth 不是主瓶颈 |

分支决策规则：

- 若 NE007a 已接近 NE007c：后续做 stage/role 剪枝，目标是减少 A8 count。
- 若 NE007c 明显强于 NE007a：后续按 stage2/stage3、fusion、stage_output、cnn role 拆分，定位遗漏组。
- 若 NE007a/b/c 均提升有限：暂停继续堆 A8，转向 reconstruction target、teacher 模型、activation 插入位置或 zero-point 机制。
- 若 overall 提升但 Kerry3D 继续弱：单独做 source-stability 分析，不直接把 NE007 作为稳定策略。

日志与执行要求：

- 每个 NE007 实验必须记录 config path、command、GPU 选择、run dir、checkpoint path、verification summary、full-grid overall、by-source、by-SNR、by-missing-rate、人类可读结论和后续决策。
- 涉及 GPU 实验前必须先查看 `nvidia-smi`；单卡优先级仍为 `1 -> 2 -> 3 -> 0`，偏离时记录原因。
- 失败 run 不覆盖、不静默修改变量；使用 `_rerun` / `_rerun2` 后缀，并在日志中记录失败原因。
- NE007 初步阶段先限制为 3 个锚点实验；只有锚点结果支持继续细分时，再开启下一轮 3-5 个 follow-up。

## 2026-05-14 NE007 mixed precision 基础设施实现

本次提交实现 NE007 第一阶段基础设施，不启动 GPU 实验，不产生 quant/eval run dir 或 checkpoint。目标是让 `W4A4 + selective A8 activation` 能通过配置进入 activation-only quantization、checkpoint 保存、eval reload 和 verify 报告链路。

实现内容：

- 新增 `quant/activation_precision.py`，提供 `activation_bitwidth_overrides` 标准化、selector group 选择、`bitwidth_refactor(n_bits)` 应用和 effective activation bit count summary。
- `activation_only_quantize_scrn.py` 新增 `--activation-bitwidth-overrides-json`，并在 activation init/range calibration 前应用 mixed precision override。
- activation-only checkpoint `quant_config` 现在保存 `activation_bitwidth_overrides`，metrics/summary 记录 `activation_bitwidth_summary`，包括 bit counts、disabled count 和 selected override names。
- `evaluate_quantized_scrn.py` 在 checkpoint rebuild 时于 `set_first_last_layer_to_8bit()` 之后恢复 per-activation bitwidth override，避免 eval/grid eval 回退到统一 `n_bits_a`。
- `verify_quantized_scrn.py` 报告 activation bit counts、enabled activation bit counts、disabled count、override 配置和 selected override names。
- 新增/扩展测试覆盖 selector union、exclude、默认排除 output quantizer、后置 override 覆盖前置 override、CLI JSON 解析、checkpoint config 保留、eval rebuild 恢复和 verify bit count 报告。

配置接口：

```json
"activation_bitwidth_overrides": [
  {
    "n_bits": 8,
    "selector_groups": [
      {"stage": "stage1", "module_type": "Conv2d"},
      {"stage": "stage4", "module_type": "Conv2d"},
      {"stage": "stage5", "module_type": "Conv2d"}
    ],
    "exclude_selector_groups": []
  }
]
```

验证：

- 红测：新增测试首次运行失败，失败点为缺少 `activation_precision` 模块、CLI 参数、checkpoint override 保存、eval rebuild 应用和 verify bit count 报告。
- 绿测：`conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.tests.test_activation_precision SCRN_BRECQ_app.scrn_brecq.tests.test_activation_only_quantize_scrn SCRN_BRECQ_app.scrn_brecq.tests.test_evaluate_quantized_scrn SCRN_BRECQ_app.scrn_brecq.tests.test_verify_quantized_scrn`，`Ran 30 tests ... OK`。

实验记录：

- run dir：无，本次 infra-only。
- checkpoint path：无，本次 infra-only。
- 数据协议：未运行数据实验；后续仍使用 `paper5_energy_filtered_perpatch_absmax`。
- GPU 使用情况：未使用 GPU；无 `nvidia-smi`。
- full-grid 指标：无，本次未做 `478 x 25 = 11950` eval。
- by-source / by-SNR / by-missing-rate：无，本次未评估。

后续决策：

- 下一步单独新增 NE007a/b/c 配置文件，再按 GPU 原则启动 NE007a 锚点实验。
- 正式实验时必须在 verification 和 eval 日志中记录 activation bit counts 与 selected override names，确认 checkpoint reload 后 mixed precision 生效。
