# 论文图件日志

本日志用于记录本科毕业论文图件工作区的目录变更、候选图生成、样本选择、验证结果和最终图件来源。后续新增记录原则上使用中文；命令、文件名、指标字段和实验代号保持原始英文，避免和代码、manifest 字段不一致。

## 2026-05-12 ch4_2_exp01 W4A32 视觉恢复图件工作区初始化

目的：

- 在 `paper_artifacts/` 下建立论文结果图件的统一管理目录。
- 创建第一个实验目录，用于管理第 4.2.2 节 W4A32 权重量化视觉恢复图。
- 固定 manifest 记录规则，确保每张论文候选图中的每一行都能追溯到固定测试集中的具体样本和退化条件。

创建的实验：

- 实验编号：`ch4_2_exp01_w4a32_visual_recovery`
- 论文位置：第 4.2.2 节，W4A32 权重量化结果分析
- 预期图列：`Clean`、`Degraded input`、`FP32`、`W4A32 pre-reconstruction`、`W4A32 final`
- 候选集 A：三种退化程度
- 候选集 B：三张中等退化样本

`experiment_info.json` 中记录的数据来源：

- FP32 checkpoint：normalized paper5 energy-filtered per-patch absmax SCRN checkpoint
- W4A32 checkpoint：E007 normalized W4A32 baseline run 的重建前和重建后 checkpoint
- 测试集：normalized 478-patch fixed test set
- 指标文件：E007 normalized W4A32 `per_sample_metrics.jsonl`

样本溯源规则：

- 每个候选图版本必须写入 `manifest_vXXX.json`。
- 每一行图必须记录 `testset_id`、`patch_index`、`patch_file`、`source`、`condition_index`、`snr_setting_db`、`missing_rate`，以及 FP32/W4A32 pre/final 的 SNR 字段。
- 判断论文图使用的是测试集哪个样本时，以 manifest 为准，不以图片文件名为准。

本初始化步骤没有生成图片。

## 2026-05-12 ch4_2_exp01 W4A32 视觉恢复图生成器

目的：

- 增加一个本地图件生成脚本，用于生成第 4.2.2 节 W4A32 的 3x5 视觉恢复候选图。
- 保证样本选择规则可复现，并在 `manifest_vXXX.json` 中记录精确样本来源。
- 增加针对代表样本选择和 manifest 必填字段的小测试。

本地工具文件：

- `experiments/ch4_2_exp01_w4a32_visual_recovery/scripts/make_w4a32_visual_recovery.py`
- `tests/test_w4a32_visual_recovery_selection.py`

说明：

- 这些脚本和测试文件用于本地生成与验证图件。
- 根据 2026-05-12 更新的 git 原则，后续不再把本地脚本和测试纳入 git 跟踪；git 只提交结果图、manifest/summary、README 和日志。

已实现的选样策略：

- 候选集 A：分别在以下退化条件中选择接近条件中位数的代表样本：
  - light：`snr_setting_db=10.0`，`missing_rate=0.02`
  - medium：`snr_setting_db=1.0`，`missing_rate=0.18`
  - heavy：`snr_setting_db=-2.0`，`missing_rate=0.38`
- 候选集 B：在中等退化条件中选择 3 个不同 source 的代表样本，source 优先顺序为：
  - `Anisotropic`
  - `Kerry3D`
  - `Shots0001`
- 代表性评分使用以下字段相对条件中位数的偏离：
  - `fp32_snr_db`
  - `quant_pre_minus_fp32_snr_db`
  - `quant_post_minus_fp32_snr_db`
  - `quant_post_minus_pre_snr_db`

验证：

- Red 测试：`conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.paper_artifacts.tests.test_w4a32_visual_recovery_selection` 因 `make_w4a32_visual_recovery.py` 不存在而失败。
- Green 测试：实现脚本后，同一测试命令通过，`Ran 3 tests in 2.111s`。

本脚本开发步骤没有生成图片。

## 2026-05-12 ch4_2_exp01 图生成脚本路径执行修复

问题：

- 按文件路径运行 `make_w4a32_visual_recovery.py` 时失败，报错为 `ModuleNotFoundError: No module named 'SCRN_BRECQ_app'`。
- 根因：Python 按文件路径执行脚本时，只会把脚本所在目录加入 `sys.path`，不会自动加入仓库根目录。

修复：

- 在脚本启动阶段自动定位仓库根目录。
- 在导入项目模块前把仓库根目录加入 `sys.path`。
- 增加一个回归测试，覆盖 `python make_w4a32_visual_recovery.py --help` 这种文件路径执行方式。

验证：

- 修复前：`test_script_help_runs_when_executed_by_path` 复现同样的 `ModuleNotFoundError`。
- 修复后：`conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.paper_artifacts.tests.test_w4a32_visual_recovery_selection` 通过，`Ran 4 tests in 3.371s`。

本修复步骤没有生成图片。

## 2026-05-12 ch4_2_exp01 W4A32 视觉恢复候选图 v001

生成命令：

```bash
conda run -n quant python SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_2_exp01_w4a32_visual_recovery/scripts/make_w4a32_visual_recovery.py --candidate-set all --device cuda --cuda-device-index 1
```

运行信息：

- 设备：`cuda:1`
- 生成候选集：`set_a_three_degradation_levels`、`set_b_three_medium_samples`
- 用时：`50.89s`

生成文件：

- `experiments/ch4_2_exp01_w4a32_visual_recovery/candidates/set_a_three_degradation_levels/fig_ch4_2_w4a32_3x5_levels_v001.png`
- `experiments/ch4_2_exp01_w4a32_visual_recovery/candidates/set_a_three_degradation_levels/fig_ch4_2_w4a32_3x5_levels_v001.pdf`
- `experiments/ch4_2_exp01_w4a32_visual_recovery/candidates/set_a_three_degradation_levels/manifest_v001.json`
- `experiments/ch4_2_exp01_w4a32_visual_recovery/candidates/set_a_three_degradation_levels/selection_summary_v001.md`
- `experiments/ch4_2_exp01_w4a32_visual_recovery/candidates/set_b_three_medium_samples/fig_ch4_2_w4a32_3x5_medium_samples_v001.png`
- `experiments/ch4_2_exp01_w4a32_visual_recovery/candidates/set_b_three_medium_samples/fig_ch4_2_w4a32_3x5_medium_samples_v001.pdf`
- `experiments/ch4_2_exp01_w4a32_visual_recovery/candidates/set_b_three_medium_samples/manifest_v001.json`
- `experiments/ch4_2_exp01_w4a32_visual_recovery/candidates/set_b_three_medium_samples/selection_summary_v001.md`

候选集 A 的选样结果：

| 行标签 | source | patch_index | patch_file | condition_index | SNR setting | missing | FP32 SNR | W4A32 pre SNR | W4A32 final SNR | final - FP32 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| light | Shots0001 | 100 | `test_000101.npy` | 20 | 10.0 | 0.02 | 22.5835 | 20.1609 | 22.5351 | -0.0484 |
| medium | Shots0001 | 296 | `test_000297.npy` | 12 | 1.0 | 0.18 | 16.6976 | 15.9910 | 16.6553 | -0.0423 |
| heavy | Shots0001 | 303 | `test_000304.npy` | 4 | -2.0 | 0.38 | 15.1011 | 14.4356 | 15.0577 | -0.0433 |

候选集 B 的选样结果：

| 行标签 | source | patch_index | patch_file | condition_index | SNR setting | missing | FP32 SNR | W4A32 pre SNR | W4A32 final SNR | final - FP32 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| sample_1 | Anisotropic | 50 | `test_000051.npy` | 12 | 1.0 | 0.18 | 21.9377 | 21.1601 | 21.9119 | -0.0258 |
| sample_2 | Kerry3D | 82 | `test_000083.npy` | 12 | 1.0 | 0.18 | 10.8369 | 10.1483 | 10.8059 | -0.0309 |
| sample_3 | Shots0001 | 296 | `test_000297.npy` | 12 | 1.0 | 0.18 | 16.6976 | 15.9910 | 16.6553 | -0.0423 |

验证：

- 两个 `manifest_v001.json` 均通过 `jq empty`。
- `file` 检查显示两张 PNG 均为 `4500 x 2280` RGBA 图像。
- 人工打开检查确认两张 PNG 都是非空的 3x5 对比网格。

说明：

- 候选集 A 的 v001 策略是按退化条件分别选择接近中位数的代表样本，因此三行不是同一个 patch。
- 候选集 B 的 v001 策略是中等退化条件下 source 多样化，分别来自 Anisotropic、Kerry3D 和 Shots0001。
- 当前还没有把任何候选图移动到 `shortlisted/` 或 `final/`。

## 2026-05-12 图件日志语言与 git 提交范围规范

根据新的工作原则，后续图件相关日志尽量使用中文记录；命令、文件名、指标字段、实验代号保留原始形式，避免和代码或 manifest 字段不一致。

git 提交范围调整为：

- 提交：结果图文件，例如 `.png`、`.pdf`
- 提交：样本追踪文件，例如 `manifest_vXXX.json`
- 提交：选择说明文件，例如 `selection_summary_vXXX.md`
- 提交：README、`ARTIFACTS_LOG.md`、`DEVELOPMENT_LOG.md`
- 不提交：本地生成脚本、测试脚本、`__pycache__`、`.pyc`、临时缓存

本次规范调整会把已跟踪的本地生成脚本和测试从 git 索引中移除，但保留工作区本地文件，后续仍可继续用于生成图件。
