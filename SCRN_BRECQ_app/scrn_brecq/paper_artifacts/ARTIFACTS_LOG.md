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
- 根据 2026-05-12 进一步修订后的 git 原则，后续保留脚本和测试代码的可追踪性；git 不再提交候选图、manifest/summary 等按版本生成的结果文件。

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

- 提交：README、`experiment_info.json`、`ARTIFACTS_LOG.md`、`DEVELOPMENT_LOG.md`
- 提交：本地图件生成脚本和测试脚本，保证后续图件可复现
- 不提交：候选图和最终图，例如 `.png`、`.pdf`
- 不提交：按版本生成的候选结果元数据，例如 `manifest_vXXX.json`、`selection_summary_vXXX.md`
- 不提交：`__pycache__`、`.pyc`、临时缓存

本次规范调整会把已跟踪的候选图、manifest 和 selection summary 从 git 索引中移除，但保留工作区本地文件，后续仍可继续查看和挑选。

## 2026-05-12 ch4_2_exp01 W4A32 视觉恢复候选图 v002-v005 版式修正

背景：

- v001 图左侧行说明包含 source、patch 文件名和退化参数，文字过多，不适合直接放入论文正文。
- 本次仅修正版式和增加候选图，不改变 W4A32 checkpoint、测试集、seed 或 fixed-grid 指标来源。
- 本地图件脚本作为可复现代码纳入 git 跟踪；候选图、manifest 和 summary 作为生成结果本地保留，不纳入后续提交。

本地图件脚本更新：

- 新增 `row_label_style`：
  - `compact`：左侧只显示 `Light` / `Medium` / `Heavy` 或 `Sample 1` / `Sample 2` / `Sample 3`
  - `none`：不显示左侧行标签
- 新增 `panel_metric_style`：
  - `snr`：小图标题显示 SNR
  - `none`：小图标题只保留列名，不显示 SNR
- 新增 `figure_title_style=none`，新版图不再显示顶部大标题，便于论文中用图题说明。
- 新增 `set_a_selection=fixed_patch_from_medium`，用于生成同一个测试 patch 在轻度、中度、重度退化下的三行对比。

本地测试：

- 命令：`conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.paper_artifacts.tests.test_w4a32_visual_recovery_selection`
- 结果：通过，`Ran 7 tests in 3.403s`

生成命令与版本：

```bash
conda run -n quant python SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_2_exp01_w4a32_visual_recovery/scripts/make_w4a32_visual_recovery.py --candidate-set all --device cuda --cuda-device-index 1 --row-label-style compact --panel-metric-style snr --figure-title-style none --set-a-selection condition_median
```

- 生成 `set_a_three_degradation_levels` v002：条件中位代表样本，compact 左侧标签，显示 SNR。
- 生成 `set_b_three_medium_samples` v002：三 source 中等退化样本，compact 左侧标签，显示 SNR。

```bash
conda run -n quant python SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_2_exp01_w4a32_visual_recovery/scripts/make_w4a32_visual_recovery.py --candidate-set set_a_three_degradation_levels --device cuda --cuda-device-index 1 --row-label-style compact --panel-metric-style snr --figure-title-style none --set-a-selection fixed_patch_from_medium
```

- 生成 `set_a_three_degradation_levels` v003：固定同一个 patch，compact 左侧标签，显示 SNR。

```bash
conda run -n quant python SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_2_exp01_w4a32_visual_recovery/scripts/make_w4a32_visual_recovery.py --candidate-set set_a_three_degradation_levels --device cuda --cuda-device-index 1 --row-label-style compact --panel-metric-style none --figure-title-style none --set-a-selection fixed_patch_from_medium
```

- 生成 `set_a_three_degradation_levels` v004：固定同一个 patch，compact 左侧标签，不显示小图 SNR。

```bash
conda run -n quant python SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_2_exp01_w4a32_visual_recovery/scripts/make_w4a32_visual_recovery.py --candidate-set set_b_three_medium_samples --device cuda --cuda-device-index 1 --row-label-style compact --panel-metric-style none --figure-title-style none
```

- 生成 `set_b_three_medium_samples` v003：三 source 中等退化样本，compact 左侧标签，不显示小图 SNR。

```bash
conda run -n quant python SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_2_exp01_w4a32_visual_recovery/scripts/make_w4a32_visual_recovery.py --candidate-set set_a_three_degradation_levels --device cuda --cuda-device-index 1 --row-label-style none --panel-metric-style none --figure-title-style none --set-a-selection fixed_patch_from_medium
```

- 生成 `set_a_three_degradation_levels` v005：固定同一个 patch，无左侧标签，不显示小图 SNR。

```bash
conda run -n quant python SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_2_exp01_w4a32_visual_recovery/scripts/make_w4a32_visual_recovery.py --candidate-set set_b_three_medium_samples --device cuda --cuda-device-index 1 --row-label-style none --panel-metric-style none --figure-title-style none
```

- 生成 `set_b_three_medium_samples` v004：三 source 中等退化样本，无左侧标签，不显示小图 SNR。

新增候选图：

| 候选集 | 版本 | 选样 | 左侧标签 | 小图 SNR | 说明 |
|---|---|---|---|---|---|
| set_a | v002 | 不同退化条件分别选代表样本 | compact | yes | v001 的版式修正版 |
| set_a | v003 | 固定 `test_000297.npy` | compact | yes | 控制变量更强，保留 SNR |
| set_a | v004 | 固定 `test_000297.npy` | compact | no | 当前更适合正文图 |
| set_a | v005 | 固定 `test_000297.npy` | none | no | 极简版，样本说明完全依赖图注/manifest |
| set_b | v002 | Anisotropic/Kerry3D/Shots0001 | compact | yes | v001 的版式修正版 |
| set_b | v003 | Anisotropic/Kerry3D/Shots0001 | compact | no | 更适合正文或附图 |
| set_b | v004 | Anisotropic/Kerry3D/Shots0001 | none | no | 极简版，样本说明完全依赖图注/manifest |

关键选样：

- set_a v003-v005 固定 patch：`test_000297.npy`，`patch_index=296`，`source=Shots0001`。
- set_a 三行退化条件：
  - Light：`condition_index=20`，`snr_setting_db=10.0`，`missing_rate=0.02`
  - Medium：`condition_index=12`，`snr_setting_db=1.0`，`missing_rate=0.18`
  - Heavy：`condition_index=4`，`snr_setting_db=-2.0`，`missing_rate=0.38`
- set_b v002-v004 三个中等退化样本：
  - Sample 1：Anisotropic，`patch_index=50`，`test_000051.npy`
  - Sample 2：Kerry3D，`patch_index=82`，`test_000083.npy`
  - Sample 3：Shots0001，`patch_index=296`，`test_000297.npy`

验证：

- `manifest_v002` 至 `manifest_v005` 均通过 `jq empty`。
- 新版 PNG 均为 `4260 x 2160` RGBA 图像。
- 人工打开检查：v003/v004/v005 和 set_b v003/v004 均可正常显示，左侧冗长文字已移除。

初步建议：

- 正文优先考虑 `set_a_three_degradation_levels` v004：固定同一 patch，控制变量清楚，图面简洁。
- 如果希望小图上直接显示 SNR，则考虑 `set_a_three_degradation_levels` v003。
- 如果要展示不同数据来源/结构差异，则考虑 `set_b_three_medium_samples` v003。

提交前复核：

- `git diff --check -- SCRN_BRECQ_app/scrn_brecq/paper_artifacts/ARTIFACTS_LOG.md` 无输出，日志空白检查通过。
- `manifest_v002` 至 `manifest_v005` 重新通过 `jq empty` 检查。
- 新版 PNG 重新通过 `file` 检查，分辨率均为 `4260 x 2160`，格式为 RGBA PNG。
- 本地选择逻辑测试重新通过：`Ran 7 tests in 3.448s`。
- 按当前“提交代码和日志、结果本地保留”的原则，本地画图脚本与测试文件纳入 git 跟踪，候选图和按版本生成的结果元数据不纳入提交。

## 2026-05-12 ch4_2_exp01 W4A32 视觉恢复纯数据版 v005/v006

背景：

- 论文图件需要进一步去除图内冗余文字。用户明确指出 `set_b_three_medium_samples` v002 左侧 `Sample 1/2/3` 不适合直接放入论文。
- 本轮版式原则调整为“只保留 3×5 图像数据”，图内不显示行标签、列标题、小图 SNR 或色标文字。
- 样本身份、列含义、退化条件与指标继续通过 `manifest_vXXX.json` 和 `selection_summary_vXXX.md` 追踪，避免把解释性文字塞进图内。

本地工具更新：

- 本地画图脚本新增 `--column-label-style {labels,none}`，用于关闭顶部列标题。
- 本地画图脚本新增 `--colorbar-style {per_row,none}`，用于关闭右侧色标。
- 本地脚本和测试文件按当前 `.gitignore` 规则作为可复现代码纳入提交；候选图和按版本生成的结果元数据不纳入提交。

测试：

- 先新增测试约束“关闭列标题”和“关闭色标”两个行为。
- 初次测试按预期失败，原因是脚本尚无 `column_title_text` 和 `should_draw_colorbar`。
- 实现后重新运行：

```bash
conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.paper_artifacts.tests.test_w4a32_visual_recovery_selection
```

- 结果：通过，`Ran 9 tests in 3.228s`。

生成命令与版本：

```bash
conda run -n quant python SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_2_exp01_w4a32_visual_recovery/scripts/make_w4a32_visual_recovery.py --candidate-set set_b_three_medium_samples --device cuda --cuda-device-index 1 --row-label-style none --panel-metric-style none --figure-title-style none --column-label-style none --colorbar-style none
```

- 生成 `set_b_three_medium_samples` v005：三张中等退化样本，图内无 `Sample 1/2/3`，无列标题，无 SNR，无色标。

```bash
conda run -n quant python SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_2_exp01_w4a32_visual_recovery/scripts/make_w4a32_visual_recovery.py --candidate-set set_a_three_degradation_levels --device cuda --cuda-device-index 1 --set-a-selection fixed_patch_from_medium --row-label-style none --panel-metric-style none --figure-title-style none --column-label-style none --colorbar-style none
```

- 生成 `set_a_three_degradation_levels` v006：固定同一 patch 的三种退化程度，图内无 `Light/Medium/Heavy`，无列标题，无 SNR，无色标。

新增候选图：

| 候选集 | 版本 | 选样 | 图内文字 | 说明 |
|---|---|---|---|---|
| set_b | v005 | Anisotropic/Kerry3D/Shots0001 中等退化样本 | none | 对应用户指出的 v002，去除 `Sample 1/2/3` 以及其他图内文字 |
| set_a | v006 | 固定 `test_000297.npy`，三种退化程度 | none | 用图注说明三行退化程度，图内只保留数据图像 |

人工检查：

- `set_b_three_medium_samples` v005 图内已无 `Sample 1/2/3`。
- `set_a_three_degradation_levels` v006 图内已无 `Light/Medium/Heavy`。
- 两张图均只保留 3×5 恢复结果图像，样本来源以后续 manifest 和图注说明为准。

提交前复核：

- `manifest_v005.json` 和 `manifest_v006.json` 通过 `jq empty`。
- 新版 PNG 均为 `4260 x 2160` RGBA 图像，对应 PDF 均为 1 页。
- 本地选择与版式开关测试重新通过：`Ran 9 tests in 3.438s`。

## 2026-05-12 ch4_2_exp01 W4A32 视觉恢复保留 SNR 与列标题版

背景：

- 用户进一步明确版式需求：不要最左侧行文字，但需要保留 SNR 数据以及顶部各列图像含义。
- 本轮图件因此采用：`row_label_style=none`、`panel_metric_style=snr`、`column_label_style=labels`。
- 样本来源和退化条件仍通过 manifest 与 selection summary 追踪，图中不再放 `Sample 1/2/3` 或 `Light/Medium/Heavy`。

生成命令与版本：

```bash
conda run -n quant python SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_2_exp01_w4a32_visual_recovery/scripts/make_w4a32_visual_recovery.py --candidate-set all --device cuda --cuda-device-index 1 --set-a-selection fixed_patch_from_medium --row-label-style none --panel-metric-style snr --figure-title-style none --column-label-style labels --colorbar-style per_row
```

- 生成 `set_a_three_degradation_levels` v007：无左侧行文字，保留列标题与 SNR，保留每行色标。
- 生成 `set_b_three_medium_samples` v006：无左侧行文字，保留列标题与 SNR，保留每行色标。
- 人工检查后发现色标会额外占用横向版面，因此继续生成无色标版本作为正文优先候选。

```bash
conda run -n quant python SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_2_exp01_w4a32_visual_recovery/scripts/make_w4a32_visual_recovery.py --candidate-set all --device cuda --cuda-device-index 1 --set-a-selection fixed_patch_from_medium --row-label-style none --panel-metric-style snr --figure-title-style none --column-label-style labels --colorbar-style none
```

- 生成 `set_a_three_degradation_levels` v008：无左侧行文字，保留列标题与 SNR，不显示色标。
- 生成 `set_b_three_medium_samples` v007：无左侧行文字，保留列标题与 SNR，不显示色标。

新增候选图：

| 候选集 | 版本 | 左侧行文字 | 顶部列标题 | SNR | 色标 | 建议 |
|---|---|---|---|---|---|---|
| set_a | v007 | no | yes | yes | yes | 备用，色标占版面 |
| set_a | v008 | no | yes | yes | no | 正文优先 |
| set_b | v006 | no | yes | yes | yes | 备用，色标占版面 |
| set_b | v007 | no | yes | yes | no | 正文优先 |

人工检查：

- `set_a_three_degradation_levels` v008 图内没有 `Light/Medium/Heavy` 行文字。
- `set_b_three_medium_samples` v007 图内没有 `Sample 1/2/3` 行文字。
- v008/v007 顶部保留 `Clean / Degraded input / FP32 / W4A32 pre / W4A32 final`。
- v008/v007 对非 clean 图像保留 SNR 数值。

提交前复核：

- `set_a` 的 `manifest_v007.json`、`manifest_v008.json` 与 `set_b` 的 `manifest_v006.json`、`manifest_v007.json` 均通过 `jq empty`。
- 新增 PNG 均为 `4260 x 2160` RGBA 图像，对应 PDF 均为 1 页。
- `git diff --check -- SCRN_BRECQ_app/scrn_brecq/paper_artifacts/ARTIFACTS_LOG.md` 无输出。
- 本地选择与版式开关测试重新通过：`Ran 9 tests in 3.230s`。

## 2026-05-12 ch4_3 W4A8/W4A4 激活量化 3×6 视觉恢复候选图

背景：

- 用户要求为 W4A8 和 W4A4 两个激活量化实验各画一组视觉恢复图。
- 每行 6 列，列顺序固定为：`Clean / Degraded input / FP32 / W4A32 final / W4A8 或 W4A4 pre-act / W4A8 或 W4A4 final`。
- 本轮先为每个实验生成若干张 3 行 × 6 列候选图，用于后续挑选。

文件管理：

- 新建 W4A8 实验目录：`ch4_3_exp01_w4a8_visual_recovery`。
- 新建 W4A4 实验目录：`ch4_3_exp02_w4a4_visual_recovery`。
- 两个实验均包含 `README.md`、`experiment_info.json`、候选集目录、`shortlisted/` 和 `final/`。
- 本地生成脚本位于 W4A8 实验的 `scripts/` 下，按当前 `.gitignore` 规则纳入 git 提交。
- 本地测试文件位于 `paper_artifacts/tests/`，同样作为可复现代码纳入 git 提交。

数据来源：

- W4A8 使用 `NE000_0_normalized_w4a8_activation_reconstruction`：
  - pre-act checkpoint：`quantized_scrn_brecq_pre_act_recon.pth`
  - final checkpoint：`quantized_scrn_brecq.pth`
  - full-grid metrics：`20260509_221644_normalized_w4a8_tensor_a5000_grid478_seed20260507`
- W4A4 使用 baseline `NE000_2_normalized_w4a4_activation_reconstruction_probe`：
  - pre-act checkpoint：`quantized_scrn_brecq_pre_act_recon.pth`
  - final checkpoint：`quantized_scrn_brecq.pth`
  - full-grid metrics：`20260509_234948_normalized_w4a4_tensor_a5000_grid478_seed20260507`
- W4A32 对照列复用 `20260509_144941_normalized_w4a32_1024cali_w20000_single_gpu1/checkpoints/quantized_scrn_brecq.pth`。

本地测试：

- 先新增脚本测试，约束 6 列顺序、W4A8/W4A4 标签和 manifest 字段。
- 初次测试按预期失败，原因是新脚本尚不存在。
- 实现脚本后测试通过：`Ran 4 tests in 3.404s`。
- 之后增加固定 patch 选择测试，用于强制使用 `test_000297.npy` 生成三种退化程度候选；初次失败后实现 `--fixed-patch-file`，最终测试通过。
- 最终联合测试：

```bash
conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.paper_artifacts.tests.test_w4a32_visual_recovery_selection SCRN_BRECQ_app.scrn_brecq.paper_artifacts.tests.test_activation_visual_recovery
```

- 结果：通过，`Ran 14 tests in 5.238s`。

生成命令：

```bash
conda run -n quant python SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_3_exp01_w4a8_visual_recovery/scripts/make_activation_visual_recovery.py --activation-experiment all --candidate-set all --device cuda --cuda-device-index 1 --set-a-selection fixed_patch_from_medium --row-label-style none --panel-metric-style snr --column-label-style labels --colorbar-style none
```

- 生成 W4A8 `set_a` v001、W4A8 `set_b` v001、W4A4 `set_a` v001、W4A4 `set_b` v001。

```bash
conda run -n quant python SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_3_exp01_w4a8_visual_recovery/scripts/make_activation_visual_recovery.py --activation-experiment all --candidate-set set_a_three_degradation_levels --fixed-patch-file test_000297.npy --device cuda --cuda-device-index 1 --row-label-style none --panel-metric-style snr --column-label-style labels --colorbar-style none
```

- 生成 W4A8 `set_a` v002 和 W4A4 `set_a` v002。

新增候选图：

| 实验 | 候选集 | 版本 | 选样 | 说明 |
|---|---|---|---|---|
| W4A8 | set_a | v001 | 自动选择同一 patch 三退化程度 | 选中 Kerry3D `test_000077.npy`，结构偏弱，保留作候选 |
| W4A8 | set_a | v002 | 固定 `test_000297.npy` 三退化程度 | 视觉结构更清楚，当前优先推荐 |
| W4A8 | set_b | v001 | 三个中等退化样本 | 覆盖 Anisotropic/Kerry3D/Shots0001 |
| W4A4 | set_a | v001 | 自动选择同一 patch 三退化程度 | 选中 Shots0001 `test_000100.npy`，W4A4 损失明显 |
| W4A4 | set_a | v002 | 固定 `test_000297.npy` 三退化程度 | 与 W4A8 v002 对齐，便于横向比较 |
| W4A4 | set_b | v001 | 三个中等退化样本 | 覆盖 Anisotropic/Kerry3D/Shots0001 |

关键样本：

- W4A8 `set_a` v002：`test_000297.npy`，`patch_index=296`，`source=Shots0001`。
- W4A4 `set_a` v002：`test_000297.npy`，`patch_index=296`，`source=Shots0001`。
- 三种退化条件：
  - light：`snr_setting_db=10.0`，`missing_rate=0.02`
  - medium：`snr_setting_db=1.0`，`missing_rate=0.18`
  - heavy：`snr_setting_db=-2.0`，`missing_rate=0.38`

人工检查：

- 六张图均为 3 行 × 6 列。
- 图内无最左侧行标签。
- 顶部保留列含义和 SNR。
- W4A8 v002 与 W4A4 v002 使用同一 patch，适合对比 W4A8/W4A4 激活量化差异。

提交前复核：

- 两个 `experiment_info.json` 和 6 个 manifest 均通过 `jq empty`。
- 6 张 PNG 均为 `5040 x 2040` RGBA 图像。
- 6 张 PDF 均为 1 页 PDF。
- 本地联合测试通过：`Ran 14 tests in 5.238s`。

## 2026-05-12 paper_artifacts git 提交范围修订

背景：

- 用户进一步明确 `paper_artifacts` 的 git 提交范围：该工作区下很多生成内容不需要提交，后续应参考仓库 `.gitignore` 的管理方式，只提交代码文件和日志内容。
- 因此本轮将候选图、最终图、按版本生成的 manifest 和 selection summary 统一视为本地生成结果。

规则调整：

- 提交：图件生成脚本、测试脚本、README、`experiment_info.json`、`ARTIFACTS_LOG.md`、`DEVELOPMENT_LOG.md`。
- 不提交：`.png`、`.pdf`、`manifest_vXXX.json`、`selection_summary_vXXX.md`。
- 不提交：Python 缓存和临时文件。

索引处理：

- 更新 `paper_artifacts/.gitignore`，加入候选集、筛选集和最终集下图件与按版本元数据的忽略规则。
- 只使用 `git rm --cached` 从 git 索引移除已跟踪的生成结果，保留工作区本地文件，便于继续挑图和查看。
- 后续每次提交前需要用 `git status --short` 和 `git ls-files SCRN_BRECQ_app/scrn_brecq/paper_artifacts` 检查是否误把生成结果加入索引。
