# 论文图件工作区

本目录用于保存 SCRN-BRECQ 实验的论文图件、样本追踪文件和选择记录。

## 管理原则

- 一个实验使用 `experiments/` 下的一个独立目录。
- 候选图、较优候选图和最终论文图分开保存。
- 每个候选图版本必须有对应的 manifest，记录精确测试样本和退化条件。
- 图件生成、筛选和最终选择过程必须记录在 `ARTIFACTS_LOG.md`。
- 图件文件名使用版本号，例如 `fig_ch4_2_w4a32_3x5_levels_v001.png`。

## git 提交范围

后续图件工作只提交结果文件和记录文件：

- 提交：`.png`、`.pdf` 等结果图文件。
- 提交：`manifest_vXXX.json`、`selection_summary_vXXX.md`。
- 提交：README 和日志文件。
- 不提交：本地生成脚本、测试脚本、Python 缓存和临时文件。

如果需要本地脚本来生成图件，可以放在实验目录的 `scripts/` 下；这些脚本只作为本地工具使用，不纳入 git 跟踪，除非之后明确要求提交。

## 目录结构

```text
paper_artifacts/
  ARTIFACTS_LOG.md
  experiments/
    <experiment_id>/
      README.md
      experiment_info.json
      scripts/
      candidates/
      shortlisted/
      final/
```

## manifest 要求

每个候选图集合必须包含 `manifest_vXXX.json`。对于视觉恢复网格图，每一行至少记录：

- `testset_id`
- `patch_index`
- `patch_file`
- `source`
- `condition_index`
- `snr_setting_db`
- `missing_rate`
- `fp32_snr_db`
- `quant_pre_recon_snr_db`
- `quant_post_recon_snr_db`
- `quant_post_minus_fp32_snr_db`

判断论文图中某一行来自测试集哪个样本时，以 manifest 为准。
