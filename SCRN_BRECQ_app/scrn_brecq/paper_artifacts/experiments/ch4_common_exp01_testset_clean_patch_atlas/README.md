# ch4_common_exp01_testset_clean_patch_atlas

本实验用于生成第 4 章结果展示前的测试集浏览图册，覆盖
`scrn_paper5_energy_filtered_perpatch_absmax_test_478` 的 478 个 clean patch。

图册只用于人工挑选视觉结构清楚的测试样本，不作为论文最终正文图。正式论文图仍由后续
W4A32、W4A8、W4A4 视觉恢复实验针对指定 `test_XXXXX.npy` 单独生成。

默认输出：

- `candidates/clean_patch_atlas/atlas_clean_test478_v001.pdf`
- `candidates/clean_patch_atlas/atlas_clean_test478_v001_page_001.png` 至 page 010
- `candidates/clean_patch_atlas/selection_index_v001.csv`
- `candidates/clean_patch_atlas/manifest_v001.json`

生成结果按 `.gitignore` 规则保留在本地，不提交到 git；脚本、测试、README 和日志纳入 git。
