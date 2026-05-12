# ch4_3_exp01_w4a8_visual_recovery

本实验服务于论文第 4 章激活量化结果展示，用于生成 W4A8 视觉恢复候选图。

图件采用 3 行 x 6 列布局：

1. Clean
2. Degraded input
3. FP32
4. W4A32 final
5. W4A8 pre-act
6. W4A8 final

候选图只在图内展示列含义和 SNR，不在最左侧放行标签。样本来源、patch 编号、退化条件和指标以 `manifest_vXXX.json` 和 `selection_summary_vXXX.md` 为准。
