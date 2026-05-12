# clean_patch_atlas

本目录存放 478 张 clean patch 浏览图册候选结果。

默认版式：

- 每页 6 列 × 8 行，共 48 张 patch。
- 全测试集 478 张 patch 生成 10 页。
- 每张小图显示 `test_XXXXX`、`idx` 和 `source`。
- 图像使用 normalized clean patch，色图为 `seismic`，振幅范围固定为 `[-1, 1]`。

生成的 PNG、PDF、`manifest_vXXX.json` 和 `selection_index_vXXX.csv` 不纳入 git。
