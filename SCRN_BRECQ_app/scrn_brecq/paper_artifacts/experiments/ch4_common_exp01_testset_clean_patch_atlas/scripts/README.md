# scripts

本目录存放 clean patch 浏览图册生成脚本。

默认脚本：

```bash
conda run -n quant python SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_common_exp01_testset_clean_patch_atlas/scripts/make_testset_clean_atlas.py
```

脚本读取固定 478 patch 测试集，生成 6×8 分页缩略图和样本索引表。
