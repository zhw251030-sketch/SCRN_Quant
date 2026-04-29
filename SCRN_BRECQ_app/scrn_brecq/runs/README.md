# SCRN-BRECQ 运行产物目录

本目录用于保存后续量化实验产生的配置快照、指标摘要和结果说明。

请不要提交以下运行产物：

- 模型权重：`.pth`、`.pt`、`.ckpt`
- 数据和中间数组：`.npy`、`.segy`
- 日志、缓存和临时文件
- `__pycache__`
- `.ipynb_checkpoints`

后续脚本应在这里按时间戳创建子目录，例如：

```text
runs/quant/20260426_120000_w4a4_example/
```

当前约定的 run 根目录:

- `runs/quant/`: 完整量化、重建、评估和 checkpoint 保存。
- `runs/quant_eval/`: 已保存量化 checkpoint 的单样本评估。
- `runs/generalization_eval/`: 已保存量化 checkpoint 的多样本泛化评估。

量化 run 中的 checkpoint 约定：

- `checkpoints/quantized_scrn_brecq_pre_recon.pth`: 权重量化初始化后、BRECQ reconstruction 前的量化模型。
- `checkpoints/quantized_scrn_brecq.pth`: BRECQ reconstruction 后的最终量化模型。

多样本泛化评估如果只传 `--checkpoint`，报告中的 `quant_*` 旧字段表示最终量化模型；如果同时传
`--pre-recon-checkpoint`，报告会额外写入 `quant_pre_recon_*`、`quant_post_recon_*` 和
`quant_post_minus_pre_*` 指标。
