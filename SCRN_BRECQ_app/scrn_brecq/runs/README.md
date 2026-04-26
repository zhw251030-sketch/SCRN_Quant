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

