import torch
from model.SCRN import SCRN

# 初始化模型
model = SCRN()
# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params } ")