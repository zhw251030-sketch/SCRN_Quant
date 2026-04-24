import torch

model = torch.load(
    '/home/zhangxin/hanwen/Github/SCRN-main/trained_model/model.pth',
    map_location='cpu',
    weights_only=False,   # 关键
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params } ")

