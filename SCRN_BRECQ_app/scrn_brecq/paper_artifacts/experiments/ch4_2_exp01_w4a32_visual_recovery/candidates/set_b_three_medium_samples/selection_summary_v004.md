# set_b_three_medium_samples v004

## 图件文件

- PNG：`SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_2_exp01_w4a32_visual_recovery/candidates/set_b_three_medium_samples/fig_ch4_2_w4a32_3x5_medium_samples_v004.png`
- PDF：`SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_2_exp01_w4a32_visual_recovery/candidates/set_b_three_medium_samples/fig_ch4_2_w4a32_3x5_medium_samples_v004.pdf`

## 选中样本

| 行标签 | source | patch_index | patch_file | condition_index | SNR setting | missing | FP32 SNR | W4A32 pre SNR | W4A32 final SNR | final - FP32 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| sample_1 | Anisotropic | 50 | test_000051.npy | 12 | 1.0 | 0.18 | 21.9377 | 21.1601 | 21.9119 | -0.0258 |
| sample_2 | Kerry3D | 82 | test_000083.npy | 12 | 1.0 | 0.18 | 10.8369 | 10.1483 | 10.8059 | -0.0309 |
| sample_3 | Shots0001 | 296 | test_000297.npy | 12 | 1.0 | 0.18 | 16.6976 | 15.9910 | 16.6553 | -0.0423 |

## 选样策略

完整选样策略见下方 JSON 记录：

```json
{
  "condition": {
    "missing_rate": 0.18,
    "snr_setting_db": 1.0
  },
  "description": "Select three source-diverse median-like representatives under medium degradation.",
  "score_fields": [
    "fp32_snr_db",
    "quant_pre_minus_fp32_snr_db",
    "quant_post_minus_fp32_snr_db",
    "quant_post_minus_pre_snr_db"
  ],
  "source_priority": [
    "Anisotropic",
    "Kerry3D",
    "Shots0001"
  ]
}
```
