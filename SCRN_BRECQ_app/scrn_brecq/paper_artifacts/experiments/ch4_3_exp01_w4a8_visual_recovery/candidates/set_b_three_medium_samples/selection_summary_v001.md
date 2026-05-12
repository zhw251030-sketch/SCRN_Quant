# W4A8 set_b_three_medium_samples v001

## 图件文件

- PNG：`SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_3_exp01_w4a8_visual_recovery/candidates/set_b_three_medium_samples/fig_ch4_3_w4a8_3x6_medium_samples_v001.png`
- PDF：`SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_3_exp01_w4a8_visual_recovery/candidates/set_b_three_medium_samples/fig_ch4_3_w4a8_3x6_medium_samples_v001.pdf`

## 选中样本

| 行标签 | source | patch_index | patch_file | condition_index | SNR setting | missing | FP32 SNR | W4A32 final SNR | pre-act SNR | final SNR | final - FP32 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sample_1 | Anisotropic | 71 | test_000072.npy | 12 | 1.0 | 0.18 | 23.3082 | 23.3363 | 22.9519 | 22.9609 | -0.3473 |
| sample_2 | Kerry3D | 76 | test_000077.npy | 12 | 1.0 | 0.18 | 9.8024 | 9.7757 | 9.6571 | 9.6606 | -0.1418 |
| sample_3 | Shots0001 | 151 | test_000152.npy | 12 | 1.0 | 0.18 | 17.0117 | 16.9512 | 16.9180 | 16.9205 | -0.0911 |

## 选样策略

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
