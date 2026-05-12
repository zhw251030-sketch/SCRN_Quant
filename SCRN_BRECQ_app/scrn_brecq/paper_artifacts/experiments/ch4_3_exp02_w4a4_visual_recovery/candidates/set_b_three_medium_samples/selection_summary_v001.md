# W4A4 set_b_three_medium_samples v001

## 图件文件

- PNG：`SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_3_exp02_w4a4_visual_recovery/candidates/set_b_three_medium_samples/fig_ch4_3_w4a4_3x6_medium_samples_v001.png`
- PDF：`SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_3_exp02_w4a4_visual_recovery/candidates/set_b_three_medium_samples/fig_ch4_3_w4a4_3x6_medium_samples_v001.pdf`

## 选中样本

| 行标签 | source | patch_index | patch_file | condition_index | SNR setting | missing | FP32 SNR | W4A32 final SNR | pre-act SNR | final SNR | final - FP32 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sample_1 | Anisotropic | 40 | test_000041.npy | 12 | 1.0 | 0.18 | 18.7976 | 18.6380 | 12.6110 | 12.8661 | -5.9315 |
| sample_2 | Kerry3D | 75 | test_000076.npy | 12 | 1.0 | 0.18 | 10.1332 | 10.0926 | 5.4040 | 5.4454 | -4.6878 |
| sample_3 | Shots0001 | 99 | test_000100.npy | 12 | 1.0 | 0.18 | 19.8472 | 19.7202 | 14.1338 | 16.0076 | -3.8396 |

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
