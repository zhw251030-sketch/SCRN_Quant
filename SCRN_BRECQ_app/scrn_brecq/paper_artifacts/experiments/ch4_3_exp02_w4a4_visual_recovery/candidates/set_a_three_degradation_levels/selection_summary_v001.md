# W4A4 set_a_three_degradation_levels v001

## 图件文件

- PNG：`SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_3_exp02_w4a4_visual_recovery/candidates/set_a_three_degradation_levels/fig_ch4_3_w4a4_3x6_levels_v001.png`
- PDF：`SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_3_exp02_w4a4_visual_recovery/candidates/set_a_three_degradation_levels/fig_ch4_3_w4a4_3x6_levels_v001.pdf`

## 选中样本

| 行标签 | source | patch_index | patch_file | condition_index | SNR setting | missing | FP32 SNR | W4A32 final SNR | pre-act SNR | final SNR | final - FP32 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| light | Shots0001 | 99 | test_000100.npy | 20 | 10.0 | 0.02 | 24.8207 | 24.7659 | 17.5561 | 18.4841 | -6.3366 |
| medium | Shots0001 | 99 | test_000100.npy | 12 | 1.0 | 0.18 | 19.8472 | 19.7202 | 14.1338 | 16.0076 | -3.8396 |
| heavy | Shots0001 | 99 | test_000100.npy | 4 | -2.0 | 0.38 | 16.4028 | 16.3276 | 11.1348 | 13.6758 | -2.7270 |

## 选样策略

```json
{
  "conditions": [
    {
      "missing_rate": 0.02,
      "row_label": "light",
      "snr_setting_db": 10.0
    },
    {
      "missing_rate": 0.18,
      "row_label": "medium",
      "snr_setting_db": 1.0
    },
    {
      "missing_rate": 0.38,
      "row_label": "heavy",
      "snr_setting_db": -2.0
    }
  ],
  "description": "Select one median-like representative row for light, medium, and heavy degradation.",
  "score_fields": [
    "fp32_snr_db",
    "quant_pre_minus_fp32_snr_db",
    "quant_post_minus_fp32_snr_db",
    "quant_post_minus_pre_snr_db"
  ]
}
```
