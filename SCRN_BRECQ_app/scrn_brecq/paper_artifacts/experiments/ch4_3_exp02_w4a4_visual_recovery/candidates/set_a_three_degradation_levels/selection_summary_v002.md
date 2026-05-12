# W4A4 set_a_three_degradation_levels v002

## 图件文件

- PNG：`SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_3_exp02_w4a4_visual_recovery/candidates/set_a_three_degradation_levels/fig_ch4_3_w4a4_3x6_levels_v002.png`
- PDF：`SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_3_exp02_w4a4_visual_recovery/candidates/set_a_three_degradation_levels/fig_ch4_3_w4a4_3x6_levels_v002.pdf`

## 选中样本

| 行标签 | source | patch_index | patch_file | condition_index | SNR setting | missing | FP32 SNR | W4A32 final SNR | pre-act SNR | final SNR | final - FP32 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| light | Shots0001 | 296 | test_000297.npy | 20 | 10.0 | 0.02 | 22.1983 | 22.2017 | 15.1432 | 17.0392 | -5.1591 |
| medium | Shots0001 | 296 | test_000297.npy | 12 | 1.0 | 0.18 | 16.6976 | 16.6553 | 12.3935 | 14.1128 | -2.5848 |
| heavy | Shots0001 | 296 | test_000297.npy | 4 | -2.0 | 0.38 | 13.0495 | 12.9614 | 9.0836 | 10.9864 | -2.0631 |

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
