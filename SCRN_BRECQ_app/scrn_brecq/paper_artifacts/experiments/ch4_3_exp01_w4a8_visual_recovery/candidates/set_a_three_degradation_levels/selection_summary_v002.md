# W4A8 set_a_three_degradation_levels v002

## 图件文件

- PNG：`SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_3_exp01_w4a8_visual_recovery/candidates/set_a_three_degradation_levels/fig_ch4_3_w4a8_3x6_levels_v002.png`
- PDF：`SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_3_exp01_w4a8_visual_recovery/candidates/set_a_three_degradation_levels/fig_ch4_3_w4a8_3x6_levels_v002.pdf`

## 选中样本

| 行标签 | source | patch_index | patch_file | condition_index | SNR setting | missing | FP32 SNR | W4A32 final SNR | pre-act SNR | final SNR | final - FP32 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| light | Shots0001 | 296 | test_000297.npy | 20 | 10.0 | 0.02 | 22.1983 | 22.2017 | 21.8699 | 21.9331 | -0.2652 |
| medium | Shots0001 | 296 | test_000297.npy | 12 | 1.0 | 0.18 | 16.6976 | 16.6553 | 16.5786 | 16.6139 | -0.0837 |
| heavy | Shots0001 | 296 | test_000297.npy | 4 | -2.0 | 0.38 | 13.0495 | 12.9614 | 12.9911 | 12.9621 | -0.0874 |

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
