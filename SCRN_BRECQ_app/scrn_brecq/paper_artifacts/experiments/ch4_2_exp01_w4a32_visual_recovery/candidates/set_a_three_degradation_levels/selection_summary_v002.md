# set_a_three_degradation_levels v002

## 图件文件

- PNG：`SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_2_exp01_w4a32_visual_recovery/candidates/set_a_three_degradation_levels/fig_ch4_2_w4a32_3x5_levels_v002.png`
- PDF：`SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_2_exp01_w4a32_visual_recovery/candidates/set_a_three_degradation_levels/fig_ch4_2_w4a32_3x5_levels_v002.pdf`

## 选中样本

| 行标签 | source | patch_index | patch_file | condition_index | SNR setting | missing | FP32 SNR | W4A32 pre SNR | W4A32 final SNR | final - FP32 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| light | Shots0001 | 100 | test_000101.npy | 20 | 10.0 | 0.02 | 22.5835 | 20.1609 | 22.5351 | -0.0484 |
| medium | Shots0001 | 296 | test_000297.npy | 12 | 1.0 | 0.18 | 16.6976 | 15.9910 | 16.6553 | -0.0423 |
| heavy | Shots0001 | 303 | test_000304.npy | 4 | -2.0 | 0.38 | 15.1011 | 14.4356 | 15.0577 | -0.0433 |

## 选样策略

完整选样策略见下方 JSON 记录：

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
