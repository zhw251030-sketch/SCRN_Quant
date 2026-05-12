# W4A8 set_a_three_degradation_levels v001

## 图件文件

- PNG：`SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_3_exp01_w4a8_visual_recovery/candidates/set_a_three_degradation_levels/fig_ch4_3_w4a8_3x6_levels_v001.png`
- PDF：`SCRN_BRECQ_app/scrn_brecq/paper_artifacts/experiments/ch4_3_exp01_w4a8_visual_recovery/candidates/set_a_three_degradation_levels/fig_ch4_3_w4a8_3x6_levels_v001.pdf`

## 选中样本

| 行标签 | source | patch_index | patch_file | condition_index | SNR setting | missing | FP32 SNR | W4A32 final SNR | pre-act SNR | final SNR | final - FP32 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| light | Kerry3D | 76 | test_000077.npy | 20 | 10.0 | 0.02 | 10.6322 | 10.6366 | 10.4902 | 10.4857 | -0.1465 |
| medium | Kerry3D | 76 | test_000077.npy | 12 | 1.0 | 0.18 | 9.8024 | 9.7757 | 9.6571 | 9.6606 | -0.1418 |
| heavy | Kerry3D | 76 | test_000077.npy | 4 | -2.0 | 0.38 | 3.9454 | 4.3260 | 4.3519 | 4.2485 | 0.3030 |

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
