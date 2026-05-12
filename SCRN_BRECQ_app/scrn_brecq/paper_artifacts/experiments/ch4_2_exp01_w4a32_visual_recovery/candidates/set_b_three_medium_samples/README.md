# Candidate Set B: Three Medium-Degradation Samples

This candidate set is for a 3x5 visual recovery figure where all rows use the same medium-degradation condition but different test patches.

Planned rows:

| Row | Input SNR setting | Missing rate | Purpose |
|---|---:|---:|---|
| Sample 1 | 1 dB | 0.18 | Representative medium-degradation patch |
| Sample 2 | 1 dB | 0.18 | Different seismic structure under the same condition |
| Sample 3 | 1 dB | 0.18 | Additional medium-degradation candidate |

Planned columns:

1. Clean
2. Degraded input
3. FP32
4. W4A32 pre-reconstruction
5. W4A32 final

For every generated version, this directory must contain:

- `fig_ch4_2_w4a32_3x5_medium_samples_vXXX.png`
- `fig_ch4_2_w4a32_3x5_medium_samples_vXXX.pdf`
- `manifest_vXXX.json`

The manifest must record the exact `patch_index`, `patch_file`, `source`, and degradation condition for each displayed row.
