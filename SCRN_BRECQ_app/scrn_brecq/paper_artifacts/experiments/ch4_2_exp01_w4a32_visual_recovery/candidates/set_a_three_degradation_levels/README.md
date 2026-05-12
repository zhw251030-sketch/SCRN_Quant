# Candidate Set A: Three Degradation Levels

This candidate set is for a 3x5 visual recovery figure where rows represent different degradation strengths.

Planned rows:

| Row | Input SNR setting | Missing rate | Purpose |
|---|---:|---:|---|
| Light | 10 dB | 0.02 | Show performance under mild degradation |
| Medium | 1 dB | 0.18 | Show representative mid-level degradation |
| Heavy | -2 dB | 0.38 | Show performance under difficult degradation |

Planned columns:

1. Clean
2. Degraded input
3. FP32
4. W4A32 pre-reconstruction
5. W4A32 final

For every generated version, this directory must contain:

- `fig_ch4_2_w4a32_3x5_levels_vXXX.png`
- `fig_ch4_2_w4a32_3x5_levels_vXXX.pdf`
- `manifest_vXXX.json`

The manifest must record the exact `patch_index`, `patch_file`, `source`, and degradation condition for each displayed row.
