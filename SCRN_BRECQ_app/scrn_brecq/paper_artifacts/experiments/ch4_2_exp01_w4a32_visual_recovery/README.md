# ch4_2_exp01_w4a32_visual_recovery

This experiment manages candidate visual recovery figures for thesis Chapter 4.2.2, "W4A32 weight quantization result analysis".

## Figure Goal

The figures compare:

1. Clean reference patch
2. Degraded input
3. FP32 SCRN restoration
4. W4A32 pre-reconstruction restoration
5. W4A32 final restoration

The intended argument is that direct 4-bit weight quantization causes visible and quantitative degradation, while BRECQ weight reconstruction restores W4A32 output close to FP32.

## Candidate Sets

- `candidates/set_a_three_degradation_levels/`: one representative sample each for light, medium, and heavy degradation.
- `candidates/set_b_three_medium_samples/`: three different test samples under the same medium-degradation condition.

## Selection Workflow

1. Generate versioned candidate figures and matching `manifest_vXXX.json` files in `candidates/`.
2. Copy or regenerate promising versions into `shortlisted/`.
3. Place only the final thesis figure in `final/`.
4. Record generation and selection decisions in `../../ARTIFACTS_LOG.md`.

## Provenance Requirement

Each candidate manifest must identify every displayed test-set row by `patch_index`, `patch_file`, `source`, `condition_index`, `snr_setting_db`, and `missing_rate`. This requirement makes it possible to state exactly which test-set sample appears in the thesis figure.
