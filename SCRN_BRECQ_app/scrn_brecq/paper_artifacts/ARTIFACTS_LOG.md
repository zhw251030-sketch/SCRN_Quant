# Paper Artifacts Log

This log records thesis figure workspace changes, generated figure candidates, selection decisions, and final figure provenance.

## 2026-05-12 ch4_2_exp01 W4A32 visual recovery workspace initialization

Purpose:

- Initialize a managed workspace for thesis result figures under `paper_artifacts/`.
- Create the first experiment directory for Chapter 4.2 W4A32 weight-quantization visual recovery figures.
- Establish mandatory manifest rules so every selected image can be traced back to a specific test-set sample and degradation condition.

Created experiment:

- Experiment id: `ch4_2_exp01_w4a32_visual_recovery`
- Thesis target: Chapter 4.2.2, W4A32 weight quantization result analysis
- Intended visual columns: `Clean`, `Degraded input`, `FP32`, `W4A32 pre-reconstruction`, `W4A32 final`
- Candidate set A: three degradation levels
- Candidate set B: three medium-degradation samples

Source artifacts recorded in `experiment_info.json`:

- FP32 checkpoint: normalized paper5 energy-filtered per-patch absmax SCRN checkpoint
- W4A32 checkpoints: pre-reconstruction and final BRECQ checkpoints from the E007 normalized W4A32 baseline run
- Test set: normalized 478-patch fixed test set
- Metrics: E007 normalized W4A32 `per_sample_metrics.jsonl`

Sample provenance rule:

- Each generated candidate version must write `manifest_vXXX.json`.
- For every displayed row, the manifest must record `testset_id`, `patch_index`, `patch_file`, `source`, `condition_index`, `snr_setting_db`, `missing_rate`, and the FP32/W4A32 pre/final SNR fields.
- The manifest, not the image filename, is the authoritative record of which test-set sample was used.

No figures were generated in this initialization step.
