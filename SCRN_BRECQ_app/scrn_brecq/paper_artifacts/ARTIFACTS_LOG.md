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

## 2026-05-12 ch4_2_exp01 W4A32 visual recovery generator

Purpose:

- Add a reusable script for generating Chapter 4.2.2 W4A32 3x5 visual recovery candidate figures.
- Keep sample selection reproducible and record exact test-set provenance in `manifest_vXXX.json`.
- Add focused tests for representative selection and required manifest fields.

Added files:

- `experiments/ch4_2_exp01_w4a32_visual_recovery/scripts/make_w4a32_visual_recovery.py`
- `tests/test_w4a32_visual_recovery_selection.py`

Selection policy implemented:

- Candidate set A selects one median-like representative for each planned degradation condition:
  - light: `snr_setting_db=10.0`, `missing_rate=0.02`
  - medium: `snr_setting_db=1.0`, `missing_rate=0.18`
  - heavy: `snr_setting_db=-2.0`, `missing_rate=0.38`
- Candidate set B selects three medium-degradation samples with source diversity using source priority:
  - `Anisotropic`
  - `Kerry3D`
  - `Shots0001`
- Representative score uses condition medians of:
  - `fp32_snr_db`
  - `quant_pre_minus_fp32_snr_db`
  - `quant_post_minus_fp32_snr_db`
  - `quant_post_minus_pre_snr_db`

Verification:

- Red test first: `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.paper_artifacts.tests.test_w4a32_visual_recovery_selection` failed because `make_w4a32_visual_recovery.py` did not exist.
- Green test after implementation: same command passed, `Ran 3 tests in 2.111s`.

No figures were generated in this script-development step.

## 2026-05-12 ch4_2_exp01 generator path-execution fix

Issue:

- Running `make_w4a32_visual_recovery.py` by file path failed with `ModuleNotFoundError: No module named 'SCRN_BRECQ_app'`.
- Root cause: Python adds the script directory to `sys.path`, but not the repository root, when a script is executed by file path.

Fix:

- Add repository-root discovery at script startup.
- Insert the repository root into `sys.path` before importing project modules.
- Add a regression test that executes `python make_w4a32_visual_recovery.py --help` by file path.

Verification:

- Before fix: `test_script_help_runs_when_executed_by_path` failed with the same `ModuleNotFoundError`.
- After fix: `conda run -n quant python -m unittest SCRN_BRECQ_app.scrn_brecq.paper_artifacts.tests.test_w4a32_visual_recovery_selection` passed, `Ran 4 tests in 3.371s`.

No figures were generated in this fix step.
