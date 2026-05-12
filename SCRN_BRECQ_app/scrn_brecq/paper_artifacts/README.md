# Paper Artifacts

This directory stores thesis figures and their selection metadata for the SCRN-BRECQ experiments.

## Principles

- One experiment uses one directory under `experiments/`.
- Candidate figures, shortlisted figures, and final paper figures are stored separately.
- Every generated figure version must have a manifest that records the exact test sample and degradation condition.
- Development or figure-selection changes must be recorded in `ARTIFACTS_LOG.md`.
- Figure files should use versioned names such as `fig_ch4_2_w4a32_3x5_levels_v001.png`.

## Directory Layout

```text
paper_artifacts/
  ARTIFACTS_LOG.md
  experiments/
    <experiment_id>/
      README.md
      experiment_info.json
      scripts/
      candidates/
      shortlisted/
      final/
```

## Manifest Requirement

Each candidate figure set must include a `manifest_vXXX.json` file. For visual recovery grids, every displayed row must record:

- `testset_id`
- `patch_index`
- `patch_file`
- `source`
- `condition_index`
- `snr_setting_db`
- `missing_rate`
- `fp32_snr_db`
- `quant_pre_recon_snr_db`
- `quant_post_recon_snr_db`
- `quant_post_minus_fp32_snr_db`

The manifest is the authoritative source for identifying which test-set sample appears in a paper figure.
