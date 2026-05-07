# SCRN Paper-Style Energy-Filtered 478 Clean Test Patches

This directory uses deterministic post-training source regions for the three locally available SCRN Table 3 test sources, with near-zero clean patches hard-filtered and no test augmentation.

## Summary

- Sample count: `478`
- Seed: `20260507`
- Manifest: `manifest.json`
- Data files are ignored by Git and should remain local.

## Per-source Counts

- `Anisotropic`: `75`
- `Kerry3D`: `16`
- `Shots0001`: `387`

## Energy Filter

- Minimum patch std: `0.001`
- Minimum patch absmax: `0.001`
- Non-finite patches: rejected
- All-zero / near-zero patches: rejected

## Filtering Statistics

- `Anisotropic`: scanned_regions=`4`, candidates=`300`, low_energy_rejected=`204`, train_hash_excluded=`0`
- `Kerry3D`: scanned_regions=`1`, candidates=`24`, low_energy_rejected=`3`, train_hash_excluded=`0`
- `Shots0001`: scanned_regions=`5`, candidates=`1955`, low_energy_rejected=`1545`, train_hash_excluded=`0`

## Train/Test Boundaries

- `Anisotropic` starts after training boundary `7`
- `Kerry3D` starts after training boundary `1435`
- `Shots0001` starts after training boundary `15`
