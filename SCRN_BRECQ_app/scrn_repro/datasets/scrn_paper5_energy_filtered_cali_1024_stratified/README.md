# SCRN Paper-Style 1024 Stratified Calibration Clean Patches

This directory is stratified from the paper-style 5-source training clean patch set. The clean targets are intended for SCRN-BRECQ calibration pipelines that generate degraded inputs online.

## Summary

- Sample count: `1024`
- Seed: `20260507`
- Manifest: `manifest.json`
- Data files are ignored by Git and should remain local.

## Per-source Counts

- `1997_2.5D_shots`: `28`
- `7m_shots_0201`: `320`
- `Anisotropic_FD_Model`: `71`
- `Kerry3D`: `46`
- `Shots0001_0200`: `559`

## Energy Filter

- Minimum patch std: `0.001`
- Minimum patch absmax: `0.001`
- Non-finite patches: rejected
- All-zero / near-zero patches: rejected

## Filtering Statistics

- `1997_2.5D_shots`: scanned_regions=`42`, candidates=`252`, low_energy_rejected=`192`
- `7m_shots_0201`: scanned_regions=`5`, candidates=`3355`, low_energy_rejected=`2573`
- `Anisotropic_FD_Model`: scanned_regions=`7`, candidates=`525`, low_energy_rejected=`357`
- `Kerry3D`: scanned_regions=`5`, candidates=`480`, low_energy_rejected=`384`
- `Shots0001_0200`: scanned_regions=`15`, candidates=`5865`, low_energy_rejected=`4635`
