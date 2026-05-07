# SCRN Paper-Style 5-Source Energy-Filtered 10750 Training Clean Patches

This directory follows a deterministic paper-style protocol for the five locally available SCRN Table 2 training sources, with near-zero clean patches hard-filtered before seeded source-wise selection and augmentation.

## Summary

- Sample count: `10750`
- Seed: `20260507`
- Manifest: `manifest.json`
- Data files are ignored by Git and should remain local.

## Per-source Counts

- `1997_2.5D_shots`: `300`
- `7m_shots_0201`: `3355`
- `Anisotropic_FD_Model`: `750`
- `Kerry3D`: `480`
- `Shots0001_0200`: `5865`

## Energy Filter

- Minimum patch std: `0.001`
- Minimum patch absmax: `0.001`
- Non-finite patches: rejected
- All-zero / near-zero patches: rejected

## Filtering Statistics

- `1997_2.5D_shots`: scanned_regions=`42`, candidates=`252`, low_energy_rejected=`192`, selected_raw=`60`
- `7m_shots_0201`: scanned_regions=`5`, candidates=`3355`, low_energy_rejected=`2573`, selected_raw=`671`
- `Anisotropic_FD_Model`: scanned_regions=`7`, candidates=`525`, low_energy_rejected=`357`, selected_raw=`150`
- `Kerry3D`: scanned_regions=`5`, candidates=`480`, low_energy_rejected=`384`, selected_raw=`96`
- `Shots0001_0200`: scanned_regions=`15`, candidates=`5865`, low_energy_rejected=`4635`, selected_raw=`1173`
