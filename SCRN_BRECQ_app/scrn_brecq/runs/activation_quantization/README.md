# Activation Quantization Runs

This directory is reserved for SCRN-BRECQ activation quantization diagnostics,
small smoke runs, and experiment outputs.

Suggested layout:

```text
runs/activation_quantization/
  E001_diagnostics/
  E002_positive_scale/
  E003_init_lr_sweep/
  E004_sensitivity/
  E005_outlier_granularity/
  E006_reconstruction_target/
```

Only this README is intended to be tracked. Do not commit generated run outputs,
checkpoints, arrays, logs, cache files, or figures unless a future task
explicitly designates a small text summary as a source artifact.
