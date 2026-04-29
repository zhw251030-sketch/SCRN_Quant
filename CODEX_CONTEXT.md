# Codex Shared Context for SCRN_BRECQ

Last updated: 2026-04-29

This file is the fixed handoff context for different Codex/OpenAI accounts working on the same server user in this repository. Read this file first when starting a new Codex conversation, then inspect the current Git state because this document may lag behind the latest commits.

## Project

- Repository root: `/home/data1/hanwen/project/Project/SCRN_Quant`
- Goal: migrate and validate the BRECQ post-training quantization algorithm on the SCRN seismic reconstruction model.
- Source reference repositories:
  - `SCRN-main/`: original SCRN source and local data artifacts.
  - `BRECQ-main/`: original BRECQ source.
- Integration code:
  - `SCRN_BRECQ_app/scrn_repro/`: SCRN reproduction code.
  - `SCRN_BRECQ_app/scrn_brecq/`: BRECQ migration and SCRN quantization code.
- Do not modify `SCRN-main/` or `BRECQ-main/` unless explicitly requested.

## Workflow Rules

- At the start of each task run:
  - `git status`
  - `git branch --show-current`
  - `git rev-parse --show-toplevel`
- Prefer changes under `SCRN_BRECQ_app/`.
- When modifying files under `SCRN_BRECQ_app/scrn_brecq/`, update `SCRN_BRECQ_app/scrn_brecq/DEVELOPMENT_LOG.md`.
- Do not use `git add .`.
- Do not commit datasets, weights, logs, caches, `.npy`, `.segy`, `.pth`, `.pt`, `.ckpt`, `__pycache__`, or `.ipynb_checkpoints`.
- Before each commit show:
  - `git status`
  - `git diff`
  - `git diff --staged`
- Run a minimal check before committing, such as `py_compile`, `json.tool`, a CLI `--help`, or a small smoke run.
- Do not run `git push` unless explicitly requested.
- Avoid destructive commands such as `git reset --hard`, `git clean -fd`, `rm -rf` on data directories, or force push.

## Current Code Capabilities

- SCRN reproduction:
  - Model implementation in `SCRN_BRECQ_app/scrn_repro/model/`.
  - Clean patch dataset and online degradation in `SCRN_BRECQ_app/scrn_repro/data/`.
  - Training and single-sample test CLIs in `SCRN_BRECQ_app/scrn_repro/cli/`.
- BRECQ migration:
  - SCRN loader in `SCRN_BRECQ_app/scrn_brecq/model/scrn_loader.py`.
  - Calibration loader uses `SCRNPatchDataset` degraded inputs.
  - Quantization modules, AdaRound, BN folding, QuantModel, and reconstruction are under `SCRN_BRECQ_app/scrn_brecq/quant/`.
  - Main quantization CLI: `SCRN_BRECQ_app.scrn_brecq.cli.quantize_scrn`.
  - Quantized checkpoint evaluation CLI: `SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn`.
  - Quantization truth verification CLI: `SCRN_BRECQ_app.scrn_brecq.cli.verify_quantized_scrn`.
  - Multi-sample generalization evaluation CLI: `SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_multi`.
- Distributed BRECQ:
  - `quantize_scrn.py` supports torchrun multi-GPU W-only reconstruction.
  - Distributed activation quantization is intentionally unsupported for now.

## Important Runs

- Single-GPU W4A32 baseline:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260427_152554_w4_recon_1024samples_20000iters`
  - post-recon SNR `11.5909`, SSIM `0.8385`.
- Four-GPU W4A32 global batch 64:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260427_165123_w4_recon_1024samples_20000iters_dist4_compare`
  - post-recon SNR `11.6952`, SSIM `0.8608`.
- Four-GPU W4A32 global batch 128:
  - `SCRN_BRECQ_app/scrn_brecq/runs/quant/20260427_192819_w4_recon_1024samples_20000iters_dist4_bsz32_global128`
  - post-recon SNR `11.7469`, SSIM `0.8675`.
- Multi-sample generalization evaluation on 128 `scrn_quant_10750_0_patches` samples:
  - `SCRN_BRECQ_app/scrn_brecq/runs/generalization_eval/20260427_222925_global128_quant10750_eval128`
  - FP32 mean SNR `6.0901`, quant mean SNR `4.8802`, mean SNR gap `-1.2099 dB`.
  - FP32 mean SSIM `0.7562`, quant mean SSIM `0.7161`, mean SSIM gap `-0.0401`.

## Model Size Interpretation

- Current `quantized_scrn_brecq.pth` is a recoverable PyTorch checkpoint, not a bit-packed deployment file.
- Actual quant checkpoint size is about `5.07 MiB`, close to the FP32 checkpoint.
- Theoretical packed model size is about `0.2427 MiB`.
- Estimated model compression ratio is about `6.77x`.
- Estimated quantized-weight compression ratio is about `7.98x`.
- Typical W4A32 bit distribution is `{"4": 50, "8": 2}`.

## Useful Commands

Run W4A32 four-GPU reconstruction:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n quant torchrun --standalone --nproc_per_node=4 \
  -m SCRN_BRECQ_app.scrn_brecq.cli.quantize_scrn \
  --distributed \
  --num-samples 1024 \
  --batch-size 32 \
  --iters-w 20000 \
  --run-name w4_recon_1024samples_20000iters_dist4_bsz32_global128 \
  --device cuda
```

Verify a quantized checkpoint:

```bash
conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.verify_quantized_scrn \
  --checkpoint <run>/checkpoints/quantized_scrn_brecq.pth \
  --output-json <run>/verification.json \
  --device cpu
```

Run multi-sample generalization evaluation:

```bash
conda run -n quant python -m SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_multi \
  --checkpoint <run>/checkpoints/quantized_scrn_brecq.pth \
  --num-eval-samples 128 \
  --batch-size 16 \
  --device auto
```

## Handoff Notes

- Latest notable commits at the time this file was created:
  - `be85cbc Add multi-sample quantized SCRN evaluation`
  - `b938329 Add quantized model size metrics`
  - `ced6c09 Add quantization runtime metrics`
  - `34a9ad1 Add distributed BRECQ reconstruction support`
- The local branch may be ahead of `origin/main`; do not push unless requested.
- For future cross-account handoff, update this file with:
  - new commit hashes,
  - important run directories,
  - key metrics,
  - unresolved questions,
  - next recommended task.
