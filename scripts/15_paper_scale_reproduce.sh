#!/usr/bin/env bash
set -euo pipefail

# Paper-faithful GMC reproduction pipeline for Farmer (2026), Chapters 2--3.
# Run from repository root:  bash scripts/15_paper_scale_reproduce.sh
# Requires a CUDA GPU for practical wall time.

mkdir -p outputs/paper_scale

# 1. Train boundary and internal CFM models at the paper configuration:
#    9e5 train + 1e5 validation, batch 1024, AdamW, max_lr=4e-4, 100 epochs.
python scripts/02_train_boundary_cfm.py \
  --generate-if-missing \
  --paper-preset \
  --seed 1234 \
  --out outputs/paper_scale/boundary_cfm_paper.pt

python scripts/09_train_internal_cfm.py \
  --generate-if-missing \
  --paper-preset \
  --seed 4321 \
  --out outputs/paper_scale/internal_cfm_paper.pt

# 2. Reproduce Chapter 2 single-cell validation figures at the same 4x4 optical grid.
python scripts/03_validate_boundary_figures.py \
  --ckpt outputs/paper_scale/boundary_cfm_paper.pt \
  --n 20000 \
  --outdir outputs/paper_scale/fig2_boundary_validation

python scripts/10_validate_internal_figures.py \
  --ckpt outputs/paper_scale/internal_cfm_paper.pt \
  --n 20000 \
  --outdir outputs/paper_scale/fig2_internal_validation

# 3. Reproduce Figure 3.1-level cloud maps: 112x112, 10^6 histories.
python scripts/13_run_article_like_cloud_maps.py \
  --boundary-ckpt outputs/paper_scale/boundary_cfm_paper.pt \
  --internal-ckpt outputs/paper_scale/internal_cfm_paper.pt \
  --n 1000000 \
  --outdir outputs/paper_scale/fig31_cloudmaps_n1e6 \
  --device cuda \
  --max-steps 200000 \
  --inference-batch-size 65536

# 4. Reproduce Figure 3.2-style runtime scaling and convergence diagnostics.
python scripts/17_run_runtime_scaling.py \
  --boundary-ckpt outputs/paper_scale/boundary_cfm_paper.pt \
  --outdir outputs/paper_scale/fig32_runtime_scaling \
  --device cuda

python scripts/18_run_convergence_study.py \
  --boundary-ckpt outputs/paper_scale/boundary_cfm_paper.pt \
  --internal-ckpt outputs/paper_scale/internal_cfm_paper.pt \
  --outdir outputs/paper_scale/fig32_convergence \
  --device cuda

# 5. Acceptance report.
python scripts/19_acceptance_report.py \
  --root outputs/paper_scale \
  --out outputs/paper_scale/ACCEPTANCE_REPORT.md
