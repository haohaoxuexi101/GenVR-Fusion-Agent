#!/usr/bin/env bash
#SBATCH --job-name=gmc-paper-scale
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/gmc-paper-scale-%j.out
#SBATCH --error=logs/gmc-paper-scale-%j.err

set -euo pipefail
mkdir -p logs outputs/paper_scale

# Adjust these lines to your cluster environment.
# module load cuda/12.1
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate gmc

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export PYTHONUNBUFFERED=1

bash scripts/15_paper_scale_reproduce.sh
