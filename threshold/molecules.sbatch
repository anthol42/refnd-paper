#!/bin/bash
#SBATCH --job-name=thresholdv2-molecules
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00

# BELKA threshold sweep (thresholdv2/molecules.py): full training set (train)
# vs. the non-triazine subset of the real Kaggle test set (production).
#
# Every stage (SMILES extraction, fingerprinting, HNSW build+index cache,
# layer-0 edge weights, null model) is cached under $REFND_CACHE_DIR, so a
# requeued/restarted job resumes instead of recomputing from scratch.
#
# Usage:
#   sbatch thresholdv2/molecules.sbatch

set -euo pipefail

export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
export REFND_CACHE_DIR=/home/lavertu/scratch/refnd_cache

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

uv run python -m thresholdv2.molecules
