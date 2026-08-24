#!/bin/bash
#SBATCH --job-name=scaling-smoketest-atlas
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Minimal end-to-end smoke test: atlas dataset, both methods, smallest
# subset only (5,000). Run this before submitting the full
# scaling_benchmark_array.sbatch gridsearch, to confirm the pipeline
# (including the psutil-based peak-RSS polling that replaced
# /usr/bin/time, which isn't available on this cluster) works.
#
# Usage:
#   sbatch scaling_benchmark_smoketest_atlas.sbatch

set -euo pipefail

export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
export REFND_CACHE_DIR=/home/lavertu/scratch/refnd_cache

mkdir -p logs results/scaling

uv run python scaling_benchmark.py \
    --dataset atlas \
    --method refnd \
    --size 5000 \
    --output results/scaling/smoketest_atlas_refnd.json

uv run python scaling_benchmark.py \
    --dataset atlas \
    --method hestia \
    --size 5000 \
    --output results/scaling/smoketest_atlas_hestia.json \
    --no-prepare
