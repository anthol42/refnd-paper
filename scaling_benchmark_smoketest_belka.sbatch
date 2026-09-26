#!/bin/bash
#SBATCH --job-name=scaling-smoketest-belka
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Minimal end-to-end smoke test: belka dataset, both methods, smallest
# subset only (25,000). Run this before submitting the full
# scaling_benchmark_array.sbatch gridsearch, to confirm the pipeline
# (including the psutil-based peak-RSS polling that replaced
# /usr/bin/time, which isn't available on this cluster) works.
#
# Usage:
#   sbatch scaling_benchmark_smoketest_belka.sbatch

set -euo pipefail

export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
export RELAG_CACHE_DIR=/home/lavertu/scratch/refnd_cache

mkdir -p logs results/scaling

uv run python scaling_benchmark.py \
    --dataset belka \
    --method refnd \
    --size 25000 \
    --output results/scaling/smoketest_belka_refnd.json

uv run python scaling_benchmark.py \
    --dataset belka \
    --method hestia \
    --size 25000 \
    --output results/scaling/smoketest_belka_hestia.json \
    --no-prepare
