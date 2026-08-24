#!/bin/bash
#SBATCH --job-name=scaling-benchmark
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-01:00:00
#SBATCH --array=0-19

# One (dataset, method, size) triple per array task, run in parallel.
# Each task writes its own results file, so there's no shared-file race
# between concurrently-running tasks (scaling_benchmark.py --output).
#
# Requires scaling_benchmark_prepare.sbatch to have already built every
# subset file -- this script always passes --no-prepare, so a task will
# crash loudly (FileNotFoundError) instead of racing to write a subset
# concurrently with another task, if prepare wasn't run first.
#
# TASKS below must stay in sync with DATASET_SIZES/METHODS in
# scaling_benchmark.py: atlas/belka x [5 sizes] x [refnd, hestia] = 20 tasks.
#
# --cpus-per-task / --mem are placeholders -- adjust to match what
# split_refnd.py / split_hestia.py actually need at the largest sizes
# (3.125M / 15.625M); hestia in particular may need more of both.
#
# Usage:
#   PREP=$(sbatch --parsable scaling_benchmark_prepare.sbatch)
#   sbatch --dependency=afterok:$PREP scaling_benchmark_array.sbatch

set -euo pipefail

module load mmseqs2/17-b804f

export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Must match scaling_benchmark_prepare.sbatch -- that job populated the
# subset files this array reads with --no-prepare.
export REFND_CACHE_DIR=/home/lavertu/scratch/refnd_cache

mkdir -p logs results/scaling

TASKS=(
    "atlas refnd  5000"
    "atlas refnd  25000"
    "atlas refnd  125000"
    "atlas refnd  625000"
    "atlas refnd  3125000"
    "atlas hestia 5000"
    "atlas hestia 25000"
    "atlas hestia 125000"
    "atlas hestia 625000"
    "atlas hestia 3125000"
    "belka refnd  25000"
    "belka refnd  125000"
    "belka refnd  625000"
    "belka refnd  3125000"
    "belka refnd  15625000"
    "belka hestia 25000"
    "belka hestia 125000"
    "belka hestia 625000"
    "belka hestia 3125000"
    "belka hestia 15625000"
)

read -r DATASET METHOD SIZE <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT="results/scaling/runtime_${DATASET}_${METHOD}_${SIZE}.json"

echo "Task $SLURM_ARRAY_TASK_ID: dataset=$DATASET method=$METHOD size=$SIZE -> $OUTPUT"

uv run python scaling_benchmark.py \
    --dataset "$DATASET" \
    --method "$METHOD" \
    --size "$SIZE" \
    --output "$OUTPUT" \
    --no-prepare
