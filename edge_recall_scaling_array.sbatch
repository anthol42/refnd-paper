#!/bin/bash
#SBATCH --job-name=edge-recall-scaling
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-5

# One (dataset, size) pair per array task -- each task runs both the fixed-ef
# and log-scaled-ef HNSW builds for that size (see edge_recall_scaling.py's
# --size docstring: the log-scale baseline is a fixed reference size, not
# something derived from other tasks, so sizes parallelize cleanly).
#
# Reuses the subset files scaling_benchmark_prepare.sbatch already builds --
# atlas's 5000/25000/125000 and belka's 25000/125000/625000 are all subsets
# of DATASET_SIZES in scaling_benchmark.py, so no separate prepare job is
# needed here. Submit with:
#
#   PREP=$(sbatch --parsable scaling_benchmark_prepare.sbatch)
#   sbatch --dependency=afterok:$PREP edge_recall_scaling_array.sbatch
#
# (or reuse the same $PREP job id from a scaling_benchmark_array.sbatch
# submission -- both arrays can depend on the same prepare job)

set -euo pipefail

cd "$(dirname "$0")"

export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Must match scaling_benchmark_prepare.sbatch -- that job populated the
# subset files this array reads with --no-prepare.
export REFND_CACHE_DIR=/home/lavertu/scratch/refnd_cache

mkdir -p logs results/edge_recall

TASKS=(
    "atlas 5000"
    "atlas 25000"
    "atlas 125000"
    "belka 25000"
    "belka 125000"
    "belka 625000"
)

read -r DATASET SIZE <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT="results/edge_recall/edge_recall_scaling_${DATASET}_${SIZE}.json"

echo "Task $SLURM_ARRAY_TASK_ID: dataset=$DATASET size=$SIZE -> $OUTPUT"

uv run python edge_recall_scaling.py \
    --dataset "$DATASET" \
    --size "$SIZE" \
    --output "$OUTPUT" \
    --no-prepare
