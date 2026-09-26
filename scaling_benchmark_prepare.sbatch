#!/bin/bash
#SBATCH --job-name=scaling-benchmark-prepare
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Builds every subset file ($RELAG_CACHE_DIR/scaling_tmp/{dataset}_{size}.fasta)
# for atlas and belka up front, once, sequentially -- so the parallel array in
# scaling_benchmark_array.sbatch can run with --no-prepare and never race on
# writing the same subset file.
#
# Usage:
#   PREP=$(sbatch --parsable scaling_benchmark_prepare.sbatch)
#   sbatch --dependency=afterok:$PREP scaling_benchmark_array.sbatch

set -euo pipefail

cd "$(dirname "$0")"

# .cache under $HOME won't hold data of this size on the cluster -- point it
# at scratch. Must match the value exported in scaling_benchmark_array.sbatch.
export RELAG_CACHE_DIR=/home/lavertu/scratch/refnd_cache

mkdir -p logs "$RELAG_CACHE_DIR"

uv run python scaling_benchmark.py --dataset atlas --prepare-only
uv run python scaling_benchmark.py --dataset belka --prepare-only
