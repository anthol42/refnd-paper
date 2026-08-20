#!/usr/bin/env bash
# Submits the shared prepare job (builds every subset file used by both
# scaling_benchmark_array.sbatch and edge_recall_scaling_array.sbatch --
# edge_recall_scaling's sizes are all subsets of scaling_benchmark's
# DATASET_SIZES, so one prepare job covers both), then both parallel arrays
# chained via --dependency=afterok so neither starts until prepare succeeds.
#
# A single sbatch job can't do this itself -- array tasks all start
# independently/in parallel, so there's no way for one to reliably "go
# first" and block the rest without a real inter-job dependency.
#
# Usage: ./submit_scaling_benchmark.sh

set -euo pipefail

cd "$(dirname "$0")"

PREP=$(sbatch --parsable scaling_benchmark_prepare.sbatch)
SCALING_ARRAY=$(sbatch --parsable --dependency=afterok:"$PREP" scaling_benchmark_array.sbatch)
RECALL_ARRAY=$(sbatch --parsable --dependency=afterok:"$PREP" edge_recall_scaling_array.sbatch)

echo "Prepare job:            $PREP"
echo "Scaling benchmark array: $SCALING_ARRAY (starts after $PREP succeeds)"
echo "Edge recall array:       $RECALL_ARRAY (starts after $PREP succeeds)"
