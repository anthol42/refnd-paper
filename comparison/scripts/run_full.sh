#!/bin/bash
# Full re-run: refnd 0.0.3 CPM splits (no subsampling) + split-cost tracking.
# Clears all splits/metrics, re-splits every method (emitting split_metrics),
# retrains all methods, recomputes leakage for all, and finalizes (figures + CSVs
# including split_metrics.csv).
set -euo pipefail
BASE="${REFND_EXP_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BASE"

# Log dirs must exist before submit: Slurm opens #SBATCH --output (now relative
# to BASE) at job start, and those paths won't be auto-created.
mkdir -p logs/splits logs/training logs/leakage

echo "Clearing splits, split_metrics, stale aggregated results..."
rm -rf splits/refnd splits/hestia splits/random splits/mmseqs2 splits/datasail
rm -rf results/split_metrics results/leakage smoke_splits
rm -f cache/mol_null_random_*.npy  # force fresh random-molecule null (cache is threshold-agnostic)

echo "Submitting split jobs..."
JID_RND=$(sbatch  --parsable random_split/submit.sh)
JID_RFND=$(sbatch --parsable refnd_split/submit.sh)
JID_MMQ=$(sbatch  --parsable mmseqs_split/submit.sh)
JID_HST=$(sbatch  --parsable hestia_split/submit.sh)
JID_DS=$(sbatch   --parsable --array=1-10 datasail_split/submit.sh)
echo "  random=$JID_RND refnd=$JID_RFND mmseqs=$JID_MMQ hestia=$JID_HST datasail=$JID_DS"

# Core methods must succeed; datasail is allowed to partially fail (ILP infeasibility)
# without blocking the whole pipeline -- only its own train/leakage tasks would skip.
CORE="afterok:${JID_RND}:${JID_RFND}:${JID_MMQ}:${JID_HST}"
DEP="${CORE},afterany:${JID_DS}"

echo "Submitting training + leakage (after splits)..."
JID_TR_PROT=$(sbatch --parsable --dependency=$DEP deep_learning/submit_protein.sh)
JID_TR_MOL=$(sbatch  --parsable --dependency=$DEP deep_learning/submit_molecule.sh)
JID_TR_DNA=$(sbatch  --parsable --dependency=$DEP deep_learning/submit_dna.sh)
JID_LEAK=$(sbatch    --parsable --dependency=$DEP deep_learning/submit_leakage_all.sh)
echo "  train: prot=$JID_TR_PROT mol=$JID_TR_MOL dna=$JID_TR_DNA  leakage=$JID_LEAK"

echo "Submitting finalize (after training + leakage)..."
JID_FIN=$(sbatch --parsable \
  --dependency=afterok:${JID_TR_PROT}:${JID_TR_MOL}:${JID_TR_DNA}:${JID_LEAK} \
  deep_learning/submit_finalize.sh)
echo "  finalize=$JID_FIN"
echo "Done. Monitor: squeue -u jacobc"
