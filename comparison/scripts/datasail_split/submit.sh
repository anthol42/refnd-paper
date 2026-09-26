#!/bin/bash
#SBATCH --job-name=datasail_split
#SBATCH --account=def-sgobeil
#SBATCH --array=1-10
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/splits/datasail_%A_%a.out

# REAL DataSAIL splits (library, in Apptainer) for the data types where it works:
# molecules (ecfp) and peptides (mmseqs, e_clusters=200). DNA is intentionally
# excluded here -- native cd-hit-est yields an unsplittable distribution; that
# needs the precomputed-matrix path (separate, still in progress).
# One array task per seed. Each (seed,dataset) gets its own working dir so the
# clustering tools never collide.
SEED=$SLURM_ARRAY_TASK_ID
BASE="${RELAG_EXP_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SIF="$BASE/datasail.sif"
RUN="$BASE/datasail_split/run_datasail.py"
module load apptainer/1.4.5

run_one() {  # dtype dataset
    local dtype="$1" ds="$2"
    local wd="$BASE/tmp/datasail_${SLURM_ARRAY_JOB_ID}_${SEED}_${ds}"
    mkdir -p "$wd"
    apptainer exec --bind $BASE --pwd "$wd" "$SIF" \
        python "$RUN" --dtype "$dtype" --dataset "$ds" --seed "$SEED" --threads "$SLURM_CPUS_PER_TASK"
    rm -rf "$wd"
}

echo "=== DataSAIL molecule splits [seed=$SEED] ==="
for ds in caco2_wang pgp_broccatelli ames lipophilicity sr_are cyp2c19_veith; do
    echo "  -> $ds"; run_one molecule "$ds"
done

echo "=== DataSAIL peptide split [seed=$SEED] ==="
run_one protein dbaasp_amp
run_one protein enzyme_topt

echo "DataSAIL seed=$SEED complete."
