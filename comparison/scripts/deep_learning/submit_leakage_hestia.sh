#!/bin/bash
#SBATCH --job-name=leakage_hestia
#SBATCH --account=def-sgobeil
#SBATCH --array=0-12
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/leakage/hestia_%A_%a.out

# Recompute Hestia leakage (seed 1) after the ccpart_random re-split.
# 6 molecule + 1 protein + 6 dna = 13 tasks.
DTYPES=(molecule molecule molecule molecule molecule molecule protein dna dna dna dna dna dna)
DSETS=(caco2_wang pgp_broccatelli ames lipophilicity sr_are cyp2c19_veith \
       dbaasp_amp gue_prom_core_all gue_prom_300_all gue_emp_h3 gue_emp_h4 gue_mouse_0 deeppromoter)
DT=${DTYPES[$SLURM_ARRAY_TASK_ID]}
DS=${DSETS[$SLURM_ARRAY_TASK_ID]}

BASE="${RELAG_EXP_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
module load python/3.13.2 rdkit/2024.09.6 clang/18.1.8
export LIBCLANG_PATH=$EBROOTCLANG/lib
source "$HOME/venvs/refnd_exp/deep_learning/bin/activate"
cd "$BASE"
echo "=== leakage hestia: $DT / $DS ==="
python deep_learning/compute_leakage.py --dtype "$DT" --dataset "$DS" --method hestia --seed 1
