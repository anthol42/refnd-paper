#!/bin/bash
#SBATCH --job-name=leakage_prot
#SBATCH --account=def-sgobeil
#SBATCH --array=0-4
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/leakage/prot_%A_%a.out

# 1 dataset (dbaasp_amp) x 5 methods = 5 tasks. Exact global-alignment
# test->train identity, seed 1.
METHODS=(refnd random mmseqs2 hestia datasail)
M=${METHODS[$SLURM_ARRAY_TASK_ID]}

BASE="${REFND_EXP_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
module load python/3.13.2 rdkit/2024.09.6 clang/18.1.8
export LIBCLANG_PATH=$EBROOTCLANG/lib
source "$HOME/venvs/refnd_exp/deep_learning/bin/activate"

cd "$BASE"
echo "=== leakage protein: dbaasp_amp / $M ($(nproc) cores) ==="
python deep_learning/compute_leakage.py --dtype protein --dataset dbaasp_amp --method "$M" --seed 1
