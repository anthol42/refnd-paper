#!/bin/bash
#SBATCH --job-name=leakage_dna
#SBATCH --account=def-sgobeil
#SBATCH --array=0-17
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/leakage/dna_%A_%a.out

# 6 datasets x 3 methods = 18 tasks (exact NN test->train identity, seed 1)
DATASETS=(gue_prom_core_all gue_prom_300_all gue_emp_h3 gue_emp_h4 gue_mouse_0 deeppromoter)
METHODS=(refnd random hestia)
DS=${DATASETS[$((SLURM_ARRAY_TASK_ID / 3))]}
M=${METHODS[$((SLURM_ARRAY_TASK_ID % 3))]}

BASE="${RELAG_EXP_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
module load python/3.13.2 rdkit/2024.09.6 clang/18.1.8
export LIBCLANG_PATH=$EBROOTCLANG/lib
source "$HOME/venvs/refnd_exp/deep_learning/bin/activate"

cd "$BASE"
echo "=== leakage DNA: $DS / $M ($(nproc) cores) ==="
python deep_learning/compute_leakage.py --dtype dna --dataset "$DS" --method "$M" --seed 1
