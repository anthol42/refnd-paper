#!/bin/bash
#SBATCH --job-name=train_dna
#SBATCH --account=def-sgobeil
#SBATCH --array=0-179
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/training/dna_%A_%a.out

# Array layout: 3 methods x 6 datasets x 10 seeds = 180 tasks
# (datasail DNA excluded for now -- exact/real-DataSAIL work is pinned)
METHODS=(refnd random hestia)
DATASETS=(gue_prom_core_all gue_prom_300_all gue_emp_h3 gue_emp_h4 gue_mouse_0 deeppromoter)

METHOD=${METHODS[$((SLURM_ARRAY_TASK_ID / 60))]}
DS_SEED=$((SLURM_ARRAY_TASK_ID % 60))
DATASET=${DATASETS[$((DS_SEED / 10))]}
SEED=$(( (DS_SEED % 10) + 1 ))

BASE="${RELAG_EXP_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="$HOME/venvs/refnd_exp/deep_learning"

module load python/3.13.2
source "$VENV/bin/activate"

export HF_HOME="$BASE/hf_cache"
export TRANSFORMERS_OFFLINE=1

cd "$BASE"
echo "Training DNA: method=$METHOD dataset=$DATASET seed=$SEED"
python deep_learning/train_dna.py --method "$METHOD" --dataset "$DATASET" --seed "$SEED"
