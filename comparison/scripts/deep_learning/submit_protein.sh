#!/bin/bash
#SBATCH --job-name=train_protein
#SBATCH --account=def-sgobeil
#SBATCH --array=0-99
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/training/protein_%A_%a.out

# Array layout: 2 datasets × 5 methods × 10 seeds = 100 tasks
DATASETS=(dbaasp_amp enzyme_topt)
METHODS=(refnd random mmseqs2 hestia datasail)
DATASET=${DATASETS[$((SLURM_ARRAY_TASK_ID / 50))]}
SUB=$((SLURM_ARRAY_TASK_ID % 50))
METHOD=${METHODS[$((SUB / 10))]}
SEED=$(( (SUB % 10) + 1 ))

BASE="${REFND_EXP_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="$HOME/venvs/refnd_exp/deep_learning"

module load python/3.13.2
source "$VENV/bin/activate"

export HF_HOME="$BASE/hf_cache"
export TRANSFORMERS_OFFLINE=1

cd "$BASE"
echo "Training protein: dataset=$DATASET method=$METHOD seed=$SEED"
python deep_learning/train_protein.py --dataset "$DATASET" --method "$METHOD" --seed "$SEED"
