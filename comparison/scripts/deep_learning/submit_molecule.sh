#!/bin/bash
#SBATCH --job-name=train_molecule
#SBATCH --account=def-sgobeil
#SBATCH --array=0-239
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/training/molecule_%A_%a.out

# Array layout: 4 methods × 6 datasets × 10 seeds = 240 tasks
METHODS=(refnd random hestia datasail)
DATASETS=(cyp2c19_veith caco2_wang pgp_broccatelli ames lipophilicity sr_are)

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
echo "Training molecule: method=$METHOD dataset=$DATASET seed=$SEED"
python deep_learning/train_molecule.py --method "$METHOD" --dataset "$DATASET" --seed "$SEED"
