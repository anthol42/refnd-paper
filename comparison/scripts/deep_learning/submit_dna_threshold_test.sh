#!/bin/bash
#SBATCH --job-name=dna_thr_test
#SBATCH --account=def-sgobeil
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/dna_thr_test_%j.out

BASE="${REFND_EXP_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
module load python/3.13.2 rdkit/2024.09.6 clang/18.1.8
export LIBCLANG_PATH=$EBROOTCLANG/lib
source "$HOME/venvs/refnd_exp/refnd_split/bin/activate"
cd "$BASE"
python deep_learning/dna_threshold_test.py
