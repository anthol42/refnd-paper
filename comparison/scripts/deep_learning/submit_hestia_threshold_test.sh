#!/bin/bash
#SBATCH --job-name=hestia_thr_test
#SBATCH --account=def-sgobeil
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/hestia_thr_test_%j.out

BASE="${RELAG_EXP_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BASE"

echo "########## Hestia partitions per threshold (hestia venv) ##########"
( module load python/3.13.2 rdkit/2024.09.6 mmseqs2/17-b804f
  source "$HOME/venvs/refnd_exp/hestia_split/bin/activate"
  python deep_learning/hestia_threshold_split.py )

echo "########## test->train identity via relag exact NN (relag venv) ##########"
( module load python/3.13.2 rdkit/2024.09.6 clang/18.1.8
  export LIBCLANG_PATH=$EBROOTCLANG/lib
  source "$HOME/venvs/refnd_exp/refnd_split/bin/activate"
  python deep_learning/dna_threshold_leakage.py )
