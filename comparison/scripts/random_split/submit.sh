#!/bin/bash
#SBATCH --job-name=random_split
#SBATCH --account=def-sgobeil
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/splits/random_%j.out

BASE="${REFND_EXP_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="$HOME/venvs/refnd_exp/random_split"
SPLITS="$BASE/splits"

module load python/3.13.2
source "$VENV/bin/activate"

cd "$BASE"
python random_split/run_all.py --splits-dir "$SPLITS"
