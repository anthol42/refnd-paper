#!/bin/bash
#SBATCH --job-name=relag_missing
#SBATCH --account=def-sgobeil
#SBATCH --output=logs/splits/relag_missing_%j.out
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

BASE="${RELAG_EXP_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPLITS="$BASE/splits"

module load python/3.13.2 rdkit/2024.09.6 clang/18.1.8
export LIBCLANG_PATH=$EBROOTCLANG/lib
source "$HOME/venvs/refnd_exp/refnd_split/bin/activate"

cd "$BASE"

echo "=== relag molecule: sr_are ==="
python relag_split/run_molecule.py --splits-dir "$SPLITS" --dataset sr_are

echo "=== relag DNA: deeppromoter ==="
python relag_split/run_dna.py --splits-dir "$SPLITS" --dataset deeppromoter

echo "relag missing splits complete."
