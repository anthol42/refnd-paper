#!/bin/bash
#SBATCH --job-name=relag_dna
#SBATCH --account=def-sgobeil
#SBATCH --output=logs/splits/relag_dna_%j.out
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

BASE="${RELAG_EXP_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPLITS="$BASE/splits"

module load python/3.13.2 rdkit/2024.09.6 clang/18.1.8
export LIBCLANG_PATH=$EBROOTCLANG/lib
source "$HOME/venvs/refnd_exp/refnd_split/bin/activate"

cd "$BASE"

echo "=== relag DNA splits (threshold=0.37) ==="
for ds in gue_prom_core_all gue_prom_300_all gue_emp_h3 gue_emp_h4 gue_mouse_0 deeppromoter; do
    echo "  -> $ds"
    python relag_split/run_dna.py --splits-dir "$SPLITS" --dataset "$ds"
done

echo "relag DNA splits complete."
