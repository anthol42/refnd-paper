#!/bin/bash
#SBATCH --job-name=hestia_dna
#SBATCH --account=def-sgobeil
#SBATCH --output=logs/splits/hestia_dna_%j.out
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

BASE="${REFND_EXP_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPLITS="$BASE/splits"

module load python/3.13.2 rdkit/2024.09.6 mmseqs2/17-b804f
source "$HOME/venvs/refnd_exp/hestia_split/bin/activate"

cd "$BASE"

echo "=== Hestia DNA splits (threshold=0.95) ==="
for ds in gue_prom_core_all gue_prom_300_all gue_emp_h3 gue_emp_h4 gue_mouse_0; do
    echo "  -> $ds"
    python hestia_split/run_dna.py --splits-dir "$SPLITS" --dataset "$ds"
done

echo "Hestia DNA splits complete."
