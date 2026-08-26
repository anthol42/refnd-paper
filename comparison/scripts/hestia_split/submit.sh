#!/bin/bash
#SBATCH --job-name=hestia_split
#SBATCH --account=def-sgobeil
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/splits/hestia_%j.out

BASE="${REFND_EXP_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="$HOME/venvs/refnd_exp/hestia_split"
SPLITS="$BASE/splits"

module load python/3.13.2 rdkit/2024.09.6 mmseqs2/17-b804f
source "$VENV/bin/activate"

cd "$BASE"

echo "=== Hestia protein splits ==="
for ds in dbaasp_amp enzyme_topt; do
    echo "  -> $ds"
    python hestia_split/run_protein.py --splits-dir "$SPLITS" --dataset "$ds"
done

echo "=== Hestia molecule splits ==="
for ds in cyp2c19_veith caco2_wang pgp_broccatelli ames lipophilicity sr_are; do
    echo "  -> $ds"
    python hestia_split/run_molecule.py --splits-dir "$SPLITS" --dataset "$ds"
done

echo "=== Hestia DNA splits ==="
for ds in gue_prom_core_all gue_prom_300_all gue_emp_h3 gue_emp_h4 gue_mouse_0 deeppromoter; do
    echo "  -> $ds"
    python hestia_split/run_dna.py --splits-dir "$SPLITS" --dataset "$ds"
done

echo "Hestia splits complete."
