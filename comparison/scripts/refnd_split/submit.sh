#!/bin/bash
#SBATCH --job-name=refnd_split
#SBATCH --account=def-sgobeil
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/splits/refnd_%j.out

BASE="${REFND_EXP_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="$HOME/venvs/refnd_exp/refnd_split"
SPLITS="$BASE/splits"

module load python/3.13.2 rdkit/2024.09.6
source "$VENV/bin/activate"

export HF_HOME="$BASE/hf_cache"

cd "$BASE"

echo "=== Refnd protein splits ==="
for ds in dbaasp_amp enzyme_topt; do
    echo "  -> $ds"
    python refnd_split/run_protein.py --splits-dir "$SPLITS" --dataset "$ds"
done

echo "=== Refnd molecule splits ==="
for ds in cyp2c19_veith caco2_wang pgp_broccatelli ames lipophilicity sr_are; do
    echo "  -> $ds"
    python refnd_split/run_molecule.py --splits-dir "$SPLITS" --dataset "$ds"
done

echo "=== Refnd DNA splits ==="
for ds in gue_prom_core_all gue_prom_300_all gue_emp_h3 gue_emp_h4 gue_mouse_0 deeppromoter; do
    echo "  -> $ds"
    python refnd_split/run_dna.py --splits-dir "$SPLITS" --dataset "$ds"
done

echo "Refnd splits complete."
