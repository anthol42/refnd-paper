#!/bin/bash
#SBATCH --job-name=leak_mol_ds
#SBATCH --account=def-sgobeil
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/leakage/mol_datasail_%j.out

# Recompute leakage (max test->train Tanimoto, exact NN, seed 1) for the REAL
# DataSAIL molecule splits, then regenerate all leakage figures. Molecule NN is
# cheap; the old datasail_seed1.npy (homemade splits) were deleted before submit.
BASE="${REFND_EXP_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
module load python/3.13.2 rdkit/2024.09.6 clang/18.1.8
export LIBCLANG_PATH=$EBROOTCLANG/lib
source "$HOME/venvs/refnd_exp/deep_learning/bin/activate"
cd "$BASE"
mkdir -p logs/leakage

for ds in caco2_wang pgp_broccatelli ames lipophilicity sr_are cyp2c19_veith; do
    echo "=== leakage molecule/datasail: $ds ($(date '+%H:%M:%S')) ==="
    python deep_learning/compute_leakage.py --dtype molecule --dataset "$ds" --method datasail --seed 1
done

echo "=== regenerate leakage figures ==="
python deep_learning/plot_leakage.py
echo "=== done ==="
ls -la results/figures_leakage/molecule_*.png 2>/dev/null
