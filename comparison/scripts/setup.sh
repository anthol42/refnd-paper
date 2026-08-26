#!/bin/bash
# Run once on login node after rsyncing refnd_exp/ to $BASE/
set -euo pipefail

BASE="${REFND_EXP_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VENV_BASE="$HOME/venvs/refnd_exp"

module load python/3.13.2 rdkit/2024.09.6 clang/18.1.8
export LIBCLANG_PATH=$EBROOTCLANG/lib

mkdir -p "$VENV_BASE"
# SLURM log dirs (relative #SBATCH --output paths resolve under BASE at submit time).
mkdir -p "$BASE/logs/splits" "$BASE/logs/training" "$BASE/logs/leakage"

echo "=== refnd_split ==="
uv venv "$VENV_BASE/refnd_split" --python python3.13
source "$VENV_BASE/refnd_split/bin/activate"
uv pip install refnd==0.0.3 numpy pandas pyarrow
deactivate

echo "=== mmseqs_split ==="
uv venv "$VENV_BASE/mmseqs_split" --python python3.13
source "$VENV_BASE/mmseqs_split/bin/activate"
uv pip install biopython numpy pandas pyarrow
deactivate

echo "=== hestia_split ==="
uv venv "$VENV_BASE/hestia_split" --python python3.13
source "$VENV_BASE/hestia_split/bin/activate"
uv pip install hestia-good biopython numpy pandas pyarrow networkx rdkit
deactivate

echo "=== random_split ==="
uv venv "$VENV_BASE/random_split" --python python3.13
source "$VENV_BASE/random_split/bin/activate"
uv pip install numpy scikit-learn pandas pyarrow
deactivate

echo "=== deep_learning ==="
uv venv "$VENV_BASE/deep_learning" --python python3.13
source "$VENV_BASE/deep_learning/bin/activate"
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install scikit-learn scipy numpy pandas pyarrow transformers tqdm gdown qmap-benchmark pytdc einops esm refnd==0.0.3
deactivate

echo ""
echo "All venvs ready under $VENV_BASE"
echo "Next: bash download_data.sh"
