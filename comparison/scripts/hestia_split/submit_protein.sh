#!/bin/bash
#SBATCH --job-name=hestia_prot_split
#SBATCH --account=def-sgobeil
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/splits/hestia_prot_%j.out
BASE="${REFND_EXP_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
module load python/3.13.2 rdkit/2024.09.6 mmseqs2/17-b804f
source "$HOME/venvs/refnd_exp/hestia_split/bin/activate"
cd "$BASE"
python hestia_split/run_protein.py --splits-dir "$BASE/splits"
