#!/bin/bash
#SBATCH --job-name=mmseqs_split
#SBATCH --account=def-sgobeil
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/splits/mmseqs_%j.out

BASE="${RELAG_EXP_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="$HOME/venvs/refnd_exp/mmseqs_split"
SPLITS="$BASE/splits"

module load python/3.13.2 mmseqs2/17-b804f
source "$VENV/bin/activate"

mkdir -p $BASE/tmp

cd "$BASE"
echo "=== MMseqs2 protein splits (10 seeds each) ==="
for ds in dbaasp_amp enzyme_topt; do
    echo "  -> $ds"
    python mmseqs_split/run_protein.py --splits-dir "$SPLITS" --dataset "$ds" --threads "$SLURM_CPUS_PER_TASK"
done

echo "MMseqs2 splits complete."
