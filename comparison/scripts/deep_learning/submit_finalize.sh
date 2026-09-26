#!/bin/bash
#SBATCH --job-name=finalize
#SBATCH --account=def-sgobeil
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/finalize_%j.out

# Aggregate per-seed results -> summaries + Wilcoxon table, then regenerate the
# overfitting and leakage figures (now including protein/peptide).
BASE="${RELAG_EXP_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
module load python/3.13.2
source "$HOME/venvs/refnd_exp/deep_learning/bin/activate"
cd "$BASE"

python deep_learning/aggregate_results.py
python deep_learning/aggregate_split_metrics.py
python deep_learning/plot_results.py
python deep_learning/plot_leakage.py
echo "=== finalize done ==="
