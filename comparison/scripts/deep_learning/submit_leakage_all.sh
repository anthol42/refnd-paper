#!/bin/bash
#SBATCH --job-name=leakage_all
#SBATCH --account=def-sgobeil
#SBATCH --array=0-51
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/leakage/all_%A_%a.out

# Full leakage sweep (max test->train identity, exact NN, seed 1) over every
# (method, dtype, dataset) the comparison includes. 24 molecule + 10 protein +
# 18 dna = 52 tasks. DataSAIL has no DNA (cd-hit-est 80% floor -> infeasible).
DT=(); M=(); DS=()
add(){ DT+=("$1"); M+=("$2"); DS+=("$3"); }
for d in cyp2c19_veith caco2_wang pgp_broccatelli ames lipophilicity sr_are; do
  for m in refnd random hestia datasail; do add molecule "$m" "$d"; done
done
for d in dbaasp_amp enzyme_topt; do
  for m in refnd random mmseqs2 hestia datasail; do add protein "$m" "$d"; done
done
for d in gue_prom_core_all gue_prom_300_all gue_emp_h3 gue_emp_h4 gue_mouse_0 deeppromoter; do
  for m in refnd random hestia; do add dna "$m" "$d"; done
done

i=$SLURM_ARRAY_TASK_ID
BASE="${RELAG_EXP_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
module load python/3.13.2 rdkit/2024.09.6 clang/18.1.8
export LIBCLANG_PATH=$EBROOTCLANG/lib
source "$HOME/venvs/refnd_exp/deep_learning/bin/activate"

cd "$BASE"
echo "=== leakage: ${DT[$i]} / ${DS[$i]} / ${M[$i]} ($(nproc) cores) ==="
python deep_learning/compute_leakage.py --dtype "${DT[$i]}" --dataset "${DS[$i]}" --method "${M[$i]}" --seed 1
