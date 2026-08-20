#!/bin/bash
#SBATCH --job-name=uniref-benchmark-hnsw
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00

# Vars:
INPATH=/home/lavertu/scratch/refnd_data
OUTPATH=/home/lavertu/scratch/refnd_output/uniref

# Builds the ProtSpaM HNSW index over UniRef50, extends it with the
# ProteinGym/PEER/CASP15 benchmarks, and exports the layer-0 weighted edge
# graph -- see threshold/build_uniref_benchmark_index.py's module docstring
# for the full pipeline description.
#
# 128 CPUs / 120GB is a safety margin over the estimated ~90GB (n=1 pattern,
# weight=3, dont_care=20, length-capped at 2048) -- see the memory analysis in
# the session that produced this script. Adjust --mem down if you rerun that
# estimate for different ProtSpaM parameters.
#
# Resumable: every stage is skipped if its output file already exists, so if
# the job times out, gets preempted, or is requeued, just `sbatch` this same
# script again -- it picks up at the first incomplete stage instead of
# recomputing everything from scratch.
#
# Usage: sbatch threshold/build_uniref_benchmark_index.sbatch

set -euo pipefail

# Explicit, not just relying on Rayon's cgroup autodetection -- matches
# --cpus-per-task above exactly.
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK

# $0 is this script *file*, not a directory -- `cd $0/..` fails outright
# ("Not a directory") since you can't descend into a file to go `..` from
# it, which under `set -e` would kill the job before it does anything.
# dirname first, then go up one level from the script's own directory
# (threshold/) to the repo root -- works regardless of the submission cwd
# or whether $0 ends up relative or absolute.
cd "$(dirname "$0")/.."

mkdir -p "$OUTPATH"

# refnd's HNSW build/export progress bars are indicatif, which only draws
# when stderr is a real tty (console::Term::is_term(), a genuine isatty()
# check -- no env var can fake this). SLURM redirects --output/--error to a
# log file, so without a pty indicatif silently disables itself entirely
# (no output, not garbled output). `script` allocates a pty for the child so
# indicatif thinks it's interactive and draws normally (\r + ANSI codes);
# `-e` propagates the child's real exit code so `set -e` above still works.
# Note: the progress bars steady-tick at 10Hz, so over a multi-day run this
# log file will grow continuously (roughly proportional to elapsed time) --
# expected, not a leak.
CMD=(uv run python -m threshold.build_uniref_benchmark_index
    --uniref50-fasta "$INPATH/uniref50.fasta.gz"
    --benchmark-cache-dir "$INPATH/protein_benchmarks"
    --uniref-sequences-out "$OUTPATH/uniref50.swseq"
    --benchmark-sequences-out "$OUTPATH/benchmarks.swseq"
    --offsets-out "$OUTPATH/offsets.txt"
    --patterns-out "$OUTPATH/patterns.bin"
    --uniref-index-out "$OUTPATH/uniref50.hnsw"
    --extended-index-out "$OUTPATH/unirefxbenchmark.hnsw"
    --edgestore-out "$OUTPATH/unirefxbenchmark_layer0.edgestr"
)

# script -c runs the reconstructed command through $SHELL, and the
# printf %q quoting above is bash-specific -- pin it explicitly rather than
# relying on the job environment happening to have $SHELL=/bin/bash.
SHELL=/bin/bash script -qec "$(printf '%q ' "${CMD[@]}")" /dev/null
