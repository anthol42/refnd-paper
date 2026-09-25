#!/bin/bash
# Build the DataSAIL Apptainer image from datasail.def into $BASE/datasail.sif,
# where submit.sh and synthetic/datasail.py look for it. Needs internet; on the
# cluster run it on a login node after `module load apptainer/1.4.5`.
#
#   build_sif.sh [--relock] [OUT.sif]
#     default   install the exact packages pinned in datasail.lock
#     --relock  re-resolve the specs in datasail.def and rewrite datasail.lock
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${REFND_EXP_BASE:-$(cd "$HERE/.." && pwd)}"
RELOCK=0
if [ "${1:-}" = "--relock" ]; then RELOCK=1; shift; fi
SIF="${1:-$BASE/datasail.sif}"

RUNTIME=$(command -v apptainer || command -v singularity) || {
    echo "Neither apptainer nor singularity is on PATH." >&2; exit 1; }

# %files paths in the def are relative to the build's cwd.
cd "$HERE"
# Unprivileged build (fakeroot) works on Alliance clusters and most workstations.
"$RUNTIME" build --fakeroot --force --build-arg relock=$RELOCK "$SIF" datasail.def
"$RUNTIME" test "$SIF"
if [ "$RELOCK" = 1 ]; then
    "$RUNTIME" exec "$SIF" micromamba env export -n base --explicit > datasail.lock
    echo "Rewrote $HERE/datasail.lock -- commit it."
fi
echo "Built $SIF"
