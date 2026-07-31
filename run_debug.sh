#!/usr/bin/env bash
# Fast smoke test: same scripts as run_ablation.sh's scaling/edge-recall/
# in-distribution block, but with --debug so it finishes quickly. Run this
# before run_ablation.sh to make sure everything works end to end.
#
# Every command's full stdout+stderr is teed live to logs/debug_<timestamp>/<label>.log.
#
# scaling_benchmark.py shells out to runtime_scripts.split_refnd/split_hestia
# and swallows their failures itself (records a "status" field in
# results/runtime_<dataset>.json instead of raising), so it always exits 0
# even when a method failed. run_scaling() below detects that case by diffing
# the results file before/after and checking for any non-"ok" status among
# the newly appended records.
#
# Exits 0 only if every command succeeded and every scaling_benchmark method
# reported "ok"; otherwise prints a summary of failures (with log paths) and
# exits 1.

set -u

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

LOG_DIR="logs/debug_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

FAILURES=()

HAVE_JQ=true
command -v jq >/dev/null 2>&1 || HAVE_JQ=false
if ! $HAVE_JQ; then
    echo -e "${RED}[warn] jq not found — skipping the scaling_benchmark.py subprocess-status check${NC}" >&2
fi

# Run a labeled command; tee full stdout+stderr to a log file, keep going on failure.
run() {
    local label="$1"; shift
    local log="$LOG_DIR/${label}.log"
    echo -e "\n${GREEN}>> ${label}${NC}"
    echo "\$ $*" > "$log"
    if "$@" > >(tee -a "$log") 2> >(tee -a "$log" >&2); then
        echo -e "${GREEN}OK: ${label}${NC}"
    else
        echo -e "${RED}FAILED: ${label} (log: ${log})${NC}" >&2
        FAILURES+=("${label} -> ${log}")
    fi
}

# Like run(), but for scaling_benchmark.py specifically: also diffs
# results/runtime_<dataset>.json before/after the call and flags any newly
# appended record whose "status" isn't "ok" (scaling_benchmark.py itself
# exits 0 in that case, so run()'s own exit-code check won't catch it).
run_scaling() {
    local dataset="$1" label="$2"; shift 2
    local results="results/runtime_${dataset}.json"
    local before=0
    if $HAVE_JQ && [ -f "$results" ]; then
        before=$(jq 'length' "$results" 2>/dev/null || echo 0)
    fi

    run "$label" "$@"

    if $HAVE_JQ && [ -f "$results" ]; then
        local bad
        bad=$(jq -c ".[${before}:] | .[] | select(.status != \"ok\")" "$results" 2>/dev/null)
        if [ -n "$bad" ]; then
            echo -e "${RED}FAILED: ${label} — new run(s) in ${results} did not report \"ok\":${NC}" >&2
            echo "$bad" | tee -a "$LOG_DIR/${label}.log" >&2
            FAILURES+=("${label} (subprocess status) -> ${results}")
        fi
    fi
}

# Last untested hyperparameter modality
run hyperparameters_prom_core_all uv run python hyperparameters.py --dataset prom_core_all --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm

# Scaling experiment (runtime/memory vs. dataset size, atlas + belka)
run_scaling atlas scaling_atlas uv run python scaling_benchmark.py --dataset atlas --debug
run_scaling belka scaling_belka uv run python scaling_benchmark.py --dataset belka --debug

# Per-stage timing breakdown of the refnd pipeline
run debug_scaling_refnd uv run python debug_scaling_refnd.py --debug

# Effect of dataset size (and ef_construction) on HNSW edge recall
run edge_recall_atlas uv run python edge_recall_scaling.py --dataset atlas --debug
run edge_recall_belka uv run python edge_recall_scaling.py --dataset belka --sizes 100000,500000

# In-distribution test (refnd + hestia split methods) for every dataset with an encoder
run in_distribution_dbaasp        uv run python in_distribution_test.py --dataset dbaasp        --method both --debug
run in_distribution_ld50_zhu      uv run python in_distribution_test.py --dataset ld50_zhu      --method both --debug
run in_distribution_prom_core_all uv run python in_distribution_test.py --dataset prom_core_all --method both --debug

echo
echo "==================== Summary ===================="
if [ ${#FAILURES[@]} -eq 0 ]; then
    echo -e "${GREEN}All commands succeeded. Logs: ${LOG_DIR}${NC}"
    exit 0
else
    echo -e "${RED}${#FAILURES[@]} failure(s):${NC}"
    for f in "${FAILURES[@]}"; do
        echo "  - $f"
    done
    echo "Logs: ${LOG_DIR}"
    exit 1
fi
