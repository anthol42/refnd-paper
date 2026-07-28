#!/usr/bin/env bash

RED='\033[0;31m'
NC='\033[0m'

# Run a command; on failure, print it in red instead of aborting the sweep.
run() {
    if ! "$@"; then
        echo -e "${RED}FAILED: $*${NC}" >&2
    fi
}


# DBAASP
# Baseline
run uv run python hyperparameters.py --dataset dbaasp        --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm
# ef-construction
run uv run python hyperparameters.py --dataset dbaasp        --ef-construction 4 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset dbaasp        --ef-construction 8 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset dbaasp        --ef-construction 16 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset dbaasp        --ef-construction 32 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset dbaasp        --ef-construction 128 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset dbaasp        --ef-construction 256 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm
# ef-init
run uv run python hyperparameters.py --dataset dbaasp        --ef-construction 64 --ef-init 2 --keep-pruned-connections --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset dbaasp        --ef-construction 64 --ef-init 4 --keep-pruned-connections --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset dbaasp        --ef-construction 64 --ef-init 8 --keep-pruned-connections --use-heuristic --leiden-objective cpm
# Turn off options
run uv run python hyperparameters.py --dataset dbaasp        --ef-construction 64 --ef-init 1 --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset dbaasp        --ef-construction 64 --ef-init 1 --keep-pruned-connections --leiden-objective cpm
# Turn on options
run uv run python hyperparameters.py --dataset dbaasp        --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --extend-candidates --leiden-objective cpm
run uv run python hyperparameters.py --dataset dbaasp        --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --strict-ef --leiden-objective cpm
run uv run python hyperparameters.py --dataset dbaasp        --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --threshold-based-neighbourhood --leiden-objective cpm


# LD50
# Baseline
run uv run python hyperparameters.py --dataset ld50_zhu      --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm
# ef-construction
run uv run python hyperparameters.py --dataset ld50_zhu        --ef-construction 4 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset ld50_zhu        --ef-construction 8 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset ld50_zhu        --ef-construction 16 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset ld50_zhu        --ef-construction 32 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset ld50_zhu        --ef-construction 128 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset ld50_zhu        --ef-construction 256 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm
# ef-init
run uv run python hyperparameters.py --dataset ld50_zhu        --ef-construction 64 --ef-init 2 --keep-pruned-connections --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset ld50_zhu        --ef-construction 64 --ef-init 4 --keep-pruned-connections --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset ld50_zhu        --ef-construction 64 --ef-init 8 --keep-pruned-connections --use-heuristic --leiden-objective cpm
# Turn off options
run uv run python hyperparameters.py --dataset ld50_zhu        --ef-construction 64 --ef-init 1 --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset ld50_zhu        --ef-construction 64 --ef-init 1 --keep-pruned-connections --leiden-objective cpm
# Turn on options
run uv run python hyperparameters.py --dataset ld50_zhu        --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --extend-candidates --leiden-objective cpm
run uv run python hyperparameters.py --dataset ld50_zhu        --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --strict-ef --leiden-objective cpm
run uv run python hyperparameters.py --dataset ld50_zhu        --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --threshold-based-neighbourhood --leiden-objective cpm

# DNA
# Baseline
run uv run python hyperparameters.py --dataset prom_core_all --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm
# ef-construction
run uv run python hyperparameters.py --dataset prom_core_all        --ef-construction 4 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset prom_core_all        --ef-construction 8 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset prom_core_all        --ef-construction 16 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset prom_core_all        --ef-construction 32 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset prom_core_all        --ef-construction 128 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset prom_core_all        --ef-construction 256 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm
# ef-init
run uv run python hyperparameters.py --dataset prom_core_all        --ef-construction 64 --ef-init 2 --keep-pruned-connections --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset prom_core_all        --ef-construction 64 --ef-init 4 --keep-pruned-connections --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset prom_core_all        --ef-construction 64 --ef-init 8 --keep-pruned-connections --use-heuristic --leiden-objective cpm
# Turn off options
run uv run python hyperparameters.py --dataset prom_core_all        --ef-construction 64 --ef-init 1 --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset prom_core_all        --ef-construction 64 --ef-init 1 --keep-pruned-connections --leiden-objective cpm
# Turn on options
run uv run python hyperparameters.py --dataset prom_core_all        --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --extend-candidates --leiden-objective cpm
run uv run python hyperparameters.py --dataset prom_core_all        --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --strict-ef --leiden-objective cpm
run uv run python hyperparameters.py --dataset prom_core_all        --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --threshold-based-neighbourhood --leiden-objective cpm

# Scaling experiment (runtime/memory vs. dataset size, atlas + belka)
run uv run python scaling_benchmark.py --dataset atlas
run uv run python scaling_benchmark.py --dataset belka

# Per-stage timing breakdown of the refnd pipeline
run uv run python debug_scaling_refnd.py

# Effect of dataset size (and ef_construction) on HNSW edge recall
run uv run python edge_recall_scaling.py --dataset atlas
run uv run python edge_recall_scaling.py --dataset belka

# mmseqs2 vs. refnd split leakage comparison (dbaasp)
run uv run python max_identity_comparison.py

# In-distribution test (refnd + hestia split methods) for every dataset with an encoder
run uv run python in_distribution_test.py --dataset dbaasp        --method both
run uv run python in_distribution_test.py --dataset ld50_zhu      --method both
run uv run python in_distribution_test.py --dataset prom_core_all --method both
run uv run python in_distribution_test.py --dataset belka         --method both

# Peptide Atlas - ef-init only
run uv run python hyperparameters.py --dataset dbaasp        --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset dbaasp        --ef-construction 64 --ef-init 2 --keep-pruned-connections --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset dbaasp        --ef-construction 64 --ef-init 4 --keep-pruned-connections --use-heuristic --leiden-objective cpm
run uv run python hyperparameters.py --dataset dbaasp        --ef-construction 64 --ef-init 8 --keep-pruned-connections --use-heuristic --leiden-objective cpm

