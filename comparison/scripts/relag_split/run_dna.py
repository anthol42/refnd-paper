#!/usr/bin/env python3
"""Refnd splits for GUE DNA datasets (CPM + null-model gamma, no post-filtering)."""
import os
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from split_utils import (save_split, save_refnd_sizes, save_community_stats,
                         split_train_val, track_split, SEEDS, DNA_DATASETS)

# refnd proximity_threshold is a DISTANCE (= 1 - identity). Target 60% identity
# for DNA (Hestia gets threshold=0.60) -> 1 - 0.60 = 0.40. This sits just above
# the ~0.6 identity noise floor of the 4-letter alphabet, so watch the community
# stats: if refnd collapses to one giant component (degenerate train=0), back off
# toward 0.32 (~68% identity, the previous known-splittable setting).
THRESHOLD = 0.40
TEST_RATIO = 0.20
VAL_RATIO = 0.10
NULL_SAMPLES = 1_000_000  # alignment kernel is expensive; 1M keeps the GPD tail fit reliable


def build_graph_and_communities(sequences: list):
    """HNSW → distance graph → CPM communities (all seed-independent)."""
    from refnd import KernelVariant
    from refnd.core import (HNSWState, INWeightType, LeidenObjective,
                            find_communities, find_components)
    from refnd.kernels.alignments import LocalIdentityMode, CoverageMode, ScoringMatrix
    from null_model import null_model, NullCfg

    # Match Hestia's mmseqs DNA alignment settings:
    # --alignment-mode 3 (local/SW), --seq-id-mode 1 (MinSeqLength),
    # --cov-mode 5 (ShorterSeq), -c 0.7 (min_coverage), DNAFull matrix.
    kernel_params = dict(
        identity_mode=LocalIdentityMode.MinSeqLength,
        cov_mode=CoverageMode.ShorterSeq,
        min_coverage=0.7,
        matrix=ScoringMatrix.Dnafull,
    )
    cfg = NullCfg(modality=KernelVariant.AlignmentLocal,
                  proximity_threshold=THRESHOLD, kernel_params=kernel_params)
    gamma = null_model(sequences, cfg, n_samples=NULL_SAMPLES)

    hnsw = HNSWState(KernelVariant.AlignmentLocal, sequences,
                     proximity_threshold=THRESHOLD, n_threads=16, **kernel_params)
    hnsw.build(progress=True)
    graph = hnsw.edges().graph(inweight_type=INWeightType.Distance)
    communities = find_communities(graph, gamma=gamma, objective=LeidenObjective.CPM)
    components = find_components(graph)  # connected components: giant-component diagnostic
    return graph, communities, components


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default=str(Path(os.environ.get("REFND_EXP_BASE", str(Path(__file__).resolve().parent.parent))) / "splits"))
    parser.add_argument("--dataset", choices=DNA_DATASETS, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    import pandas as pd
    from refnd.core import partition

    df = pd.read_csv(Path(args.splits_dir).parent / "data" / "dna" / args.dataset / "pooled.csv")
    sequences = df["sequence"].tolist()
    print(f"[{args.dataset}] {len(sequences)} sequences")

    sizes = {}
    per_seed_local = {}
    with track_split("refnd", args.dataset, args.splits_dir):
        graph, communities, components = build_graph_and_communities(sequences)
        for seed in args.seeds:
            train_val_idx, test_idx = partition(
                communities, graph, test_ratio=TEST_RATIO, seed=seed, post_filtering=False)
            train_val_idx = list(train_val_idx)
            test_idx = list(test_idx)
            per_seed_local[seed] = (train_val_idx, test_idx)
            train_idx, val_idx = split_train_val(train_val_idx, VAL_RATIO, seed)
            save_split(args.splits_dir, "refnd", args.dataset, seed, train_idx, val_idx, test_idx)
            print(f"[{args.dataset}][seed={seed}] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
            sizes[str(seed)] = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}

    save_refnd_sizes(args.splits_dir, args.dataset, sizes)
    save_community_stats(args.splits_dir, "refnd", args.dataset,
                         communities, per_seed_local, components=components)
    print(f"Saved sizes.json for {args.dataset}")


if __name__ == "__main__":
    main()
