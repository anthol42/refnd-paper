#!/usr/bin/env python3
"""relag splits for DBAASP protein dataset (CPM + null-model gamma, no post-filtering)."""
import os
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from split_utils import (save_split, save_relag_sizes, save_community_stats,
                         split_train_val, track_split, load_protein_sequences,
                         REPO_ROOT, SEEDS, PROTEIN_DATASETS)

# relag proximity_threshold is a DISTANCE (= 1 - identity). 50% identity target
# -> 1 - 0.50 = 0.50 (matches Hestia's THRESHOLD below).
THRESHOLD = 0.50
TEST_RATIO = 0.20
VAL_RATIO = 0.10  # fraction of (train+val) kept as val
NULL_SAMPLES = 10_000_000  # 10M for the GPD tail fit


def build_graph_and_communities(sequences: list):
    """HNSW → distance graph → CPM communities (all seed-independent)."""
    from refnd import KernelVariant
    from refnd.core import (HNSWState, INWeightType, LeidenObjective,
                            find_communities, find_components)
    from null_model import null_model, NullCfg
    from src.cache import CacheStore
    from src.datasets import load_dataset

    atlas_sequences, _ = load_dataset("peptide_atlas", CacheStore(root=str(REPO_ROOT / ".cache")))

    cfg = NullCfg(modality=KernelVariant.AlignmentGlobal, proximity_threshold=THRESHOLD)
    gamma = null_model(atlas_sequences, cfg, n_samples=NULL_SAMPLES)

    hnsw = HNSWState(KernelVariant.AlignmentGlobal, sequences, proximity_threshold=THRESHOLD)
    hnsw.build(progress=True)
    graph = hnsw.edges().graph(inweight_type=INWeightType.Distance)
    communities = find_communities(graph, gamma=gamma, objective=LeidenObjective.CPM)
    components = find_components(graph)  # connected components: giant-component diagnostic
    return graph, communities, components


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default=str(Path(os.environ.get("RELAG_EXP_BASE", str(Path(__file__).resolve().parent.parent))) / "splits"))
    parser.add_argument("--dataset", choices=PROTEIN_DATASETS, default="dbaasp_amp")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    from refnd.core import partition

    sequences = load_protein_sequences(args.dataset)
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

    save_relag_sizes(args.splits_dir, args.dataset, sizes)
    save_community_stats(args.splits_dir, "refnd", args.dataset,
                         communities, per_seed_local, components=components)
    print(f"Saved sizes.json for {args.dataset}")


if __name__ == "__main__":
    main()
