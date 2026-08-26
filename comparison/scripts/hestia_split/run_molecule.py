#!/usr/bin/env python3
"""Hestia splits for TDC molecule datasets.

Uses ccpart_random (whole connected components assigned to test, seeded) instead of
the default ccpart, which is deterministic -> identical test sets across seeds.
NOTE: HestiaGenerator.calculate_partitions cannot be used for this: it rejects
'ccpart_random' in validation, and even its dead ccpart_random branch never forwards
the seed. So we compute similarity once and call ccpart_random directly with seed=seed.
Validation is carved from train via split_train_val (consistent with refnd/mmseqs).
"""
import os
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from split_utils import (save_split, save_community_stats, split_train_val,
                         track_split, SEEDS, MOLECULE_DATASETS)

THRESHOLD = 0.40
TEST_RATIO = 0.20
VAL_RATIO = 0.10


def run(splits_dir: str, dataset: str, seeds: list):
    import pandas as pd
    from hestia.dataset_generator import HestiaGenerator, SimArguments
    from hestia.partition import ccpart_random

    df = pd.read_parquet(Path(splits_dir).parent / "data" / "molecule" / f"{dataset}.parquet")
    df = df.reset_index(drop=True)

    with track_split("hestia", dataset, splits_dir):
        # Similarity computed ONCE (avoids the O(n^2) recompute per seed).
        gen = HestiaGenerator(df, verbose=False)
        sim_df = gen.calculate_similarity(
            SimArguments(data_type="molecule", field_name="smiles", min_threshold=THRESHOLD))

        clusters = None
        per_seed_local = {}
        for seed in seeds:
            split_path = Path(splits_dir) / "hestia" / dataset / f"{seed}.json"
            if split_path.exists():
                print(f"[{dataset}][seed={seed}] Already computed, skipping.")
                continue
            # ccpart_random's third return is the per-node connected-component labels.
            train_val, test, clusters = ccpart_random(
                df, label_name=None, test_size=TEST_RATIO, threshold=THRESHOLD,
                n_bins=10, sim_df=sim_df, seed=seed, verbose=0)
            train_val = [int(x) for x in train_val]
            test_idx = [int(x) for x in test]
            per_seed_local[seed] = (train_val, test_idx)
            train_idx, val_idx = split_train_val(train_val, VAL_RATIO, seed)
            save_split(splits_dir, "hestia", dataset, seed, train_idx, val_idx, test_idx)
            print(f"[{dataset}][seed={seed}] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    if clusters is not None:
        save_community_stats(splits_dir, "hestia", dataset,
                             clusters, per_seed_local, components=clusters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default=str(Path(os.environ.get("REFND_EXP_BASE", str(Path(__file__).resolve().parent.parent))) / "splits"))
    parser.add_argument("--dataset", choices=MOLECULE_DATASETS, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    run(args.splits_dir, args.dataset, args.seeds)


if __name__ == "__main__":
    main()
