#!/usr/bin/env python3
"""Hestia splits for GUE DNA datasets.

Uses ccpart_random (seeded, whole-component test assignment) called directly, because
calculate_partitions' ccpart is deterministic and its dead ccpart_random branch drops
the seed. See hestia_split/run_molecule.py for the rationale.
"""
import os
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from split_utils import (save_split, save_community_stats, split_train_val,
                         track_split, SEEDS, DNA_DATASETS)

THRESHOLD = 0.60  # 60% identity (was 0.95; now matches refnd's 0.40 distance)
TEST_RATIO = 0.20
VAL_RATIO = 0.10


def run(splits_dir: str, dataset: str, seeds: list):
    import pandas as pd
    from hestia.dataset_generator import HestiaGenerator, SimArguments
    from hestia.partition import ccpart_random

    df = pd.read_csv(Path(splits_dir).parent / "data" / "dna" / dataset / "pooled.csv")
    df = df.reset_index(drop=True)

    with track_split("hestia", dataset, splits_dir):
        gen = HestiaGenerator(df, verbose=False)
        sim_df = gen.calculate_similarity(
            SimArguments(data_type="dna sequence", field_name="sequence", min_threshold=THRESHOLD))

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
    parser.add_argument("--dataset", choices=DNA_DATASETS, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    run(args.splits_dir, args.dataset, args.seeds)


if __name__ == "__main__":
    main()
