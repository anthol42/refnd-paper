#!/usr/bin/env python3
"""Hestia splits for the DBAASP peptide dataset.

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
                         track_split, load_protein_sequences, SEEDS, PROTEIN_DATASETS)

THRESHOLD = 0.50  # 50% identity, matches refnd_split/run_protein.py's distance threshold
TEST_RATIO = 0.20
VAL_RATIO = 0.10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default=str(Path(os.environ.get("REFND_EXP_BASE", str(Path(__file__).resolve().parent.parent))) / "splits"))
    parser.add_argument("--dataset", choices=PROTEIN_DATASETS, default="dbaasp_amp")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    import pandas as pd
    from hestia.dataset_generator import HestiaGenerator, SimArguments
    from hestia.partition import ccpart_random

    sequences = load_protein_sequences(args.dataset)
    df = pd.DataFrame({"sequence": sequences})
    print(f"[{args.dataset}] {len(df)} sequences")

    with track_split("hestia", args.dataset, args.splits_dir):
        gen = HestiaGenerator(df, verbose=False)
        sim_df = gen.calculate_similarity(
            SimArguments(data_type="sequence", field_name="sequence", min_threshold=THRESHOLD))

        clusters = None
        per_seed_local = {}
        for seed in args.seeds:
            split_path = Path(args.splits_dir) / "hestia" / args.dataset / f"{seed}.json"
            if split_path.exists():
                print(f"[seed={seed}] Already computed, skipping.")
                continue
            # ccpart_random's third return is the per-node connected-component labels
            # (seed-independent; only the test assignment uses the seed).
            train_val, test, clusters = ccpart_random(
                df, label_name=None, test_size=TEST_RATIO, threshold=THRESHOLD,
                n_bins=10, sim_df=sim_df, seed=seed, verbose=0)
            train_val = [int(x) for x in train_val]
            test_idx = [int(x) for x in test]
            per_seed_local[seed] = (train_val, test_idx)
            train_idx, val_idx = split_train_val(train_val, VAL_RATIO, seed)
            save_split(args.splits_dir, "hestia", args.dataset, seed, train_idx, val_idx, test_idx)
            print(f"[seed={seed}] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    if clusters is not None:
        save_community_stats(args.splits_dir, "hestia", args.dataset,
                             clusters, per_seed_local, components=clusters)


if __name__ == "__main__":
    main()
