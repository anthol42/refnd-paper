#!/usr/bin/env python3
"""Random splits for all datasets."""
import os
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from split_utils import (save_split, track_split, SEEDS,
                         PROTEIN_DATASETS, MOLECULE_DATASETS, DNA_DATASETS)

TEST_RATIO = 0.20
VAL_RATIO = 0.10


def random_split(n: int, test_ratio: float, val_ratio: float, seed: int) -> tuple:
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n).tolist()
    n_test = int(n * test_ratio)
    n_val = int((n - n_test) * val_ratio)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return train_idx, val_idx, test_idx


def run_protein(splits_dir: str, dataset: str, seeds: list):
    import pandas as pd
    df = pd.read_parquet(Path(splits_dir).parent / "data" / "protein" / f"{dataset}.parquet")
    n = len(df)
    with track_split("random", dataset, splits_dir):
        for seed in seeds:
            train, val, test = random_split(n, TEST_RATIO, VAL_RATIO, seed)
            save_split(splits_dir, "random", dataset, seed, train, val, test)
            print(f"[{dataset}][seed={seed}] train={len(train)} val={len(val)} test={len(test)}")


def run_molecule(splits_dir: str, dataset: str, seeds: list):
    import pandas as pd
    df = pd.read_parquet(Path(splits_dir).parent / "data" / "molecule" / f"{dataset}.parquet")
    n = len(df)
    with track_split("random", dataset, splits_dir):
        for seed in seeds:
            train, val, test = random_split(n, TEST_RATIO, VAL_RATIO, seed)
            save_split(splits_dir, "random", dataset, seed, train, val, test)
            print(f"[{dataset}][seed={seed}] train={len(train)} val={len(val)} test={len(test)}")


def run_dna(splits_dir: str, dataset: str, seeds: list):
    import pandas as pd
    df = pd.read_csv(Path(splits_dir).parent / "data" / "dna" / dataset / "pooled.csv")
    n = len(df)
    with track_split("random", dataset, splits_dir):
        for seed in seeds:
            train, val, test = random_split(n, TEST_RATIO, VAL_RATIO, seed)
            save_split(splits_dir, "random", dataset, seed, train, val, test)
            print(f"[{dataset}][seed={seed}] train={len(train)} val={len(val)} test={len(test)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default=str(Path(os.environ.get("REFND_EXP_BASE", str(Path(__file__).resolve().parent.parent))) / "splits"))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    for ds in PROTEIN_DATASETS:
        run_protein(args.splits_dir, ds, args.seeds)
    for ds in MOLECULE_DATASETS:
        run_molecule(args.splits_dir, ds, args.seeds)
    for ds in DNA_DATASETS:
        run_dna(args.splits_dir, ds, args.seeds)
    print("Random splits complete.")


if __name__ == "__main__":
    main()
