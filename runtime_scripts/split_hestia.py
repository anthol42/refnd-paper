#!/usr/bin/env python3
"""
Split a subset file into train/test using Hestia (sequence identity / molecular similarity).

Usage:
    uv run python -m runtime_scripts.split_hestia atlas input.fasta --threshold 0.5 --test-size 0.2
    uv run python -m runtime_scripts.split_hestia belka input.fasta --threshold 0.4 --test-size 0.2
"""
import argparse
import os.path
import sys

import pandas as pd
from hestia.dataset_generator import HestiaGenerator, SimArguments

from src.datasets import DATASETS, SCALING_DATASET_KEY

# hestia's SimArguments.data_type per --dataset key ("molecule" -> RDKit/Tanimoto similarity path)
DATA_TYPE = {
    "atlas": "sequence",
    "belka": "molecule",
}
FIELD_NAME = {
    "atlas": "sequence",
    "belka": "smiles",
}


def read_fasta(path: str) -> list[tuple[str, str]]:
    records = []
    header, seq_parts = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts)))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        records.append((header, "".join(seq_parts)))
    return records


def write_fasta(path: str, records: list[tuple[str, str]]):
    with open(path, "w") as f:
        for header, seq in records:
            f.write(f">{header}\n{seq}\n")


def main():
    parser = argparse.ArgumentParser(description="Hestia identity/similarity-based splitter")
    parser.add_argument("dataset", choices=list(DATA_TYPE), help="Dataset key (atlas|belka)")
    parser.add_argument("input", help="Input subset file (FASTA-formatted sequences or SMILES)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Similarity threshold (default: dataset's proximity_threshold)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data for test set (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-out", default="tmp/hestia_train.fasta")
    parser.add_argument("--test-out", default="tmp/hestia_test.fasta")
    args = parser.parse_args()
    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    field_name = FIELD_NAME[args.dataset]
    data_type  = DATA_TYPE[args.dataset]
    threshold  = args.threshold
    if threshold is None:
        threshold = DATASETS[SCALING_DATASET_KEY[args.dataset]].proximity_threshold
    # This repo's thresholds are distances (lower = more similar; edges where
    # distance <= threshold). hestia partitions on similarity (higher = more
    # similar; kept where similarity >= min_threshold) — convert. Also,
    # calculate_partitions() overwrites sim_args.min_threshold with its own
    # `min_threshold` kwarg (default 0.0) before computing similarity, so it
    # must be passed here too or hestia silently computes the full O(n^2)
    # pairwise similarity with no cutoff at all.
    sim_threshold = round(1.0 - threshold, 4)

    records = read_fasta(args.input)
    if not records:
        print(f"Error: no items found in {args.input}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(records)} items from {args.input}")

    df = pd.DataFrame({"header": [h for h, _ in records],
                       field_name: [s for _, s in records]})
    df["idx"] = df.index

    gen = HestiaGenerator(df, verbose=False)
    # By default calculate_partitions() sweeps ~20 thresholds (min_threshold=0.0,
    # threshold_step=0.05) to trace a whole OOD-difficulty curve. We only want the
    # one threshold matching this dataset's proximity_threshold, so threshold_step=1.0
    # collapses its internal range(min_threshold_int, 100, threshold_step_int) to a
    # single value — verified empirically to yield exactly one non-"random" key.
    gen.calculate_partitions(
        sim_args=SimArguments(data_type=data_type, field_name=field_name,
                              min_threshold=sim_threshold),
        min_threshold=sim_threshold + 1e-6, threshold_step=1.0,
        test_size=args.test_size,
        valid_size=0.0,
        random_state=args.seed,
        verbose=1,
    )
    parts_dict = gen.get_partitions(return_dict=True)
    # That single key is computed internally as int(min_threshold * 100) / 100, which
    # can drift a float ULP from sim_threshold — take the closest numeric key rather
    # than an exact match. (Also filters out hestia's extra 'random'-baseline key,
    # which isn't numeric and would break the comparison.)
    numeric_keys = [k for k in parts_dict if isinstance(k, (int, float))]
    key = min(numeric_keys, key=lambda k: abs(k - sim_threshold))
    parts = parts_dict[key]

    train_idx = [int(x) for x in parts["train"]]
    test_idx  = [int(x) for x in parts["test"]]

    write_fasta(args.train_out, [records[i] for i in train_idx])
    write_fasta(args.test_out,  [records[i] for i in test_idx])

    print(f"Train: {len(train_idx)} items -> {args.train_out}")
    print(f"Test:  {len(test_idx)} items -> {args.test_out}")


if __name__ == "__main__":
    main()
