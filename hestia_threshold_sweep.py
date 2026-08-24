"""Efficient Hestia threshold sweep: compute the pairwise similarity matrix
ONCE (via calculate_partitions' built-in sim_df reuse), then bucket it at
many thresholds without recomputing mmseqs/alignment each time.

Usage:
    uv run python hestia_threshold_sweep.py --dataset prom_core_all \
        --min-sim-threshold 0.65 --max-sim-threshold 0.99 --step 0.02
"""
import argparse

import numpy as np
import pandas as pd
from hestia.dataset_generator import HestiaGenerator, SimArguments

from src.cache import CacheStore
from src.datasets import DATASETS, load_dataset
from refnd.kernels import KernelVariant

MODALITY_TO_HESTIA_DATA_TYPE = {
    str(KernelVariant.AlignmentGlobal): "sequence",
    str(KernelVariant.AlignmentLocal):  "dna sequence",
    str(KernelVariant.TanimotoBit):     "molecule",
}
FIELD_NAME_BY_HESTIA_DATA_TYPE = {
    "sequence":     "sequence",
    "dna sequence": "sequence",
    "molecule":     "smiles",
}


def get_items(dataset_key, cfg, cache):
    data, _ = load_dataset(dataset_key, cache)
    if cfg.modality is KernelVariant.TanimotoBit:
        smiles_entry = cache.get_dataset(f"{dataset_key}_smiles")
        return smiles_entry[0]
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--min-sim-threshold", type=float, default=0.65,
                     help="Lowest similarity threshold to sweep (also used to prefilter mmseqs alignment)")
    ap.add_argument("--max-sim-threshold", type=float, default=0.99)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--test-size", type=float, default=0.5)
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    cache = CacheStore()
    items = get_items(args.dataset, cfg, cache)
    n = len(items)
    data_type = MODALITY_TO_HESTIA_DATA_TYPE[str(cfg.modality)]
    field_name = FIELD_NAME_BY_HESTIA_DATA_TYPE[data_type]

    df = pd.DataFrame({field_name: items})
    df["idx"] = df.index

    gen = HestiaGenerator(df, verbose=True)
    # Similarity is calculated ONCE here (filtered to pairs >= min_sim_threshold),
    # then every threshold in [min_sim_threshold, max_sim_threshold) reuses it --
    # no repeated mmseqs/alignment calls.
    gen.calculate_partitions(
        sim_args=SimArguments(data_type=data_type, field_name=field_name,
                              min_threshold=args.min_sim_threshold),
        min_threshold=args.min_sim_threshold, threshold_step=args.step,
        test_size=args.test_size, valid_size=0.0, random_state=0, verbose=1,
        partition_algorithm="ccpart",
    )
    # calculate_partitions loops range(int(min*100), 100, int(step*100)) -- clip
    # to caller's requested max.
    max_key = int(args.max_sim_threshold * 100)

    print(f"\n{'sim_threshold':>14} | {'dist_threshold':>14} | {'n_components':>12} | "
          f"{'largest_cc':>10} | {'largest_cc_%':>12} | {'train':>8} | {'test':>8} | {'test_%':>8}")
    print("-" * 100)
    rows = []
    for key in sorted(k for k in gen.partitions if isinstance(k, (int, float))):
        if key * 100 > max_key + 1e-6:
            continue
        part = gen.partitions[key]
        clusters = np.asarray(part["clusters"])
        _, counts = np.unique(clusters, return_counts=True)
        n_comp = len(counts)
        largest = counts.max()
        train_n = len(part["train"])
        test_n = len(part["test"])
        dist_threshold = round(1.0 - key, 4)
        rows.append((key, dist_threshold, n_comp, largest, train_n, test_n))
        print(f"{key:>14.2f} | {dist_threshold:>14.2f} | {n_comp:>12} | {largest:>10} | "
              f"{largest / n:>11.1%} | {train_n:>8} | {test_n:>8} | {test_n / n:>7.1%}")


if __name__ == "__main__":
    main()
