"""In-distribution test variant, run at a custom *distance* threshold:

  - hestia: 50/50 label split via ccpart (sim_threshold = 1 - distance_threshold,
    same conversion in_distribution_test.py already uses), then a random 80/20
    split (not hestia) for the classifier itself, since ccpart is deterministic
    and re-running it just reproduces the same tiny test set.
  - refnd: full self-consistent in_distribution_test methodology (both the
    label split and the classifier split done by refnd's own HNSW+Leiden
    partition) at the SAME distance threshold, for a like-for-like comparison.

Usage:
    uv run python hestia_lowthr_test.py --dataset dbaasp --threshold 0.2
    uv run python hestia_lowthr_test.py --dataset prom_core_all --threshold 0.35
"""
import argparse

import numpy as np
import pandas as pd
from hestia.dataset_generator import HestiaGenerator, SimArguments
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from src.cache import CacheStore
from src.datasets import DATASETS, load_dataset
from src.embeddings import compute_embeddings
from refnd.kernels import KernelVariant

from in_distribution_test import (
    MODALITY_TO_HESTIA_DATA_TYPE, FIELD_NAME_BY_HESTIA_DATA_TYPE,
    train_test_split_refnd, train_test_split_hestia, _is_degenerate_split,
)

N_REPEATS = 10
LABEL_SPLIT_RATIO = 0.5
CLASSIFIER_SPLIT_RATIO = 0.2
SEED_OFFSET = 10_000


def get_items(dataset_key, cfg, cache):
    data, _ = load_dataset(dataset_key, cache)
    if cfg.modality is KernelVariant.TanimotoBit:
        smiles_entry = cache.get_dataset(f"{dataset_key}_smiles")
        return smiles_entry[0]
    return data


def run_hestia_label_plus_random_clf(dataset_items, cfg, threshold, embs, n, hestia_modality=None):
    # hestia_modality: use the dataset's *original* modality (e.g. AlignmentLocal for DNA)
    # for the hestia data_type/is_nucleotide lookup, independent of whatever modality
    # refnd itself is run with (e.g. when --global-alignment overrides cfg.modality).
    hestia_modality = hestia_modality if hestia_modality is not None else cfg.modality
    sim_threshold = round(1.0 - threshold, 4)
    print(f"  distance_threshold={threshold} -> sim_threshold={sim_threshold}")
    _, label_test_idx = train_test_split_hestia(
        dataset_items, hestia_modality, threshold, cfg.kernel_params,
        test_ratio=LABEL_SPLIT_RATIO, seed=0,
    )
    label = np.zeros(n, dtype=np.int64)
    label[label_test_idx] = 1
    n_pos = len(label_test_idx)
    print(f"  label split: {n - n_pos} / {n_pos}  ({n_pos / n:.1%} positive)")
    if n_pos == 0 or n_pos == n:
        print("  Degenerate label split (one side empty) -- aborting.")
        return

    scores, n_failed = [], 0
    for seed in range(N_REPEATS):
        idx = np.arange(n)
        try:
            train_idx, test_idx = train_test_split(
                idx, test_size=CLASSIFIER_SPLIT_RATIO, random_state=seed, stratify=label,
            )
        except ValueError:
            train_idx, test_idx = train_test_split(idx, test_size=CLASSIFIER_SPLIT_RATIO, random_state=seed)
        if len(np.unique(label[train_idx])) < 2:
            scores.append(float("nan")); n_failed += 1
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(embs[train_idx], label[train_idx])
        preds = clf.predict(embs[test_idx])
        scores.append(balanced_accuracy_score(label[test_idx], preds))

    scores_arr = np.array(scores)
    all_nan = n_failed == N_REPEATS
    mean = float("nan") if all_nan else float(np.nanmean(scores_arr))
    std = float("nan") if all_nan else float(np.nanstd(scores_arr))
    print(f"  [hestia label + random clf] balanced_accuracy = {mean:.4f} +/- {std:.4f}  "
          f"({n_failed}/{N_REPEATS} failed)")


def run_refnd_selfconsistent(dataset_items, cfg, threshold, embs, n,
                              ef_construction, ef_init):
    from functools import partial
    split_fn = partial(
        train_test_split_refnd, dataset_items, cfg.modality, threshold,
        cfg.kernel_params, post_filtering=True,
        ef_construction=ef_construction, ef_init=ef_init,
    )
    scores, n_failed = [], 0
    for seed in range(N_REPEATS):
        _, label_test_idx = split_fn(test_ratio=LABEL_SPLIT_RATIO, seed=seed)
        label = np.zeros(n, dtype=np.int64)
        label[label_test_idx] = 1
        clf_train_idx, clf_test_idx = split_fn(test_ratio=CLASSIFIER_SPLIT_RATIO, seed=seed + SEED_OFFSET)
        if seed == 0:
            print(f"  label split (seed 0): {n - len(label_test_idx)} / {len(label_test_idx)}  "
                  f"({len(label_test_idx) / n:.1%} positive)")
        if _is_degenerate_split(clf_train_idx, clf_test_idx, label):
            scores.append(float("nan")); n_failed += 1
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(embs[clf_train_idx], label[clf_train_idx])
        preds = clf.predict(embs[clf_test_idx])
        scores.append(balanced_accuracy_score(label[clf_test_idx], preds))

    scores_arr = np.array(scores)
    all_nan = n_failed == N_REPEATS
    mean = float("nan") if all_nan else float(np.nanmean(scores_arr))
    std = float("nan") if all_nan else float(np.nanstd(scores_arr))
    print(f"  [refnd self-consistent]     balanced_accuracy = {mean:.4f} +/- {std:.4f}  "
          f"({n_failed}/{N_REPEATS} failed)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--threshold", type=float, required=True,
                     help="Distance threshold (this repo's convention) to test at")
    ap.add_argument("--ef-construction", type=int, default=64)
    ap.add_argument("--ef-init", type=int, default=2)
    ap.add_argument("--global-alignment", action="store_true",
                     help="Override the dataset's sequence modality to AlignmentGlobal "
                          "(drops local-only kernel params: identity_mode, cov_mode, min_coverage)")
    args = ap.parse_args()

    from dataclasses import replace

    cfg = DATASETS[args.dataset]
    hestia_modality = cfg.modality
    if args.global_alignment:
        if cfg.modality is not KernelVariant.AlignmentLocal:
            raise ValueError(f"--global-alignment only makes sense for AlignmentLocal datasets, "
                              f"{args.dataset!r} is {cfg.modality!r}")
        cfg = replace(cfg, modality=KernelVariant.AlignmentGlobal,
                       kernel_params={"matrix": cfg.kernel_params["matrix"]})
    cache = CacheStore()
    data, _ = load_dataset(args.dataset, cache)
    n = len(data)
    items = get_items(args.dataset, cfg, cache)

    print(f"=== {args.dataset}: distance_threshold={args.threshold}"
          f"{' (refnd using AlignmentGlobal)' if args.global_alignment else ''} ===")
    embs = compute_embeddings(args.dataset, data, cfg, cache).numpy()

    print("\n-- hestia (ccpart label split + random classifier split) --")
    run_hestia_label_plus_random_clf(items, cfg, args.threshold, embs, n, hestia_modality=hestia_modality)

    print("\n-- refnd (self-consistent, both splits via HNSW+Leiden) --")
    run_refnd_selfconsistent(items, cfg, args.threshold, embs, n, args.ef_construction, args.ef_init)


if __name__ == "__main__":
    main()
