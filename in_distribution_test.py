"""In-distribution test: can a linear classifier tell which side of a split
a sample came from?

Method (see README "In-distribution test"):
  1. Split the dataset 50/50 (train/test) with a given split method, and
     label each sample by which side it landed on.
  2. Shuffle everything back together and take a fresh 80/20 split of *all*
     samples with the *same* split method, for the classifier itself.
  3. Fit a linear (logistic regression) classifier on embeddings to predict
     the split-membership label, evaluated on the 20% held out.

If the two halves of the split are in-distribution, the classifier shouldn't
be able to do better than chance -> balanced accuracy ~= 50%.
Repeated over several seeds.

Two split methods are compared, "for now": refnd's community-based
`partition()`, and Hestia's identity/similarity-based clustering. Both splits
(the 50/50 label split and the 80/20 classifier split) reuse the *same*
method — never a plain random split — so a degenerate collapse (e.g. a
single-class fold) is a real signal about the split method and is allowed to
crash rather than being papered over with a NaN. If this crashes with
"needs samples of at least 2 classes", it means the split method collapsed
the whole dataset into (effectively) one cluster at the configured
proximity_threshold, so no cluster-preserving split can produce two
non-trivial sides — worth checking the threshold/data, not silently retried.

Note: the Hestia branch reclusters (RDKit/alignment similarity, O(n^2)) twice
per repeat, so it's noticeably slower than the refnd branch for larger n.

Usage:
    uv run python in_distribution_test.py --dataset dbaasp
    uv run python in_distribution_test.py --dataset belka --method refnd
    uv run python in_distribution_test.py --dataset ld50_zhu --method hestia
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from hestia.dataset_generator import HestiaGenerator, SimArguments
from rich import print
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from src.cache import CacheStore
from src.datasets import DATASETS, load_dataset
from src.embeddings import compute_embeddings
from refnd.core import HNSWState, INWeightType, LeidenObjective, find_communities, partition

N_REPEATS = 10
LABEL_SPLIT_RATIO = 0.5
CLASSIFIER_SPLIT_RATIO = 0.2
SEED_OFFSET = 10_000  # keeps the label split and classifier split independent

# Hestia's SimArguments per main DATASETS key (distinct from scaling_benchmark's
# atlas/belka namespace) — molecule datasets need raw SMILES, not fingerprints.
HESTIA_DATA_TYPE = {
    "dbaasp": "sequence",
    "prom_core_all": "dna sequence",
    "ld50_zhu": "molecule",
    "belka": "molecule",
}
HESTIA_FIELD_NAME = {
    "dbaasp": "sequence",
    "prom_core_all": "sequence",
    "ld50_zhu": "smiles",
    "belka": "smiles",
}


def _fit_and_score(embs: np.ndarray, label: np.ndarray,
                    train_idx: list[int], test_idx: list[int]) -> float:
    clf = LogisticRegression(max_iter=1000)
    clf.fit(embs[train_idx], label[train_idx])
    preds = clf.predict(embs[test_idx])
    return balanced_accuracy_score(label[test_idx], preds)


def _run_refnd(communities: list[int], graph, embs: np.ndarray,
               n: int, post_filtering: bool) -> dict:
    scores = []
    for seed in range(N_REPEATS):
        label_train_idx, label_test_idx = partition(
            communities, graph, test_ratio=LABEL_SPLIT_RATIO, seed=seed,
            post_filtering=post_filtering,
        )
        label = np.zeros(n, dtype=np.int64)
        label[list(label_test_idx)] = 1

        clf_train_idx, clf_test_idx = partition(
            communities, graph, test_ratio=CLASSIFIER_SPLIT_RATIO, seed=seed + SEED_OFFSET,
            post_filtering=post_filtering,
        )
        scores.append(_fit_and_score(embs, label, list(clf_train_idx), list(clf_test_idx)))

    scores_arr = np.array(scores)
    return {
        "balanced_accuracy_mean": float(scores_arr.mean()),
        "balanced_accuracy_std":  float(scores_arr.std()),
    }


def _run_hestia(dataset: str, items: list, embs: np.ndarray, n: int, threshold: float) -> dict:
    field_name = HESTIA_FIELD_NAME[dataset]
    data_type  = HESTIA_DATA_TYPE[dataset]
    df = pd.DataFrame({field_name: items})
    df["idx"] = df.index

    gen = HestiaGenerator(df, verbose=False)

    scores = []
    for seed in range(N_REPEATS):
        gen.calculate_partitions(
            sim_args=SimArguments(data_type=data_type, field_name=field_name,
                                  min_threshold=threshold),
            test_size=LABEL_SPLIT_RATIO, valid_size=0.0, random_state=seed, verbose=0,
        )
        parts1 = gen.get_partitions(return_dict=True)[threshold]
        label = np.zeros(n, dtype=np.int64)
        label[[int(x) for x in parts1["test"]]] = 1

        gen.calculate_partitions(
            sim_args=SimArguments(data_type=data_type, field_name=field_name,
                                  min_threshold=threshold),
            test_size=CLASSIFIER_SPLIT_RATIO, valid_size=0.0,
            random_state=seed + SEED_OFFSET, verbose=0,
        )
        parts2 = gen.get_partitions(return_dict=True)[threshold]
        clf_train_idx = [int(x) for x in parts2["train"]]
        clf_test_idx  = [int(x) for x in parts2["test"]]

        scores.append(_fit_and_score(embs, label, clf_train_idx, clf_test_idx))

    scores_arr = np.array(scores)
    return {
        "balanced_accuracy_mean": float(scores_arr.mean()),
        "balanced_accuracy_std":  float(scores_arr.std()),
    }


def _print_result(key: str, r: dict) -> None:
    print(f"  [{key}] balanced_accuracy = {r['balanced_accuracy_mean']:.4f} "
          f"+/- {r['balanced_accuracy_std']:.4f}  (50% = in-distribution)")


def main() -> None:
    parser = argparse.ArgumentParser(description="In-distribution test for dataset splits")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--method", choices=["refnd", "hestia", "both"], default="both")
    parser.add_argument("--ef-construction", type=int, default=64)
    parser.add_argument("--ef-init", type=int, default=1)
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    if cfg.encoder is None:
        raise ValueError(
            f"{args.dataset!r} has no encoder configured — the in-distribution "
            f"test needs embeddings to classify on."
        )
    if args.method in ("hestia", "both") and args.dataset not in HESTIA_DATA_TYPE:
        raise ValueError(f"No Hestia data_type/field_name mapping for {args.dataset!r}")
    cache = CacheStore()

    print(f"[bold][orange2]=== In-Distribution Test: {args.dataset} ({args.method}) ===[/][/]")

    print("Loading dataset...")
    data, _ = load_dataset(args.dataset, cache)
    n = len(data)
    print(f"  {n:,} samples")

    print("Loading/computing embeddings...")
    embs = compute_embeddings(args.dataset, data, cfg, cache).numpy()

    results = {}

    if args.method in ("refnd", "both"):
        print("\n[green]-- refnd --[/]")
        print("Building HNSW...")
        hnsw = HNSWState(
            cfg.modality, data,
            proximity_threshold=cfg.proximity_threshold,
            ef_construction=args.ef_construction,
            ef_init=args.ef_init,
            **cfg.kernel_params,
        )
        hnsw.build(progress=True)
        graph = hnsw.edges().graph(inweight_type=INWeightType.Distance)
        communities = find_communities(graph, gamma=1.0, objective=LeidenObjective.Modularity)
        print(f"  {len(set(communities))} communities")

        results["refnd"] = {}
        for post_filtering in (False, True):
            key = "postfilter" if post_filtering else "no_postfilter"
            r = _run_refnd(communities, graph, embs, n, post_filtering)
            results["refnd"][key] = r
            _print_result(key, r)

    if args.method in ("hestia", "both"):
        print("\n[green]-- hestia --[/]")
        field_name = HESTIA_FIELD_NAME[args.dataset]
        if field_name == "smiles":
            smiles_entry = cache.get_dataset(f"{args.dataset}_smiles")
            if smiles_entry is None:
                raise RuntimeError(f"SMILES cache missing for {args.dataset}")
            items = smiles_entry[0]
        else:
            items = data

        results["hestia"] = _run_hestia(args.dataset, items, embs, n, cfg.proximity_threshold)
        _print_result("hestia", results["hestia"])

    out_path = Path(f"results/in_distribution_{args.dataset}.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[bold]Results saved to {out_path}[/]")


if __name__ == "__main__":
    main()
