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

Note: both split strategies rebuild everything from scratch on every call
(HNSW+Leiden for relag; RDKit/alignment reclustering for Hestia) — that's
2 * N_REPEATS rebuilds per method. Fine for these dataset sizes; keeps the
split functions simple, self-contained, and sklearn-shaped.

Usage:
    uv run python in_distribution_test.py --dataset dbaasp
    uv run python in_distribution_test.py --dataset belka --method refnd
    uv run python in_distribution_test.py --dataset ld50_zhu --method hestia
"""

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from hestia.dataset_generator import HestiaGenerator, SimArguments
from rich import print
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from src.cache import CacheStore
from src.datasets import DATASETS, load_dataset
from src.embeddings import compute_embeddings
from src.fingerprints import compute_fingerprints
from refnd.core import HNSWState, INWeightType, LeidenObjective, find_communities, partition
from refnd.kernels import KernelVariant

SplitFn = Callable[..., tuple[list[int], list[int]]]

N_REPEATS = 10
LABEL_SPLIT_RATIO = 0.5
CLASSIFIER_SPLIT_RATIO = 0.2
SEED_OFFSET = 10_000  # keeps the label split and classifier split independent

# Bridges relag's KernelVariant (cfg.modality) to Hestia's SimArguments.data_type
# string, so both split strategies are driven off the same DatasetConfig.modality
# instead of a second, hand-maintained per-dataset-key table.
# Keyed by str(modality) since KernelVariant (a PyO3 enum) isn't hashable.
MODALITY_TO_HESTIA_DATA_TYPE: dict[str, str] = {
    str(KernelVariant.AlignmentGlobal): "sequence",
    str(KernelVariant.AlignmentLocal):  "dna sequence",
    str(KernelVariant.TanimotoBit):     "molecule",
}
# Hestia's DataFrame column name it expects for a given data_type.
FIELD_NAME_BY_HESTIA_DATA_TYPE = {
    "sequence":     "sequence",
    "dna sequence": "sequence",
    "molecule":     "smiles",
}


def _fit_and_score(embs: np.ndarray, label: np.ndarray,
                    train_idx: list[int], test_idx: list[int]) -> float:
    clf = LogisticRegression(max_iter=1000)
    clf.fit(embs[train_idx], label[train_idx])
    preds = clf.predict(embs[test_idx])
    return balanced_accuracy_score(label[test_idx], preds)


def _to_kernel_items(dataset: list[str], modality: KernelVariant) -> list[Any]:
    """Convert raw items (sequences or SMILES) into what HNSWState expects."""
    if modality is not KernelVariant.TanimotoBit:
        return dataset

    from refnd.utils import BitFingerprint

    fps = compute_fingerprints(dataset)
    missing = sum(1 for fp in fps if fp is None)
    if missing:
        raise ValueError(f"{missing} SMILES could not be parsed by RDKit")
    return [BitFingerprint.from_np(fp) for fp in fps]


# ── Per-method split strategies (sklearn-shaped: same (dataset, modality, threshold,
# kernel_params) inputs, differing only in keyword-only extras and internals) ──

def train_test_split_relag(
    dataset: list[str], modality: KernelVariant, threshold: float, kernel_params: dict,
    *, test_ratio: float, seed: int, post_filtering: bool,
    ef_construction: int = 64, ef_init: int = 2,
) -> tuple[list[int], list[int]]:
    items = _to_kernel_items(dataset, modality)
    hnsw = HNSWState(
        modality, items, proximity_threshold=threshold,
        ef_construction=ef_construction, ef_init=ef_init, **kernel_params,
    )
    hnsw.build(progress=False)
    graph = hnsw.edges().graph(inweight_type=INWeightType.Distance)
    communities = find_communities(graph, gamma=1.0, objective=LeidenObjective.Modularity)
    train_idx, test_idx = partition(
        communities, graph, test_ratio=test_ratio, seed=seed, post_filtering=post_filtering,
    )
    return list(train_idx), list(test_idx)


def train_test_split_hestia(
    dataset: list[str], modality: KernelVariant, threshold: float, kernel_params: dict,
    *, test_ratio: float, seed: int,
) -> tuple[list[int], list[int]]:
    data_type  = MODALITY_TO_HESTIA_DATA_TYPE[str(modality)]
    field_name = FIELD_NAME_BY_HESTIA_DATA_TYPE[data_type]
    # This repo's thresholds are distances (lower = more similar; edges where
    # distance <= threshold). hestia partitions on similarity (higher = more
    # similar; kept where similarity >= min_threshold) — convert. Also,
    # calculate_partitions() overwrites sim_args.min_threshold with its own
    # `min_threshold` kwarg (default 0.0) before computing similarity, so it
    # must be passed here too or hestia silently computes the full O(n^2)
    # pairwise similarity with no cutoff at all.
    sim_threshold = round(1.0 - threshold, 4)

    df = pd.DataFrame({field_name: dataset})
    df["idx"] = df.index
    gen = HestiaGenerator(df, verbose=False)
    # By default calculate_partitions() sweeps ~20 thresholds (min_threshold=0.0,
    # threshold_step=0.05) to trace a whole OOD-difficulty curve. We only want the
    # one threshold matching this dataset's proximity_threshold, so threshold_step=1.0
    # collapses its internal range(min_threshold_int, 100, threshold_step_int) to a
    # single value — verified empirically to yield exactly one non-"random" key.
    gen.calculate_partitions(
        sim_args=SimArguments(data_type=data_type, field_name=field_name, min_threshold=sim_threshold),
        min_threshold=sim_threshold + 1e-6, threshold_step=1.0,
        test_size=test_ratio, valid_size=0.0, random_state=seed, verbose=0,
    )
    parts_dict = gen.get_partitions(return_dict=True)
    # That single key is computed internally as int(min_threshold * 100) / 100, which
    # can drift a float ULP from sim_threshold — take the closest numeric key rather
    # than an exact match. (Also filters out hestia's extra 'random'-baseline key,
    # which isn't numeric and would break the comparison.)
    numeric_keys = [k for k in parts_dict if isinstance(k, (int, float))]
    key = min(numeric_keys, key=lambda k: abs(k - sim_threshold))
    parts = parts_dict[key]
    return [int(x) for x in parts["train"]], [int(x) for x in parts["test"]]


def _is_degenerate_split(
    train_idx: list[int], test_idx: list[int], label: np.ndarray,
) -> bool:
    """True if a split can't be fit/scored: an empty side, or a train set
    that only contains one label class (LogisticRegression can't fit that)."""
    if len(train_idx) == 0 or len(test_idx) == 0:
        return True
    return len(np.unique(label[train_idx])) < 2


# ── Generic runner: label split + classifier split + score, over N_REPEATS seeds ──

def _run(split_fn: SplitFn, embs: np.ndarray, n: int) -> dict:
    scores = []
    n_failed = 0
    for seed in range(N_REPEATS):
        _, label_test_idx = split_fn(test_ratio=LABEL_SPLIT_RATIO, seed=seed)
        label = np.zeros(n, dtype=np.int64)
        label[label_test_idx] = 1

        clf_train_idx, clf_test_idx = split_fn(
            test_ratio=CLASSIFIER_SPLIT_RATIO, seed=seed + SEED_OFFSET,
        )

        if _is_degenerate_split(clf_train_idx, clf_test_idx, label):
            print(f"  [yellow]seed {seed}: degenerate split (empty side or single "
                  f"class) — scoring as NaN[/]")
            scores.append(float("nan"))
            n_failed += 1
            continue

        scores.append(_fit_and_score(embs, label, clf_train_idx, clf_test_idx))

    scores_arr = np.array(scores)
    all_nan = n_failed == N_REPEATS
    return {
        "balanced_accuracy_mean": float("nan") if all_nan else float(np.nanmean(scores_arr)),
        "balanced_accuracy_std":  float("nan") if all_nan else float(np.nanstd(scores_arr)),
        "n_failed_repeats": n_failed,
    }


def _print_result(key: str, r: dict) -> None:
    failed_note = f"  [red]({r['n_failed_repeats']}/{N_REPEATS} repeats failed)[/]" if r["n_failed_repeats"] else ""
    print(f"  [{key}] balanced_accuracy = {r['balanced_accuracy_mean']:.4f} "
          f"+/- {r['balanced_accuracy_std']:.4f}  (50% = in-distribution){failed_note}")


def main() -> None:
    parser = argparse.ArgumentParser(description="In-distribution test for dataset splits")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--method", choices=["refnd", "hestia", "both"], default="both")
    parser.add_argument("--ef-construction", type=int, default=64)
    parser.add_argument("--ef-init", type=int, default=2)
    parser.add_argument("--debug", action="store_true",
                        help="Only run 2 repeats (instead of 10) for a fast smoke test")
    args = parser.parse_args()

    global N_REPEATS
    if args.debug:
        N_REPEATS = 2

    cfg = DATASETS[args.dataset]
    if cfg.encoder is None:
        raise ValueError(
            f"{args.dataset!r} has no encoder configured — the in-distribution "
            f"test needs embeddings to classify on."
        )
    if args.method in ("hestia", "both") and str(cfg.modality) not in MODALITY_TO_HESTIA_DATA_TYPE:
        raise ValueError(f"No Hestia data_type mapping for modality {cfg.modality!r}")
    cache = CacheStore()

    print(f"[bold][orange2]=== In-Distribution Test: {args.dataset} ({args.method}) ===[/][/]")

    print("Loading dataset...")
    data, _ = load_dataset(args.dataset, cache)
    n = len(data)
    print(f"  {n:,} samples")

    print("Loading/computing embeddings...")
    embs = compute_embeddings(args.dataset, data, cfg, cache).numpy()

    # Both split functions want raw items (sequences or SMILES), not the
    # BitFingerprint objects `data` holds for molecule datasets.
    if cfg.modality is KernelVariant.TanimotoBit:
        smiles_entry = cache.get_dataset(f"{args.dataset}_smiles")
        if smiles_entry is None:
            raise RuntimeError(f"SMILES cache missing for {args.dataset}")
        dataset_items = smiles_entry[0]
    else:
        dataset_items = data

    results = {}

    if args.method in ("refnd", "both"):
        print("\n[green]-- relag --[/]")
        split_fn = partial(
            train_test_split_relag, dataset_items, cfg.modality, cfg.proximity_threshold,
            cfg.kernel_params, post_filtering=True,
            ef_construction=args.ef_construction, ef_init=args.ef_init,
        )


        results["refnd"] = _run(split_fn, embs, n)
        _print_result("refnd", results["refnd"])

    if args.method in ("hestia", "both"):
        print("\n[green]-- hestia --[/]")
        split_fn = partial(
            train_test_split_hestia, dataset_items, cfg.modality, cfg.proximity_threshold,
            cfg.kernel_params,
        )
        results["hestia"] = _run(split_fn, embs, n)
        _print_result("hestia", results["hestia"])

    out_path = Path(f"results/in_distribution_{args.dataset}.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[bold]Results saved to {out_path}[/]")


if __name__ == "__main__":
    main()
