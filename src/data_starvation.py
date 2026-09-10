"""Shared helpers for the data-starvation experiment.

The community structure is computed ONCE per dataset, from the EXACT
(brute-force, O(n^2)) proximity graph rather than an HNSW approximation, and
then reused across every split seed. Only `partition` is re-seeded, so the
whole experiment is reproducible from the seeds alone: HNSW's build order and
Leiden's refinement randomness no longer vary the communities from split to
split. Exact edges are cached on disk (`CacheStore`), so the O(n^2) pass is
paid once per (dataset, threshold).

Given that fixed structure, the experiment offers two ways to downsample the
train set by a target fraction:

- community-based: drop whole communities at random until at least the
  target fraction of train samples has been removed (communities are
  indivisible, so this can overshoot slightly).
- random: drop exactly as many samples, chosen uniformly at random, as the
  community-based method actually dropped — so both methods remove the same
  number of samples for a fair comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from refnd.core import (
    CsrGraph,
    INWeightType,
    LeidenObjective,
    exact_edges,
    find_communities,
    partition,
)

from .cache import CacheStore
from .datasets import DATASETS, DatasetConfig, load_dataset
from .embeddings import compute_embeddings
from .mlp import train_eval_mlp

DEFAULT_DROP_FRACTIONS = [0.0, 0.05, 0.10, 0.20, 0.40, 0.60]
DEFAULT_N_REPEATS = 30
DEFAULT_N_SPLITS = 5


def build_exact_communities(
    name: str,
    data: list[Any],
    prox_threshold: float,
    gamma: float,
    cfg: DatasetConfig,
    cache: CacheStore,
    *,
    objective: LeidenObjective = LeidenObjective.CPM,
) -> tuple[list[int], CsrGraph]:
    """Compute the exact proximity graph and its communities, once.

    Uses `exact_edges` (brute force) instead of an HNSW approximation so the
    graph -- and therefore the community structure every split is drawn from
    -- is a deterministic function of (data, threshold). The EdgeStore is
    cached under `{name}_exact` (or `{name}_exact_thr{t}` when the threshold
    differs from the dataset's registered default, as it does for most
    data-starvation runs), so the O(n^2) pass happens only the first time.

    Returns `(communities, graph)`; feed both to `split_from_communities`.
    """
    native = prox_threshold == cfg.proximity_threshold
    key = f"{name}_exact" if native else f"{name}_exact_thr{prox_threshold}"
    es = cache.get_edges(key)
    if es is None:
        es = exact_edges(
            cfg.modality, data, proximity_threshold=prox_threshold,
            progress=True, **cfg.kernel_params,
        )
        cache.store_edges(key, es)

    graph = es.graph(inweight_type=INWeightType.SimilarityComplement)
    communities = list(find_communities(graph, gamma=gamma, objective=objective))
    return communities, graph


def split_from_communities(
    communities: list[int],
    graph: CsrGraph,
    *,
    test_ratio: float = 0.2,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """Split a fixed community structure into train/val/test for one seed.

    val is carved out of the train side using the same community-based
    `partition`, mirroring the outer test split (see `split_and_violations`
    in src/metrics.py). `communities` and `graph` come from
    `build_exact_communities` and are shared across seeds -- only the
    partition shuffling varies here, which is what makes a run reproducible.

    Returns a dict with keys: communities (list[int], one per full-dataset
    index), train_idx, val_idx, test_idx (lists of full-dataset indices).
    """
    train_full_idx, test_idx = partition(
        communities, graph, test_ratio=test_ratio, seed=seed, post_filtering=False,
    )
    train_full_idx = list(train_full_idx)
    test_idx = list(test_idx)

    train_sub_graph, _ = graph.subgraph(train_full_idx)
    train_communities_global = [communities[i] for i in train_full_idx]
    relabel = {old: new for new, old in enumerate(sorted(set(train_communities_global)))}
    train_communities_local = [relabel[c] for c in train_communities_global]
    inner_train_local, val_local = partition(
        train_communities_local, train_sub_graph, test_ratio=val_ratio, seed=seed, post_filtering=False,
    )
    train_idx = [train_full_idx[i] for i in inner_train_local]
    val_idx = [train_full_idx[i] for i in val_local]

    return {
        "communities": communities,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


def community_downsample(
    train_idx: list[int],
    communities: list[int],
    drop_frac: float,
    seed: int,
    singletons_are_com: bool = False,
) -> list[int]:
    """Drop whole communities (restricted to `train_idx`) at random until at
    least `drop_frac` of `train_idx` has been removed.

    `singletons_are_com` decides whether singleton communities (size 1) are
    eligible to be dropped. Most datasets' singletons behave as outliers
    (see `run_singleton_power_experiment`), so by default (False) they're
    never touched -- only communities of size >=2 are eligible. For the few
    datasets where singletons carry more per-sample generalization value than
    non-singleton clusters (empirically, lipophilicity, caco2_wang, and sr_are),
    `singletons_are_com=True` makes them eligible like any other community.
    If the eligible communities can't reach target_drop on their own (only
    possible with singletons_are_com=False, if non-singleton mass runs out),
    this undershoots -- otherwise it only ever overshoots.

    Since communities can't be split, the actual dropped fraction may exceed
    `drop_frac` slightly (or fall short, in the undershoot case above).
    Returns the kept indices (a subset of train_idx).
    """
    rng = np.random.default_rng(seed)
    n_train = len(train_idx)
    target_drop = int(round(drop_frac * n_train))

    by_community: dict[int, list[int]] = {}
    for idx in train_idx:
        by_community.setdefault(communities[idx], []).append(idx)

    min_size = 0 if singletons_are_com else 1
    community_ids = [cid for cid, members in by_community.items() if len(members) > min_size]
    rng.shuffle(community_ids)

    dropped: set[int] = set()
    for cid in community_ids:
        if len(dropped) >= target_drop:
            break
        dropped.update(by_community[cid])

    return [i for i in train_idx if i not in dropped], len(dropped)


def random_downsample(train_idx: list[int], n_drop: int, seed: int) -> list[int]:
    """Drop exactly `n_drop` samples chosen uniformly at random from train_idx."""
    rng = np.random.default_rng(seed)
    n_drop = min(n_drop, len(train_idx))
    dropped = set(rng.choice(train_idx, size=n_drop, replace=False).tolist())
    return [i for i in train_idx if i not in dropped]


def run_starvation_repeat(
    embs: torch.Tensor,
    labels: np.ndarray,
    metric: str,
    communities: list[int],
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    drop_frac: float,
    seed: int,
    singletons_are_com: bool = False,
) -> dict:
    """One repeat of both downsampling methods at `drop_frac`, sharing the
    same number of dropped samples (community-based decides it first)."""
    community_train_idx, n_dropped = community_downsample(
        train_idx, communities, drop_frac, seed, singletons_are_com=singletons_are_com,
    )
    random_train_idx = random_downsample(train_idx, n_dropped, seed)

    community_result = train_eval_mlp(
        embs, labels, train_idx=community_train_idx, val_idx=val_idx, test_idx=test_idx,
        metric=metric, seed=seed,
    )
    random_result = train_eval_mlp(
        embs, labels, train_idx=random_train_idx, val_idx=val_idx, test_idx=test_idx,
        metric=metric, seed=seed,
    )

    # n_eff counts communities that are eligible to be dropped by
    # community_downsample -- non-singleton only by default (singletons are
    # outliers, don't contribute to n_eff), or ALL communities including
    # singletons when singletons_are_com=True (their singletons carry real
    # per-sample generalization value -- see run_singleton_power_experiment).
    from collections import Counter
    train_sizes = Counter(communities[i] for i in train_idx)
    if singletons_are_com:
        eligible_ids = set(train_sizes.keys())
    else:
        eligible_ids = {cid for cid, sz in train_sizes.items() if sz > 1}
    n_eff_community = len({communities[i] for i in community_train_idx} & eligible_ids)
    n_eff_random = len({communities[i] for i in random_train_idx} & eligible_ids)

    return {
        "n_dropped": n_dropped,
        "n_train_community": len(community_train_idx),
        "n_train_random": len(random_train_idx),
        "n_eff_community": n_eff_community,
        "n_eff_random": n_eff_random,
        "community_score": community_result["score"],
        "random_score": random_result["score"],
    }


def run_starvation_experiment(
    name: str,
    threshold: float,
    gamma: float,
    out_path: str | Path,
    *,
    split_seed: int = 7,
    n_splits: int = DEFAULT_N_SPLITS,
    drop_fractions: list[float] | None = None,
    n_repeats: int = DEFAULT_N_REPEATS,
    objective: LeidenObjective = LeidenObjective.CPM,
    singletons_are_com: bool = False,
) -> dict:
    """End-to-end data-starvation experiment for one registered dataset.

    Loads the dataset + embeddings through `src/` (download + cache, same path as
    dbaasp), computes the exact proximity graph and its communities ONCE, then
    builds `n_splits` community-based train/val/test splits from that one fixed
    structure (different `split_seed`s -- different communities land in
    train/val/test each time, but the communities themselves never change), then at each drop fraction downsamples the train set two ways (whole
    communities vs. the same number of random samples) and trains an MLP head,
    scoring the fixed test set with the dataset's metric.

    `singletons_are_com`: whether singleton communities count as droppable
    communities (and contribute to n_eff) alongside non-singleton ones. False
    (default) for datasets where singletons behave as outliers; True for
    datasets where singletons carry MORE per-sample generalization value than
    non-singleton clusters (empirically: lipophilicity, caco2_wang -- see
    `run_singleton_power_experiment`).

    `n_repeats` is spread evenly across the `n_splits` splits (n_repeats must be
    divisible by n_splits), so the reported mean/std reflect both split-to-split
    variance (different communities held out) and repeat-to-repeat variance
    (downsample + MLP-training randomness within a fixed split) instead of just
    the latter. Writes JSON and returns the results dict.
    """
    from rich import print

    if n_repeats % n_splits != 0:
        raise ValueError(f"n_repeats ({n_repeats}) must be divisible by n_splits ({n_splits})")
    repeats_per_split = n_repeats // n_splits
    split_seeds = [split_seed + 10 * i for i in range(n_splits)]

    drop_fractions = drop_fractions if drop_fractions is not None else list(DEFAULT_DROP_FRACTIONS)
    cfg = DATASETS[name]
    cache = CacheStore()

    print(f"[bold][orange2]=== Data Starvation: {name} ===[/][/]")
    data, labels = load_dataset(name, cache)
    print(f"  {len(data):,} samples")
    embs = compute_embeddings(name, data, cfg, cache)

    print("Computing exact proximity graph + communities (shared across all splits)...")
    communities_shared, graph_shared = build_exact_communities(
        name, data, threshold, gamma, cfg, cache, objective=objective,
    )
    n_com = len(set(communities_shared))
    print(f"  {n_com:,} communities over {len(communities_shared):,} samples")

    print(f"Building {n_splits} community-based train/val/test splits (seeds {split_seeds})...")
    splits = []
    for s_seed in split_seeds:
        split = split_from_communities(communities_shared, graph_shared, seed=s_seed)
        splits.append(split)
        print(f"  seed={s_seed}: train={len(split['train_idx'])}  val={len(split['val_idx'])}  test={len(split['test_idx'])}")

    results: dict = {
        "dataset": name,
        "n_train": int(np.mean([len(s["train_idx"]) for s in splits])),
        "n_val": int(np.mean([len(s["val_idx"]) for s in splits])),
        "n_test": int(np.mean([len(s["test_idx"]) for s in splits])),
        "drop_fractions": drop_fractions,
        "n_repeats": n_repeats,
        "n_splits": n_splits,
        "repeats_per_split": repeats_per_split,
        "split_seeds": split_seeds,
        "gamma": gamma,
        "threshold": threshold,
        "metric": cfg.metric,
        "objective": str(objective),
        "graph": "exact",
        "n_communities": n_com,
        "singletons_are_com": singletons_are_com,
        "by_drop_fraction": {},
    }

    for drop_frac in drop_fractions:
        print(f"\n[green]-- drop_frac={drop_frac:.0%} --[/]")
        community_scores, random_scores, n_dropped_list = [], [], []
        n_eff_community_list, n_eff_random_list = [], []
        for split_idx, split in enumerate(splits):
            communities = split["communities"]
            train_idx, val_idx, test_idx = split["train_idx"], split["val_idx"], split["test_idx"]
            for local_seed in range(repeats_per_split):
                seed = split_idx * repeats_per_split + local_seed
                r = run_starvation_repeat(
                    embs, labels, cfg.metric, communities,
                    train_idx, val_idx, test_idx, drop_frac, seed,
                    singletons_are_com=singletons_are_com,
                )
                community_scores.append(r["community_score"])
                random_scores.append(r["random_score"])
                n_dropped_list.append(r["n_dropped"])
                n_eff_community_list.append(r["n_eff_community"])
                n_eff_random_list.append(r["n_eff_random"])
                print(f"  split={split_idx} (seed={split_seeds[split_idx]}) repeat={local_seed}: dropped={r['n_dropped']}  "
                      f"community={r['community_score']:.4f} (n_eff={r['n_eff_community']})  "
                      f"random={r['random_score']:.4f} (n_eff={r['n_eff_random']})")

        community_arr = np.array(community_scores)
        random_arr = np.array(random_scores)
        results["by_drop_fraction"][f"{drop_frac}"] = {
            "n_dropped_mean": float(np.mean(n_dropped_list)),
            "n_eff_community_mean": float(np.mean(n_eff_community_list)),
            "n_eff_community_std": float(np.std(n_eff_community_list)),
            "n_eff_random_mean": float(np.mean(n_eff_random_list)),
            "n_eff_random_std": float(np.std(n_eff_random_list)),
            "community_score_mean": float(community_arr.mean()),
            "community_score_std": float(community_arr.std()),
            "random_score_mean": float(random_arr.mean()),
            "random_score_std": float(random_arr.std()),
        }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[bold]Results saved to {out_path}[/]")
    return results


def run_singleton_power_experiment(
    name: str,
    threshold: float,
    gamma: float,
    out_path: str | Path,
    *,
    split_seed: int = 7,
    n_splits: int = DEFAULT_N_SPLITS,
    n_repeats: int = DEFAULT_N_REPEATS,
    objective: LeidenObjective = LeidenObjective.CPM,
) -> dict:
    """Singleton-vs-non-singleton per-sample information content, for one dataset.

    This trains an MLP on ONLY the singleton subset of train, and separately
    on a size-matched random subsample of the non-singleton subset (matched
    so the comparison isolates per-sample information content rather than
    just favoring whichever subset has more data), evaluates both on the same
    fixed test set, across `n_splits` independent community splits x
    (n_repeats // n_splits) repeats each -- same structure as
    `run_starvation_experiment`. Reports mean/std and a paired t-test.
    Writes JSON, returns the results dict.
    """
    from collections import Counter

    from rich import print
    from scipy import stats

    if n_repeats % n_splits != 0:
        raise ValueError(f"n_repeats ({n_repeats}) must be divisible by n_splits ({n_splits})")
    repeats_per_split = n_repeats // n_splits
    split_seeds = [split_seed + 10 * i for i in range(n_splits)]

    cfg = DATASETS[name]
    cache = CacheStore()

    print(f"[bold][orange2]=== Singleton Power: {name} ===[/][/]")
    data, labels = load_dataset(name, cache)
    print(f"  {len(data):,} samples")
    embs = compute_embeddings(name, data, cfg, cache)

    print("Computing exact proximity graph + communities (shared across all splits)...")
    communities_shared, graph_shared = build_exact_communities(
        name, data, threshold, gamma, cfg, cache, objective=objective,
    )
    print(f"  {len(set(communities_shared)):,} communities over {len(communities_shared):,} samples")

    print(f"Building {n_splits} community-based train/val/test splits (seeds {split_seeds})...")
    single_scores, nonsingle_scores = [], []
    single_ns, nonsingle_ns_matched = [], []
    for split_idx, s_seed in enumerate(split_seeds):
        split = split_from_communities(communities_shared, graph_shared, seed=s_seed)
        communities = split["communities"]
        train_idx, val_idx, test_idx = split["train_idx"], split["val_idx"], split["test_idx"]
        sizes = Counter(communities[i] for i in train_idx)

        singleton_idx_full = [i for i in train_idx if sizes[communities[i]] == 1]
        nonsingleton_idx_full = [i for i in train_idx if sizes[communities[i]] >= 2]
        # Match both subsets down to the smaller of the two -- singletons
        # outnumber non-singletons for some datasets.
        n_match = min(len(singleton_idx_full), len(nonsingleton_idx_full))
        single_ns.append(n_match)
        print(f"  seed={s_seed}: train={len(train_idx)}  singleton={len(singleton_idx_full)}  "
              f"non-singleton={len(nonsingleton_idx_full)}  matched_to={n_match}")

        for local_seed in range(repeats_per_split):
            seed = split_idx * repeats_per_split + local_seed
            rng = np.random.default_rng(seed)
            singleton_idx = list(rng.choice(singleton_idx_full, size=n_match, replace=False)) \
                if len(singleton_idx_full) > n_match else singleton_idx_full
            nonsingleton_idx_matched = list(rng.choice(nonsingleton_idx_full, size=n_match, replace=False)) \
                if len(nonsingleton_idx_full) > n_match else nonsingleton_idx_full
            nonsingle_ns_matched.append(len(nonsingleton_idx_matched))

            r_single = train_eval_mlp(embs, labels, train_idx=singleton_idx, val_idx=val_idx,
                                       test_idx=test_idx, metric=cfg.metric, seed=seed)
            r_nonsingle = train_eval_mlp(embs, labels, train_idx=nonsingleton_idx_matched, val_idx=val_idx,
                                          test_idx=test_idx, metric=cfg.metric, seed=seed)
            single_scores.append(r_single["score"])
            nonsingle_scores.append(r_nonsingle["score"])
            print(f"  split={split_idx} (seed={s_seed}) repeat={local_seed}: "
                  f"singleton={r_single['score']:.4f}  non-singleton(matched)={r_nonsingle['score']:.4f}")

    single_arr = np.array(single_scores)
    nonsingle_arr = np.array(nonsingle_scores)
    diffs = single_arr - nonsingle_arr
    n = len(single_arr)
    if diffs.std() > 0:
        t_stat, p_value = stats.ttest_rel(single_arr, nonsingle_arr)
    else:
        t_stat, p_value = float("nan"), float("nan")

    results: dict = {
        "dataset": name,
        "threshold": threshold,
        "gamma": gamma,
        "objective": str(objective),
        "metric": cfg.metric,
        "n_splits": n_splits,
        "repeats_per_split": repeats_per_split,
        "n_repeats": n_repeats,
        "split_seeds": split_seeds,
        "n_singleton_train_mean": float(np.mean(single_ns)),
        "n_nonsingleton_train_matched_mean": float(np.mean(nonsingle_ns_matched)),
        "singleton_score_mean": float(single_arr.mean()),
        "singleton_score_std": float(single_arr.std(ddof=1)),
        "singleton_score_sem": float(single_arr.std(ddof=1) / np.sqrt(n)),
        "nonsingleton_score_mean": float(nonsingle_arr.mean()),
        "nonsingleton_score_std": float(nonsingle_arr.std(ddof=1)),
        "nonsingleton_score_sem": float(nonsingle_arr.std(ddof=1) / np.sqrt(n)),
        "diff_mean": float(diffs.mean()),
        "diff_std": float(diffs.std(ddof=1)),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value == p_value and p_value < 0.05),  # p_value != nan
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  singleton:            {results['singleton_score_mean']:.4f} +- {results['singleton_score_std']:.4f}")
    print(f"  non-singleton(matched): {results['nonsingleton_score_mean']:.4f} +- {results['nonsingleton_score_std']:.4f}")
    print(f"  diff={results['diff_mean']:+.4f}  t={results['t_stat']:+.2f}  p={results['p_value']:.4f}"
          f"  {'***SIGNIFICANT***' if results['significant'] else ''}")
    print(f"[bold]Results saved to {out_path}[/]")
    return results
