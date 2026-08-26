"""Shared helpers for the data-starvation experiment.

Builds a community-based train/val/test split for a dataset, then offers two
ways to downsample the train set by a target fraction:

- community-based: drop whole communities at random until at least the
  target fraction of train samples has been removed (communities are
  indivisible, so this can overshoot slightly).
- random: drop exactly as many samples, chosen uniformly at random, as the
  community-based method actually dropped — so both methods remove the same
  number of samples for a fair comparison.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from refnd.core import HNSWState, INWeightType, LeidenObjective, find_communities, partition

from .datasets import DatasetConfig
from .mlp import train_eval_mlp


def build_community_split(
    data: list[Any],
    prox_threshold: float,
    gamma: float,
    cfg: DatasetConfig,
    *,
    ef_construction: int = 64,
    ef_init: int = 2,
    test_ratio: float = 0.2,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """Build an HNSW graph, find communities, and split into train/val/test.

    val is carved out of the train side using the same community-based
    `partition`, mirroring the outer test split (see `split_and_violations`
    in src/metrics.py).

    Returns a dict with keys: communities (list[int], one per full-dataset
    index), train_idx, val_idx, test_idx (lists of full-dataset indices).
    """
    hnsw = HNSWState(
        cfg.modality, data, proximity_threshold=prox_threshold,
        ef_construction=ef_construction, ef_init=ef_init, **cfg.kernel_params,
    )
    hnsw.build(progress=True)
    graph = hnsw.edges().graph(inweight_type=INWeightType.SimilarityComplement)
    communities = find_communities(graph, gamma=gamma, objective=LeidenObjective.CPM)

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
        "communities": list(communities),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


def community_downsample(
    train_idx: list[int],
    communities: list[int],
    drop_frac: float,
    seed: int,
) -> list[int]:
    """Drop whole communities (restricted to `train_idx`) at random until at
    least `drop_frac` of `train_idx` has been removed.

    Since communities can't be split, the actual dropped fraction may exceed
    `drop_frac` slightly. Returns the kept indices (a subset of train_idx).
    """
    rng = np.random.default_rng(seed)
    n_train = len(train_idx)
    target_drop = int(round(drop_frac * n_train))

    by_community: dict[int, list[int]] = {}
    for idx in train_idx:
        by_community.setdefault(communities[idx], []).append(idx)

    community_ids = list(by_community)
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
) -> dict:
    """One repeat of both downsampling methods at `drop_frac`, sharing the
    same number of dropped samples (community-based decides it first)."""
    community_train_idx, n_dropped = community_downsample(train_idx, communities, drop_frac, seed)
    random_train_idx = random_downsample(train_idx, n_dropped, seed)

    community_result = train_eval_mlp(
        embs, labels, train_idx=community_train_idx, val_idx=val_idx, test_idx=test_idx,
        metric=metric, seed=seed,
    )
    random_result = train_eval_mlp(
        embs, labels, train_idx=random_train_idx, val_idx=val_idx, test_idx=test_idx,
        metric=metric, seed=seed,
    )

    # n_eff counts only non-singleton communities: singletons dominate the raw
    # community count and do not contribute to final accuracy (They are outliers)
    from collections import Counter
    train_sizes = Counter(communities[i] for i in train_idx)
    non_singleton_ids = {cid for cid, sz in train_sizes.items() if sz > 1}
    n_eff_community = len({communities[i] for i in community_train_idx} & non_singleton_ids)
    n_eff_random = len({communities[i] for i in random_train_idx} & non_singleton_ids)

    return {
        "n_dropped": n_dropped,
        "n_train_community": len(community_train_idx),
        "n_train_random": len(random_train_idx),
        "n_eff_community": n_eff_community,
        "n_eff_random": n_eff_random,
        "community_score": community_result["score"],
        "random_score": random_result["score"],
    }
