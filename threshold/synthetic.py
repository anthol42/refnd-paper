"""Synthetic 2D isotropic-Gaussian-mixture data for HNSW threshold recovery.

We sample a population of well-separated 2D isotropic Gaussian blobs on the
unit square. Each blob gets its own sample count and spread (sigma). Blob
centers are placed with a CONSTANT minimum gap of
`2 * sigma_multiplier * max(sigmas)` between every pair -- using the single
largest sigma across the whole population, not the pair's own two sigmas.

This matters: a per-pair gap (sigma_multiplier * (sigma_i + sigma_j)) lets a
high-sigma blob sit close to a low-sigma neighbor, and a point drawn from
that high-sigma blob's tail can then land closer to the NEIGHBOR's mean
than to its own -- i.e. its label ("blob i") is wrong by the very geometry
that generated it, independent of anything a clustering algorithm does.
Using the population's max sigma for every pair's required gap rules that
out everywhere at once: even the widest blob in the dataset can't reach a
neighbor closer than sigma_multiplier sigmas of its OWN spread, so no point
is ever, by construction, nearer to another blob's mean than its own.

This gives us a dataset with a known ground-truth clustering (one community
per Gaussian) and a known separation scale, so a threshold sweep on the
HNSW graph should recover an edge threshold around
~2*sigma_multiplier*max(sigma) (where within-blob pairs stay connected but
cross-blob pairs don't).

Usage:
    uv run python -m threshold.synthetic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rich import print

from refnd import exact_edges
from refnd.core import EdgeStore, HNSWState, INWeightType, LeidenObjective, find_communities
from refnd.kernels import KernelVariant, zip_kernel

from src.cache import CacheStore
from src.metrics import null_model_cdf

from threshold.fit import sweep_thresholds

OUT_DIR = Path(__file__).parent.parent / "results" / "thresholds"
GRAPH_CACHE_KEY = "threshold_sweep_synthetic"
MODALITY = KernelVariant.L2


def sample_synthetic_gaussians(
    n_gaussians: int = 1000,
    map_size: float = 1.0,
    n_samples_mean: float = 50.0,
    n_samples_std: float = 15.0,
    n_samples_min: int = 5,
    sigma_mean: float = 0.001,
    sigma_std: float = 0.0004,
    sigma_min: float = 0.0002,
    sigma_max: float = 0.0032,
    sigma_multiplier: float = 3.0,
    max_attempts: int = 500,
    seed: int = 0,
) -> dict:
    """Sample a mixture of isotropic 2D Gaussians on [0, map_size]^2.

    Every blob's sigma (spread) and sample count are drawn up front --
    sigma from N(sigma_mean, sigma_std) clipped to [sigma_min, sigma_max],
    count from N(n_samples_mean, n_samples_std) floored at n_samples_min.
    Centers are then reject-sampled one at a time, uniformly on the map,
    until each is >= 2 * sigma_multiplier * max(sigmas) away from every
    already-placed center. A blob that can't be placed within max_attempts
    is dropped.

    Returns a dict with:
      points:  (N, 2) float64 array of sampled points, all blobs concatenated
      labels:  (N,) int64 array, blob id each point belongs to
      means:   (n_placed, 2) float64 array of blob centers
      sigmas:  (n_placed,) float64 array of blob spreads
      counts:  (n_placed,) int64 array of per-blob sample counts
    """
    rng = np.random.default_rng(seed)

    sigmas_all = np.clip(rng.normal(sigma_mean, sigma_std, size=n_gaussians), sigma_min, sigma_max)
    counts_all = np.maximum(
        np.round(rng.normal(n_samples_mean, n_samples_std, size=n_gaussians)).astype(np.int64), n_samples_min,
    )
    min_gap = 2.0 * sigma_multiplier * sigmas_all.max()

    means = np.empty((n_gaussians, 2), dtype=np.float64)
    placed_mask = np.zeros(n_gaussians, dtype=bool)
    placed = 0

    for i in range(n_gaussians):
        for _ in range(max_attempts):
            center = rng.uniform(0.0, map_size, size=2)
            if placed == 0 or np.all(np.linalg.norm(means[:placed] - center, axis=1) >= min_gap):
                means[placed] = center
                placed_mask[i] = True
                placed += 1
                break
        else:
            print(f"  [dim]Warning: could not place blob {i} (map too full) -- skipping[/]")

    means = means[:placed]
    sigmas = sigmas_all[placed_mask]
    counts = counts_all[placed_mask]

    points = np.concatenate(
        [rng.normal(loc=means[i], scale=sigmas[i], size=(counts[i], 2)) for i in range(placed)],
        axis=0,
    )
    labels = np.concatenate([np.full(counts[i], i, dtype=np.int64) for i in range(placed)])

    print(
        f"  Placed {placed}/{n_gaussians} blobs, {len(points):,} points total "
        f"(sigma range [{sigmas.min():.4f}, {sigmas.max():.4f}])"
    )
    return {"points": points, "labels": labels, "means": means, "sigmas": sigmas, "counts": counts}


def split_train_prod_by_gaussian(
    data: dict, train_frac: float = 0.8, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split by WHOLE blobs (not points) into train/prod, 80/20 by default,
    then reorder points so all train points come first and all prod points
    come last -- the convention `sweep_thresholds` expects.

    Returns (points_reordered, labels_reordered, source) where source is 0
    for train nodes and 1 for prod nodes, in the same reordered node order.
    """
    points, labels = data["points"], data["labels"]
    n_gaussians = len(data["means"])
    rng = np.random.default_rng(seed)
    blob_ids = rng.permutation(n_gaussians)
    n_train_blobs = int(round(train_frac * n_gaussians))
    train_blobs = set(blob_ids[:n_train_blobs].tolist())

    is_train_point = np.array([lbl in train_blobs for lbl in labels])
    order = np.concatenate([np.nonzero(is_train_point)[0], np.nonzero(~is_train_point)[0]])
    points_reordered = points[order]
    labels_reordered = labels[order]
    source = np.concatenate([
        np.zeros(int(is_train_point.sum()), dtype=np.int8),
        np.ones(int((~is_train_point).sum()), dtype=np.int8),
    ])
    return points_reordered, labels_reordered, source


def build_hnsw(
    points: np.ndarray, cache: CacheStore, ef_construction: int = 64, cache_key_suffix: str = "",
) -> EdgeStore:
    """Build (or load, if cached) the layer-0 HNSW graph over all points.

    proximity_threshold is irrelevant here (get_layer(0) is unaffected by
    it), just a required HNSWState constructor arg -- see peptides.py.

    cache_key_suffix: the default cache key is only n_points + ef_construction, so
    two DIFFERENT point sets that happen to have the same size (e.g. several
    synthetic-data variants with the same blob/sample-count parameters but
    different seeds or spreads) silently collide and one gets served the
    other's stale cached graph. Pass something that distinguishes them
    (e.g. the seed, or a variant tag) whenever you build more than one
    differently-generated dataset of the same size in a single run.
    """
    cache_key = f"{GRAPH_CACHE_KEY}_n{len(points)}_ef{ef_construction}{cache_key_suffix}"
    cached = cache.get_edges(cache_key)
    if cached is not None:
        print(f"  [dim]Loading cached layer-0 graph: {cache_key}[/]")
        return cached

    print(f"  Building HNSW over {len(points):,} points...")
    hnsw = HNSWState(
        MODALITY, list(points),
        proximity_threshold=0.0,
        ef_construction=ef_construction,
        keep_all_edges=False,
        strict_ef=True,
    )
    hnsw.build(progress=True)
    es = hnsw.get_layer(0, weights=True, progress=True)
    print(f"  layer0: n_edges={len(es):,}")
    cache.store_edges(cache_key, es)
    return es


def build_exact_edges(
    points: np.ndarray, cache: CacheStore, proximity_threshold: float, cache_key_suffix: str = "",
) -> EdgeStore:
    """Brute-force (O(n^2)) exact edge set, all pairs with distance <=
    proximity_threshold.

    cache_key_suffix: the default cache key is only n_points + threshold, so
    two DIFFERENT point sets that happen to have the same size (e.g. several
    synthetic-data variants with the same blob/sample-count parameters but
    different seeds or spreads) silently collide and one gets served the
    other's stale cached graph. Pass something that distinguishes them
    (e.g. the seed, or a variant tag) whenever you build more than one
    differently-generated dataset of the same size in a single run.
    """
    cache_key = f"{GRAPH_CACHE_KEY}_exact_n{len(points)}_pt{proximity_threshold}{cache_key_suffix}"
    cached = cache.get_edges(cache_key)
    if cached is not None:
        print(f"  [dim]Loading cached exact graph: {cache_key}[/]")
        return cached

    print(f"  Computing exact edges over {len(points):,} points (brute force, pt={proximity_threshold})...")
    es = exact_edges(MODALITY, list(points), proximity_threshold=proximity_threshold, progress=True)
    print(f"  exact: n_edges={len(es):,}")
    cache.store_edges(cache_key, es)
    return es


def find_gamma_function_shuffled(points: np.ndarray, n_pairs: int = 2_000_000, seed: int = 42):
    """Null model built the same way as the other datasets (see
    `src.metrics.null_model_scores`, shuffle=True): sample random pairs from
    the full point set, element-shuffle each sampled point (swap its x/y
    coordinate -- same idea as permuting a sequence's characters), then
    score the shuffled pair's distance. p0(t) = P(shuffled distance <= t).
    """
    rng = np.random.default_rng(seed)
    n = len(points)
    idx_a = rng.integers(0, n, size=n_pairs)
    idx_b = rng.integers(0, n, size=n_pairs)

    def shuffle_point(p: np.ndarray) -> np.ndarray:
        p = p.copy()
        rng.shuffle(p)
        return p

    list_a = [shuffle_point(points[i]) for i in idx_a]
    list_b = [shuffle_point(points[i]) for i in idx_b]
    print(f"  Scoring {n_pairs:,} shuffled null pairs (L2)...")
    null_scores = np.asarray(
        zip_kernel(MODALITY, list_a, list_b, n_threads=0, progress=True),
        dtype=np.float64,
    )
    return null_model_cdf(null_scores)


def recover_partition_accuracy(
    hnsw_graph: EdgeStore, labels: np.ndarray, thresholds: np.ndarray, null_cdf,
    inweight_type: INWeightType = INWeightType.Distance,
) -> tuple[list[float], list[np.ndarray]]:
    """For each threshold, CPM-Leiden the tau-filtered graph (gamma from the
    null model) and score how well the recovered communities recover the
    TRUE blob partition via Adjusted Rand Index -- 1.0 means exact match (up
    to relabeling), 0.0 means chance level.

    Returns (ari_scores, communities_per_threshold).
    """
    from sklearn.metrics import adjusted_rand_score

    dist = np.fromiter((e[2] for e in hnsw_graph), dtype=np.float32, count=len(hnsw_graph))
    gammas = null_cdf(thresholds)

    ari_scores = []
    communities_per_threshold = []
    for tau, gamma in zip(thresholds, gammas):
        graph = hnsw_graph[dist <= tau].graph(inweight_type=inweight_type)
        communities = np.asarray(
            find_communities(graph, gamma=float(gamma), objective=LeidenObjective.CPM),
            dtype=np.int64,
        )
        ari = float(adjusted_rand_score(labels, communities))
        ari_scores.append(ari)
        communities_per_threshold.append(communities)
        print(f"  tau={tau:.5f}  gamma={gamma:.3e}  n_comm={int(communities.max()) + 1:,}  ARI={ari:.4f}")
    return ari_scores, communities_per_threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-gaussians", type=int, default=1000)
    parser.add_argument("--map-size", type=float, default=1.0)
    parser.add_argument("--n-samples-mean", type=float, default=50.0)
    parser.add_argument("--n-samples-std", type=float, default=15.0)
    parser.add_argument("--sigma-mean", type=float, default=0.0006)
    parser.add_argument("--sigma-std", type=float, default=0.0006)
    parser.add_argument("--sigma-min", type=float, default=0.00003)
    parser.add_argument("--sigma-max", type=float, default=0.01)
    parser.add_argument("--sigma-multiplier", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--thresh-lo", type=float, default=0.0)
    parser.add_argument("--thresh-hi", type=float, default=0.035)
    parser.add_argument("--n-sweep", type=int, default=25)
    parser.add_argument("--n-null-pairs", type=int, default=2_000_000)
    parser.add_argument("--ef-construction", type=int, default=64)
    args = parser.parse_args()

    print("[bold][orange2]=== Synthetic Gaussian mixture: threshold recovery ===[/][/]")
    data = sample_synthetic_gaussians(
        n_gaussians=args.n_gaussians,
        map_size=args.map_size,
        n_samples_mean=args.n_samples_mean,
        n_samples_std=args.n_samples_std,
        sigma_mean=args.sigma_mean,
        sigma_std=args.sigma_std,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma_multiplier=args.sigma_multiplier,
        seed=args.seed,
    )

    points, labels, source = split_train_prod_by_gaussian(data, train_frac=args.train_frac, seed=args.seed)
    train_nodes = np.nonzero(source == 0)[0]
    prod_nodes = np.nonzero(source == 1)[0]
    print(f"  train={len(train_nodes):,} points  prod={len(prod_nodes):,} points")

    cache = CacheStore()
    edge_graph = build_hnsw(points, cache, ef_construction=args.ef_construction)
    # edge_graph = build_exact_edges(points, cache, proximity_threshold=args.thresh_hi)

    print("  Computing gamma(tau) from a shuffled null...")
    gamma_cdf = find_gamma_function_shuffled(points, n_pairs=args.n_null_pairs, seed=args.seed)

    thresholds, split_results = sweep_thresholds(
        edge_graph, train_nodes, prod_nodes, gamma_cdf,
        args.thresh_lo, args.thresh_hi, args.n_sweep,
        inweight_type=INWeightType.SimilarityComplement,
    )

    print("\n  Scoring ground-truth partition recovery (ARI vs. true blob id)...")
    ari_scores, _ = recover_partition_accuracy(
        edge_graph, labels, thresholds, gamma_cdf,
        inweight_type=INWeightType.SimilarityComplement,
    )
    for res, ari in zip(split_results, ari_scores):
        res["ari"] = ari

    ks_stats = [r["ks_stat"] if r["ks_stat"] is not None else float("inf") for r in split_results]
    best_ks_idx = int(np.argmin(ks_stats))
    best_ari_idx = int(np.argmax(ari_scores))
    print(
        f"\n  Best by min KS: tau={float(thresholds[best_ks_idx]):.5f} "
        f"(KS={ks_stats[best_ks_idx]:.4f}, ARI={ari_scores[best_ks_idx]:.4f})"
    )
    print(f"  Best by ARI:    tau={float(thresholds[best_ari_idx]):.5f} (ARI={ari_scores[best_ari_idx]:.4f})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "synthetic.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "best_threshold_ks": float(thresholds[best_ks_idx]),
                "best_threshold_ari": float(thresholds[best_ari_idx]),
                "thresholds": thresholds.tolist(),
                "ari_scores": ari_scores,
                "ks_stats": ks_stats,
                "results": split_results,
            },
            f, indent=2,
        )
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
