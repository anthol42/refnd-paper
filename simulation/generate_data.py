"""Precompute a threshold sweep (communities, single-linkage train/prod
distance distributions, ARI, KS) across several sigma_std levels and dump it
as a self-contained JS data file for simulation/index.html to slide through.

Usage:
    uv run python -m simulation.generate_data
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rich import print
from sklearn.metrics import adjusted_rand_score
from scipy.stats import ks_2samp

from refnd.core import EdgeStore, INWeightType, LeidenObjective, find_communities

from src.cache import CacheStore
from threshold.synthetic import (
    sample_synthetic_gaussians, split_train_prod_by_gaussian, build_hnsw,
    find_gamma_function_shuffled,
)

OUT_DIR = Path(__file__).parent.parent / ".cache" / "figs"
N_SWEEP = 25
THRESH_HI = 0.05
# The min blob-center gap is now CONSTANT: 2*SIGMA_MULTIPLIER*max(sigmas across the whole population)
# (see threshold/synthetic.py's sample_synthetic_gaussians docstring for why per-pair gaps were wrong).
# That means the gap only grows with the SINGLE largest sigma sampled -- at N_GAUSSIANS=1000 a tight
# SIGMA_MAX clip was needed to keep packing feasible, but that clip SATURATED at sigma_std >= 0.0004
# (the sampled tail routinely exceeded it), so every higher-std variant silently had the SAME max
# sigma and looked identical. N_GAUSSIANS=300 (with a loose SIGMA_MAX that's never actually reached
# in this range) keeps packing feasible while max(sigma) genuinely keeps growing with sigma_std.
N_GAUSSIANS = 1000
SIGMA_MULTIPLIER = 5.0
SIGMA_MEAN = 0.0006
SIGMA_MIN = 0.00003
SIGMA_MAX = 0.01
SIGMA_STD_VALUES = [0.0, 0.0002, 0.0004, 0.0006, 0.0008, 0.0010]  # slider steps
DEFAULT_SIGMA_STD = 0.0004  # shown on page load
VIZ_N_POINTS = 8000   # subsample for browser rendering (x len(SIGMA_STD_VALUES) variants); stats use FULL data
N_NULL_PAIRS = 500_000


def plot_synthetic_gaussians(data: dict, out_path: Path, map_size: float = 1.0, sigma_multiplier: float = 3.0) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    points, labels, means, sigmas = data["points"], data["labels"], data["means"], data["sigmas"]
    n_gaussians = len(means)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="hsv", s=3, alpha=0.6, linewidths=0)
    for mean, sigma in zip(means, sigmas):
        ax.add_patch(Circle(mean, sigma_multiplier * sigma, fill=False, edgecolor="black", linewidth=0.3, alpha=0.4))
    ax.scatter(means[:, 0], means[:, 1], c="black", s=2, marker="x", linewidths=0.5)

    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_aspect("equal")
    ax.set_title(
        f"Synthetic Gaussian mixture: {n_gaussians} blobs, {len(points):,} points\n"
        f"(circles = {sigma_multiplier:g}-sigma radius)"
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved figure: {out_path}")


def min_dist_to_train_arrays(
    communities: np.ndarray, source: np.ndarray, src: np.ndarray, dst: np.ndarray, dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """For each PURE group (train-only or prod-only, by `communities`), the
    distance to its nearest OTHER train group -- exactly `fit.run_threshold`'s
    train_train/train_prod construction, generalized so it can be called
    either with Leiden's inferred communities or with the TRUE blob labels
    (the "oracle" -- what the same statistic looks like if partition
    recovery were perfect, since it's the ground truth, not inferred).
    """
    n_comm = int(communities.max()) + 1
    is_train_node = source == 0
    train_counts = np.bincount(communities[is_train_node], minlength=n_comm)
    prod_counts = np.bincount(communities[~is_train_node], minlength=n_comm)
    pure_train_mask = (prod_counts == 0) & (train_counts > 0)
    pure_prod_mask = (train_counts == 0) & (prod_counts > 0)

    c_src, c_dst = communities[src], communities[dst]
    min_dist_to_train = np.full(n_comm, np.inf, dtype=np.float32)
    mask_a = is_train_node[dst] & (c_src != c_dst)
    mask_b = is_train_node[src] & (c_src != c_dst)
    if mask_a.any():
        np.minimum.at(min_dist_to_train, c_src[mask_a], dist[mask_a])
    if mask_b.any():
        np.minimum.at(min_dist_to_train, c_dst[mask_b], dist[mask_b])

    pure_train_ids = np.nonzero(pure_train_mask)[0]
    pure_prod_ids = np.nonzero(pure_prod_mask)[0]
    tt_vals = min_dist_to_train[pure_train_ids]
    tp_vals = min_dist_to_train[pure_prod_ids]
    return tt_vals[np.isfinite(tt_vals)], tp_vals[np.isfinite(tp_vals)]


def compute_frame(
    tau: float, gamma: float, labels: np.ndarray, source: np.ndarray,
    edge_graph: EdgeStore, src: np.ndarray, dst: np.ndarray, dist: np.ndarray,
    inweight_type: INWeightType,
) -> tuple[np.ndarray, dict]:
    """Mirrors `threshold.fit.run_threshold`, but also keeps the raw
    per-community min-distance-to-train arrays (not just their summary
    stats) so the browser can render an actual histogram."""
    mask = dist <= tau
    graph = edge_graph[mask].graph(inweight_type=inweight_type)
    communities = np.asarray(find_communities(graph, gamma=gamma, objective=LeidenObjective.CPM), dtype=np.int64)
    n_comm = int(communities.max()) + 1

    is_train_node = source == 0
    train_counts = np.bincount(communities[is_train_node], minlength=n_comm)
    prod_counts = np.bincount(communities[~is_train_node], minlength=n_comm)
    pure_train_mask = (prod_counts == 0) & (train_counts > 0)
    pure_prod_mask = (train_counts == 0) & (prod_counts > 0)
    hybrid_mask = ~pure_train_mask & ~pure_prod_mask

    tt_finite, tp_finite = min_dist_to_train_arrays(communities, source, src, dst, dist)

    if len(tt_finite) > 1 and len(tp_finite) > 1:
        ks_stat, ks_p = ks_2samp(tt_finite, tp_finite)
        ks_stat, ks_p = float(ks_stat), float(ks_p)
    else:
        ks_stat, ks_p = None, None

    ari = float(adjusted_rand_score(labels, communities))

    stats = {
        "tau": float(tau), "gamma": float(gamma), "n_comm": n_comm,
        "n_pure_train": int(pure_train_mask.sum()), "n_pure_prod": int(pure_prod_mask.sum()),
        "n_hybrid": int(hybrid_mask.sum()),
        "ks_stat": ks_stat, "ks_pvalue": ks_p, "ari": ari,
        "train_train_dist": [round(float(x), 6) for x in tt_finite.tolist()],
        "train_prod_dist": [round(float(x), 6) for x in tp_finite.tolist()],
    }
    return communities, stats


def build_variant(sigma_std: float, seed: int) -> dict:
    print(f"\n[bold][orange2]--- sigma_std={sigma_std:g} ---[/][/]")
    data = sample_synthetic_gaussians(
        n_gaussians=N_GAUSSIANS, seed=seed, sigma_multiplier=SIGMA_MULTIPLIER,
        sigma_mean=SIGMA_MEAN, sigma_std=sigma_std, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX,
    )
    n_placed = len(data["sigmas"])
    print(f"  Placed {n_placed}/{N_GAUSSIANS} blobs (constant gap = 2*{SIGMA_MULTIPLIER:g}*max(sigma) -- a blob that can't fit is dropped)")
    print(f"  sigma range: [{data['sigmas'].min():.5f}, {data['sigmas'].max():.5f}]")
    sigma_std_tag = f"{sigma_std:g}".replace(".", "p")
    plot_synthetic_gaussians(
        data, OUT_DIR / f"blobs_sigma_std_{sigma_std_tag}.png",
        sigma_multiplier=SIGMA_MULTIPLIER,
    )
    points, labels, source = split_train_prod_by_gaussian(data, train_frac=0.8, seed=seed)

    cache = CacheStore()
    # edge_graph = build_exact_edges(
    #     points, cache, proximity_threshold=THRESH_HI, cache_key_suffix=f"_sigmastd{sigma_std:g}_seed{seed}",
    # )
    edge_graph = build_hnsw(
        points, cache, cache_key_suffix=f"_sigmastd{sigma_std:g}_seed{seed}",
    )
    gamma_cdf = find_gamma_function_shuffled(points, n_pairs=N_NULL_PAIRS, seed=seed)

    dist = np.fromiter((e[2] for e in edge_graph), dtype=np.float32, count=len(edge_graph))
    src = np.fromiter((e[0] for e in edge_graph), dtype=np.uint32, count=len(edge_graph))
    dst = np.fromiter((e[1] for e in edge_graph), dtype=np.uint32, count=len(edge_graph))

    thresholds = np.linspace(0.0, THRESH_HI, N_SWEEP)
    gammas = gamma_cdf(thresholds)
    inweight_type = INWeightType.SimilarityComplement

    # Oracle: the SAME train_train/train_prod statistic, but using the TRUE
    # blob labels as the "communities" -- independent of tau/gamma/Leiden,
    # since it's ground truth, not inferred. Reference for how well the
    # inferred (tau-dependent) distributions could ever match.
    oracle_tt, oracle_tp = min_dist_to_train_arrays(labels, source, src, dst, dist)
    print(f"  Oracle (true blobs): n_train_blobs={len(oracle_tt)}  n_prod_blobs={len(oracle_tp)}")
    if len(oracle_tt) > 1 and len(oracle_tp) > 1:
        oracle_ks_stat, oracle_ks_p = ks_2samp(oracle_tt, oracle_tp)
        print(f"  Oracle KS: stat={oracle_ks_stat:.4f}  p={oracle_ks_p:.4f}")

    rng = np.random.default_rng(seed)
    n = len(points)
    viz_n = min(VIZ_N_POINTS, n)
    viz_idx = np.sort(rng.choice(n, size=viz_n, replace=False))
    viz_points = points[viz_idx]
    viz_labels = labels[viz_idx]
    viz_source = source[viz_idx]

    frames_communities = []
    frames_stats = []
    for tau, gamma in zip(thresholds, gammas):
        communities, stats = compute_frame(
            float(tau), float(gamma), labels, source, edge_graph, src, dst, dist, inweight_type,
        )
        frames_communities.append(communities[viz_idx].tolist())
        frames_stats.append(stats)
        print(f"  tau={tau:.5f}  n_comm={stats['n_comm']:,}  ARI={stats['ari']:.4f}  KS={stats['ks_stat']}  p={stats['ks_pvalue']}")

    best_ari_idx = int(np.argmax([s["ari"] for s in frames_stats]))
    ks_vals = [s["ks_stat"] if s["ks_stat"] is not None else float("inf") for s in frames_stats]
    best_ks_idx = int(np.argmin(ks_vals))
    p_vals = [s["ks_pvalue"] if s["ks_pvalue"] is not None else -1.0 for s in frames_stats]
    best_p_idx = int(np.argmax(p_vals))

    return {
        "sigma_std": sigma_std,
        "thresholds": [float(t) for t in thresholds],
        "points": viz_points.round(6).tolist(),
        "true_labels": viz_labels.tolist(),
        "source": viz_source.tolist(),
        "frames_communities": frames_communities,
        "frames_stats": frames_stats,
        "oracle_train_train_dist": [round(float(x), 6) for x in oracle_tt.tolist()],
        "oracle_train_prod_dist": [round(float(x), 6) for x in oracle_tp.tolist()],
        "markers": {
            "best_ari_idx": best_ari_idx, "best_ks_idx": best_ks_idx, "best_pvalue_idx": best_p_idx,
        },
        "meta": {
            "n_gaussians": N_GAUSSIANS, "n_blobs_placed": n_placed, "sigma_multiplier": SIGMA_MULTIPLIER,
            "sigma_min": float(data["sigmas"].min()), "sigma_max": float(data["sigmas"].max()),
            "n_points_total": n, "n_points_viz": viz_n,
        },
    }


def main() -> None:
    print("[bold][orange2]=== Building simulation data (sweep over sigma_std) ===[/][/]")
    variants = [build_variant(sigma_std, seed=0) for sigma_std in SIGMA_STD_VALUES]
    default_idx = SIGMA_STD_VALUES.index(DEFAULT_SIGMA_STD)

    payload = {"sigma_std_values": SIGMA_STD_VALUES, "default_idx": default_idx, "variants": variants}

    out_path = OUT_DIR / "data.js"
    with open(out_path, "w") as f:
        f.write("const VIZ_DATA = ")
        json.dump(payload, f, separators=(",", ":"))
        f.write(";\n")
    print(f"\n  Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
