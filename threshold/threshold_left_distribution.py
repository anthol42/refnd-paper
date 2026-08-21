"""Scratch analysis of the distribution to the LEFT of the proximity threshold.

Step 1: plot the nearest-neighbor distance distribution B(y) — for each
production item y (PeptideAtlas), the distance to its nearest neighbor in
train (DBAASP) — reusing the cached ANN B(y) array from the main pipeline.
Threshold = 0.5 (DATASETS["dbaasp"].proximity_threshold == DATASETS["peptide_atlas"]
.proximity_threshold).

Usage:
    uv run python -m threshold.threshold_left_distribution
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from rich import print

from refnd.core import (
    EdgeStore,
    HNSWState,
    INWeightType,
    LeidenObjective,
    find_communities,
    partition,
)

from src.cache import CacheStore
from src.datasets import DATASETS, load_dataset
from src.metrics import null_model_scores
from threshold import prod_self_null as psn

PLOT_DIR = Path(__file__).parent
CACHE_DIR = PLOT_DIR / "cache"
THRESHOLD = 0.5

SPLIT_THRESHOLD = 0.5
TESTED_THRESHOLDS = [0.2, 0.4, 0.5, 0.55, 0.6]
SEED = 42
TEST_RATIO = 0.2
ATLAS_SUBSAMPLE_N = 1_000_000
EF_CONSTRUCTION = 64
EF_INIT = 1
SEARCH_EF = 64
N_BINS = 50
N_NULL_PAIRS_DBAASP = 5_000_000
DBAASP_NULL_PATH = CACHE_DIR / f"null_dbaasp_self_n{N_NULL_PAIRS_DBAASP}_seed{SEED}.npy"


def plot_b_distribution(B: np.ndarray, threshold: float, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frac_left = float(np.mean(B <= threshold))

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.hist(B, bins=200, color="#009E73", alpha=0.85, label="B(y) — dist. to nearest DBAASP neighbor")
    ax.axvline(threshold, color="#D55E00", linewidth=1.4, linestyle="--",
               label=f"threshold = {threshold:g}  (frac ≤ threshold = {frac_left:.4f})")
    ax.set_xlabel(r"$B(y) = \min_x d(y, x)$")
    ax.set_ylabel("count")
    ax.set_title(f"PeptideAtlas → DBAASP nearest-neighbor distance  (n={len(B):,})", loc="left")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_stacked(B_atlas: np.ndarray, B_test: np.ndarray, bins: np.ndarray, out_path: Path) -> None:
    """Both histograms on one figure, two subplots sharing the same x-axis: atlas on top, test on bottom."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 8), dpi=150, sharex=True)
    fig.patch.set_facecolor("white")

    ax_top.hist(B_atlas, bins=bins, color="#0072B2", alpha=0.85,
                label=f"PeptideAtlas → train (n={len(B_atlas):,})")
    ax_top.set_ylabel("count")
    ax_top.set_title(f"PeptideAtlas subsample → DBAASP-train NN distance (n={len(B_atlas):,})", loc="left")
    ax_top.legend(loc="upper right", fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax_top.spines[s].set_visible(False)

    ax_bot.hist(B_test, bins=bins, color="#D55E00", alpha=0.85,
                label=f"test → train (split, n={len(B_test):,})")
    ax_bot.set_xlabel(r"nearest-neighbor distance to DBAASP-train")
    ax_bot.set_ylabel("count")
    ax_bot.set_title(f"DBAASP split @ {SPLIT_THRESHOLD:g}: test → train NN distance (n={len(B_test):,})", loc="left")
    ax_bot.legend(loc="upper right", fontsize=9, frameon=False)
    ax_bot.set_xlim(bins[0], bins[-1])
    for s in ("top", "right"):
        ax_bot.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_overlay(B_test: np.ndarray, B_atlas: np.ndarray, bins: np.ndarray, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.hist(B_test, bins=bins, density=True, color="#D55E00", alpha=0.33,
            label=f"test → train (split, n={len(B_test):,})")
    ax.hist(B_atlas, bins=bins, density=True, color="#0072B2", alpha=0.33,
            label=f"PeptideAtlas → train (n={len(B_atlas):,})")
    ax.set_xlabel(r"nearest-neighbor distance to DBAASP-train")
    ax.set_ylabel("density")
    ax.set_title("DBAASP community-split test set vs. PeptideAtlas: NN distance to train", loc="left")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_xlim(bins[0], bins[-1])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


def run_split(threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (B_atlas, B_test): NN distance to DBAASP-train for a PeptideAtlas subsample and
    for the community-split DBAASP test set, at the given split threshold. Cached end-to-end."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    b_atlas_path = CACHE_DIR / f"B_atlas_thr{threshold:g}_n{ATLAS_SUBSAMPLE_N}_seed{SEED}.npy"
    b_test_path  = CACHE_DIR / f"B_test_thr{threshold:g}_seed{SEED}.npy"
    if b_atlas_path.exists() and b_test_path.exists():
        print(f"  [dim]Loading cached B arrays for threshold={threshold:g}[/]")
        return np.load(b_atlas_path), np.load(b_test_path)

    print(f"\n[bold][orange2]=== DBAASP split @ threshold={threshold:g} vs. PeptideAtlas ===[/][/]")
    cache = CacheStore()
    cfg = DATASETS["dbaasp"]

    dbaasp, _ = load_dataset("dbaasp", cache)
    atlas, _ = load_dataset("peptide_atlas", cache)
    print(f"  dbaasp: {len(dbaasp):,}   peptide_atlas: {len(atlas):,}")

    # 1. HNSW graph over all of DBAASP (approximate, same as the real pipeline uses for
    # splitting -- NOT exact_edges, which is only ever a recall reference in hyperparameters.py).
    split_edges_path = CACHE_DIR / f"dbaasp_hnsw_split_thr{threshold:g}_efc{EF_CONSTRUCTION}.edgestr"
    if split_edges_path.exists():
        print(f"  [dim]Loading cached {split_edges_path.name}[/]")
        split_es = EdgeStore.load(str(split_edges_path))
    else:
        print(f"  Building HNSW @ {threshold:g} over full DBAASP...")
        split_hnsw = HNSWState(
            cfg.modality, dbaasp,
            proximity_threshold=threshold,
            ef_construction=EF_CONSTRUCTION,
            ef_init=EF_INIT,
            **cfg.kernel_params,
        )
        split_hnsw.build(progress=True)
        split_es = split_hnsw.edges()
        split_es.save(str(split_edges_path))
    print(f"  n_edges = {len(split_es.edges()):,}")
    graph = split_es.graph(inweight_type=INWeightType.Distance)

    # 2. gamma = p0 = P(unrelated atlas pair distance <= threshold), from the already-cached
    # prod-self null (real, unmodified atlas pairs, n=10M, seed=42).
    null_scores = np.load(psn.PEPTIDES_NULL_PROD_PATH)
    gamma = float(np.mean(null_scores <= threshold))
    print(f"  gamma = p0(<= {threshold:g}) = {gamma:.6f}  (from {len(null_scores):,} atlas self-pairs)")

    # 3. CPM communities + split.
    communities = find_communities(graph, gamma=gamma, objective=LeidenObjective.CPM)
    print(f"  {len(set(communities)):,} communities")
    train_idx, test_idx = partition(communities, graph, test_ratio=TEST_RATIO, seed=SEED, post_filtering=False)
    train_idx, test_idx = list(train_idx), list(test_idx)
    print(f"  train={len(train_idx):,}  test={len(test_idx):,}")

    train_data = [dbaasp[i] for i in train_idx]
    test_data  = [dbaasp[i] for i in test_idx]

    # 4. HNSW over train only, search test (DBAASP holdout) and an atlas subsample against it.
    hnsw = HNSWState(
        cfg.modality, train_data,
        proximity_threshold=cfg.proximity_threshold,
        ef_construction=EF_CONSTRUCTION,
        strict_ef=True,
        keep_all_edges=False,
        **cfg.kernel_params,
    )
    hnsw.build(progress=True)

    print("  Searching test (DBAASP holdout) -> train...")
    test_results = hnsw.search(test_data, k=1, ef=SEARCH_EF, threads=0, progress=True)
    B_test = np.array([hits[0][1] for hits in test_results], dtype=np.float64)

    rng = np.random.default_rng(SEED)
    atlas_idx = rng.choice(len(atlas), size=ATLAS_SUBSAMPLE_N, replace=False)
    atlas_subsample = [atlas[i] for i in atlas_idx]
    print(f"  Searching PeptideAtlas subsample (n={ATLAS_SUBSAMPLE_N:,}) -> train...")
    atlas_results = hnsw.search(atlas_subsample, k=1, ef=SEARCH_EF, threads=0, progress=True)
    B_atlas = np.array([hits[0][1] for hits in atlas_results], dtype=np.float64)

    print(f"  B_test:  min={B_test.min():.4f}  median={np.median(B_test):.4f}  max={B_test.max():.4f}")
    print(f"  B_atlas: min={B_atlas.min():.4f}  median={np.median(B_atlas):.4f}  max={B_atlas.max():.4f}")

    np.save(b_atlas_path, B_atlas)
    np.save(b_test_path, B_test)
    return B_atlas, B_test


def split_and_compare(threshold: float = SPLIT_THRESHOLD) -> None:
    B_atlas, B_test = run_split(threshold)

    lo = float(min(B_test.min(), B_atlas.min()))
    hi = float(max(B_test.max(), B_atlas.max()))
    bins = np.linspace(lo, hi, N_BINS + 1)

    plot_stacked(B_atlas, B_test, bins, PLOT_DIR / "left_dist_split_stacked.png")
    plot_overlay(B_test, B_atlas, bins, PLOT_DIR / "left_dist_split_overlay.png")


def ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def plot_ecdf_compare(results: dict[float, tuple[np.ndarray, np.ndarray]], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 5.5), dpi=150, sharey=True)
    fig.patch.set_facecolor("white")
    if len(results) == 1:
        axes = [axes]

    for ax, (threshold, (B_atlas, B_test)) in zip(axes, results.items()):
        x_a, y_a = ecdf(B_atlas)
        x_t, y_t = ecdf(B_test)
        frac_a = float(np.mean(B_atlas <= threshold))
        frac_t = float(np.mean(B_test <= threshold))

        ax.plot(x_a, y_a, color="#0072B2", linewidth=2.0, label=f"PeptideAtlas → train (left={frac_a:.3f})")
        ax.plot(x_t, y_t, color="#D55E00", linewidth=2.0, label=f"test → train (left={frac_t:.3f})")
        ax.axvline(threshold, color="#6b7280", linewidth=1.2, linestyle="--", label=f"threshold={threshold:g}")
        ax.set_xlabel("NN distance to DBAASP-train")
        ax.set_title(f"threshold = {threshold:g}", loc="left")
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax.set_xlim(0.0, 0.85)
        ax.set_ylim(-0.02, 1.02)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel("ECDF")
    fig.suptitle("ECDF of NN distance to DBAASP-train: PeptideAtlas vs. community-split test set", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def compare_thresholds() -> None:
    print(f"\n[bold][orange2]=== ECDF comparison across thresholds {TESTED_THRESHOLDS} ===[/][/]")
    results: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for threshold in TESTED_THRESHOLDS:
        B_atlas, B_test = run_split(threshold)
        results[threshold] = (B_atlas, B_test)

    print("\n  [bold]Proportion of NN distances <= threshold (\"left of threshold\"):[/]")
    print(f"  {'threshold':>10}  {'atlas→train':>12}  {'test→train':>12}")
    for threshold, (B_atlas, B_test) in results.items():
        frac_a = float(np.mean(B_atlas <= threshold))
        frac_t = float(np.mean(B_test <= threshold))
        print(f"  {threshold:>10g}  {frac_a:>12.4f}  {frac_t:>12.4f}")

    plot_ecdf_compare(results, PLOT_DIR / "left_dist_ecdf_compare.png")


def dbaasp_self_null_scores() -> np.ndarray:
    """Real, unmodified random pairs sampled directly from DBAASP itself (shuffle=False) --
    same "prod-self null" methodology as threshold/prod_self_null.py, but with DBAASP as its
    own source instead of PeptideAtlas. Tests whether DBAASP's own internal background
    collision rate is higher than the PeptideAtlas-derived p0 used elsewhere in this file."""
    if DBAASP_NULL_PATH.exists():
        print(f"  [dim]Loading cached {DBAASP_NULL_PATH.name}[/]")
        return np.load(DBAASP_NULL_PATH)
    cache = CacheStore()
    cfg = DATASETS["dbaasp"]
    dbaasp, _ = load_dataset("dbaasp", cache)
    scores = null_model_scores(dbaasp, cfg, n_samples=N_NULL_PAIRS_DBAASP, seed=SEED, shuffle=False)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DBAASP_NULL_PATH, scores)
    return scores


def compute_expected_cross_community_edges(
    threshold: float, p0_source: str = "atlas", n_iterations: int = 2,
) -> dict:
    """Null-model expected number of cross-community edges (distance <= threshold) for the
    CPM community partition of DBAASP at this split threshold.

    For each unordered pair of distinct communities (A, B), the number of possible edges
    between them is n_A * n_B (every node in A against every node in B), and under the null
    model each such pair independently lands <= threshold with probability p0. So the expected
    count for that pair is p0 * n_A * n_B, and summed over all unordered community pairs:
    p0 * sum_{A<B} n_A * n_B = p0 * (N^2 - sum_A n_A^2) / 2, where N = sum_A n_A.
    Every edge actually stored in the split graph already satisfies distance <= threshold (by
    construction of the threshold graph), so "expected/actual edges to the left of the
    threshold" here just means expected/actual cross-community edges in that graph.

    p0_source="atlas" (default, matches the rest of this file): p0 from PeptideAtlas self-pairs,
    also used as gamma (CPM resolution) -- i.e. the same p0/gamma coupling used to build the
    split in run_split(). p0_source="dbaasp": communities and split graph are unchanged (both
    come from the "atlas" split graph, since that's the graph actually used everywhere else),
    but p0 for the expected-edge formula is instead DBAASP's own internal self-pair rate.
    """
    split_edges_path = CACHE_DIR / f"dbaasp_hnsw_split_thr{threshold:g}_efc{EF_CONSTRUCTION}.edgestr"
    if not split_edges_path.exists():
        raise FileNotFoundError(f"No cached split graph for threshold={threshold:g}; run run_split first.")
    split_es = EdgeStore.load(str(split_edges_path))
    graph = split_es.graph(inweight_type=INWeightType.Distance)

    atlas_null_scores = np.load(psn.PEPTIDES_NULL_PROD_PATH)
    gamma = float(np.mean(atlas_null_scores <= threshold))

    if p0_source == "atlas":
        p0 = gamma
    elif p0_source == "dbaasp":
        dbaasp_null_scores = dbaasp_self_null_scores()
        p0 = float(np.mean(dbaasp_null_scores <= threshold))
    else:
        raise ValueError(f"p0_source must be 'atlas' or 'dbaasp', got {p0_source!r}")

    # CPM communities are always detected with gamma = atlas p0 (matches run_split()); only the
    # null probability used in the *expected* formula below changes with p0_source.
    communities = find_communities(graph, gamma=gamma, objective=LeidenObjective.CPM, n_iterations=n_iterations)
    sizes = np.array(list(Counter(communities).values()), dtype=np.float64)
    n_total = float(sizes.sum())

    expected_cross = float(p0 * (n_total ** 2 - float(np.sum(sizes ** 2))) / 2.0)
    edges = split_es.edges()
    actual_cross = sum(1 for s, d, _ in edges if communities[s] != communities[d])

    return {
        "threshold": threshold,
        "p0_source": p0_source,
        "n_iterations": n_iterations,
        "p0": p0,
        "n_communities": len(sizes),
        "expected_cross_edges": expected_cross,
        "actual_cross_edges": actual_cross,
        "actual_total_edges": len(edges),
    }


def report_expected_cross_edges(thresholds: list[float], p0_source: str = "atlas", n_iterations: int = 2) -> None:
    print(f"\n[bold][orange2]=== Null-model expected cross-community edges "
          f"(p0 from {p0_source}, n_iterations={n_iterations}) ===[/][/]")
    rows = [compute_expected_cross_community_edges(t, p0_source=p0_source, n_iterations=n_iterations)
            for t in thresholds]
    print(f"  {'threshold':>10}  {'p0':>10}  {'n_comm':>8}  {'expected_cross':>15}  "
          f"{'actual_cross':>13}  {'actual_total':>13}")
    for r in rows:
        print(f"  {r['threshold']:>10g}  {r['p0']:>10.6f}  {r['n_communities']:>8,}  "
              f"{r['expected_cross_edges']:>15,.2f}  {r['actual_cross_edges']:>13,}  "
              f"{r['actual_total_edges']:>13,}")


def find_high_cross_community_pairs(
    threshold: float, n_iterations: int = 20, top_n: int = 20,
) -> None:
    """For a given split threshold, find community pairs (A, B) whose RAW cross-edge count
    exceeds the null expectation p0*n_A*n_B, and check whether CPM actually "should" have
    merged them.

    CPM's Leiden objective does NOT compare against a raw edge *count*: it compares the sum of
    edge *weights* between A and B to gamma*n_A*n_B (merging A,B is only an improving move when
    that weighted sum exceeds gamma*n_A*n_B). The graph here uses INWeightType.Distance, which
    maps a raw distance d to weight 1/(1+d) -- strictly less than 1 for every real edge (d>0).
    So a pair can have MORE raw edges than the count-based null predicts while still having a
    weighted sum below gamma*n_A*n_B (because those edges sit close to the threshold, at low
    weight) -- in which case CPM is correctly leaving them unmerged, and the "excess" we found
    with raw counts was never a real signal for Leiden to act on. If the weighted sum ALSO
    exceeds gamma*n_A*n_B, merging really would improve the CPM objective, and Leiden simply
    failed to find that move (a genuine local-optimum miss).
    """
    split_edges_path = CACHE_DIR / f"dbaasp_hnsw_split_thr{threshold:g}_efc{EF_CONSTRUCTION}.edgestr"
    if not split_edges_path.exists():
        raise FileNotFoundError(f"No cached split graph for threshold={threshold:g}; run run_split first.")
    split_es = EdgeStore.load(str(split_edges_path))
    graph = split_es.graph(inweight_type=INWeightType.Distance)

    atlas_null_scores = np.load(psn.PEPTIDES_NULL_PROD_PATH)
    gamma = float(np.mean(atlas_null_scores <= threshold))

    communities = find_communities(graph, gamma=gamma, objective=LeidenObjective.CPM, n_iterations=n_iterations)
    sizes = Counter(communities)
    print(f"\n[bold][orange2]=== High cross-community pairs @ threshold={threshold:g} "
          f"(gamma=p0={gamma:.6f}, n_iterations={n_iterations}, {len(sizes):,} communities) ===[/][/]")

    pair_dists: dict[tuple[int, int], list[float]] = {}
    for s, d, w in split_es.edges():
        ca, cb = communities[s], communities[d]
        if ca == cb:
            continue
        key = (ca, cb) if ca < cb else (cb, ca)
        pair_dists.setdefault(key, []).append(w)

    rows = []
    for (a, b), dists in pair_dists.items():
        n_a, n_b = sizes[a], sizes[b]
        raw_count = len(dists)
        expected_count = gamma * n_a * n_b
        if raw_count <= expected_count:
            continue
        weighted_sum = sum(1.0 / (1.0 + dd) for dd in dists)
        weighted_expected = gamma * n_a * n_b  # same number as expected_count (gamma==p0 here)
        rows.append({
            "a": a, "b": b, "n_a": n_a, "n_b": n_b,
            "raw_count": raw_count, "expected_count": expected_count,
            "excess_ratio": raw_count / expected_count,
            "weighted_sum": weighted_sum, "weighted_expected": weighted_expected,
            "mean_dist": float(np.mean(dists)), "min_dist": float(np.min(dists)),
            "weighted_excess": weighted_sum > weighted_expected,
        })

    rows.sort(key=lambda r: r["raw_count"] - r["expected_count"], reverse=True)
    n_weighted_excess = sum(1 for r in rows if r["weighted_excess"])
    print(f"  {len(rows):,} community pairs have raw_count > expected_count "
          f"(gamma * n_a * n_b); of those, {n_weighted_excess:,} ALSO have weighted_sum > "
          f"weighted_expected (genuine CPM local-optimum candidates)")

    print(f"\n  Top {min(top_n, len(rows))} by (raw_count - expected_count):")
    print(f"  {'A':>6} {'B':>6}  {'n_A':>5} {'n_B':>5}  {'raw':>5}  {'expected':>9}  "
          f"{'ratio':>6}  {'w_sum':>7}  {'w_exp':>7}  {'mean_d':>7}  {'min_d':>6}  verdict")
    for r in rows[:top_n]:
        verdict = "MISSED MERGE" if r["weighted_excess"] else "weight-deficit (correct)"
        print(f"  {r['a']:>6} {r['b']:>6}  {r['n_a']:>5} {r['n_b']:>5}  {r['raw_count']:>5}  "
              f"{r['expected_count']:>9.3f}  {r['excess_ratio']:>6.1f}  {r['weighted_sum']:>7.3f}  "
              f"{r['weighted_expected']:>7.3f}  {r['mean_dist']:>7.4f}  {r['min_dist']:>6.4f}  {verdict}")


def main() -> None:
    B = np.load(psn.PEPTIDES_B_PATH)
    print(f"Loaded B(y): {psn.PEPTIDES_B_PATH.name}  n={len(B):,}")
    print(f"  min={B.min():.4f}  median={np.median(B):.4f}  max={B.max():.4f}")
    print(f"  frac(B <= {THRESHOLD:g}) = {np.mean(B <= THRESHOLD):.4f}")

    plot_b_distribution(B, THRESHOLD, PLOT_DIR / "left_dist_peptides_B_hist.png")

    compare_thresholds()

    report_expected_cross_edges(TESTED_THRESHOLDS, p0_source="atlas", n_iterations=20)

    find_high_cross_community_pairs(0.55, n_iterations=20)


if __name__ == "__main__":
    main()
