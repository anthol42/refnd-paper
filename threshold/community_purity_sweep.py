"""Threshold sweep with CPM community detection on the combined DBAASP+PeptideAtlas
layer-0 graph, adapting the null model (gamma) to each threshold.

For each threshold tau in [0.35, 0.65]:
  1. Filter the master layer-0 edge list (real measured distances, see
     threshold/threshold_left_distribution.py's combined-HNSW work) to edges with
     distance <= tau.
  2. gamma(tau) = P(unrelated atlas self-pair distance <= tau) (same null model
     used throughout threshold/*.py), used both as CPM resolution and passed to
     find_communities.
  3. Run CPM Leiden on the filtered graph.
  4. Purity: for every community, count DBAASP ("train") vs PeptideAtlas ("prod")
     members. A "pure" community contains only one dataset.
  5. For each pure TRAIN community, using the FULL (unfiltered) master edge
     distances (the true measured proximity, not the tau-filtered graph):
       - min_dist_train_train: min distance to any node in a DIFFERENT pure
         train community.
       - min_dist_train_prod: min distance to any node in a pure prod community.
  6. Collect these two distributions across all pure train communities, and
     compare them (KS test) -- the threshold where they become statistically
     indistinguishable is where community-level train/prod separation "looks
     the same" as it would look for real, unrelated data.

Usage:
    uv run python -m threshold.community_purity_sweep
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from rich import print
from scipy.stats import ks_2samp

from refnd.core import EdgeStore, INWeightType, LeidenObjective, find_communities, find_components

from threshold import prod_self_null as psn

PLOT_DIR = Path(__file__).parent
CACHE_DIR = PLOT_DIR / "cache"
N_SWEEP = 20
THRESH_LO, THRESH_HI = 0.35, 0.65
N_ITERATIONS = 2


def load_master_edges(source_name: str = "layer0") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (source, u, v, w): source labels (0=dbaasp,1=atlas) and the master
    edge list (node, node, real distance).

    source_name="layer0" (default): layer-0 adjacency of the search-only combined
    HNSW (keep_all_edges=False, built @ threshold=0.5), capped at ~M_max0 edges/node.

    source_name="edges065": the recorded proximity_edges of a SEPARATE combined HNSW
    built with keep_all_edges=True @ threshold=0.65 -- not capped per-node, so denser
    (every below-0.65 pair the build actually computed, not just the navigable graph).
    """
    source = np.load(CACHE_DIR / "combined_atlas_dbaasp_source.npy")

    if source_name == "layer0":
        u_path = CACHE_DIR / "combined_atlas_dbaasp_layer0_u.npy"
        v_path = CACHE_DIR / "combined_atlas_dbaasp_layer0_v.npy"
        w_path = CACHE_DIR / "combined_atlas_dbaasp_layer0_dists.npy"
        master_edgestore = CACHE_DIR / "combined_atlas_dbaasp_layer0_measured.edgestr"
    elif source_name == "edges065":
        u_path = CACHE_DIR / "combined_atlas_dbaasp_edges_thr0.65_u.npy"
        v_path = CACHE_DIR / "combined_atlas_dbaasp_edges_thr0.65_v.npy"
        w_path = CACHE_DIR / "combined_atlas_dbaasp_edges_thr0.65_w.npy"
        master_edgestore = None
    else:
        raise ValueError(f"source_name must be 'layer0' or 'edges065', got {source_name!r}")

    if u_path.exists() and v_path.exists():
        print(f"  [dim]Loading cached {source_name} u/v/w arrays[/]")
        u = np.load(u_path)
        v = np.load(v_path)
        w = np.load(w_path)
    else:
        print(f"  Loading master EdgeStore ({source_name}) and extracting u/v/w (one-time cost)...")
        t0 = time.perf_counter()
        es = EdgeStore.load(str(master_edgestore))
        edges = es.edges()
        u = np.fromiter((e[0] for e in edges), dtype=np.uint32, count=len(edges))
        v = np.fromiter((e[1] for e in edges), dtype=np.uint32, count=len(edges))
        w = np.fromiter((e[2] for e in edges), dtype=np.float32, count=len(edges))
        del edges
        np.save(u_path, u)
        np.save(v_path, v)
        print(f"  Done in {time.perf_counter() - t0:.2f}s  n_edges={len(u):,}")

    return source, u, v, w


def run_threshold(
    tau: float, source: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray,
    gamma_scores: np.ndarray, n: int, method: str = "cpm",
) -> dict:
    t_start = time.perf_counter()

    mask = w <= tau
    edges_tau = list(zip(u[mask].tolist(), v[mask].tolist(), w[mask].tolist()))
    es_tau = EdgeStore(n, edges_tau)
    graph = es_tau.graph(inweight_type=INWeightType.Distance)
    build_t = time.perf_counter() - t_start

    if method == "cpm":
        gamma = float(np.mean(gamma_scores <= tau))
        t0 = time.perf_counter()
        communities = np.asarray(
            find_communities(graph, gamma=gamma, objective=LeidenObjective.CPM, n_iterations=N_ITERATIONS),
            dtype=np.int64,
        )
        leiden_t = time.perf_counter() - t0
    elif method == "components":
        gamma = float("nan")  # not used -- connected components has no resolution parameter
        t0 = time.perf_counter()
        communities = np.asarray(find_components(graph), dtype=np.int64)
        leiden_t = time.perf_counter() - t0
    else:
        raise ValueError(f"method must be 'cpm' or 'components', got {method!r}")

    t0 = time.perf_counter()
    n_comm_ids = int(communities.max()) + 1
    dbaasp_counts = np.bincount(communities[source == 0], minlength=n_comm_ids)
    atlas_counts = np.bincount(communities[source == 1], minlength=n_comm_ids)

    pure_train_mask = (atlas_counts == 0) & (dbaasp_counts > 0)
    pure_prod_mask = (dbaasp_counts == 0) & (atlas_counts > 0)

    type_of_comm = np.full(n_comm_ids, 2, dtype=np.int8)  # 0=pure train, 1=pure prod, 2=other(mixed)
    type_of_comm[pure_train_mask] = 0
    type_of_comm[pure_prod_mask] = 1

    cu = communities[u]
    cv = communities[v]
    tu = type_of_comm[cu]
    tv = type_of_comm[cv]

    min_tt = np.full(n_comm_ids, np.inf, dtype=np.float32)
    mask_tt = (tu == 0) & (tv == 0) & (cu != cv)
    if mask_tt.any():
        np.minimum.at(min_tt, cu[mask_tt], w[mask_tt])
        np.minimum.at(min_tt, cv[mask_tt], w[mask_tt])

    min_tp = np.full(n_comm_ids, np.inf, dtype=np.float32)
    mask_tp_a = (tu == 0) & (tv == 1)
    mask_tp_b = (tu == 1) & (tv == 0)
    if mask_tp_a.any():
        np.minimum.at(min_tp, cu[mask_tp_a], w[mask_tp_a])
    if mask_tp_b.any():
        np.minimum.at(min_tp, cv[mask_tp_b], w[mask_tp_b])

    pure_train_ids = np.nonzero(pure_train_mask)[0]
    tt_vals = min_tt[pure_train_ids]
    tp_vals = min_tp[pure_train_ids]
    tt_finite = tt_vals[np.isfinite(tt_vals)]
    tp_finite = tp_vals[np.isfinite(tp_vals)]

    if len(tt_finite) > 1 and len(tp_finite) > 1:
        ks_stat, ks_p = ks_2samp(tt_finite, tp_finite)
    else:
        ks_stat, ks_p = float("nan"), float("nan")
    analysis_t = time.perf_counter() - t0

    total_t = time.perf_counter() - t_start

    result = {
        "threshold": tau, "gamma": gamma, "n_communities": n_comm_ids,
        "n_pure_train": int(pure_train_mask.sum()), "n_pure_prod": int(pure_prod_mask.sum()),
        "n_mixed": int(n_comm_ids - pure_train_mask.sum() - pure_prod_mask.sum()),
        "tt_vals": tt_finite, "tp_vals": tp_finite,
        "ks_stat": float(ks_stat), "ks_p": float(ks_p),
        "build_t": build_t, "leiden_t": leiden_t, "analysis_t": analysis_t, "total_t": total_t,
    }
    print(
        f"  tau={tau:.4f}  gamma={gamma:.6f}  n_comm={n_comm_ids:,}  "
        f"pure_train={result['n_pure_train']:,}  pure_prod={result['n_pure_prod']:,}  "
        f"mixed={result['n_mixed']:,}  n_tt={len(tt_finite):,}  n_tp={len(tp_finite):,}  "
        f"median_tt={np.median(tt_finite) if len(tt_finite) else float('nan'):.4f}  "
        f"median_tp={np.median(tp_finite) if len(tp_finite) else float('nan'):.4f}  "
        f"KS={ks_stat:.4f} (p={ks_p:.3g})  "
        f"[dim](build={build_t:.2f}s leiden={leiden_t:.2f}s analysis={analysis_t:.2f}s total={total_t:.2f}s)[/]"
    )
    return result


def plot_summary(results: list[dict], out_path: Path, method: str = "cpm") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    taus = [r["threshold"] for r in results]
    med_tt = [np.median(r["tt_vals"]) if len(r["tt_vals"]) else np.nan for r in results]
    med_tp = [np.median(r["tp_vals"]) if len(r["tp_vals"]) else np.nan for r in results]
    q25_tt = [np.percentile(r["tt_vals"], 25) if len(r["tt_vals"]) else np.nan for r in results]
    q75_tt = [np.percentile(r["tt_vals"], 75) if len(r["tt_vals"]) else np.nan for r in results]
    q25_tp = [np.percentile(r["tp_vals"], 25) if len(r["tp_vals"]) else np.nan for r in results]
    q75_tp = [np.percentile(r["tp_vals"], 75) if len(r["tp_vals"]) else np.nan for r in results]
    ks = [r["ks_stat"] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), dpi=150, sharex=True)
    fig.patch.set_facecolor("white")

    ax1.plot(taus, med_tt, color="#D55E00", linewidth=2.0, marker="o", markersize=4,
              label="min dist: train → nearest OTHER pure train community")
    ax1.fill_between(taus, q25_tt, q75_tt, color="#D55E00", alpha=0.15)
    ax1.plot(taus, med_tp, color="#0072B2", linewidth=2.0, marker="o", markersize=4,
              label="min dist: train → nearest pure prod community")
    ax1.fill_between(taus, q25_tp, q75_tp, color="#0072B2", alpha=0.15)
    ax1.set_ylabel("min distance between communities")
    ax1.set_title(f"Median (IQR band) min inter-community distance vs. threshold ({method})", loc="left")
    ax1.legend(loc="best", fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    ax2.plot(taus, ks, color="#009E73", linewidth=2.0, marker="o", markersize=4)
    if not all(np.isnan(ks)):
        best_idx = int(np.nanargmin(ks))
        ax2.axvline(taus[best_idx], color="#6b7280", linewidth=1.2, linestyle="--",
                     label=f"min KS at threshold={taus[best_idx]:.4f}")
        ax2.legend(loc="best", fontsize=9, frameon=False)
    ax2.set_xlabel(r"CPM threshold $\tau$")
    ax2.set_ylabel("KS statistic (train-train vs. train-prod)")
    ax2.set_title("Two-sample KS distance between the two distributions (lower = more similar)", loc="left")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["cpm", "components"], default="cpm")
    parser.add_argument("--tag", default="", help="Extra suffix for output filenames, e.g. 'rerun2'.")
    parser.add_argument("--master", choices=["layer0", "edges065"], default="layer0",
                        help="Which cached master edge set to sweep over.")
    args = parser.parse_args()

    print(f"[bold][orange2]=== Community purity sweep: {args.method} on combined "
          f"DBAASP+PeptideAtlas ({args.master}) ===[/][/]")
    source, u, v, w = load_master_edges(args.master)
    n = len(source)
    gamma_scores = np.load(psn.PEPTIDES_NULL_PROD_PATH)

    thresholds = np.linspace(THRESH_LO, THRESH_HI, N_SWEEP)
    print(f"  n={n:,}  n_edges(master)={len(u):,}  sweep=[{THRESH_LO},{THRESH_HI}] n_points={N_SWEEP}")

    t_total = time.perf_counter()
    results = [run_threshold(tau, source, u, v, w, gamma_scores, n, method=args.method) for tau in thresholds]
    total_time = time.perf_counter() - t_total

    print(f"\n[bold]Total sweep time: {total_time:.2f}s ({total_time/60:.2f} min) "
          f"for {N_SWEEP} thresholds ({total_time/N_SWEEP:.2f}s/threshold avg)[/]")

    suffix = "" if args.method == "cpm" else f"_{args.method}"
    if args.master != "layer0":
        suffix += f"_{args.master}"
    if args.tag:
        suffix += f"_{args.tag}"
    plot_summary(results, PLOT_DIR / f"community_purity_sweep{suffix}.png", method=args.method)


if __name__ == "__main__":
    main()
