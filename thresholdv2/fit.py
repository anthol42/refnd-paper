from typing import Callable

import numpy as np
from rich import print
from scipy.stats import ks_2samp

from refnd.core import EdgeStore, INWeightType, LeidenObjective, find_communities



def run_threshold(
    tau: float,
    gamma: float,
    source: np.ndarray,
    hnsw_edges: EdgeStore,
    src: np.ndarray,
    dst: np.ndarray,
    dist: np.ndarray,
    inweight_type: INWeightType,
) -> dict:
    graph = hnsw_edges[dist <= tau].graph(inweight_type=inweight_type)
    communities = np.asarray(
        find_communities(graph, gamma=gamma, objective=LeidenObjective.CPM),
        dtype=np.int64,
    )
    n_comm = int(communities.max()) + 1

    train_counts = np.bincount(communities[source == 0], minlength=n_comm)
    prod_counts = np.bincount(communities[source == 1], minlength=n_comm)

    pure_train_mask = ( prod_counts== 0) & (train_counts > 0)
    pure_prod_mask = (train_counts == 0) & (prod_counts > 0)
    hybrid_mask = ~pure_train_mask & ~pure_prod_mask

    comm_type = np.full(n_comm, 2, dtype=np.int8)  # 0=pure train, 1=pure prod, 2=hybrid
    comm_type[pure_train_mask] = 0
    comm_type[pure_prod_mask] = 1

    c_src, c_dst = communities[src], communities[dst]

    is_train_node = source == 0
    min_dist_to_train = np.full(n_comm, np.inf, dtype=np.float32)
    mask_a = is_train_node[dst] & (c_src != c_dst)  # dst is train -> candidate for c_src's min
    mask_b = is_train_node[src] & (c_src != c_dst)  # src is train -> candidate for c_dst's min
    if mask_a.any():
        np.minimum.at(min_dist_to_train, c_src[mask_a], dist[mask_a])
    if mask_b.any():
        np.minimum.at(min_dist_to_train, c_dst[mask_b], dist[mask_b])

    pure_train_ids = np.nonzero(pure_train_mask)[0]
    pure_prod_ids = np.nonzero(pure_prod_mask)[0]
    tt_vals = min_dist_to_train[pure_train_ids]  # pure-train communities -> nearest OTHER train node
    tp_vals = min_dist_to_train[pure_prod_ids]   # pure-prod communities -> nearest train node
    tt_finite = tt_vals[np.isfinite(tt_vals)]
    tp_finite = tp_vals[np.isfinite(tp_vals)]

    def stats(a: np.ndarray) -> dict:
        if len(a) == 0:
            return {"median": None, "p15": None, "p85": None, "n": 0}
        return {
            "median": float(np.median(a)),
            "p15": float(np.percentile(a, 15)),
            "p85": float(np.percentile(a, 85)),
            "n": int(len(a)),
        }

    if len(tt_finite) > 1 and len(tp_finite) > 1:
        ks_stat, ks_p = ks_2samp(tt_finite, tp_finite)
        ks_stat, ks_p = float(ks_stat), float(ks_p)
    else:
        ks_stat, ks_p = None, None

    result = {
        "threshold": float(tau),
        "gamma": gamma,
        "n_communities": n_comm,
        "n_pure_train": int(pure_train_mask.sum()),
        "n_pure_prod": int(pure_prod_mask.sum()),
        "n_hybrid": int(hybrid_mask.sum()),
        "train_train": stats(tt_finite),
        "train_prod": stats(tp_finite),
        "ks_stat": ks_stat,
        "ks_pvalue": ks_p,
    }
    print(
        f"  tau={tau:.4f}  gamma={gamma:.3e}  n_comm={n_comm:,}  "
        f"pure_train={result['n_pure_train']:,}  pure_prod={result['n_pure_prod']:,}  "
        f"hybrid={result['n_hybrid']:,}  "
        f"median_tt={result['train_train']['median']}  median_tp={result['train_prod']['median']}  "
        f"KS={ks_stat}  (p={ks_p})"
    )
    return result


def sweep_thresholds(hnsw_graph: EdgeStore, train_nodes: np.ndarray[np.int32], prod_nodes: np.ndarray[np.int32],
                     null_cdf: Callable[[np.ndarray], np.ndarray], min_threshold: float, max_threshold: float,
                     num_points: int, inweight_type: INWeightType = INWeightType.Distance):
    """Threshold sweep over an already-built HNSW layer-0 graph.

    hnsw_graph: full layer-0 EdgeStore (real distances), covering every node
        in train_nodes and prod_nodes.
    train_nodes / prod_nodes: node ids (indices into hnsw_graph) belonging to
        the training set / the production sample. Disjoint, and together
        cover every node used below.
    null_cdf: p0(t) = P(null distance <= t) under a permutation null
        (unrelated pairs), as returned by `src.metrics.null_model_cdf` —
        used to derive gamma(tau) at every threshold.
    inweight_type: how the tau-filtered EdgeStore's raw distances are turned
        into CsrGraph weights (forwarded to `EdgeStore.graph`).
    """
    n = hnsw_graph.node_count()
    source = np.full(n, -1, dtype=np.int8)
    source[np.asarray(train_nodes, dtype=np.int64)] = 0
    source[np.asarray(prod_nodes, dtype=np.int64)] = 1

    src = np.fromiter((e[0] for e in hnsw_graph), dtype=np.uint32, count=len(hnsw_graph))
    dst = np.fromiter((e[1] for e in hnsw_graph), dtype=np.uint32, count=len(hnsw_graph))
    dist = np.fromiter((e[2] for e in hnsw_graph), dtype=np.float32, count=len(hnsw_graph))

    thresholds = np.linspace(min_threshold, max_threshold, num_points)
    gammas = null_cdf(thresholds)  # vectorized: one p0(tau) call for the whole grid
    results = [
        run_threshold(tau, float(gamma), source, hnsw_graph, src, dst, dist, inweight_type)
        for tau, gamma in zip(thresholds, gammas)
    ]
    return thresholds, results