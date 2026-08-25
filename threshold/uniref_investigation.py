"""Node-level train<->train vs. train<->prod nearest-neighbor ECDF for the
UniRef50 (train) / ProteinGym+PEER+CASP15 (prod) benchmark index.

`threshold/uniref.py` only reports 3 summary quantiles (p15/median/p85) per
threshold, computed PER LEIDEN COMMUNITY after re-clustering the tau-filtered
graph at each sweep point. This script instead reads the same pre-built
layer-0 EdgeStore directly as a nearest-neighbor graph (no thresholding, no
Leiden) and computes, per NODE:

  - for every train node: its distance to the nearest OTHER train node
    among its layer-0 HNSW neighbors ("train<->train")
  - for every prod node: its distance to the nearest train node among its
    layer-0 HNSW neighbors ("train<->prod")

Node/source split comes from offsets.txt, the same way threshold/uniref.py
reads it: uniref50 (train) occupies [0, offsets["proteingym"]), everything
from offsets["proteingym"] onward (proteingym+peer+casp15) is prod.

Since the raw per-node arrays can run into the tens of millions of floats,
the output stores a compact quantile-function representation (the ECDF's
inverse, sampled at --n-quantiles evenly spaced levels) rather than the raw
values -- exact for reconstructing the ECDF curve, negligible file size.

Usage:
    uv run python -m threshold.uniref_investigation \\
        --edgestore-path /path/unirefxbenchmark_layer0.edgestr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rich import print
from scipy.stats import ks_2samp

from refnd.core import EdgeStore

OUT_DIR = Path(__file__).parent.parent / "results" / "thresholds"


def load_offsets(offsets_path: Path) -> dict[str, int]:
    offsets = {}
    for line in offsets_path.read_text().splitlines():
        label, value = line.split(":")
        offsets[label.strip()] = int(value.strip())
    return offsets


def load_edges(hnsw_graph: EdgeStore) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_edges = len(hnsw_graph)
    src = np.fromiter((e[0] for e in hnsw_graph), dtype=np.uint32, count=n_edges)
    dst = np.fromiter((e[1] for e in hnsw_graph), dtype=np.uint32, count=n_edges)
    dist = np.fromiter((e[2] for e in hnsw_graph), dtype=np.float32, count=n_edges)
    return src, dst, dist


def nearest_train_distance(
    src: np.ndarray, dst: np.ndarray, dist: np.ndarray, is_train: np.ndarray,
    query_nodes: np.ndarray, n_total: int, same_group: bool,
) -> np.ndarray:
    """Per-node min distance to a train node, over `query_nodes`.

    same_group=True: query_nodes are themselves train nodes, so an edge only
        counts if BOTH endpoints are train (excludes prod entirely, and a
        node is never its own neighbor since the layer-0 graph has no
        self-loops) -- this is train<->train.
    same_group=False: query_nodes are prod nodes, edge counts if the OTHER
        endpoint is train -- this is train<->prod.

    The layer-0 EdgeStore is undirected but each edge is stored once, so
    both (src, dst) directions are scattered into the per-node minimum.
    """
    min_dist = np.full(n_total, np.inf, dtype=np.float64)
    if same_group:
        mask = is_train[src] & is_train[dst]
        if mask.any():
            np.minimum.at(min_dist, src[mask], dist[mask])
            np.minimum.at(min_dist, dst[mask], dist[mask])
    else:
        mask_a = is_train[src] & ~is_train[dst]  # dst is prod, src is train -> candidate for dst
        mask_b = is_train[dst] & ~is_train[src]  # src is prod, dst is train -> candidate for src
        if mask_a.any():
            np.minimum.at(min_dist, dst[mask_a], dist[mask_a])
        if mask_b.any():
            np.minimum.at(min_dist, src[mask_b], dist[mask_b])
    return min_dist[query_nodes]


def quantile_function(values: np.ndarray, n_quantiles: int) -> dict:
    """Compact ECDF representation: the quantile function (ECDF's inverse)
    sampled at n_quantiles evenly spaced levels in (0, 1) -- exact for
    reconstructing a step ECDF, without storing every raw value."""
    finite = values[np.isfinite(values)]
    levels = np.linspace(0.0, 1.0, n_quantiles)
    if len(finite) == 0:
        return {"n_raw": int(len(values)), "n_finite": 0, "levels": levels.tolist(), "values": []}
    return {
        "n_raw": int(len(values)),
        "n_finite": int(len(finite)),
        "levels": levels.tolist(),
        "values": np.quantile(finite, levels).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--edgestore-path", type=Path, required=True,
                        help="Layer-0 EdgeStore from build_uniref_benchmark_index "
                             "(offsets.txt is expected as a sibling in the same directory, "
                             "unless --offsets-path is given).")
    parser.add_argument("--offsets-path", type=Path, default=None,
                        help="Defaults to offsets.txt next to --edgestore-path.")
    parser.add_argument("--out-path", type=Path,
                        default=OUT_DIR / "unirefxbenchmark_ecdf.json")
    parser.add_argument("--n-quantiles", type=int, default=1000,
                        help="Number of evenly-spaced quantile levels to store per group.")
    parser.add_argument("--ks-sample-size", type=int, default=200_000,
                        help="ks_2samp is O(n log n) per side; subsample each finite array "
                             "to at most this many points (uniform, seeded) before running it, "
                             "so it stays fast on tens-of-millions-of-node arrays. "
                             "Set to 0 to disable subsampling and use every finite value.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    offsets_path = args.offsets_path or (args.edgestore_path.parent / "offsets.txt")

    print("[bold][orange2]=== UniRef50xbenchmark: node-level train<->train vs train<->prod ECDF ===[/][/]")
    print(f"  Loading layer-0 graph: {args.edgestore_path}")
    hnsw_graph = EdgeStore.load(str(args.edgestore_path))
    n_total = hnsw_graph.node_count()

    offsets = load_offsets(offsets_path)
    train_nodes = np.arange(0, offsets["proteingym"], dtype=np.int64)
    prod_nodes = np.arange(offsets["proteingym"], n_total, dtype=np.int64)
    print(f"  train(uniref50)={len(train_nodes):,}  prod(proteingym+peer+casp15)={len(prod_nodes):,}")

    is_train = np.zeros(n_total, dtype=bool)
    is_train[train_nodes] = True

    print("  Reading edges...")
    src, dst, dist = load_edges(hnsw_graph)
    print(f"  {len(dist):,} edges")

    print("  Computing per-node nearest-train distances (train<->train)...")
    tt_vals = nearest_train_distance(src, dst, dist, is_train, train_nodes, n_total, same_group=True)
    print("  Computing per-node nearest-train distances (train<->prod)...")
    tp_vals = nearest_train_distance(src, dst, dist, is_train, prod_nodes, n_total, same_group=False)

    tt_finite = tt_vals[np.isfinite(tt_vals)]
    tp_finite = tp_vals[np.isfinite(tp_vals)]
    print(f"  train<->train: {len(tt_finite):,}/{len(tt_vals):,} finite (have >=1 train neighbor)")
    print(f"  train<->prod:  {len(tp_finite):,}/{len(tp_vals):,} finite (have >=1 train neighbor)")

    rng = np.random.default_rng(args.seed)

    def subsample(a: np.ndarray) -> np.ndarray:
        if args.ks_sample_size <= 0 or len(a) <= args.ks_sample_size:
            return a
        return rng.choice(a, size=args.ks_sample_size, replace=False)

    if len(tt_finite) > 1 and len(tp_finite) > 1:
        print("  Running ks_2samp (subsampled if needed)...")
        ks_stat, ks_p = ks_2samp(subsample(tt_finite), subsample(tp_finite))
        ks_stat, ks_p = float(ks_stat), float(ks_p)
    else:
        ks_stat, ks_p = None, None
    print(f"  KS={ks_stat}  (p={ks_p})")

    result = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "n_total": n_total,
        "n_train": len(train_nodes),
        "n_prod": len(prod_nodes),
        "train_train": quantile_function(tt_vals, args.n_quantiles),
        "train_prod": quantile_function(tp_vals, args.n_quantiles),
        "ks_stat": ks_stat,
        "ks_pvalue": ks_p,
    }

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved {args.out_path}")


if __name__ == "__main__":
    main()
