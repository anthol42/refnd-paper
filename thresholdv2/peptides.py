"""Threshold sweep for peptides: DBAASP (train) vs. PeptideAtlas (production).

Usage:
    uv run python -m thresholdv2.peptides
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
from rich import print

from refnd.core import EdgeStore, HNSWState, INWeightType

from src.cache import CacheStore
from src.datasets import DATASETS, load_dataset
from src.metrics import null_model_cdf, null_model_scores

from thresholdv2.fit import sweep_thresholds

OUT_DIR = Path(__file__).parent.parent / "results" / "thresholds"
CACHE_KEY = "threshold_sweep_dbaasp_atlas"


def build_hnsw(train: list[str], prod: list[str], cache: CacheStore, ef_construction: int = 64) -> EdgeStore:
    """Build (or load, if cached) the combined train+prod HNSW's layer-0 graph.

    train leads prod in the concatenated dataset, so node ids
    [0, len(train)) are train and [len(train), len(train)+len(prod)) are prod.

    proximity_threshold is irrelevant here: get_layer(0) returns the base
    HNSW navigation layer regardless of it (unlike .edges(), which is
    threshold-filtered) -- it's only a required HNSWState constructor arg.
    """
    cached = cache.get_edges(CACHE_KEY)
    if cached is not None:
        print(f"  [dim]Loading cached layer-0 graph: {CACHE_KEY}[/]")
        return cached

    combined = train + prod
    cfg = DATASETS["dbaasp"]
    print(f"  Building combined HNSW (n_train={len(train):,}, n_prod={len(prod):,})...")
    hnsw = HNSWState(
        cfg.modality, combined,
        proximity_threshold=0.0,
        ef_construction=ef_construction,
        keep_all_edges=False,
        strict_ef=True,
        **cfg.kernel_params,
    )
    hnsw.build(progress=True)
    es = hnsw.get_layer(0, weights=True, progress=True)
    print(f"  layer0: n_edges={len(es):,}")
    cache.store_edges(CACHE_KEY, es)
    return es


def find_gamma_function(sequences: list[str], n_pairs: int = 10_000_000, seed: int = 42) -> Callable[[np.ndarray], np.ndarray]:
    """Null model p0(t) = P(distance <= t) between two unrelated PeptideAtlas
    sequences: random pairs, each element-shuffled to destroy homology,
    scored in parallel via the DBAASP/PeptideAtlas shared kernel (zip_kernel).
    """
    cfg = DATASETS["peptide_atlas"]  # peptide_atlas shares the same kernel/modality
    scores = null_model_scores(sequences, cfg, n_samples=n_pairs, seed=seed, shuffle=True)
    return null_model_cdf(scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresh-lo", type=float, default=0.3)
    parser.add_argument("--thresh-hi", type=float, default=0.7)
    parser.add_argument("--n-sweep", type=int, default=9)
    parser.add_argument("--ef-construction", type=int, default=64)
    parser.add_argument("--n-null-pairs", type=int, default=10_000_000)
    args = parser.parse_args()

    cache = CacheStore()

    print("[bold][orange2]=== Threshold sweep: DBAASP (train) vs. PeptideAtlas (prod) ===[/][/]")
    train_data, _ = load_dataset("dbaasp", cache)
    prod_data, _ = load_dataset("peptide_atlas", cache)
    print(f"  train(dbaasp)={len(train_data):,}  prod(peptide_atlas, full)={len(prod_data):,}")

    hnsw_graph = build_hnsw(train_data, prod_data, cache, ef_construction=args.ef_construction)
    train_nodes = np.arange(len(train_data), dtype=np.int32)
    prod_nodes = np.arange(len(train_data), len(train_data) + len(prod_data), dtype=np.int32)

    print("  Computing gamma(tau) from a PeptideAtlas permutation null...")
    gamma_cdf = find_gamma_function(prod_data, n_pairs=args.n_null_pairs)

    thresholds, results = sweep_thresholds(
        hnsw_graph, train_nodes, prod_nodes, gamma_cdf,
        args.thresh_lo, args.thresh_hi, args.n_sweep,
        inweight_type=INWeightType.SimilarityComplement,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"peptides.json"
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
