"""Threshold sweep on the synthetic forest: annotated (train) vs. production.

Applies the algorithm of `threshold/` (see `threshold.peptides`) to two
independent draws of the synthetic generator. Because the generative process is
known, the sweep can be judged against ground truth: the ideal threshold is the
one whose communities line up with the generated families.

Usage:
    uv run python -m synthetic.find_threshold
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
from src.datasets import DATASETS
from src.metrics import null_model_cdf, null_model_scores
from synthetic.dataset import (SyntheticConfig, add_config_arguments, build_dataset,
                               config_from_args)
from threshold.fit import sweep_thresholds

OUT_DIR = Path(__file__).parent.parent / "results" / "thresholds"
DATASET_KEY = "peptide_atlas"          # shares the peptide kernel/modality


def build_hnsw(annotated: list[str], production: list[str], cache: CacheStore,
               cache_key: str, ef_construction: int = 64) -> EdgeStore:
    """Combined annotated+production HNSW layer-0 graph (built, or loaded).

    annotated leads production in the concatenated dataset, so node ids
    [0, n_annotated) are annotated and the rest are production.

    proximity_threshold is irrelevant here: get_layer(0) returns the base HNSW
    navigation layer regardless of it (unlike .edges(), which is
    threshold-filtered) -- it's only a required HNSWState constructor arg.
    """
    cached = cache.get_edges(cache_key)
    if cached is not None:
        print(f"  [dim]Loading cached layer-0 graph: {cache_key}[/]")
        return cached

    cfg = DATASETS[DATASET_KEY]
    print(f"  Building combined HNSW (n_annotated={len(annotated):,}, "
          f"n_production={len(production):,})...")
    hnsw = HNSWState(
        cfg.modality, annotated + production,
        proximity_threshold=0.0,
        ef_construction=ef_construction,
        keep_all_edges=False,
        strict_ef=True,
        **cfg.kernel_params,
    )
    hnsw.build(progress=True)
    edges = hnsw.get_layer(0, weights=True, progress=True)
    print(f"  layer0: n_edges={len(edges):,}")
    cache.store_edges(cache_key, edges)
    return edges


def find_gamma_function(sequences: list[str], n_pairs: int = 10_000_000,
                        seed: int = 42) -> Callable[[np.ndarray], np.ndarray]:
    """Null model p0(t) = P(distance <= t) between two unrelated sequences:
    random pairs, each element-shuffled to destroy homology, scored with the
    peptide kernel. Same procedure as `threshold.peptides`."""
    scores = null_model_scores(sequences, DATASETS[DATASET_KEY], n_samples=n_pairs,
                               seed=seed, shuffle=True)
    return null_model_cdf(scores)


def run_name(config: SyntheticConfig) -> str:
    return (f"synthetic_{config.n_peptides}_{config.n_families}"
            f"_d{config.max_depth_edits}_e{config.edits_per_branch:g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--annotated-seed", type=int, default=0)
    parser.add_argument("--production-seed", type=int, default=1)
    parser.add_argument("--thresh-lo", type=float, default=0.)
    parser.add_argument("--thresh-hi", type=float, default=0.9)
    parser.add_argument("--n-sweep", type=int, default=19)
    parser.add_argument("--ef-construction", type=int, default=64)
    parser.add_argument("--n-null-pairs", type=int, default=10_000_000)
    args = parser.parse_args()
    config = config_from_args(args)

    cache = CacheStore()
    print("[bold][orange2]=== Threshold sweep: synthetic annotated vs. production ===[/][/]")
    annotated = build_dataset(config, seed=args.annotated_seed, cache=cache)
    production = build_dataset(config, seed=args.production_seed, cache=cache)
    print(f"  annotated={len(annotated):,} ({config.n_families} families)  "
          f"production={len(production):,} ({config.n_families} families)")

    name = run_name(config)
    graph = build_hnsw(annotated.sequences, production.sequences, cache,
                       cache_key=f"threshold_sweep_{name}",
                       ef_construction=args.ef_construction)
    annotated_nodes = np.arange(len(annotated), dtype=np.int32)
    production_nodes = np.arange(len(annotated), len(annotated) + len(production),
                                 dtype=np.int32)

    print("  Computing gamma(tau) from a permutation null...")
    gamma_cdf = find_gamma_function(production.sequences, n_pairs=args.n_null_pairs)

    thresholds, results = sweep_thresholds(
        graph, annotated_nodes, production_nodes, gamma_cdf,
        args.thresh_lo, args.thresh_hi, args.n_sweep,
        inweight_type=INWeightType.Distance,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{name}.json"
    with open(out_path, "w") as handle:
        json.dump({"args": vars(args), "results": results}, handle, indent=2)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
