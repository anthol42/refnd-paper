"""Threshold sweep for MNIST: train set (train) vs. test set (production).

Images are treated as flattened float32 vectors (784-d, values in [0, 1])
and compared with the Cosine kernel. The null model comes from a real,
unrelated image population -- tiny-imagenet-200, downsized to 28x28 and
grayscaled to match MNIST's shape -- rather than any kind of shuffled MNIST
(there's no meaningful "shuffle" for a fixed-length dense pixel vector the
way there is for a variable-length sequence or a sparse bit fingerprint).

Usage:
    uv run python -m thresholdv2.mnist
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
from rich import print

from refnd.core import EdgeStore, HNSWState, INWeightType
from refnd.kernels import KernelVariant

from refnd.kernels import zip_kernel

from src.cache import CacheStore
from src.datasets import DatasetConfig, mnist_download, tiny_imagenet_grayscale_vectors
from src.metrics import null_model_cdf

from thresholdv2.fit import sweep_thresholds

OUT_DIR = Path(__file__).parent.parent / "results" / "thresholds"
GRAPH_CACHE_KEY = "threshold_sweep_mnist"

CFG = DatasetConfig(
    modality=KernelVariant.Cosine,
    metric=None,
    encoder=None,
    proximity_threshold=0.3,
    kernel_params={},
)


def build_hnsw(
    train_vectors: np.ndarray, test_vectors: np.ndarray, cache: CacheStore, ef_construction: int = 64,
) -> EdgeStore:
    """Build (or load, if cached) the combined train+test HNSW's layer-0
    graph -- train leads test, so node ids [0, n_train) are train and
    [n_train, n_train+n_test) are prod (MNIST test).
    """
    cached = cache.get_edges(GRAPH_CACHE_KEY)
    if cached is not None:
        print(f"  [dim]Loading cached layer-0 graph: {GRAPH_CACHE_KEY}[/]")
        return cached

    combined = list(train_vectors) + list(test_vectors)
    print(f"  Building combined HNSW (n_train={len(train_vectors):,}, n_prod={len(test_vectors):,})...")
    hnsw = HNSWState(
        CFG.modality, combined,
        proximity_threshold=CFG.proximity_threshold,
        ef_construction=ef_construction,
        keep_all_edges=False,
        strict_ef=True,
        **CFG.kernel_params,
    )
    hnsw.build(progress=True)
    es = hnsw.get_layer(0, weights=True, progress=True)
    print(f"  layer0: n_edges={len(es):,}")
    cache.store_edges(GRAPH_CACHE_KEY, es)
    return es


def sample_cross_class_pairs(labels: np.ndarray, n_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random pairs drawn from DIFFERENT classes only -- same-class pairs
    would be spuriously similar (e.g. two images of the same tiny-imagenet
    category) and bias the null toward smaller distances than genuinely
    unrelated images. Colliding same-class draws are resampled."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    idx_a = rng.integers(0, n, size=n_pairs)
    idx_b = rng.integers(0, n, size=n_pairs)
    same = labels[idx_a] == labels[idx_b]
    while same.any():
        idx_b[same] = rng.integers(0, n, size=int(same.sum()))
        same = labels[idx_a] == labels[idx_b]
    return idx_a, idx_b


def find_gamma_function(
    cache: CacheStore, n_pool: int = 100_000, n_pairs: int = 2_000_000, seed: int = 42,
) -> Callable[[np.ndarray], np.ndarray]:
    """Null model p0(t) = P(distance <= t) between two unrelated images:
    random pairs of DIFFERENT-class, real (not shuffled) tiny-imagenet
    images, downsized + grayscaled to MNIST's 28x28 shape, scored with the
    Cosine kernel."""
    pool, labels = tiny_imagenet_grayscale_vectors(cache, n=n_pool, size=28, seed=seed)
    idx_a, idx_b = sample_cross_class_pairs(labels, n_pairs, seed)
    print(f"  Scoring {n_pairs:,} cross-class null pairs (Cosine)...")
    null_scores = np.asarray(
        zip_kernel(CFG.modality, list(pool[idx_a]), list(pool[idx_b]), n_threads=0, progress=True, **CFG.kernel_params),
        dtype=np.float64,
    )
    return null_model_cdf(null_scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresh-lo", type=float, default=0.0)
    parser.add_argument("--thresh-hi", type=float, default=0.6)
    parser.add_argument("--n-sweep", type=int, default=13)
    parser.add_argument("--ef-construction", type=int, default=64)
    parser.add_argument("--n-null-pool", type=int, default=100_000)
    parser.add_argument("--n-null-pairs", type=int, default=2_000_000)
    args = parser.parse_args()

    cache = CacheStore()

    print("[bold][orange2]=== Threshold sweep: MNIST train vs. MNIST test (prod) ===[/][/]")
    train_vectors, _, test_vectors, _ = mnist_download(cache)
    print(f"  train={len(train_vectors):,}  prod(test)={len(test_vectors):,}")

    hnsw_graph = build_hnsw(train_vectors, test_vectors, cache, ef_construction=args.ef_construction)
    train_nodes = np.arange(len(train_vectors), dtype=np.int64)
    prod_nodes = np.arange(len(train_vectors), len(train_vectors) + len(test_vectors), dtype=np.int64)

    print("  Computing gamma(tau) from a tiny-imagenet null...")
    gamma_cdf = find_gamma_function(cache, n_pool=args.n_null_pool, n_pairs=args.n_null_pairs)

    thresholds, results = sweep_thresholds(
        hnsw_graph, train_nodes, prod_nodes, gamma_cdf,
        args.thresh_lo, args.thresh_hi, args.n_sweep,
        inweight_type=INWeightType.SimilarityComplement,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"mnist.json"
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
