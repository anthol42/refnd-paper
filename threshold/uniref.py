"""Threshold sweep for UniRef50 (train) vs. the union of protein benchmarks
(ProteinGym + PEER + CASP15, production) that
build_uniref_benchmark_index.py extended the ProtSpaM HNSW index
with.

Takes that script's pre-computed layer-0 EdgeStore as input; offsets.txt and
patterns.bin (also written by that script, into the same directory) are
picked up as siblings by their fixed names.

Usage:
    uv run python -m thresholdv2.uniref \\
        --edgestore-path /path/unirefxbenchmark_layer0.edgestr \\
        --uniref50-fasta /path/uniref50.fasta.gz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
from rich import print

from refnd.core import EdgeStore, INWeightType
from refnd.kernels import KernelVariant, zip_kernel
from refnd.kernels.protspam import ProtSpamDistance
from refnd.utils import SWPatternSet, SWSequence

from src.cache import CacheStore
from src.metrics import _shuffle_sample, null_model_cdf

from threshold.build_uniref_benchmark_index import stream_uniref50_fasta

from threshold.fit import sweep_thresholds

OUT_DIR = Path(__file__).parent.parent / "results" / "thresholds"

SEED = 42
LENGTH_CAP = 2048  # same cap build_uniref_benchmark_index.py encodes with
SIGNIFICANCE_THRESHOLD = -1_000_000  # same default the build script uses


def load_offsets(offsets_path: Path) -> dict[str, int]:
    offsets = {}
    for line in offsets_path.read_text().splitlines():
        label, value = line.split(":")
        offsets[label.strip()] = int(value.strip())
    return offsets


def reservoir_sample(stream, k: int, seed: int) -> list[str]:
    """Uniform sample of k items from a one-pass stream of unknown length,
    without ever materializing it."""
    rng = np.random.default_rng(seed)
    pool: list[str] = []
    for i, item in enumerate(stream):
        if len(pool) < k:
            pool.append(item)
        else:
            j = rng.integers(0, i + 1)
            if j < k:
                pool[j] = item
    return pool


def find_gamma_function(
    fasta_path: Path, patterns: SWPatternSet, cache: CacheStore,
    n_pool: int = 20_000, n_pairs: int = 1_000_000,
) -> Callable[[np.ndarray], np.ndarray]:
    """Null model p0(t) = P(distance <= t) between two unrelated UniRef50
    sequences: random pairs, each element-shuffled to destroy homology, then
    re-encoded as ProtSpaM SWSequences and scored with the ProtSpaM kernel --
    analogous to the peptide permutation null, with a conversion step to
    ProtSpaM's sequence representation.
    """
    null_path = cache.root / f"null_uniref50_protspam_pool{n_pool}_pairs{n_pairs}_seed{SEED}.npy"
    if null_path.exists():
        print(f"  [dim]Loading cached null scores: {null_path.name}[/]")
        return null_model_cdf(np.load(null_path))

    print(f"  Reservoir-sampling {n_pool:,} raw UniRef50 sequences from {fasta_path.name}...")
    pool = reservoir_sample(stream_uniref50_fasta(fasta_path, LENGTH_CAP), n_pool, SEED)

    rng = np.random.default_rng(SEED)
    idx_a = rng.integers(0, len(pool), size=n_pairs)
    idx_b = rng.integers(0, len(pool), size=n_pairs)
    print(f"  Shuffling + encoding {n_pairs:,} random pairs into ProtSpaM sequences...")
    seq_a = [SWSequence(_shuffle_sample(pool[i], rng), patterns) for i in idx_a]
    seq_b = [SWSequence(_shuffle_sample(pool[i], rng), patterns) for i in idx_b]

    print("  Scoring null pairs (ProtSpaM, parallel)...")
    null_scores = np.asarray(
        zip_kernel(
            KernelVariant.ProtSpam, seq_a, seq_b, n_threads=0, progress=True,
            patterns=patterns, significance_threshold=SIGNIFICANCE_THRESHOLD,
            distance=ProtSpamDistance.MismatchRate,
        ),
        dtype=np.float64,
    )
    np.save(null_path, null_scores)
    return null_model_cdf(null_scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edgestore-path", type=Path, required=True,
                        help="Pre-computed layer-0 EdgeStore from build_uniref_benchmark_index "
                             "(offsets.txt and patterns.bin are expected as siblings in the same directory).")
    parser.add_argument("--uniref50-fasta", type=Path, required=True,
                        help="Raw uniref50.fasta.gz -- source of the null model's random sequence pool.")
    parser.add_argument("--thresh-lo", type=float, default=0.0)
    parser.add_argument("--thresh-hi", type=float, default=0.9)
    parser.add_argument("--n-sweep", type=int, default=10)
    parser.add_argument("--n-null-pool", type=int, default=20_000)
    parser.add_argument("--n-null-pairs", type=int, default=1_000_000)
    args = parser.parse_args()

    cache = CacheStore()
    edge_dir = args.edgestore_path.parent
    offsets_path = edge_dir / "offsets.txt"
    patterns_path = edge_dir / "patterns.bin"

    print("[bold][orange2]=== Threshold sweep: UniRef50 (train) vs. ProteinGym+PEER+CASP15 (prod) ===[/][/]")
    print(f"  Loading layer-0 graph: {args.edgestore_path}")
    hnsw_graph = EdgeStore.load(str(args.edgestore_path))

    offsets = load_offsets(offsets_path)
    n_total = hnsw_graph.node_count()
    train_nodes = np.arange(0, offsets["proteingym"], dtype=np.int64)
    prod_nodes = np.arange(offsets["proteingym"], n_total, dtype=np.int64)
    print(f"  train(uniref50)={len(train_nodes):,}  prod(proteingym+peer+casp15)={len(prod_nodes):,}")

    print(f"  Loading pattern set: {patterns_path}")
    patterns = SWPatternSet.load(str(patterns_path))

    print("  Computing gamma(tau) from a shuffled-UniRef50 ProtSpaM null...")
    gamma_cdf = find_gamma_function(
        args.uniref50_fasta, patterns, cache,
        n_pool=args.n_null_pool, n_pairs=args.n_null_pairs,
    )

    thresholds, results = sweep_thresholds(
        hnsw_graph, train_nodes, prod_nodes, gamma_cdf,
        args.thresh_lo, args.thresh_hi, args.n_sweep,
        inweight_type=INWeightType.SimilarityComplement,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"uniref.json"
    with open(out_path, "w") as f:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "results": results}, f, indent=2)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
