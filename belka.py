"""Build and save the HNSW proximity index over the full BELKA *training* set.

Two-phase pipeline, so fingerprint-computation memory and HNSWState-construction memory
never overlap and can be measured/attributed separately:

1. `compute_and_cache_fingerprints_to_disk`: Morgan fingerprints computed in parallel
   (ProcessPoolExecutor + RDKit) and packed+written to disk one at a time — no in-memory
   list/array of fingerprints ever exists. Written under `/mnt/documents` (not `.cache/`):
   the packed array is ~98.4M x 256B =~ 25GB and the repo's filesystem only had ~8GB free.
2. `stream_bitfingerprints_from_disk`: reads that file back via a memory-mapped array and
   yields packed `BitFingerprint`s one at a time, passed directly as `HNSWState`'s `data`
   (refnd>=0.0.3 accepts any Python iterable, not just a sized sequence — checked at runtime
   via `len()`). `HNSWState(...)` always builds its own Rust-owned copy of the data
   regardless of how `data` is passed in; streaming just avoids *also* holding a full
   Python-side list of `BitFingerprint`s at construction time.

(An earlier single-phase version fused computation and streaming into one generator handed
directly to HNSWState. It still ran into ~50GB+ RSS during construction — high enough to
raise real OOM risk before reaching `build()` — with no way to tell whether that came from
the ProcessPoolExecutor/RDKit worker overhead or the Rust-side copy growing concurrently.
Splitting the phases removes that ambiguity and keeps each phase's peak independently bounded.)

The index itself is built with `keep_all_edges=False` and `cache_capacity=0`: BELKA's
combinatorial synthon-sharing structure makes the raw below-threshold proximity-edge count
grow superlinearly with dataset size (measured ~3.8 avg degree at 100K molecules vs. ~18.9 at
500K, at threshold=0.4) — recording all of them would blow far past available memory. The
bounded HNSW graph links (m_max0/m_max) are unaffected by that density and stay a fixed cost
per node.

`strict_ef=True` is required for the same reason. In search_layer.rs, the
per-layer candidate set is only trimmed back to `ef` once its worst member's
distance exceeds `proximity_threshold` (`strict_ef=False`, the HNSWConfig
default) — in a dense synthon cluster, newly found candidates keep beating
that check, so the set balloons past `ef_construction` and search cost tracks
local density instead of staying O(ef). `strict_ef=True` trims unconditionally
at `ef`, decoupling search cost from the threshold. Measured effect at
N=2M: 252s -> 28s (9x). Confirmed by isolating threading (n_threads=1 gives a
build-time exponent ~1.23 close to true n log n vs ~1.61 multi-threaded) that
the residual superlinearity is concurrent add_edge/prune contention on hub
nodes, not a further algorithmic issue — n_threads=0 (all cores) still wins
on absolute throughput despite the worse exponent.

Usage:
    uv run python belka.py
    uv run python belka.py --n-molecules 49207805 --index-path .cache/belka_half_test.hnsw

`--n-molecules` caps the dataset to the first N molecules (timing tests at less than full
scale). If the on-disk fingerprint cache already has at least that many entries, phase 1 is
skipped entirely (no smiles load, no re-fingerprinting) and phase 2 streams just the first N
of the cached fingerprints.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import psutil
from rich import print

from refnd.core import HNSWState
from refnd.kernels import KernelVariant

from src.cache import CacheStore
from src.datasets import DATASETS, belka_unique_smiles
from src.fingerprints import (
    FP_N_BYTES_PACKED,
    compute_and_cache_fingerprints_to_disk,
    stream_bitfingerprints_from_disk,
)

FP_CACHE_PATH = Path("/mnt/documents/refnd_cache/belka_train_fp.bin")
INDEX_PATH = ".cache/belka_train.hnsw"


def _rss_gb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and save the BELKA train HNSW index")
    parser.add_argument("--n-molecules", type=int, default=None,
                         help="Use only the first N molecules instead of the full training set")
    parser.add_argument("--index-path", default=INDEX_PATH)
    args = parser.parse_args()

    cfg = DATASETS["belka"]
    cache = CacheStore()

    print(f"[bold][orange2]=== BELKA train HNSW index build ===[/][/]")
    print(f"RSS at start: {_rss_gb():.2f} GB")

    FP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cached_n = FP_CACHE_PATH.stat().st_size // FP_N_BYTES_PACKED if FP_CACHE_PATH.exists() else 0

    if args.n_molecules is not None and cached_n >= args.n_molecules:
        n_written = args.n_molecules
        print(f"Reusing existing fingerprint cache ({str(FP_CACHE_PATH)!r}): "
              f"using first {n_written:,} of {cached_n:,} cached fingerprints")
    else:
        smiles = belka_unique_smiles(cache)
        if args.n_molecules is not None:
            smiles = smiles[:args.n_molecules]
        print(f"  {len(smiles):,} molecules to fingerprint")
        print(f"RSS after loading smiles: {_rss_gb():.2f} GB")

        print(f"Computing Morgan fingerprints and packing to disk at {str(FP_CACHE_PATH)!r}...")
        t0 = time.time()
        n_written, n_missing = compute_and_cache_fingerprints_to_disk(
            smiles, str(FP_CACHE_PATH), progress=True,
        )
        if n_missing:
            print(f"  [yellow]Dropped {n_missing:,} SMILES RDKit couldn't parse[/]")
        print(f"  fingerprinting took {time.time() - t0:.1f}s")
        del smiles
        gc.collect()
        print(f"RSS after fingerprinting (smiles freed): {_rss_gb():.2f} GB")

    print(f"Building HNSW index (proximity_threshold={cfg.proximity_threshold}, "
          f"keep_all_edges=False, cache_capacity=0, strict_ef=True), "
          f"streaming fingerprints back from disk...")
    t0 = time.time()
    state = HNSWState(
        KernelVariant.TanimotoBit,
        stream_bitfingerprints_from_disk(str(FP_CACHE_PATH), n_written, progress=True),
        proximity_threshold=cfg.proximity_threshold,
        keep_all_edges=False, cache_capacity=0, strict_ef=True,
        **cfg.kernel_params,
    )
    print(f"  construction took {time.time() - t0:.1f}s")
    print(f"RSS after construction: {_rss_gb():.2f} GB")

    t0 = time.time()
    state.build(progress=True)
    print(f"  build() took {time.time() - t0:.1f}s")
    print(f"RSS after build(): {_rss_gb():.2f} GB")

    print(f"Saving index to {args.index_path!r}...")
    state.save(args.index_path)
    print(f"RSS after save: {_rss_gb():.2f} GB")

    print("[bold]Done.[/] Reload with "
          "HNSWState.load(KernelVariant.TanimotoBit, path, fingerprints) — "
          "the same (or freshly recomputed) BitFingerprint list must be "
          "supplied again at load time.")


if __name__ == "__main__":
    main()
