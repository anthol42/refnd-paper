"""Effect of dataset size on HNSW edge recall.

Computes exact edges (cached) and HNSW edges at several dataset sizes with
fixed default HNSW parameters, to see how `pct_edges_recovered` degrades as
n grows. Then repeats the HNSW build at each size with `ef_construction`
scaled proportionally to log(n) (relative to the smallest/baseline size) to
check whether that keeps recall roughly constant.

Usage:
    uv run python edge_recall_scaling.py --dataset atlas
    uv run python edge_recall_scaling.py --dataset belka
    uv run python edge_recall_scaling.py --dataset atlas --sizes 5000,25000,125000
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import numpy as np

from rich import print

from src.cache import CacheStore
from src.datasets import DATASETS, SCALING_DATASET_KEY, prepare_hnsw_input
from src.metrics import pct_edges_recovered
from refnd.core import HNSWState, exact_edges
from scaling_benchmark import _load_items, _subset_path, _write_fasta

DEFAULT_SIZES = [5_000, 25_000, 125_000]
DEBUG_SIZES   = [5_000, 25_000]
SEED = 42

# Fixed default HNSW params (mirrors main.py's argparse defaults) — everything
# except ef_construction is held constant across sizes.
BASE_EF_CONSTRUCTION = 32


def _prepare_subsets(dataset: str, data: list[str], sizes: list[int]) -> None:
    Path(".cache/scaling_tmp").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for size in sizes:
        path = _subset_path(dataset, size)
        if path.exists():
            continue
        if size > len(data):
            print(f"  [yellow]Skipping size {size:,}: only {len(data):,} samples available[/]")
            continue
        idx = rng.choice(len(data), size=size, replace=False)
        _write_fasta(path, [data[i] for i in idx])
        print(f"  Cached subset of {size:,} → {path}")


def _read_subset(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.startswith(">")]


def _build_and_recall(
    dataset_key: str, cfg, cache: CacheStore, size: int, items: list[str], ef_construction: int,
) -> dict:
    data = prepare_hnsw_input(dataset_key, items)

    exact_key = f"{dataset_key}_{size}_exact"
    exact_es = cache.get_edges(exact_key)
    if exact_es is None:
        print(f"    Computing exact edges for size={size:,} (cached after this)...")
        exact_es = exact_edges(
            cfg.modality, data,
            proximity_threshold=cfg.proximity_threshold,
            **cfg.kernel_params,
        )
        cache.store_edges(exact_key, exact_es)

    hnsw = HNSWState(
        cfg.modality, data,
        proximity_threshold=cfg.proximity_threshold,
        ef_construction=ef_construction,
        **cfg.kernel_params,
    )
    t0 = time.perf_counter()
    hnsw.build(progress=False)
    build_time = time.perf_counter() - t0
    hnsw_es = hnsw.edges()

    recall = pct_edges_recovered(hnsw_es, exact_es)
    return {
        "size": size,
        "ef_construction": ef_construction,
        "pct_edges_recovered": recall,
        "hnsw_build_time_s": round(build_time, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Effect of dataset size on HNSW edge recall")
    parser.add_argument("--dataset", choices=["atlas", "belka"], default="atlas")
    parser.add_argument("--sizes", type=str, default=None,
                        help="Comma-separated sizes, default: 5000,25000,125000")
    parser.add_argument("--debug", action="store_true",
                        help="Only use 5K/25K sizes for a fast smoke test (overrides --sizes)")
    args = parser.parse_args()

    if args.debug:
        sizes = DEBUG_SIZES
    elif args.sizes is not None:
        sizes = [int(s) for s in args.sizes.split(",")]
    else:
        sizes = DEFAULT_SIZES
    baseline_size = sizes[0]

    dataset = args.dataset
    cfg     = DATASETS[SCALING_DATASET_KEY[dataset]]
    cache   = CacheStore()

    print(f"[bold][orange2]=== Edge Recall vs. Dataset Size ({dataset}) ===[/][/]")
    print(f"Loading {dataset} dataset...")
    all_items = _load_items(dataset, cache)
    print(f"  {len(all_items):,} items available")

    print("\nPreparing subsets...")
    _prepare_subsets(dataset, all_items, sizes)

    records = []

    print(f"\n[green]-- Fixed ef_construction={BASE_EF_CONSTRUCTION} --[/]")
    for size in sizes:
        path = _subset_path(dataset, size)
        if not path.exists():
            print(f"  [dim]Skipping size {size:,}: subset not available[/]")
            continue
        items = _read_subset(path)
        record = _build_and_recall(dataset, cfg, cache, size, items, BASE_EF_CONSTRUCTION)
        record["scaled"] = False
        records.append(record)
        print(f"  size={size:>10,}  recall={record['pct_edges_recovered']:.4f}  "
              f"build={record['hnsw_build_time_s']:.2f}s")

    print(f"\n[green]-- ef_construction scaled by log(n)/log({baseline_size:,}) --[/]")
    for size in sizes:
        path = _subset_path(dataset, size)
        if not path.exists():
            continue
        if size == baseline_size:
            continue  # scale factor is 1.0 — identical to the fixed-ef baseline above
        ef_scaled = round(BASE_EF_CONSTRUCTION * math.log(size) / math.log(baseline_size))
        items = _read_subset(path)
        record = _build_and_recall(dataset, cfg, cache, size, items, ef_scaled)
        record["scaled"] = True
        records.append(record)
        print(f"  size={size:>10,}  ef_construction={ef_scaled}  "
              f"recall={record['pct_edges_recovered']:.4f}  "
              f"build={record['hnsw_build_time_s']:.2f}s")

    out_path = Path(f"results/edge_recall_scaling_{dataset}.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n[bold]Results saved to {out_path}[/]")


if __name__ == "__main__":
    main()
