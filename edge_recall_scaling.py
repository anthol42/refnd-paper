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

from rich import print

from src.cache import CacheStore
from src.datasets import DATASETS, SCALING_DATASET_KEY, prepare_hnsw_input
from src.metrics import pct_edges_recovered
from refnd.core import HNSWState, exact_edges
from scaling_benchmark import _load_items, _prepare_subsets, _subset_path

DEFAULT_SIZES: dict[str, list[int]] = {
    "atlas": [5_000, 25_000, 125_000],
    "belka": [25_000, 125_000, 625_000],
}
DEBUG_SIZES   = [5_000, 25_000]
SEED = 42

# Fixed default HNSW params (mirrors main.py's argparse defaults) — everything
# except ef_construction is held constant across sizes.
BASE_EF_CONSTRUCTION = 32


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
                        help="Comma-separated sizes, default: 5000,25000,125000 (atlas) / "
                             "25000,125000,625000 (belka)")
    parser.add_argument("--debug", action="store_true",
                        help="Only use 5K/25K sizes for a fast smoke test (overrides --sizes)")
    parser.add_argument("--size", type=int, default=None,
                        help="Run only this size (default: loop over all --sizes). The "
                             "log-scaled ef_construction baseline is still taken from the "
                             "full --sizes list, not just this one size.")
    parser.add_argument("--output", default=None,
                        help="Results file path (default: results/edge_recall_scaling_{dataset}.json). "
                             "Use a distinct path per task to avoid races when running in parallel.")
    prepare_group = parser.add_mutually_exclusive_group()
    prepare_group.add_argument("--prepare-only", action="store_true",
                        help="Only load the dataset and write subset files, then exit "
                             "(no benchmarking). Run this once before a parallel array "
                             "to avoid concurrent tasks racing to write the same subset file.")
    prepare_group.add_argument("--no-prepare", action="store_true",
                        help="Skip subset preparation and never fall back to it; crash if "
                             "a required subset file is missing. Use in array tasks once "
                             "--prepare-only has already built every subset.")
    args = parser.parse_args()

    dataset = args.dataset

    if args.debug:
        all_sizes = DEBUG_SIZES
    elif args.sizes is not None:
        all_sizes = [int(s) for s in args.sizes.split(",")]
    else:
        all_sizes = DEFAULT_SIZES[dataset]
    # Baseline for the log-scale factor is always the smallest size in the
    # full configured list, regardless of which size(s) this run computes --
    # it's a fixed reference point, not something derived from other tasks.
    baseline_size = all_sizes[0]
    run_sizes = [args.size] if args.size is not None else all_sizes

    cfg     = DATASETS[SCALING_DATASET_KEY[dataset]]
    cache   = CacheStore()
    out_path = Path(args.output) if args.output else Path(f"results/edge_recall_scaling_{dataset}.json")

    print(f"[bold][orange2]=== Edge Recall vs. Dataset Size ({dataset}) ===[/][/]")

    if not args.no_prepare:
        print(f"Loading {dataset} dataset...")
        all_items = _load_items(dataset, cache)
        print(f"  {len(all_items):,} items available")

        print("\nPreparing subsets...")
        _prepare_subsets(dataset, all_items, run_sizes)

        if args.prepare_only:
            print("\n[bold]Subsets prepared, exiting (--prepare-only).[/]")
            return

    records = []

    print(f"\n[green]-- Fixed ef_construction={BASE_EF_CONSTRUCTION} --[/]")
    for size in run_sizes:
        path = _subset_path(dataset, size)
        if not path.exists():
            if args.no_prepare:
                raise FileNotFoundError(
                    f"Subset not found: {path} (--no-prepare set; run with --prepare-only first)"
                )
            print(f"  [dim]Skipping size {size:,}: subset not available[/]")
            continue
        items = _read_subset(path)
        record = _build_and_recall(dataset, cfg, cache, size, items, BASE_EF_CONSTRUCTION)
        record["scaled"] = False
        records.append(record)
        print(f"  size={size:>10,}  recall={record['pct_edges_recovered']:.4f}  "
              f"build={record['hnsw_build_time_s']:.2f}s")

    print(f"\n[green]-- ef_construction scaled by log(n)/log({baseline_size:,}) --[/]")
    for size in run_sizes:
        if size == baseline_size:
            continue  # scale factor is 1.0 — identical to the fixed-ef baseline above
        path = _subset_path(dataset, size)
        if not path.exists():
            continue  # already reported (or raised) in the fixed-ef pass above
        ef_scaled = round(BASE_EF_CONSTRUCTION * math.log(size) / math.log(baseline_size))
        items = _read_subset(path)
        record = _build_and_recall(dataset, cfg, cache, size, items, ef_scaled)
        record["scaled"] = True
        records.append(record)
        print(f"  size={size:>10,}  ef_construction={ef_scaled}  "
              f"recall={record['pct_edges_recovered']:.4f}  "
              f"build={record['hnsw_build_time_s']:.2f}s")

    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n[bold]Results saved to {out_path}[/]")


if __name__ == "__main__":
    main()
