"""Per-stage timing breakdown for the refnd HNSW->Leiden pipeline, across dataset sizes.

Post-filtering is disabled so `partition()` timing isn't polluted by its
violation-checking cost.

Usage:
    uv run python test_scaling_detailed_refnd.py
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from refnd.core import HNSWState, INWeightType, LeidenObjective, find_communities, partition
from refnd import KernelVariant

from src.cache import CacheStore
from src.datasets import DATASETS, SCALING_DATASET_KEY, belka_unique_smiles, load_dataset, prepare_hnsw_input

SIZES = {
    "atlas": [5_000, 25_000, 125_000, 625_000, 3_125_000],
    "belka": [25_000, 125_000, 625_000, 3_125_000, 15_625_000],
}
DEBUG_SIZES = [5_000, 25_000]
SEED    = 42
TMP_DIR = Path(".cache/scaling_tmp")
RESULTS_DIR = Path("results")


def _results_path(dataset: str) -> Path:
    suffix = "" if dataset == "atlas" else f"_{dataset}"
    return RESULTS_DIR / f"scaling_detailed_refnd{suffix}.json"


def _subset_path(dataset: str, size: int) -> Path:
    return TMP_DIR / f"{dataset}_{size}.fasta"


def _write_fasta(path: Path, sequences: list[str]) -> None:
    with open(path, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq_{i}\n{seq}\n")


def _prepare_subsets(dataset: str, sizes: list[int]) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    needed = [s for s in sizes if not _subset_path(dataset, s).exists()]
    if not needed:
        return
    print(f"Loading {dataset} dataset to generate missing subsets...")
    cache = CacheStore()
    if dataset == "belka":
        data = belka_unique_smiles(cache)
    else:
        data, _ = load_dataset("peptide_atlas", cache)
    print(f"  {len(data):,} items")
    rng = np.random.default_rng(SEED)
    for size in needed:
        if size > len(data):
            print(f"  [skip] size={size:,}: only {len(data):,} samples available")
            continue
        idx = rng.choice(len(data), size=size, replace=False)
        _write_fasta(_subset_path(dataset, size), [data[i] for i in idx])
        print(f"  Cached subset of {size:,} → {_subset_path(dataset, size)}")


def _read_fasta(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.startswith(">")]


def _time_one(dataset: str, size: int) -> dict | None:
    path = _subset_path(dataset, size)
    if not path.exists():
        print(f"  [skip] size={size:,}: subset not available")
        return None

    items = _read_fasta(path)
    cfg  = DATASETS[SCALING_DATASET_KEY[dataset]]
    data = prepare_hnsw_input(dataset, items)

    t0 = time.perf_counter()
    hnsw = HNSWState(cfg.modality, data, proximity_threshold=cfg.proximity_threshold,
                     cache_capacity=0 if cfg.modality == KernelVariant.TanimotoBit else 2_000_000, **cfg.kernel_params)
    hnsw.build(progress=True)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    es = hnsw.edges()
    t_edges = time.perf_counter() - t0
    n_edges = len(es)

    t0 = time.perf_counter()
    graph = es.graph(inweight_type=INWeightType.Distance)
    t_graph = time.perf_counter() - t0

    t0 = time.perf_counter()
    coms = find_communities(graph, gamma=1.0, objective=LeidenObjective.Modularity)
    t_leiden = time.perf_counter() - t0

    t0 = time.perf_counter()
    partition(coms, graph, test_ratio=0.2, post_filtering=False)
    t_partition = time.perf_counter() - t0

    record = {
        "size": size,
        "n_edges": n_edges,
        "build_s": round(t_build, 3),
        "edges_s": round(t_edges, 3),
        "graph_s": round(t_graph, 3),
        "leiden_s": round(t_leiden, 3),
        "partition_s": round(t_partition, 3),
        "total_s": round(t_build + t_edges + t_graph + t_leiden + t_partition, 3),
    }
    print(f"  size={size:>10,}  n_edges={n_edges:,}  build={t_build:.2f}s  edges={t_edges:.2f}s  "
          f"graph={t_graph:.2f}s  leiden={t_leiden:.2f}s  partition={t_partition:.2f}s")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-stage timing breakdown of the refnd pipeline")
    parser.add_argument("--dataset", choices=list(SIZES), default="atlas")
    parser.add_argument("--debug", action="store_true",
                        help="Only use 5K/25K sizes for a fast smoke test")
    args = parser.parse_args()
    sizes = DEBUG_SIZES if args.debug else SIZES[args.dataset]
    results_path = _results_path(args.dataset)

    results_path.parent.mkdir(exist_ok=True)
    _prepare_subsets(args.dataset, sizes)
    records = []
    for size in sizes:
        record = _time_one(args.dataset, size)
        if record is not None:
            records.append(record)
            results_path.write_text(json.dumps(records, indent=2))
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
