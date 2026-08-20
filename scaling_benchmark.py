"""Scaling benchmark: split pipeline runtime vs. dataset size.

Usage:
    uv run python scaling_benchmark.py --dataset atlas
    uv run python scaling_benchmark.py --dataset belka

Each method in METHODS is a standalone Python script that takes the dataset
key and the input subset file as positional arguments. The orchestrator runs
it under /usr/bin/time to capture true peak RSS including Rust heap allocations.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Union

import numpy as np
from rich import print

from src.cache import CacheStore
from src.datasets import belka_unique_smiles, load_dataset

SEED    = 42
TIMEOUT = 86_400           # 1 day in seconds
TMP_DIR = Path(os.environ.get("REFND_CACHE_DIR", ".cache")) / "scaling_tmp"

Size = Union[int, str]     # int for a subsample size, or the literal "full"

# ── Dataset registry: name → subset sizes ──────────────────────────────────────

DATASET_SIZES: dict[str, list[Size]] = {
    "atlas": [5_000, 25_000, 125_000, 625_000, 3_125_000],
    "belka": [25_000, 125_000, 625_000, 3_125_000, 15_625_000],
}

DEBUG_SIZES: list[Size] = [5_000, 25_000]


def _parse_size(raw: str) -> Size:
    return raw if raw == "full" else int(raw)


def _load_items(dataset: str, cache: CacheStore) -> list[str]:
    if dataset == "atlas":
        data, _ = load_dataset("peptide_atlas", cache)
        return data
    elif dataset == "belka":
        return belka_unique_smiles(cache)
    raise ValueError(f"Unknown dataset: {dataset!r}")


# ── Method registry: name → runtime_scripts module (run via `python -m`) ──────

METHODS: dict[str, str] = {
    "refnd": "runtime_scripts.split_refnd",
    "hestia": "runtime_scripts.split_hestia",
    # "hnsw-only": "runtime_scripts.split_hnsw_only",
}

# ── FASTA helpers ──────────────────────────────────────────────────────────────
# Also used for SMILES subsets — plain text, ">header\n<sequence-or-smiles>\n".

def _write_fasta(path: Path, items: list[str]) -> None:
    with open(path, "w") as f:
        for i, item in enumerate(items):
            f.write(f">seq_{i}\n{item}\n")


# ── Subset preparation ─────────────────────────────────────────────────────────

def _subset_path(dataset: str, size: Size) -> Path:
    return TMP_DIR / f"{dataset}_{size}.fasta"


def _prepare_subsets(dataset: str, data: list[str], sizes: list[Size]) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for size in sizes:
        path = _subset_path(dataset, size)
        if path.exists():
            continue
        if size == "full":
            _write_fasta(path, data)
            print(f"  Cached full set of {len(data):,} → {path}")
            continue
        if size > len(data):
            print(f"  [yellow]Skipping size {size:,}: only {len(data):,} samples available[/]")
            continue
        idx = rng.choice(len(data), size=size, replace=False)
        _write_fasta(path, [data[i] for i in idx])
        print(f"  Cached subset of {size:,} → {path}")


# ── Results I/O ────────────────────────────────────────────────────────────────

def _results_path(dataset: str, output: str | None) -> Path:
    return Path(output) if output else Path(f"results/runtime_{dataset}.json")


def _load_results(path: Path) -> list[dict]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def _save_results(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


# ── Subprocess runner ──────────────────────────────────────────────────────────

_IS_MACOS = platform.system() == "Darwin"
_TIME_FLAG = "-l" if _IS_MACOS else "-v"


def _parse_peak_rss_bytes(stderr: str) -> int | None:
    """Parse peak RSS from /usr/bin/time stderr (bytes on macOS, kbytes on Linux)."""
    if _IS_MACOS:
        m = re.search(r"(\d+)\s+maximum resident set size", stderr)
        return int(m.group(1)) if m else None
    else:
        m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", stderr)
        return int(m.group(1)) * 1024 if m else None


def _run_subprocess(method_name: str, module: str, dataset: str, size: Size,
                     input_path: Path) -> dict:
    """Run module under /usr/bin/time in a fresh process; return timing/memory record."""
    cmd = ["/usr/bin/time", _TIME_FLAG, "uv", "run", "python", "-m", module, dataset, str(input_path)]

    t0     = time.perf_counter()
    status = "ok"
    peak_b = None
    try:
        result  = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        elapsed = time.perf_counter() - t0
        peak_b  = _parse_peak_rss_bytes(result.stderr)
        if result.returncode != 0:
            status = f"error: rc={result.returncode}"
            if result.stderr:
                print(f"    [red]{result.stderr.strip()[:400]}[/]")
    except subprocess.TimeoutExpired:
        elapsed = TIMEOUT
        status  = "timeout"

    return {
        "method":      method_name,
        "size":        size,
        "runtime_s":   round(elapsed, 3),
        "peak_mem_mb": round(peak_b / 1024 ** 2, 2) if peak_b is not None else None,
        "status":      status,
    }


# ── Orchestrator ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Split pipeline scaling benchmark")
    parser.add_argument("--dataset", choices=list(DATASET_SIZES), default="atlas")
    parser.add_argument("--debug", action="store_true",
                        help="Only use 5K/25K sizes for a fast smoke test")
    parser.add_argument("--method", choices=list(METHODS), default=None,
                        help="Run only this method (default: loop over all methods)")
    parser.add_argument("--size", type=_parse_size, default=None,
                        help="Run only this size, e.g. 25000 or 'full' "
                             "(default: loop over all sizes for --dataset)")
    parser.add_argument("--output", default=None,
                        help="Results file path (default: results/runtime_{dataset}.json). "
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
    all_sizes = DEBUG_SIZES if args.debug else DATASET_SIZES[dataset]
    sizes = [args.size] if args.size is not None else all_sizes
    methods = {args.method: METHODS[args.method]} if args.method else METHODS
    results_path = _results_path(dataset, args.output)

    cache = CacheStore()

    print(f"[bold][orange2]=== {dataset} Scaling Benchmark ===[/][/]")

    if not args.no_prepare:
        print(f"Loading {dataset} dataset...")
        data = _load_items(dataset, cache)
        print(f"  {len(data):,} items")

        print("\nPreparing subsets...")
        _prepare_subsets(dataset, data, sizes)

        if args.prepare_only:
            print("\n[bold]Subsets prepared, exiting (--prepare-only).[/]")
            return

    records = _load_results(results_path)

    for method_name, module in methods.items():
        print(f"\n[green]-- Method: {method_name} --[/]")
        for size in sizes:
            input_path = _subset_path(dataset, size)
            if not input_path.exists():
                if args.no_prepare:
                    raise FileNotFoundError(
                        f"Subset not found: {input_path} (--no-prepare set; "
                        f"run with --prepare-only first)"
                    )
                print(f"  [dim]Skipping size {size}: subset not available[/]")
                continue

            size_s = f"{size:,}" if isinstance(size, int) else size
            print(f"  size={size_s:>10} ... ", end="", flush=True)
            record = _run_subprocess(method_name, module, dataset, size, input_path)
            records.append(record)
            _save_results(results_path, records)

            s     = record["status"]
            mem   = record["peak_mem_mb"]
            color = "green" if s == "ok" else "yellow" if s == "timeout" else "red"
            mem_s = f"{mem:.0f} MB" if mem is not None else "N/A"
            print(f"[{color}]{record['runtime_s']:.1f}s  {mem_s}  ({s})[/]")

            if s == "timeout":
                print(f"  [yellow]Timeout — skipping larger sizes for {method_name}[/]")
                break

    print(f"\n[bold]Results saved to {results_path}[/]")


if __name__ == "__main__":
    main()
