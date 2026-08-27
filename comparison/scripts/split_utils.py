"""Shared utilities for all splitting scripts."""
import json
import os
import platform
import resource
import time
from contextlib import contextmanager
from pathlib import Path

# Repo root (refnd-paper/), independent of cwd -- split_utils.py lives at
# comparison/scripts/split_utils.py.
REPO_ROOT = Path(__file__).resolve().parents[2]

# comparison/ protein dataset key -> src/datasets.py loader key. dbaasp_amp has
# no dedicated loader; src.datasets' "dbaasp" (DBAASP, L-amino/canonical only,
# E. coli MIC labels) is the closest auto-downloadable equivalent.
_PROTEIN_SRC_DATASET = {
    "dbaasp_amp": "dbaasp",
    "enzyme_topt": "enzyme_topt",
}


def load_protein_sequences(dataset: str) -> list[str]:
    """Sequences for a comparison/ protein dataset key, via src/datasets.py's
    auto-downloading loaders (no comparison/data/ parquet required, no
    hardcoded paths -- runs on any machine)."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from src.cache import CacheStore
    from src.datasets import load_dataset

    src_name = _PROTEIN_SRC_DATASET[dataset]
    sequences, _labels = load_dataset(src_name, CacheStore(root=str(REPO_ROOT / ".cache")))
    return list(sequences)


def _peak_rss_mb() -> float:
    """Process high-water RSS in MB, including any child processes (e.g. the
    mmseqs2 binary). ru_maxrss is KB on Linux, bytes on macOS."""
    peak = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    div = 1024 if platform.system() == "Linux" else 1024 * 1024
    return peak / div


@contextmanager
def track_split(method: str, dataset: str, splits_dir: str, seed: int = None):
    """Record wall-time + peak RSS for a split computation to
    results/split_metrics/{method}/{dataset}[_seed{seed}].json.

    Pass seed only for methods that split one seed per process (datasail); the
    build-once methods wrap the whole dataset (all seeds) and omit it."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        seconds = time.perf_counter() - t0
        out_dir = Path(splits_dir).parent / "results" / "split_metrics" / method
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"{dataset}_seed{seed}.json" if seed is not None else f"{dataset}.json"
        rec = {"method": method, "dataset": dataset, "seed": seed,
               "seconds": round(seconds, 2), "peak_rss_mb": round(_peak_rss_mb(), 1)}
        tmp = out_dir / (name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=2)
        os.replace(tmp, out_dir / name)
        print(f"[metrics] {method}/{dataset}"
              f"{'' if seed is None else f' seed{seed}'}: "
              f"{seconds:.1f}s, peak {rec['peak_rss_mb']:.0f}MB")


def save_split(splits_dir: str, method: str, dataset: str, seed: int,
               train: list, val: list, test: list):
    out_dir = Path(splits_dir) / method / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{seed}.json"
    # Write atomically: a killed job must never leave a truncated stub that
    # later gets skipped by the "if path.exists()" logic.
    tmp = out_dir / f"{seed}.json.tmp"
    with open(tmp, "w") as f:
        json.dump({"train": train, "val": val, "test": test}, f)
    os.replace(tmp, path)


def load_split(splits_dir: str, method: str, dataset: str, seed: int):
    path = Path(splits_dir) / method / dataset / f"{seed}.json"
    with open(path) as f:
        d = json.load(f)
    return d["train"], d["val"], d["test"]


def _label_stats(labels):
    """Count / largest-fraction / singleton summary for a per-node label array."""
    import numpy as np
    lab = np.asarray(labels, dtype=np.int64)
    n = lab.size
    uniq, counts = np.unique(lab, return_counts=True)
    return {
        "n": int(n),
        "count": int(uniq.size),
        "largest_frac": round(float(counts.max()) / n, 4) if n else None,
        "singletons": int((counts == 1).sum()),
    }, lab


def save_community_stats(splits_dir: str, method: str, dataset: str,
                         communities, per_seed_local: dict, components=None):
    """Persist cluster/community counts for a split to
    results/split_metrics/{method}/{dataset}_communities.json (atomic).

    communities: 1-D per-node community labels (Leiden/CPM), in local node order.
    per_seed_local: {seed: (train_val_local_idx, test_local_idx)} -- node indices
    (same local order as `communities`) so we can count distinct communities per side.
    components: optional 1-D per-node connected-component labels; its largest_frac
    is the giant-component diagnostic (the collapse signal)."""
    import numpy as np
    comm_summary, comm = _label_stats(communities)
    stats = {
        "method": method, "dataset": dataset,
        "n_nodes": comm_summary["n"],
        "n_communities": comm_summary["count"],
        "largest_community_frac": comm_summary["largest_frac"],
        "singleton_communities": comm_summary["singletons"],
        "per_seed": {},
    }
    if components is not None:
        comp_summary, _ = _label_stats(components)
        stats["n_components"] = comp_summary["count"]
        stats["largest_component_frac"] = comp_summary["largest_frac"]
    for seed, (tr_local, te_local) in per_seed_local.items():
        tr = np.asarray(tr_local, dtype=np.int64)
        te = np.asarray(te_local, dtype=np.int64)
        stats["per_seed"][str(seed)] = {
            "train_communities": int(np.unique(comm[tr]).size) if tr.size else 0,
            "test_communities": int(np.unique(comm[te]).size) if te.size else 0,
        }
    out_dir = Path(splits_dir).parent / "results" / "split_metrics" / method
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dataset}_communities.json"
    tmp = out_dir / f"{dataset}_communities.json.tmp"
    with open(tmp, "w") as f:
        json.dump(stats, f, indent=2)
    os.replace(tmp, path)
    print(f"[communities] {method}/{dataset}: {stats['n_communities']} communities"
          + (f", largest component {stats['largest_component_frac']}"
             if components is not None else ""))


def save_refnd_sizes(splits_dir: str, dataset: str, sizes: dict):
    """sizes: {seed: {"train": N, "val": N, "test": N}}"""
    path = Path(splits_dir) / "refnd" / dataset / "sizes.json"
    with open(path, "w") as f:
        json.dump(sizes, f, indent=2)


def load_refnd_sizes(splits_dir: str, dataset: str) -> dict:
    path = Path(splits_dir) / "refnd" / dataset / "sizes.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Refnd sizes not found at {path}. Run refnd_split first."
        )
    with open(path) as f:
        d = json.load(f)
    if len(d) < 10:
        raise ValueError(
            f"sizes.json for {dataset} only has {len(d)}/10 seeds. "
            "Refnd split may be incomplete."
        )
    return d


def split_train_val(train_idx: list, val_ratio: float, seed: int) -> tuple:
    """Split train indices into train/val."""
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = np.array(train_idx)
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_ratio))
    return idx[n_val:].tolist(), idx[:n_val].tolist()


SEEDS = list(range(1, 11))

PROTEIN_DATASETS = [
    "dbaasp_amp",
    "enzyme_topt",
]

MOLECULE_DATASETS = [
    "cyp2c19_veith",
    "caco2_wang",
    "pgp_broccatelli",
    "ames",
    "lipophilicity",
    "sr_are",  # Tox21 SR-ARE assay (imbalanced, benefits from graph-based splitting)
]

DNA_DATASETS = [
    "gue_prom_core_all",   # core promoter regions
    "gue_prom_300_all",    # larger (300 bp) promoter regions
    "gue_emp_h3",          # epigenetic markers H3
    "gue_emp_h4",          # epigenetic markers H4
    "gue_mouse_0",         # mouse TF-binding regions
    "deeppromoter",        # E. coli strong vs. weak promoters (khanhlee/deepPromoter)
]
