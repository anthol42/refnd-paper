"""Threshold sweep with community detection on the combined BELKA train+test
layer-0 graph, adapting the null model (gamma) to each threshold. Belka analog
of `threshold/community_purity_sweep.py` (see that module's docstring for the
general method) -- differences specific to Belka:

- "train": a random half (~49.2M, seed=42) of BELKA's unique train molecules,
  DNA-conjugation tag [Dy] stripped -- the SAME cached fingerprint file used by
  `threshold/belka.py` (`belka_train_half_stripped_fp_seed42.bin`).
- "prod": the REAL Kaggle BELKA test set in full (878,022 molecules, triazine +
  OOD mixed, unlabeled), reusing `threshold/belka.py`'s cached stripped
  fingerprints.
- gamma(tau) = P(random-atom SELFIES null pair distance <= tau), reusing
  `threshold.belka.build_null_scores` (the "syntax always valid" alternative
  molecule representation null already implemented there) unchanged.
- HNSW build: keep_all_edges=False, strict_ef=True (required at Belka's
  combinatorial density, see belka.py's docstring), cache_capacity=0 (refnd's
  internal distance cache disabled -- Belka's own convention). proximity_threshold
  is a no-op here (only matters for keep_all_edges=True / non-strict_ef) so it's
  passed as 0.
- Threshold sweep: 0.30 to 0.90 in steps of 0.05 (13 points), vs. peptides' 0.35-0.65.
- Plot: same two panels as the peptide version, plus a third panel with the
  number of pure-train / pure-prod communities (or components) vs. threshold.

Usage:
    uv run python -m threshold.belka_purity_sweep --stage build
    uv run python -m threshold.belka_purity_sweep --stage extract
    uv run python -m threshold.belka_purity_sweep --stage sweep --method both
    uv run python -m threshold.belka_purity_sweep --stage all --method both
    uv run python -m threshold.belka_purity_sweep --stage all --test-subset ood --method both

--test-subset selects what "prod" means:
    all (default): full Kaggle test set (triazine + OOD)
    ood:           OOD-only test molecules (triazine excluded) -- train <-> ood test
Each subset gets its own cache files (index, source labels, master/sweep edges,
plots), keyed by a subset-specific filename prefix, so switching subsets never
clobbers another subset's cached artifacts.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
from rich import print

from refnd.core import HNSWState

from src.datasets import DATASETS
from src.fingerprints import FP_N_BYTES_PACKED, stream_bitfingerprints_from_disk

from threshold.belka import (
    NULL_MEAN_SIZE,
    N_NULL_MOLECULES,
    OUT_DIR,
    SEED,
    TEST_OOD_FP_PATH,
    TEST_TRIAZINE_FP_PATH,
    TRAIN_FP_PATH,
    build_null_scores,
)
from threshold.community_purity_sweep import run_threshold

PLOT_DIR = Path(__file__).parent
EF_CONSTRUCTION = 64

THRESH_LO, THRESH_HI, THRESH_STEP = 0.0, 0.40, 0.05
SUBSAMPLE_CAP = 150_000_000

TEST_SUBSET_PREFIXES = {
    "all": "combined_train_test",   # train <-> full test (triazine + OOD)
    "ood": "combined_train_ood",    # train <-> OOD test only (triazine excluded)
}


class Paths:
    """Cache file paths for one --test-subset choice, keyed by a subset-specific
    filename prefix so `all` and `ood` never share/clobber each other's artifacts.
    """

    def __init__(self, test_subset: str):
        prefix = TEST_SUBSET_PREFIXES[test_subset]
        self.test_subset = test_subset
        self.index = OUT_DIR / f"{prefix}_seed42.hnsw"
        self.source = OUT_DIR / f"{prefix}_source_seed42.npy"
        self.edgestr = OUT_DIR / f"{prefix}_layer0.edgestr"
        self.master_u = OUT_DIR / f"{prefix}_layer0_u.npy"
        self.master_v = OUT_DIR / f"{prefix}_layer0_v.npy"
        self.master_w = OUT_DIR / f"{prefix}_layer0_w.npy"
        self.sweep_u = OUT_DIR / f"{prefix}_layer0_u_sub{SUBSAMPLE_CAP}.npy"
        self.sweep_v = OUT_DIR / f"{prefix}_layer0_v_sub{SUBSAMPLE_CAP}.npy"
        self.sweep_w = OUT_DIR / f"{prefix}_layer0_w_sub{SUBSAMPLE_CAP}.npy"


def mem_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1024 / 1024
    return float("nan")


def _fp_count(path: Path) -> int:
    return path.stat().st_size // FP_N_BYTES_PACKED


class _ChainedFPStream:
    """Concatenates multiple `_DiskFingerprintStream`s into one iterable,
    exposing `__length_hint__` (but not `__len__`) so `HNSWState(...)` stays
    on its unsized streaming path -- same convention as `_DiskFingerprintStream`
    itself (see src/fingerprints.py's docstring for why that matters at scale).
    """

    def __init__(self, streams: list, total: int):
        import itertools
        self._chain = itertools.chain(*streams)
        self._total = total

    def __iter__(self):
        return self

    def __length_hint__(self) -> int:
        return self._total

    def __next__(self):
        return next(self._chain)


def build_combined_index(cfg, paths: Paths) -> None:
    if paths.index.exists() and paths.source.exists():
        print(f"  [dim]Cached combined index + source labels already exist, skipping build[/]")
        return

    n_train = _fp_count(TRAIN_FP_PATH)
    n_ood = _fp_count(TEST_OOD_FP_PATH)
    n_tri = _fp_count(TEST_TRIAZINE_FP_PATH) if paths.test_subset == "all" else 0
    n_test = n_ood + n_tri
    n = n_train + n_test
    print(f"n_train={n_train:,}  n_test={n_test:,} (ood={n_ood:,} + triazine={n_tri:,})  n_total={n:,}")

    source = np.zeros(n, dtype=np.uint8)
    source[n_train:] = 1  # 0 = train, 1 = prod (test)
    np.save(paths.source, source)
    print(f"Saved source labels: {paths.source}")

    streams = [
        stream_bitfingerprints_from_disk(str(TRAIN_FP_PATH), n_train, progress=True),
        stream_bitfingerprints_from_disk(str(TEST_OOD_FP_PATH), n_ood, progress=True),
    ]
    if paths.test_subset == "all":
        streams.append(stream_bitfingerprints_from_disk(str(TEST_TRIAZINE_FP_PATH), n_tri, progress=True))
    combined_stream = _ChainedFPStream(streams, n)

    cfg_kwargs = cfg.kernel_params
    print(
        f"Building combined HNSW (proximity_threshold=0 [no-op: keep_all_edges=False, "
        f"strict_ef=True], ef_construction={EF_CONSTRUCTION}, keep_all_edges=False, "
        f"cache_capacity=0, strict_ef=True, n_threads=0)... MemAvailable: {mem_gb():.2f}GB"
    )
    t0 = time.perf_counter()
    hnsw = HNSWState(
        cfg.modality, combined_stream,
        proximity_threshold=0.0,
        ef_construction=EF_CONSTRUCTION,
        keep_all_edges=False,
        cache_capacity=0,
        strict_ef=True,
        **cfg_kwargs,
    )
    print(f"  construction (streamed) took {time.perf_counter()-t0:.1f}s. MemAvailable: {mem_gb():.2f}GB")

    t0 = time.perf_counter()
    hnsw.build(progress=True)
    build_time = time.perf_counter() - t0
    print(f"  build() took {build_time:.1f}s ({build_time/3600:.2f}h). MemAvailable: {mem_gb():.2f}GB")

    hnsw.save(str(paths.index))
    print(f"Saved index: {paths.index}")
    print(f"\nSUMMARY: n_nodes={n:,}  build_time={build_time:.2f}s ({build_time/3600:.2f}h)")


def _load_combined_data(paths: Paths) -> list:
    n_train = _fp_count(TRAIN_FP_PATH)
    n_ood = _fp_count(TEST_OOD_FP_PATH)
    n_tri = _fp_count(TEST_TRIAZINE_FP_PATH) if paths.test_subset == "all" else 0
    print(f"Materializing combined BitFingerprint list (n={n_train+n_ood+n_tri:,})... "
          f"MemAvailable: {mem_gb():.2f}GB")
    combined = []
    combined.extend(stream_bitfingerprints_from_disk(str(TRAIN_FP_PATH), n_train, progress=True))
    combined.extend(stream_bitfingerprints_from_disk(str(TEST_OOD_FP_PATH), n_ood, progress=True))
    if paths.test_subset == "all":
        combined.extend(stream_bitfingerprints_from_disk(str(TEST_TRIAZINE_FP_PATH), n_tri, progress=True))
    print(f"  done. MemAvailable: {mem_gb():.2f}GB")
    return combined


def extract_layer0_master_edges(cfg, paths: Paths) -> None:
    if paths.master_u.exists() and paths.master_v.exists() and paths.master_w.exists():
        print(f"  [dim]Cached master layer0 u/v/w arrays already exist, skipping extraction[/]")
        return

    combined = _load_combined_data(paths)

    print(f"Loading combined HNSW index ({paths.index.name})...")
    t0 = time.perf_counter()
    hnsw = HNSWState.load(cfg.modality, str(paths.index), combined)
    print(f"  index load: {time.perf_counter()-t0:.2f}s. MemAvailable: {mem_gb():.2f}GB")

    print("Extracting layer0 edges (deduped, real distances computed Rust-side)...")
    t0 = time.perf_counter()
    es = hnsw.get_layer(0, directed=False, weights=True, progress=True)
    n_edges = len(es)
    print(f"  get_layer: {time.perf_counter()-t0:.2f}s  n_edges={n_edges:,}")
    del hnsw, combined
    gc.collect()
    print(f"  freed HNSW index + data copy. MemAvailable: {mem_gb():.2f}GB")

    t0 = time.perf_counter()
    es.save(str(paths.edgestr))
    print(f"  Saved raw EdgeStore (safety checkpoint): {paths.edgestr}  ({time.perf_counter()-t0:.2f}s)")

    print(f"Streaming {n_edges:,} edges into preallocated numpy arrays...")
    u = np.empty(n_edges, dtype=np.uint32)
    v = np.empty(n_edges, dtype=np.uint32)
    w = np.empty(n_edges, dtype=np.float32)
    t0 = time.perf_counter()
    for i, (s, d, dist) in enumerate(es):
        u[i] = s
        v[i] = d
        w[i] = dist
    print(f"  streamed to numpy: {time.perf_counter()-t0:.2f}s")
    print(f"  dist range: min={w.min():.4f}  median={np.median(w):.4f}  max={w.max():.4f}")
    del es
    gc.collect()

    np.save(paths.master_u, u)
    np.save(paths.master_v, v)
    np.save(paths.master_w, w)
    print(f"Saved master u/v/w arrays ({paths.master_u.name} etc, n_edges={n_edges:,})")


def derive_ood_master_edges(paths_ood: Paths) -> None:
    """Restrict the already-extracted 'all' master layer0 edges + source labels to
    train + OOD-test nodes only (drop triazine), instead of rebuilding an HNSW index
    and re-extracting layer0 from scratch. OOD-test molecules were appended right
    after train and before triazine when the 'all' combined index was built (see
    `build_combined_index`'s stream order), so "train + OOD" is exactly the
    contiguous node-id prefix [0, n_train + n_ood) -- no id remapping needed.
    """
    if paths_ood.master_u.exists() and paths_ood.master_v.exists() and paths_ood.master_w.exists():
        print(f"  [dim]Cached ood master layer0 u/v/w arrays already exist, skipping derivation[/]")
        return

    paths_all = Paths("all")
    n_train = _fp_count(TRAIN_FP_PATH)
    n_ood = _fp_count(TEST_OOD_FP_PATH)
    n_keep = n_train + n_ood
    print(f"Deriving OOD-only subset from cached 'all' master edges "
          f"(n_keep={n_keep:,} = train {n_train:,} + ood {n_ood:,}, dropping triazine nodes)...")

    source_all = np.load(paths_all.source)
    np.save(paths_ood.source, source_all[:n_keep])
    print(f"  Saved source labels: {paths_ood.source}")

    u = np.load(paths_all.master_u)
    v = np.load(paths_all.master_v)
    w = np.load(paths_all.master_w)
    t0 = time.perf_counter()
    keep = (u < n_keep) & (v < n_keep)
    u, v, w = u[keep], v[keep], w[keep]
    print(f"  filtered {keep.sum():,} / {len(keep):,} edges ({time.perf_counter()-t0:.2f}s)")
    del keep
    gc.collect()

    np.save(paths_ood.master_u, u)
    np.save(paths_ood.master_v, v)
    np.save(paths_ood.master_w, w)
    print(f"Saved OOD master u/v/w arrays ({paths_ood.master_u.name} etc, n_edges={len(u):,})")


def _load_sweep_edges(paths: Paths) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = np.load(paths.source)
    n = len(source)

    if paths.sweep_u.exists():
        print(f"  [dim]Loading cached subsampled sweep edges[/]")
        u = np.load(paths.sweep_u)
        v = np.load(paths.sweep_v)
        w = np.load(paths.sweep_w)
        return source, u, v, w

    u = np.load(paths.master_u)
    v = np.load(paths.master_v)
    w = np.load(paths.master_w)
    n_edges = len(u)
    if n_edges > SUBSAMPLE_CAP:
        print(f"  Subsampling master layer0 edges {n_edges:,} -> {SUBSAMPLE_CAP:,} "
              f"(per-threshold EdgeStore construction needs a Python list of tuples; "
              f"the full master set would risk OOM, same as the earlier peptide edges065 case)...")
        rng = np.random.default_rng(SEED)
        idx = rng.choice(n_edges, size=SUBSAMPLE_CAP, replace=False)
        idx.sort()
        u, v, w = u[idx], v[idx], w[idx]
    np.save(paths.sweep_u, u)
    np.save(paths.sweep_v, v)
    np.save(paths.sweep_w, w)
    return source, u, v, w


def plot_summary_belka(results: list[dict], out_path: Path, method: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    taus = [r["threshold"] for r in results]
    med_tt = [np.median(r["tt_vals"]) if len(r["tt_vals"]) else np.nan for r in results]
    med_tp = [np.median(r["tp_vals"]) if len(r["tp_vals"]) else np.nan for r in results]
    q25_tt = [np.percentile(r["tt_vals"], 25) if len(r["tt_vals"]) else np.nan for r in results]
    q75_tt = [np.percentile(r["tt_vals"], 75) if len(r["tt_vals"]) else np.nan for r in results]
    q25_tp = [np.percentile(r["tp_vals"], 25) if len(r["tp_vals"]) else np.nan for r in results]
    q75_tp = [np.percentile(r["tp_vals"], 75) if len(r["tp_vals"]) else np.nan for r in results]
    ks = [r["ks_stat"] for r in results]
    n_pure_train = [r["n_pure_train"] for r in results]
    n_pure_prod = [r["n_pure_prod"] for r in results]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 12.5), dpi=150, sharex=True)
    fig.patch.set_facecolor("white")

    ax1.plot(taus, med_tt, color="#D55E00", linewidth=2.0, marker="o", markersize=4,
              label="min dist: train → nearest OTHER pure train community")
    ax1.fill_between(taus, q25_tt, q75_tt, color="#D55E00", alpha=0.15)
    ax1.plot(taus, med_tp, color="#0072B2", linewidth=2.0, marker="o", markersize=4,
              label="min dist: train → nearest pure prod community")
    ax1.fill_between(taus, q25_tp, q75_tp, color="#0072B2", alpha=0.15)
    ax1.set_ylabel("min distance between communities")
    ax1.set_title(f"BELKA: Median (IQR band) min inter-community distance vs. threshold ({method})", loc="left")
    ax1.legend(loc="best", fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    ax2.plot(taus, ks, color="#009E73", linewidth=2.0, marker="o", markersize=4)
    if not all(np.isnan(ks)):
        best_idx = int(np.nanargmin(ks))
        ax2.axvline(taus[best_idx], color="#6b7280", linewidth=1.2, linestyle="--",
                     label=f"min KS at threshold={taus[best_idx]:.4f}")
        ax2.legend(loc="best", fontsize=9, frameon=False)
    ax2.set_ylabel("KS statistic (train-train vs. train-prod)")
    ax2.set_title("Two-sample KS distance between the two distributions (lower = more similar)", loc="left")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    ax3.plot(taus, n_pure_train, color="#D55E00", linewidth=2.0, marker="o", markersize=4,
              label="# pure train communities")
    ax3.plot(taus, n_pure_prod, color="#0072B2", linewidth=2.0, marker="o", markersize=4,
              label="# pure prod communities")
    ax3.set_xlabel(r"threshold $\tau$")
    ax3.set_ylabel("# pure communities")
    ax3.set_title(f"Pure train / pure prod community count vs. threshold ({method})", loc="left")
    ax3.legend(loc="best", fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax3.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


def run_sweep(cfg, paths: Paths, methods: list[str]) -> None:
    source, u, v, w = _load_sweep_edges(paths)
    n = len(source)
    print(f"  n={n:,}  n_edges(sweep)={len(u):,}")

    null_scores = build_null_scores(cfg)
    print(f"  null (random-atom, always-valid-SELFIES pairs): n={len(null_scores):,} "
          f"min={null_scores.min():.4f} median={np.median(null_scores):.4f} max={null_scores.max():.4f}")

    n_steps = int(round((THRESH_HI - THRESH_LO) / THRESH_STEP)) + 1
    thresholds = np.round(np.linspace(THRESH_LO, THRESH_HI, n_steps), 4)
    print(f"  sweep=[{THRESH_LO},{THRESH_HI}] step={THRESH_STEP} n_points={len(thresholds)}")

    for method in methods:
        print(f"\n[bold][orange2]=== BELKA community purity sweep: {method} ===[/][/]")
        t_total = time.perf_counter()
        results = [
            run_threshold(tau, source, u, v, w, null_scores, n, method=method)
            for tau in thresholds
        ]
        total_time = time.perf_counter() - t_total
        print(f"[bold]Total sweep time ({method}): {total_time:.2f}s ({total_time/60:.2f} min) "
              f"for {len(thresholds)} thresholds ({total_time/len(thresholds):.2f}s/threshold avg)[/]")

        subset_tag = "" if paths.test_subset == "all" else f"_{paths.test_subset}"
        suffix = "" if method == "cpm" else f"_{method}"
        plot_summary_belka(results, PLOT_DIR / f"belka_purity_sweep{subset_tag}{suffix}.png", method=method)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["build", "extract", "sweep", "all"], default="all")
    parser.add_argument("--method", choices=["cpm", "components", "both"], default="both")
    parser.add_argument("--test-subset", choices=["all", "ood"], default="all")
    args = parser.parse_args()

    cfg = DATASETS["belka"]
    paths = Paths(args.test_subset)
    methods = ["cpm", "components"] if args.method == "both" else [args.method]

    if paths.test_subset == "ood":
        if args.stage in ("build", "extract", "all"):
            print("[bold][orange2]=== Stage: derive OOD-only subset from cached 'all' master edges ===[/][/]")
            derive_ood_master_edges(paths)
    else:
        if args.stage in ("build", "all"):
            print("[bold][orange2]=== Stage: build combined train+test HNSW index ===[/][/]")
            build_combined_index(cfg, paths)
        if args.stage in ("extract", "all"):
            print("[bold][orange2]=== Stage: extract + measure layer0 master edges ===[/][/]")
            extract_layer0_master_edges(cfg, paths)
    if args.stage in ("sweep", "all"):
        print("[bold][orange2]=== Stage: threshold sweep ===[/][/]")
        run_sweep(cfg, paths, methods)


if __name__ == "__main__":
    main()
