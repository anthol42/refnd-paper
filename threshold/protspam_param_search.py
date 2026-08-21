"""Parameter search for the ProtSpaM kernel (`refnd.kernels.protspam`) on very
distant protein pairs, ahead of using it to build the real UniRef50 HNSW index
(analog to `threshold/uniref50_benchmarks.py`, but with ProtSpaM instead of
GlobalAligner/AlignmentLocal).

ProtSpaM's own defaults (weight=6, dont_care=40, significance_threshold=0) are
tuned for detecting homologs: `weight` counts *exact-match* residues hashed
together into a spaced-word key, so two unrelated/distant proteins essentially
never collide at weight=6 -- confirmed empirically here (99.9% of random
UniRef50 pairs land at mismatch_rate=1.0, the kernel's "no match at all"
sentinel). Since HNSW makes mostly very-distant comparisons while building the
index, the default parameters would report almost every candidate edge as
maximally (and uninformatively) dissimilar.

This script:
1. Samples a pool of `--pool-size` UniRef50 sequences and `--n-pairs` random
   index pairs from it (cached, so every parameter combo is scored against
   the exact same pairs).
2. Computes a ground-truth "Global identity" distance for every pair via
   `GlobalAligner` (Needleman-Wunsch, BLOSUM62, `GlobalIdentityMode.MaxLength`
   -- the same convention `dbaasp` already uses in `src/datasets.py`).
3. Grid-searches ProtSpaM (n patterns, weight, dont_care, significance_threshold),
   RasBhari-optimized per combo (confirmed <1s/combo, so no need to fall back
   to unoptimized `SWPatternSet.random`), reporting `ProtSpamDistance.MismatchRate`
   (not `Evolutionary` -- the Kimura correction blows up to infinity past
   ~85% mismatch, which is the *typical* regime for random distant pairs, so it
   would collapse the ranking that matters here).
4. Scores each combo by Spearman rank correlation (rank-based, since we only
   care whether ProtSpaM's ordering of pairs by distance agrees with the
   ground truth, not the absolute scale) between its mismatch-rate distances
   and the ground-truth distances, alongside the fraction of pairs still stuck
   at the degenerate ceiling (mismatch_rate == 1.0).

Usage:
    uv run python -m threshold.protspam_param_search
    uv run python -m threshold.protspam_param_search --stage prepare
    uv run python -m threshold.protspam_param_search --stage search
"""

from __future__ import annotations

import argparse
import pickle
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rich import print
from rich.table import Table
from rich.console import Console
from scipy import stats

from refnd.kernels import KernelVariant, zip_kernel
from refnd.kernels.alignments import GlobalIdentityMode, ScoringMatrix
from refnd.kernels.protspam import ProtSpamDistance
from refnd.utils import SWPatternSet, SWSequence

UNIREF_POOL_PATH = Path("/mnt/documents/refnd_cache/uniref50/uniref50_random6m.pkl")
OUT_DIR = Path("/mnt/documents/refnd_cache/threshold_protspam")
PLOT_DIR = Path(__file__).parent

SEED = 42
POOL_SIZE = 100_000
N_PAIRS = 20_000

# (n patterns, weight, dont_care, significance_threshold). weight=2 is skipped:
# with 0 interior match positions there's only 1 distinct pattern possible, and
# SWPatternSet.random/rejection-sampling hangs for n>1 (see SWPatternSet docs).
WEIGHTS = [3, 4, 5, 6]
DONT_CARES = [10, 20, 40, 80, 120, 160]
N_PATTERNS = [5, 10]
THRESHOLDS = [0, -1_000_000]  # ProtSpaM default (BLOSUM-filtered) vs. disabled filter

C_GT, C_HI, MUTED, GRID, TEXT = "#0072B2", "#D55E00", "#6b7280", "#e5e7eb", "#374151"


def _cached(path: Path, compute) -> np.ndarray:
    if path.exists():
        print(f"  [dim]Loading cached {path.name}[/]")
        return np.load(path)
    arr = compute()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


def load_pool() -> list[str]:
    cache_path = OUT_DIR / f"pool_n{POOL_SIZE}_seed{SEED}.pkl"
    if cache_path.exists():
        print(f"  [dim]Loading cached pool ({cache_path.name})[/]")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    print(f"Loading full UniRef50 6M subsample ({UNIREF_POOL_PATH.name})...")
    with open(UNIREF_POOL_PATH, "rb") as f:
        full = pickle.load(f)
    # A handful of UniRef50 entries use non-standard codes (e.g. 'O'/pyrrolysine,
    # 'U'/selenocysteine) outside ProtSpaM's 25-symbol alphabet (encode_residue) --
    # ~0.003% of sequences. Filter before sampling so every pool member is valid
    # for every parameter combo (alphabet validity doesn't depend on weight/dont_care).
    valid_alphabet = set("ARNDCQEGHILKMFPSTWYVBZXJ*")
    rng = random.Random(SEED)
    shuffled = full[:]
    rng.shuffle(shuffled)
    pool: list[str] = []
    n_rejected = 0
    for s in shuffled:
        if len(pool) >= POOL_SIZE:
            break
        if set(s.upper()) <= valid_alphabet:
            pool.append(s)
        else:
            n_rejected += 1
    if n_rejected:
        print(f"  [dim]skipped {n_rejected} sequences with non-ProtSpaM-alphabet characters[/]")
    lens = np.array([len(s) for s in pool])
    print(f"  pool: n={len(pool):,} mean_len={lens.mean():.1f} median_len={np.median(lens):.0f} "
          f"max_len={lens.max()}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(pool, f)
    return pool


def sample_pairs() -> tuple[np.ndarray, np.ndarray]:
    path1 = OUT_DIR / f"pairs_idx1_n{N_PAIRS}_pool{POOL_SIZE}_seed{SEED}.npy"
    path2 = OUT_DIR / f"pairs_idx2_n{N_PAIRS}_pool{POOL_SIZE}_seed{SEED}.npy"
    if path1.exists() and path2.exists():
        print(f"  [dim]Loading cached pairs ({path1.name}, {path2.name})[/]")
        return np.load(path1), np.load(path2)
    rng = np.random.default_rng(SEED)
    idx1 = rng.integers(0, POOL_SIZE, size=N_PAIRS)
    idx2 = rng.integers(0, POOL_SIZE, size=N_PAIRS)
    same = idx1 == idx2
    while same.any():
        idx2[same] = rng.integers(0, POOL_SIZE, size=int(same.sum()))
        same = idx1 == idx2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path1, idx1)
    np.save(path2, idx2)
    return idx1, idx2


def ground_truth_distances(pool: list[str], idx1: np.ndarray, idx2: np.ndarray) -> np.ndarray:
    path = OUT_DIR / f"ground_truth_global_identity_n{N_PAIRS}_seed{SEED}.npy"

    def compute():
        print(f"  Computing Global identity (Needleman-Wunsch/BLOSUM62/MaxLength) over {N_PAIRS:,} pairs...")
        seqs1 = [pool[i] for i in idx1]
        seqs2 = [pool[i] for i in idx2]
        t0 = time.time()
        dist = zip_kernel(KernelVariant.AlignmentGlobal, seqs1, seqs2, progress=False,
                           matrix=ScoringMatrix.Blosum62, identity_mode=GlobalIdentityMode.MaxLength)
        print(f"  done in {time.time()-t0:.1f}s")
        return np.array(dist, dtype=np.float64)

    return _cached(path, compute)


def run_combo(pool: list[str], idx1: np.ndarray, idx2: np.ndarray,
              n: int, weight: int, dont_care: int, threshold: int) -> np.ndarray:
    tag = f"n{n}_w{weight}_dc{dont_care}_thr{threshold}"
    path = OUT_DIR / f"protspam_dist_{tag}.npy"
    if path.exists():
        return np.load(path)

    patterns = SWPatternSet(n, weight, dont_care)
    swseqs = [SWSequence(s, patterns) for s in pool]
    a = [swseqs[i] for i in idx1]
    b = [swseqs[i] for i in idx2]
    dist = zip_kernel(KernelVariant.ProtSpam, a, b, progress=False,
                       patterns=patterns, significance_threshold=threshold,
                       distance=ProtSpamDistance.MismatchRate)
    dist = np.array(dist, dtype=np.float64)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, dist)
    patterns.save(str(OUT_DIR / f"patterns_{tag}.bin"))
    return dist


def plot_scatter(tag: str, gt: np.ndarray, dist: np.ndarray, rho: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    fig.patch.set_facecolor("white")
    ax.scatter(gt, dist, s=4, alpha=0.25, color=C_GT, edgecolors="none")
    ax.set_xlabel("Global identity distance (1 - identity)", fontsize=10, color=MUTED)
    ax.set_ylabel("ProtSpaM mismatch rate", fontsize=10, color=MUTED)
    ax.set_title(f"{tag}\nSpearman rho={rho:.3f}", fontsize=11, color=TEXT, loc="left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["prepare", "search", "all"], default="all")
    args = parser.parse_args()

    print("[bold][orange2]=== ProtSpaM parameter search (UniRef50, random distant pairs) ===[/][/]")
    pool = load_pool()
    idx1, idx2 = sample_pairs()
    gt = ground_truth_distances(pool, idx1, idx2)
    print(f"  ground truth: n={len(gt):,} mean={gt.mean():.4f} median={np.median(gt):.4f} "
          f"min={gt.min():.4f} max={gt.max():.4f}")

    if args.stage == "prepare":
        return

    combos = [
        (n, w, dc, thr)
        for n in N_PATTERNS
        for w in WEIGHTS
        for dc in DONT_CARES
        for thr in THRESHOLDS
    ]
    print(f"\n[bold][orange2]=== Grid search: {len(combos)} combos ===[/][/]")

    results = []
    for n, w, dc, thr in combos:
        tag = f"n{n}_w{w}_dc{dc}_thr{thr}"
        t0 = time.time()
        dist = run_combo(pool, idx1, idx2, n, w, dc, thr)
        rho, pval = stats.spearmanr(gt, dist)
        frac_ceiling = float(np.mean(dist >= 0.9999))
        frac_zero = float(np.mean(dist <= 1e-9))
        dt = time.time() - t0
        results.append({
            "tag": tag, "n": n, "weight": w, "dont_care": dc, "threshold": thr,
            "rho": float(rho), "pval": float(pval),
            "frac_ceiling": frac_ceiling, "frac_zero": frac_zero, "time_s": dt,
        })
        print(f"  {tag:<28} rho={rho:+.4f}  frac_at_1.0={frac_ceiling:.3f}  "
              f"frac_at_0.0={frac_zero:.3f}  ({dt:.1f}s)")

    results.sort(key=lambda r: r["rho"], reverse=True)

    console = Console()
    table = Table(title="ProtSpaM parameter search — ranked by Spearman rho vs. Global identity")
    for col in ["rank", "n", "weight", "dont_care", "threshold", "rho", "p-value", "frac@1.0", "frac@0.0"]:
        table.add_column(col)
    for i, r in enumerate(results[:15]):
        table.add_row(str(i + 1), str(r["n"]), str(r["weight"]), str(r["dont_care"]), str(r["threshold"]),
                      f"{r['rho']:+.4f}", f"{r['pval']:.2e}", f"{r['frac_ceiling']:.3f}", f"{r['frac_zero']:.3f}")
    console.print(table)

    import csv
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "grid_search_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nFull results: {OUT_DIR / 'grid_search_results.csv'}")

    for r in results[:3]:
        dist = run_combo(pool, idx1, idx2, r["n"], r["weight"], r["dont_care"], r["threshold"])
        plot_scatter(r["tag"], gt, dist, r["rho"], PLOT_DIR / f"protspam_search_{r['tag']}.png")
    print(f"Saved scatter plots for top-3 combos to {PLOT_DIR}/protspam_search_*.png")

    best = results[0]
    print(f"\n[bold]Best combo: n={best['n']} weight={best['weight']} dont_care={best['dont_care']} "
          f"significance_threshold={best['threshold']}  rho={best['rho']:+.4f}[/]")


if __name__ == "__main__":
    main()
