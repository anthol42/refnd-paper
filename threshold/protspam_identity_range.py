"""Re-test of the ProtSpaM parameter search (`threshold/protspam_param_search.py`)
on pairs that actually span the identity range, instead of uniformly random
UniRef50 pairs.

`protspam_param_search.py` found only a weak Spearman correlation (best
rho=+0.34) against Global identity -- but its random pairs turned out to sit
almost entirely in the "twilight zone" (0.3-25% identity, see that script's
docstring): there's no real evolutionary signal there for *any* method to
recover, since random UniRef50 pairs are essentially all non-homologous.
ProtSpaM is designed to estimate distance between actually-related sequences,
not to discriminate among noise.

This script builds a fairer test: for each of `--n-pairs` seed sequences drawn
from the UniRef50 pool, generate a same-length synthetic "variant" by mutating
a fraction of positions to a different one of the 20 standard amino acids,
with that fraction drawn uniformly from `--lo` to `--hi`. Because the mutated
sequence has the same length as the seed, Needleman-Wunsch's global alignment
lines residues up 1:1 with no gap incentive, so the resulting measured Global
identity tracks the design target closely -- giving direct, roughly-uniform
coverage of the full identity range (~0 to ~1), unlike real random pairs which
clump near 0.

Reuses `threshold.protspam_param_search`'s cached UniRef50 pool and its
`ProtSpaM(MismatchRate)` / `GlobalAligner` conventions. Evaluates a small set
of representative parameter combos (the winner from the random-pairs search,
ProtSpaM's own defaults, and a couple of points in between) rather than the
full grid, since the point here is to check whether the *shape* of the
identity-range problem is fundamentally different, not to re-tune from
scratch.

Usage:
    uv run python -m threshold.protspam_identity_range
"""

from __future__ import annotations

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

from threshold.protspam_param_search import load_pool, OUT_DIR

PLOT_DIR = Path(__file__).parent
SEED = 42
N_PAIRS = 5000
LO, HI = 0.02, 0.98  # target identity range (fraction of positions left unmutated)

AA20 = "ARNDCQEGHIKLMFPSTWYV"  # 20 standard amino acids, ProtSpaM-alphabet subset

# (label, n, weight, dont_care, significance_threshold)
COMBOS = [
    ("random-pairs winner (w=3, dc=160, thr=off)", 10, 3, 160, -1_000_000),
    ("random-pairs best@dc=20 (w=3, dc=20, thr=off)", 10, 3, 20, -1_000_000),
    ("ProtSpaM default (w=6, dc=40, thr=0)", 5, 6, 40, 0),
    ("default weight, filter off (w=6, dc=40, thr=off)", 5, 6, 40, -1_000_000),
]

C_GT = "#0072B2"
MUTED, GRID, TEXT = "#6b7280", "#e5e7eb", "#374151"
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


def make_synthetic_pairs(pool: list[str], n_pairs: int, lo: float, hi: float, seed: int
                          ) -> tuple[list[str], list[str], np.ndarray]:
    """`n_pairs` (seed, mutated) pairs, one per distinct pool sequence, with a
    per-pair target identity drawn uniformly from `[lo, hi]`. Mutation is
    point-substitution only (no indels), at exactly
    `round((1 - target) * len(seq))` positions, each replaced by a uniformly
    random *different* residue from the 20 standard amino acids."""
    rng = random.Random(seed)
    seeds = rng.sample(pool, n_pairs)
    seqs1, seqs2, targets = [], [], []
    for seq in seeds:
        target = rng.uniform(lo, hi)
        chars = list(seq)
        n_mut = round((1.0 - target) * len(chars))
        n_mut = min(n_mut, len(chars))
        positions = rng.sample(range(len(chars)), n_mut)
        for pos in positions:
            orig = chars[pos].upper()
            choices = [c for c in AA20 if c != orig]
            chars[pos] = rng.choice(choices)
        seqs1.append(seq)
        seqs2.append("".join(chars))
        targets.append(target)
    return seqs1, seqs2, np.array(targets)


def ground_truth(seqs1: list[str], seqs2: list[str]) -> np.ndarray:
    path = OUT_DIR / f"identity_range_ground_truth_n{N_PAIRS}_lo{LO}_hi{HI}_seed{SEED}.npy"
    if path.exists():
        print(f"  [dim]Loading cached {path.name}[/]")
        return np.load(path)
    print(f"  Computing Global identity over {len(seqs1):,} synthetic pairs...")
    t0 = time.time()
    dist = zip_kernel(KernelVariant.AlignmentGlobal, seqs1, seqs2, progress=False,
                       matrix=ScoringMatrix.Blosum62, identity_mode=GlobalIdentityMode.MaxLength)
    dist = np.array(dist, dtype=np.float64)
    print(f"  done in {time.time()-t0:.1f}s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, dist)
    return dist


def run_combo(seqs1: list[str], seqs2: list[str], n: int, weight: int, dont_care: int, threshold: int
              ) -> np.ndarray:
    tag = f"n{n}_w{weight}_dc{dont_care}_thr{threshold}"
    path = OUT_DIR / f"identity_range_protspam_dist_n{N_PAIRS}_{tag}.npy"
    if path.exists():
        return np.load(path)
    patterns = SWPatternSet(n, weight, dont_care)
    a = [SWSequence(s, patterns) for s in seqs1]
    b = [SWSequence(s, patterns) for s in seqs2]
    dist = zip_kernel(KernelVariant.ProtSpam, a, b, progress=False,
                       patterns=patterns, significance_threshold=threshold,
                       distance=ProtSpamDistance.MismatchRate)
    dist = np.array(dist, dtype=np.float64)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, dist)
    return dist


def plot_all(gt: np.ndarray, results: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    fig.patch.set_facecolor("white")
    for ax, r, color in zip(axes.flat, results, PALETTE):
        ax.scatter(gt, r["dist"], s=4, alpha=0.25, color=color, edgecolors="none")
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", alpha=0.6)
        ax.set_xlabel("Global identity distance (1 - identity)", fontsize=9, color=MUTED)
        ax.set_ylabel("ProtSpaM mismatch rate", fontsize=9, color=MUTED)
        ax.set_title(f"{r['label']}\nrho={r['rho']:+.3f}  (frac@1.0={r['frac_ceiling']:.2f})",
                     fontsize=9.5, color=TEXT, loc="left")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(GRID)
    fig.suptitle(f"ProtSpaM vs. Global identity -- synthetic pairs uniform over identity ∈ [{LO},{HI}] (n={N_PAIRS:,})",
                 fontsize=12, color=TEXT)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    print("[bold][orange2]=== ProtSpaM vs. Global identity: synthetic pairs spanning the full identity range ===[/][/]")
    pool = load_pool()

    pairs_cache = OUT_DIR / f"identity_range_pairs_n{N_PAIRS}_lo{LO}_hi{HI}_seed{SEED}.npz"
    if pairs_cache.exists():
        print(f"  [dim]Loading cached pairs ({pairs_cache.name})[/]")
        npz = np.load(pairs_cache, allow_pickle=True)
        seqs1, seqs2, targets = list(npz["seqs1"]), list(npz["seqs2"]), npz["targets"]
    else:
        seqs1, seqs2, targets = make_synthetic_pairs(pool, N_PAIRS, LO, HI, SEED)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(pairs_cache, seqs1=np.array(seqs1, dtype=object),
                  seqs2=np.array(seqs2, dtype=object), targets=targets)
    print(f"  {len(seqs1):,} synthetic pairs, target identity range [{targets.min():.3f}, {targets.max():.3f}], "
          f"mean={targets.mean():.3f}")

    gt = ground_truth(seqs1, seqs2)
    print(f"  measured ground truth (1-identity): mean={gt.mean():.4f} median={np.median(gt):.4f} "
          f"min={gt.min():.4f} max={gt.max():.4f}")
    design_vs_measured_rho, _ = stats.spearmanr(1 - targets, gt)
    print(f"  [dim]sanity check -- design target vs. measured Global identity: rho={design_vs_measured_rho:.4f} "
          f"(should be close to 1.0)[/]")

    results = []
    for label, n, w, dc, thr in COMBOS:
        t0 = time.time()
        dist = run_combo(seqs1, seqs2, n, w, dc, thr)
        rho, pval = stats.spearmanr(gt, dist)
        frac_ceiling = float(np.mean(dist >= 0.9999))
        dt = time.time() - t0
        results.append({"label": label, "n": n, "weight": w, "dont_care": dc, "threshold": thr,
                         "dist": dist, "rho": float(rho), "pval": float(pval),
                         "frac_ceiling": frac_ceiling})
        print(f"  {label:<50} rho={rho:+.4f}  frac_at_1.0={frac_ceiling:.3f}  ({dt:.1f}s)")

    console = Console()
    table = Table(title="ProtSpaM vs. Global identity -- synthetic full-identity-range pairs")
    for col in ["combo", "n", "weight", "dont_care", "threshold", "rho", "p-value", "frac@1.0"]:
        table.add_column(col)
    for r in sorted(results, key=lambda r: r["rho"], reverse=True):
        table.add_row(r["label"], str(r["n"]), str(r["weight"]), str(r["dont_care"]), str(r["threshold"]),
                      f"{r['rho']:+.4f}", f"{r['pval']:.2e}", f"{r['frac_ceiling']:.3f}")
    console.print(table)

    out_path = PLOT_DIR / "protspam_identity_range.png"
    plot_all(gt, results, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
