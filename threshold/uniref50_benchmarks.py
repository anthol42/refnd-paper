"""Empirical test of the "Finding optimal threshold" theory (see CLAUDE.md /
the paper's methods note) for two standard PLM-evaluation benchmarks against
UniRef50 (the ESM2/ESM-C pretraining-data proxy) -- analog to
threshold/peptides.py and threshold/belka.py.

Train: the random 6M-sequence UniRef50 subsample already indexed in
threshold/uniref50_build6m.py's HNSW build (AlignmentLocal / Blosum62 /
min_coverage=0.8 / proximity_threshold=0.5 / ef_construction=64 /
strict_ef=True / keep_all_edges=False / cache_capacity=2_000_000).

Production (two independent benchmarks, analyzed separately):
- ProteinGym: 187 unique DMS wild-type sequences (one per distinct assayed
  protein -- NOT the ~2.47M single/multi-mutant variant rows, which would
  violate the i.i.d. assumption behind the u-distribution/KS fit, since a
  1-3 residue mutation barely moves alignment distance from its wildtype).
- CASP15 protein targets (87 sequences): the contact-prediction benchmark
  ESM-C actually reports on (P@L on CASP15), not the older
  TAPE/ProteinNet-CASP12 benchmark.

Null model: per-sequence amino-acid shuffling (`null_model_scores(...,
shuffle=True)`), exactly as used for DBAASP/PeptideAtlas in
threshold/peptides.py -- the user's explicit starting point ("like for
peptides!"). Sampled from the train pool with length capped at 2048 residues
(drops only 0.57% of train, all in the pathological long tail that caused
the HNSW build's late slowdown) to keep the O(L1*L2) alignment cost bounded.

n_eff is fit via KS-minimization restricted to u > h=0.3 (never by looking
at c, which would be circular) -- same protocol as peptides.py/belka.py.
After fitting, uniformity of u above h is checked with a one-sample KS test
against Uniform(h, 1); if that check fails, the null model is swapped out
(see `NULL_STRATEGIES`) and everything downstream (n_eff, c, tau) is
re-fit from scratch for the new null -- n_eff is NEVER reused across a null
model change.

Long steps (B(y) search, the null) are cached to disk under
/mnt/documents/refnd_cache/threshold_uniref50/.

Usage:
    uv run python -m threshold.uniref50_benchmarks
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from refnd.core import HNSWState
from refnd.kernels import KernelVariant
from refnd.kernels.alignments import ScoringMatrix
from rich import print
from scipy import stats

from src.datasets import DatasetConfig
from src.metrics import null_model_cdf, null_model_scores

from threshold.peptides import (
    fit_c_from_ecdf,
    find_best_n_eff,
    plot_eq_c_solution,
    solve_tau_eq_c,
    u_from_p0,
)

UNIREF_DIR = Path("/mnt/documents/refnd_cache/uniref50")
OUT_DIR = Path("/mnt/documents/refnd_cache/threshold_uniref50")
PLOT_DIR = Path(__file__).parent
BENCH_DIR = Path("/mnt/documents/refnd_cache/protein_benchmarks")

SEED = 42
H = 0.3
SEARCH_EF = 64
NULL_LEN_CAP = 2048  # drops 0.57% of train (the pathological long tail)

TRAIN_SEQS_PATH = UNIREF_DIR / "uniref50_random6m.pkl"
INDEX_PATH = UNIREF_DIR / "uniref50_random6m.hnsw"

CFG = DatasetConfig(
    modality=KernelVariant.AlignmentLocal,
    metric=None,
    encoder=None,
    proximity_threshold=0.5,
    kernel_params={"matrix": ScoringMatrix.Blosum62, "min_coverage": 0.8},
)

C_P0, C_G0, C_G = "#0072B2", "#E69F00", "#009E73"
GRID, TEXT, MUTED = "#e5e7eb", "#374151", "#6b7280"

# Null-model strategies to try in order, until u above H passes a uniformity
# check. Each is (name, n_samples, shuffle, tail_quantile, min_tail_samples).
#
# shuffle_capped2048 (tried first, "like for peptides!"): FAILS immediately --
# with AlignmentLocal + min_coverage=0.8, a fully shuffled sequence destroys
# every local motif, so it essentially never clears the 80%-coverage gate.
# Verified empirically: 20,000/20,000 shuffled pairs landed at EXACTLY
# distance=1.0 (the kernel's "no alignment found" sentinel) -- p0(tau)=0 for
# all tau<1, which would collapse every u_y to 0 regardless of B(y).
#
# real_pairs (fallback): real (unshuffled) random train-train pairs. Also
# mostly land at 1.0 (unrelated real proteins usually share no significant
# local homology either -- rate ~0.007%), but unlike the shuffle case this
# is a genuine, non-degenerate spike: a 20M-pair calibration found 1,419
# informative sub-1.0 hits, including a few down at tau<=0.3-0.5, enough to
# resolve p0 in the region that matters. tail_quantile is set far below the
# 0.01 default (which would land the GPD cutoff inside the degenerate
# 1.0-spike, given how rare sub-1.0 hits are) so it floors at
# min_tail_samples instead, keeping the tail fit inside the genuinely
# informative sub-1.0 population (~60M * 7e-5 ~= 4,200 expected hits vs a
# min_tail_samples floor of 1,500).
NULL_STRATEGIES = [
    ("shuffle_capped2048", 300_000, True, 0.01, 1000),
    ("real_pairs", 60_000_000, False, 1e-6, 1500),
]


def mem_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1024 / 1024
    return float("nan")


def load_train_and_index() -> tuple[list[str], HNSWState]:
    print(f"Loading cached train sequences ({TRAIN_SEQS_PATH.name})...")
    with open(TRAIN_SEQS_PATH, "rb") as f:
        train = pickle.load(f)
    print(f"  {len(train):,} train sequences. MemAvailable: {mem_gb():.2f}GB")
    print(f"Loading cached HNSW index ({INDEX_PATH.name})...")
    t0 = time.time()
    hnsw = HNSWState.load(CFG.modality, str(INDEX_PATH), train)
    print(f"  loaded in {time.time()-t0:.1f}s. MemAvailable: {mem_gb():.2f}GB")
    return train, hnsw


def load_proteingym_wildtypes() -> list[str]:
    import csv
    path = BENCH_DIR / "proteingym_wildtypes.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    seqs = [r["target_seq"] for r in rows]
    print(f"ProteinGym: {len(seqs)} unique wild-type sequences")
    return seqs


def load_casp15_targets() -> list[str]:
    path = BENCH_DIR / "casp15_protein_only.fasta"
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            else:
                cur.append(line)
        if cur:
            seqs.append("".join(cur))
    print(f"CASP15: {len(seqs)} protein target sequences")
    return seqs


def compute_B(hnsw: HNSWState, prod: list[str], cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        print(f"  [dim]Loading cached B(y) ({cache_path.name})[/]")
        return np.load(cache_path)
    print(f"  Searching {len(prod):,} sequences against the {INDEX_PATH.name} index...")
    t0 = time.time()
    results = hnsw.search(prod, k=1, ef=SEARCH_EF, threads=0, progress=True)
    B = np.array([hits[0][1] for hits in results], dtype=np.float64)
    print(f"  search done in {time.time()-t0:.1f}s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, B)
    return B


def build_null_scores(train: list[str], name: str, n_samples: int, shuffle: bool) -> np.ndarray:
    cache_path = OUT_DIR / f"null_{name}_n{n_samples}_seed{SEED}.npy"
    if cache_path.exists():
        print(f"  [dim]Loading cached null ({cache_path.name})[/]")
        return np.load(cache_path)
    pool = [s for s in train if len(s) <= NULL_LEN_CAP] if shuffle else train
    print(f"  Null model '{name}': sampling {n_samples:,} pairs from a {len(pool):,}-sequence "
          f"pool (shuffle={shuffle})...")
    t0 = time.time()
    scores = null_model_scores(pool, CFG, n_samples=n_samples, seed=SEED, shuffle=shuffle)
    print(f"  null done in {time.time()-t0:.1f}s. MemAvailable: {mem_gb():.2f}GB")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, scores)
    return scores


def fit_and_check(B: np.ndarray, null_scores: np.ndarray, n_train: int,
                   tail_quantile: float = 0.01, min_tail_samples: int = 1000) -> dict:
    """Fit n_eff (KS-minimized on u>H, non-circular), fit c two ways, and
    check whether u above H actually passes a uniformity test."""
    p0 = null_model_cdf(null_scores, tail_quantile=tail_quantile, min_tail_samples=min_tail_samples)
    p0_B = p0(B)
    n_eff_grid = np.unique(np.round(np.geomspace(1.0, float(n_train), 100)))
    ks_stats = find_best_n_eff(p0_B, n_eff_grid, min_u=H)
    best_idx = int(np.nanargmin(ks_stats))
    n_eff = float(n_eff_grid[best_idx])
    u = u_from_p0(p0_B, n_eff)

    u_tail = u[u > H]
    if len(u_tail) >= 8:
        ks_res = stats.kstest(u_tail, "uniform", args=(H, 1.0 - H))
        ks_pvalue = float(ks_res.pvalue)
        ks_tail_stat = float(ks_res.statistic)
    else:
        ks_pvalue, ks_tail_stat = float("nan"), float("nan")

    # A degenerate null (e.g. every score landing at the kernel's "no match"
    # sentinel) can send every u_y to 0, leaving nothing above H at all --
    # fit_c_from_ecdf's polyfit needs >=2 points and would raise on an empty
    # tail, so treat that case as an outright uniformity-check failure
    # instead of letting it crash a strategy that's about to be discarded
    # anyway.
    if len(u_tail) < 2:
        return {
            "p0": p0, "p0_B": p0_B, "n_eff_grid": n_eff_grid, "ks_stats": ks_stats,
            "n_eff": n_eff, "u": u, "c_ols": float("nan"), "c_r2": float("nan"),
            "c_n_points": len(u_tail), "c_surv": float("nan"),
            "ks_tail_stat": ks_tail_stat, "ks_pvalue": ks_pvalue,
            "n_tail": len(u_tail), "passes": False,
        }

    c_fit = fit_c_from_ecdf(u, h=H)
    frac_above = float(np.mean(u > H))
    c_surv = 1 - frac_above / (1 - H)

    return {
        "p0": p0, "p0_B": p0_B, "n_eff_grid": n_eff_grid, "ks_stats": ks_stats,
        "n_eff": n_eff, "u": u, "c_ols": c_fit["c"], "c_r2": c_fit["r2"],
        "c_n_points": c_fit["n_points"], "c_surv": c_surv,
        "ks_tail_stat": ks_tail_stat, "ks_pvalue": ks_pvalue,
        "n_tail": len(u_tail), "passes": (not np.isnan(ks_pvalue)) and ks_pvalue > 0.05,
    }


def plot_p0_g0_g(name: str, p0, B: np.ndarray, n_eff: float, null_scores: np.ndarray, out_path: Path) -> None:
    B_sorted = np.sort(B)
    n_B = len(B_sorted)

    def G_emp(tau):
        return np.searchsorted(B_sorted, tau, side="right") / n_B

    tau_grid = np.linspace(min(B.min(), null_scores.min()) - 0.03, max(B.max(), null_scores.max()) + 0.03, 500)
    p0_vals = p0(tau_grid)
    G0_vals = 1 - (1 - p0_vals) ** n_eff
    G_vals = G_emp(tau_grid)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    fig.patch.set_facecolor("white")
    ax.plot(tau_grid, p0_vals, color=C_P0, linewidth=1.6, linestyle=":", label=r"$p_0(\tau)$ — shuffled-AA null")
    ax.plot(tau_grid, G0_vals, color=C_G0, linewidth=2.2, label=rf"$G_0(\tau)$, n_eff={n_eff:,.0f}")
    ax.plot(tau_grid, G_vals, color=C_G, linewidth=2.2, label=rf"$G(\tau)$ — empirical ({name})")
    ax.set_xlabel(r"$\tau$", fontsize=10, color=MUTED)
    ax.set_ylabel("cumulative probability", fontsize=10, color=MUTED)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{name} vs UniRef50 (6M): $p_0$, $G_0$, $G$", fontsize=11, color=TEXT, loc="left")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_u_pdf(name: str, u: np.ndarray, n_eff: float, c_ols: float, c_surv: float,
               ks_pvalue: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor("white")
    n_bins = min(30, max(8, len(u) // 5))
    ax.hist(u, bins=n_bins, range=(0, 1), density=True, color=C_G0, alpha=0.85, label=f"u density ({name})")
    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle="--", label="Uniform(0,1)")
    ax.axhline(1 - c_ols, color=C_G, linewidth=1.4, linestyle="-.", label=rf"(1-c) = {1-c_ols:.3f}")
    ax.axvline(H, color=MUTED, linewidth=1, linestyle=":", label=f"h={H}")
    ax.set_xlabel(r"$u_y$", fontsize=11, color=MUTED)
    ax.set_ylabel("density", fontsize=11, color=MUTED)
    ax.set_title(
        f"u PDF — {name} (n={len(u):,}), n_eff={n_eff:,.0f}\n"
        f"c_ols={c_ols:+.3f}, c_surv={c_surv:+.3f}, KS-uniformity p={ks_pvalue:.3f}",
        fontsize=11, color=TEXT, loc="left",
    )
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 1)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


def run_benchmark(name: str, prod: list[str], train: list[str], hnsw: HNSWState, n_train: int) -> dict:
    print(f"\n[bold][orange2]=== {name} vs UniRef50 (6M train) ===[/][/]")
    B_path = OUT_DIR / f"B_{name}_vs_uniref50_6m_seed{SEED}.npy"
    B = compute_B(hnsw, prod, B_path)
    print(f"  B(y): n={len(B):,} min={B.min():.4f} median={np.median(B):.4f} max={B.max():.4f}")

    result = None
    for strat_name, n_samples, shuffle, tail_q, min_tail in NULL_STRATEGIES:
        null_scores = build_null_scores(train, strat_name, n_samples, shuffle)
        print(f"  null ({strat_name}): n={len(null_scores):,} min={null_scores.min():.4f} "
              f"median={np.median(null_scores):.4f} max={null_scores.max():.4f}  "
              f"n_sub1.0={int((null_scores<1.0).sum()):,}")
        fit = fit_and_check(B, null_scores, n_train, tail_quantile=tail_q, min_tail_samples=min_tail)
        print(f"  n_eff={fit['n_eff']:,.0f}  c_ols={fit['c_ols']:+.4f} (R2={fit['c_r2']:.3f}, "
              f"n_points={fit['c_n_points']})  c_surv={fit['c_surv']:+.4f}")
        print(f"  uniformity check (u>{H:g}, n={fit['n_tail']}): KS_stat={fit['ks_tail_stat']:.4f}  "
              f"p-value={fit['ks_pvalue']:.4f}  -> {'PASS' if fit['passes'] else 'FAIL'}")
        result = {"strategy": strat_name, "null_scores": null_scores, **fit, "B": B}
        if fit["passes"]:
            print(f"  [green]Null model '{strat_name}' passes the uniformity check.[/]")
            break
        else:
            print(f"  [yellow]Null model '{strat_name}' failed uniformity -- trying next strategy if available.[/]")

    tau_grid = np.linspace(0.0, 1.0, 2000)
    tau_result = solve_tau_eq_c(result["B"], result["p0"], result["n_eff"], result["c_ols"], tau_grid)
    result["tau_star"] = tau_result["tau_star"]
    if tau_result["tau_star"] is not None:
        print(f"  [bold]tau* = {tau_result['tau_star']:.4f}[/]")
    else:
        print("  [yellow]No tau reached the fitted c before G0(tau) saturated -- no solution found.[/]")

    plot_eq_c_solution(tau_result, result["c_ols"], result["n_eff"], PLOT_DIR / f"uniref50_{name}_tau_solution.png")
    plot_p0_g0_g(name, result["p0"], result["B"], result["n_eff"], result["null_scores"],
                 PLOT_DIR / f"uniref50_{name}_p0_g0_g.png")
    plot_u_pdf(name, result["u"], result["n_eff"], result["c_ols"], result["c_surv"], result["ks_pvalue"],
               PLOT_DIR / f"uniref50_{name}_u_pdf.png")
    print(f"  Saved plots: uniref50_{name}_{{tau_solution,p0_g0_g,u_pdf}}.png")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    print("[bold][orange2]=== UniRef50 (6M train) vs ProteinGym / CASP15 (production) ===[/][/]")
    print(f"MemAvailable at start: {mem_gb():.2f}GB")

    train, hnsw = load_train_and_index()
    n_train = len(train)

    proteingym = load_proteingym_wildtypes()
    casp15 = load_casp15_targets()

    results = {}
    results["proteingym"] = run_benchmark("proteingym", proteingym, train, hnsw, n_train)
    results["casp15"] = run_benchmark("casp15", casp15, train, hnsw, n_train)

    print("\n[bold][orange2]=== Final summary ===[/][/]")
    for name, r in results.items():
        print(f"{name}: n_eff={r['n_eff']:,.0f}  c_ols={r['c_ols']:+.4f}  c_surv={r['c_surv']:+.4f}  "
              f"tau*={r['tau_star']}  uniformity_p={r['ks_pvalue']:.4f} ({r['strategy']})")


if __name__ == "__main__":
    main()
