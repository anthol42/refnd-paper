"""Sanity test: instead of the elaborate synthetic null models used in
threshold/peptides.py (permutation-shuffled DBAASP) and threshold/belka.py
(purely-random-atom molecules), build p0 from the simplest possible null —
random, UNMODIFIED pairs sampled directly from the production set itself
(shuffle=False). No relation to train assumed or injected; this is just
"how similar are two random real production items to each other."

Reuses the already-cached B(y) (production -> train nearest-neighbor
distance) from each dataset's main run, so this only needs to (re)compute
the prod-self null scores, refit n_eff (KS-minimized on u > h, same
procedure as the main scripts), and report c.

Usage:
    uv run python -m threshold.prod_self_null
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rich import print

from src.cache import CacheStore
from src.datasets import DATASETS, load_dataset
from src.metrics import null_model_cdf, null_model_scores

from threshold.peptides import fit_c_from_ecdf, find_best_n_eff, u_from_p0
from threshold import belka as belka_mod

PLOT_DIR = Path(__file__).parent
PEPTIDES_CACHE = Path(__file__).parent / "cache"
BELKA_CACHE = belka_mod.OUT_DIR

SEED = 42
H = 0.3
N_NULL_PAIRS_PEPTIDES = 10_000_000
N_NULL_PAIRS_BELKA = 2_000_000

# Largest cached peptide B(y) (full 1M production subsample) for max power.
PEPTIDES_B_PATH = PEPTIDES_CACHE / "B_ann_peptide_atlas_dbaasp_n1000000_seed42_efc64_ef64.npy"
PEPTIDES_NULL_PROD_PATH = PEPTIDES_CACHE / f"null_prodself_peptide_atlas_n{N_NULL_PAIRS_PEPTIDES}_seed{SEED}.npy"

BELKA_B_PATH = belka_mod.B_PATH
BELKA_NULL_PROD_PATH = BELKA_CACHE / f"null_prodself_belka_test_n{N_NULL_PAIRS_BELKA}_seed{SEED}.npy"


def _cached(path: Path, compute) -> np.ndarray:
    if path.exists():
        print(f"  [dim]Loading cached {path.name}[/]")
        return np.load(path)
    arr = compute()
    np.save(path, arr)
    return arr


def run(name: str, B: np.ndarray, null_scores: np.ndarray, n_ref: int) -> dict:
    print(f"  prod-self null: n={len(null_scores):,} min={null_scores.min():.4f} "
          f"median={np.median(null_scores):.4f} max={null_scores.max():.4f}")

    p0 = null_model_cdf(null_scores)
    p0_B = p0(B)

    n_eff_grid = np.unique(np.round(np.geomspace(1.0, float(n_ref), 100)))
    ks_stats = find_best_n_eff(p0_B, n_eff_grid, min_u=H)
    best_idx = int(np.nanargmin(ks_stats))
    n_eff = float(n_eff_grid[best_idx])

    u = u_from_p0(p0_B, n_eff)
    c_fit = fit_c_from_ecdf(u, h=H)
    frac_above = float(np.mean(u > H))
    c_surv = 1 - frac_above / (1 - H)

    print(f"  n_eff={n_eff:,.0f}  KS={ks_stats[best_idx]:.4f}")
    print(f"  [bold]c_ols={c_fit['c']:+.4f}[/] (R2={c_fit['r2']:.3f}, n_points={c_fit['n_points']:,})   "
          f"[bold]c_surv={c_surv:+.4f}[/]")
    print(f"  u: mean={u.mean():.4f}  frac(u<0.05)={np.mean(u < 0.05):.4f}  frac(u>0.95)={np.mean(u > 0.95):.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.hist(u, bins=50, range=(0, 1), density=True, color="#E69F00", alpha=0.85, label="u density")
    ax.axhline(1.0, color="#6b7280", linewidth=1, linestyle="--", label="Uniform(0,1)")
    ax.axhline(1 - c_fit["c"], color="#009E73", linewidth=1.4, linestyle="-.", label=f"(1-c_ols) = {1 - c_fit['c']:.3f}")
    ax.axvline(H, color="#6b7280", linewidth=1, linestyle=":", label=f"h={H}")
    ax.set_xlabel("$u_y$")
    ax.set_ylabel("density")
    ax.set_title(
        f"u PDF — {name}, prod-self null (n={len(u):,}), n_eff={n_eff:,.0f}\n"
        f"c_ols={c_fit['c']:+.3f}, c_surv={c_surv:+.3f}",
        loc="left",
    )
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_xlim(0, 1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out_path = PLOT_DIR / f"prodself_{name}_u_pdf.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")

    return {"n_eff": n_eff, "c_ols": c_fit["c"], "c_r2": c_fit["r2"], "c_surv": c_surv}


def run_peptides() -> dict:
    print("[bold][orange2]=== PeptideAtlas: prod-self null test ===[/][/]")
    cfg = DATASETS["dbaasp"]  # peptide_atlas shares the same kernel/modality
    cache = CacheStore()
    prod, _ = load_dataset("peptide_atlas", cache)
    print(f"  production (peptide_atlas): {len(prod):,}")

    B = np.load(PEPTIDES_B_PATH)
    print(f"  Loaded cached B(y): {PEPTIDES_B_PATH.name}  n={len(B):,}")

    null_scores = _cached(
        PEPTIDES_NULL_PROD_PATH,
        lambda: null_model_scores(prod, cfg, n_samples=N_NULL_PAIRS_PEPTIDES, seed=SEED, shuffle=False),
    )
    return run("peptides", B, null_scores, n_ref=len(prod))


def run_belka() -> dict:
    print("[bold][orange2]=== BELKA: prod-self null test ===[/][/]")
    cfg = DATASETS["belka"]

    B = np.load(BELKA_B_PATH)
    print(f"  Loaded cached B(y): {BELKA_B_PATH.name}  n={len(B):,}")

    if BELKA_NULL_PROD_PATH.exists():
        null_scores = np.load(BELKA_NULL_PROD_PATH)
        print(f"  [dim]Loading cached {BELKA_NULL_PROD_PATH.name}[/]")
    else:
        test_fps = belka_mod.load_full_test_fps()
        print(f"  production (belka full test set): {len(test_fps):,}")
        null_scores = null_model_scores(test_fps, cfg, n_samples=N_NULL_PAIRS_BELKA, seed=SEED, shuffle=False)
        np.save(BELKA_NULL_PROD_PATH, null_scores)

    return run("belka", B, null_scores, n_ref=len(B))


def main() -> None:
    peptides_result = run_peptides()
    belka_result = run_belka()

    print("\n[bold][orange2]=== Summary: collision rate via prod-self (real, unmodified) null ===[/][/]")
    for name, res in (("peptides", peptides_result), ("belka", belka_result)):
        print(f"  {name}: n_eff={res['n_eff']:,.0f}  c_ols={res['c_ols']:+.4f} (R2={res['c_r2']:.3f})  "
              f"c_surv={res['c_surv']:+.4f}")


if __name__ == "__main__":
    main()
