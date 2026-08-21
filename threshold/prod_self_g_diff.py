"""G(tau) - G0(tau) under the prod-self null (threshold/prod_self_null.py),
but with a different n_eff-fitting procedure than the rest of this repo.

Everywhere else, n_eff is picked by KS-minimizing u = G0(B(y)) against
Uniform(0,1) restricted to u > h=0.3 (a fixed cutoff in u-space, motivated by
"collisions cluster near u=0"). That is sensitive to exactly how the tail is
defined and, for peptides, to B(y)'s heavy discretization (see conversation:
huge non-collision atoms scattered across the whole u range wreck both the
KS fit and the OLS/survival c estimates).

Here instead: fit n_eff directly in tau-space, least-squares aligning G(tau)
to G0(tau) restricted to tau above the 25th percentile of the observed B(y)
values (not a fixed u cutoff, not a KS statistic) -- i.e. "make the null
model match the observed middle-to-upper part of the distribution, then look
at where the low end departs from that fit." No linear-above-h assumption,
no atom-sensitive u-transform for the fitting step itself; G-G0 is reported
directly in tau-space, which is exact (steps, no aliasing) regardless of how
discretized B(y) is.

Usage:
    uv run python -m threshold.prod_self_g_diff
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rich import print

from src.metrics import null_model_cdf

from threshold import prod_self_null as psn

PLOT_DIR = Path(__file__).parent
PERCENTILE = 25.0

# Upper bound for the n_eff search grid -- same convention as peptides.py/belka.py
# (defaults to n_train when no override given).
N_TRAIN = {"peptides": 9_796, "belka": 49_207_805}


def fit_n_eff_middle_alignment(
    B: np.ndarray, p0, n_train: int, percentile: float = PERCENTILE,
    n_fit_points: int = 300, n_eff_steps: int = 200, side: str = "right",
) -> dict:
    """Least-squares n_eff fit: match G0(tau)=1-(1-p0(tau))^n_eff to the
    empirical G(tau), restricted to one side of a percentile cutoff of B(y).

    side="right" (original): fit on tau in [q_percentile, max(B)] -- excludes
    the bottom `percentile`% (small tau / close matches, where collisions are
    assumed to live), aligns on the clean majority above it.

    side="left": mirror image -- fit on tau in [min(B), q_(100-percentile)],
    i.e. excludes the top `percentile`% (large tau / far matches) and aligns
    on the small-to-medium-distance majority instead.
    """
    sorted_B = np.sort(B)
    n = len(B)

    def G_emp(tau):
        return np.searchsorted(sorted_B, tau, side="right") / n

    if side == "right":
        q = float(np.percentile(B, percentile))
        tau_fit = np.linspace(q, B.max(), n_fit_points)
    elif side == "left":
        q = float(np.percentile(B, 100.0 - percentile))
        tau_fit = np.linspace(B.min(), q, n_fit_points)
    elif side == "full":
        q = float(B.min())
        tau_fit = np.linspace(B.min(), B.max(), n_fit_points)
    else:
        raise ValueError(f"side must be 'left', 'right' or 'full', got {side!r}")

    G_fit = G_emp(tau_fit)
    p0_fit = p0(tau_fit)

    n_eff_grid = np.unique(np.round(np.geomspace(1.0, float(n_train), n_eff_steps)))
    sses = np.empty(len(n_eff_grid))
    for i, n_eff in enumerate(n_eff_grid):
        G0_fit = 1.0 - (1.0 - p0_fit) ** n_eff
        sses[i] = np.mean((G_fit - G0_fit) ** 2)
    best_idx = int(np.nanargmin(sses))
    n_eff = float(n_eff_grid[best_idx])

    return {"n_eff": n_eff, "q": q, "percentile": percentile, "n_eff_grid": n_eff_grid,
            "sses": sses, "sse_best": float(sses[best_idx])}


def compute_G_G0(B: np.ndarray, p0, n_eff: float, n_grid: int = 1000) -> dict:
    sorted_B = np.sort(B)
    n = len(B)
    tau_grid = np.linspace(0.0, B.max(), n_grid)
    G_tau = np.searchsorted(sorted_B, tau_grid, side="right") / n
    p0_tau = p0(tau_grid)
    G0_tau = 1.0 - (1.0 - p0_tau) ** n_eff
    return {"tau": tau_grid, "G": G_tau, "G0": G0_tau, "diff": G_tau - G0_tau}


def plot_c_tau(name: str, fit: dict, result: dict, out_path: Path, g0_cutoff: float = 0.99) -> None:
    """c(tau) = (G(tau)-G0(tau)) / (1-G0(tau)) -- the point-estimate of the
    collision rate you'd get by assuming F_coll(tau)=1 exactly at that tau
    (same R(tau) as threshold/peptides.py's solve_tau_eq_c). Restricted to
    G0(tau) < g0_cutoff: once the null saturates to ~1, the denominator
    collapses and amplifies noise into meaningless swings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tau, G, G0 = result["tau"], result["G"], result["G0"]
    with np.errstate(divide="ignore", invalid="ignore"):
        c_tau = (G - G0) / (1.0 - G0)
    valid = G0 < g0_cutoff

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.plot(tau[valid], c_tau[valid], color="#D55E00", linewidth=2.0,
            label=r"$c(\tau) = \frac{G(\tau)-G_0(\tau)}{1-G_0(\tau)}$")
    ax.axhline(0.0, color="#6b7280", linewidth=1.0, linestyle="--", label="c=0 (no collision)")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$c(\tau)$")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{name}: collision-rate point-estimate c(τ), n_eff={fit['n_eff']:,.0f} "
                 f"(fit on full distribution)", loc="left")
    ax.legend(loc="best", fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_g_g0_diff(name: str, fit: dict, result: dict, side: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tau, G, G0, diff = result["tau"], result["G"], result["G0"], result["diff"]
    q = fit["q"]
    q_pct = fit["percentile"]
    q_label = (f"{q_pct:g}th" if side == "right" else f"{100.0 - q_pct:g}th")
    region_label = (f"{q_label} pct of B(y) = {q:.4f}  (fit region: τ ≥ this)" if side == "right"
                     else f"{q_label} pct of B(y) = {q:.4f}  (fit region: τ ≤ this)")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), dpi=150, sharex=True)
    fig.patch.set_facecolor("white")

    ax1.plot(tau, G, color="#009E73", linewidth=2.0, label=r"$G(\tau)$ — empirical")
    ax1.plot(tau, G0, color="#E69F00", linewidth=2.0, linestyle="--",
              label=rf"$G_0(\tau)$ — prod-self null, n_eff={fit['n_eff']:,.0f}")
    if side != "full":
        ax1.axvline(q, color="#6b7280", linewidth=1.2, linestyle=":", label=region_label)
    ax1.set_ylabel("cumulative probability")
    side_desc = {"left": "left part (small τ)", "right": "right part (large τ)", "full": "full distribution"}[side]
    ax1.set_title(f"{name}: G(τ) vs G0(τ) — n_eff aligned on the {side_desc}", loc="left")
    ax1.legend(loc="lower right", fontsize=9, frameon=False)
    ax1.set_ylim(-0.02, 1.02)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    ax2.plot(tau, diff, color="#D55E00", linewidth=2.0, label=r"$G(\tau)-G_0(\tau)$")
    ax2.axhline(0.0, color="#6b7280", linewidth=1.0, linestyle="--")
    if side != "full":
        ax2.axvline(q, color="#6b7280", linewidth=1.2, linestyle=":")
    ax2.set_xlabel(r"$\tau$")
    ax2.set_ylabel(r"$G(\tau)-G_0(\tau)$")
    ax2.legend(loc="upper right", fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


def run(
    name: str, B: np.ndarray, null_scores: np.ndarray, n_train: int,
    side: str = "right", percentile: float = PERCENTILE,
) -> dict:
    print(f"[bold][orange2]=== {name}: G-G0 via prod-self null, n_eff aligned on the {side} "
          f"(percentile={percentile:g}) ===[/][/]")
    p0 = null_model_cdf(null_scores)

    fit = fit_n_eff_middle_alignment(B, p0, n_train, percentile=percentile, side=side)
    if side != "full":
        q_label = f"{100.0 - percentile:g}th" if side == "left" else f"{percentile:g}th"
        print(f"  {q_label} percentile of B(y) = {fit['q']:.4f}")
    print(f"  best n_eff = {fit['n_eff']:,.0f}  (SSE={fit['sse_best']:.3e})")

    result = compute_G_G0(B, p0, fit["n_eff"])
    diff = result["diff"]
    max_diff = float(diff.max())
    tau_at_max = float(result["tau"][np.argmax(diff)])
    diff_at_0 = float(diff[0])
    print(f"  G(τ)-G0(τ): max={max_diff:+.4f} at τ={tau_at_max:.4f}   at τ=0: {diff_at_0:+.4f}   "
          f"at τ=max(B): {float(diff[-1]):+.4f}")

    side_suffix = {"left": "_left", "right": "", "full": "_full"}[side]
    pct_suffix = "" if percentile == PERCENTILE else f"_p{percentile:g}"
    plot_g_g0_diff(name, fit, result, side, PLOT_DIR / f"prodself_{name}_g_g0_diff{side_suffix}{pct_suffix}.png")
    if side == "full":
        plot_c_tau(name, fit, result, PLOT_DIR / f"prodself_{name}_c_tau_full{pct_suffix}.png")

    return {"n_eff": fit["n_eff"], "q25": fit["q"], "max_diff": max_diff, "tau_at_max": tau_at_max}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=["left", "right", "full"], default="right",
                        help="Which part of B(y)'s range to align n_eff on: 'right' (original, "
                             "fit above the percentile) or 'left' (fit below 100-percentile).")
    parser.add_argument("--percentile", type=float, default=PERCENTILE,
                        help="Percentile cutoff of B(y) used to define the fit region (ignored for --side full).")
    args = parser.parse_args()

    B_pep = np.load(psn.PEPTIDES_B_PATH)
    null_pep = np.load(psn.PEPTIDES_NULL_PROD_PATH)
    res_pep = run("peptides", B_pep, null_pep, N_TRAIN["peptides"], side=args.side, percentile=args.percentile)

    print()
    B_belka = np.load(psn.BELKA_B_PATH)
    null_belka = np.load(psn.BELKA_NULL_PROD_PATH)
    res_belka = run("belka", B_belka, null_belka, N_TRAIN["belka"], side=args.side, percentile=args.percentile)

    print("\n[bold][orange2]=== Summary ===[/][/]")
    for name, res in (("peptides", res_pep), ("belka", res_belka)):
        print(f"  {name}: n_eff={res['n_eff']:,.0f}  q(B)={res['q25']:.4f}  "
              f"max[G-G0]={res['max_diff']:+.4f} at τ={res['tau_at_max']:.4f}")


if __name__ == "__main__":
    main()
