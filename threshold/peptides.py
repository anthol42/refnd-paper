"""Empirical test of the "Finding optimal threshold" theory (see CLAUDE.md /
the paper's methods note): DBAASP is the train set, PeptideAtlas is production.

For each production sample y, B(y) is its nearest-neighbor distance to DBAASP,
found by approximate search (HNSW, not exact brute-force) against an HNSW index
built on DBAASP alone — same approximation the real pipeline uses, `strict_ef`
enabled and no proximity-edge bookkeeping since only `search()` is needed here.
p_0 is the permutation-null CDF of "two unrelated samples land within distance t
of each other by chance", fit on DBAASP itself and reused from src.metrics (the
same GPD peaks-over-threshold estimator already used for CPM gamma elsewhere in
this repo). Under the no-collision null, u_y = 1 - (1 - p_0(B(y)))^n_eff should
be ~Uniform(0,1); a spike near 0 signals production samples that collide with
graphs already observed in train.

The HNSW index and B(y) are cached to disk (`threshold/cache/`), so sweeping
`--n-eff` or `--search-ef` after the first run doesn't rebuild/re-search.

Usage:
    uv run python -m threshold.peptides
    uv run python -m threshold.peptides --n-eff 500
    uv run python -m threshold.peptides --recompute
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich import print

from refnd.core import HNSWState

from src.cache import CacheStore
from src.datasets import DATASETS, DatasetConfig, load_dataset
from src.metrics import null_model_cdf, null_model_scores

CACHE_DIR = Path(__file__).parent / "cache"
TRAIN_KEY = "dbaasp"
PROD_KEY  = "peptide_atlas"


def _cached_array(path: Path, compute) -> np.ndarray:
    if path.exists():
        print(f"  [dim]Loading cached {path.name}[/]")
        return np.load(path)
    arr = compute()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


def build_or_load_hnsw(
    path: Path, train: list[str], cfg: DatasetConfig, ef_construction: int,
) -> HNSWState:
    """HNSW index over train (DBAASP) only — no exact_edges. `strict_ef=True` and
    `keep_all_edges=False` since we only ever call `.search()` on this index, never
    `.edges()`."""
    if path.exists():
        print(f"  [dim]Loading cached HNSW index {path.name}[/]")
        return HNSWState.load(cfg.modality, str(path), train)
    hnsw = HNSWState(
        cfg.modality, train,
        proximity_threshold=cfg.proximity_threshold,
        ef_construction=ef_construction,
        strict_ef=True,
        keep_all_edges=False,
        **cfg.kernel_params,
    )
    hnsw.build(progress=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hnsw.save(str(path))
    return hnsw


def compute_B(hnsw: HNSWState, prod_subset: list[str], search_ef: int) -> np.ndarray:
    """B(y): approximate nearest-neighbor distance from each production sample
    to the train (DBAASP) HNSW index."""
    results = hnsw.search(prod_subset, k=1, ef=search_ef, threads=0, progress=True)
    return np.array([hits[0][1] for hits in results], dtype=np.float64)


def u_from_p0(p0_B: np.ndarray, n_eff: float) -> np.ndarray:
    """u_y = 1 - (1 - p0(B(y)))^n_eff, via log1p/expm1 for numerical stability
    when p0 is tiny. p0_B == 1.0 exactly (e.g. a sample tied with the null's
    most extreme observed value) sends log1p(-1) to -inf — correctly resolves
    to u=1 after clipping, just noisily; silenced here since it's expected,
    not a bug."""
    with np.errstate(divide="ignore"):
        return np.clip(-np.expm1(n_eff * np.log1p(-p0_B)), 0.0, 1.0)


def find_best_n_eff(p0_B: np.ndarray, n_eff_grid: np.ndarray, min_u: float = 0.0) -> np.ndarray:
    """KS statistic of u_y against Uniform(0,1) for each candidate n_eff.

    Cheap: u is a pointwise transform of the already-computed p0(B) array, so
    no re-search / re-fit is needed per candidate. This is the theory note's
    "search n_eff to make the u histogram as flat as possible" step, made
    precise via a CDF-distance (KS statistic) instead of eyeballing bins.

    If min_u > 0, the test is restricted to the u > min_u tail, compared
    against the conditional Uniform(min_u, 1) — not Uniform(0, 1), since
    that's the correct null once you've conditioned on u > min_u. This
    mirrors the theory note's h cutoff: collisions are assumed to live near
    u = 0 (F_coll(t) ~= 1 above h), so restricting to u > h keeps the test
    on the region that should be clean uniform even when collisions exist.
    """
    from scipy import stats

    def ks_stat(n_eff: float) -> float:
        u = u_from_p0(p0_B, n_eff)
        if min_u > 0:
            u = u[u > min_u]
            if len(u) == 0:
                return np.nan
            return stats.kstest(u, "uniform", args=(min_u, 1.0 - min_u)).statistic
        return stats.kstest(u, "uniform").statistic

    return np.array([ks_stat(n_eff) for n_eff in n_eff_grid])


def plot_n_eff_search(
    n_eff_grid: np.ndarray, ks_stats: np.ndarray, best_n_eff: float, out_path: Path, min_u: float = 0.0,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(n_eff_grid, ks_stats, color="#2a78d6", linewidth=1.8, marker="o", markersize=3)
    ax.axvline(best_n_eff, color="#8a8a86", linestyle="--", linewidth=1.2,
               label=f"best n_eff = {best_n_eff:,.0f}")

    ref = f"Uniform({min_u:g},1)" if min_u > 0 else "Uniform(0,1)"
    region = f", u > {min_u:g} only" if min_u > 0 else ""
    ax.set_xscale("log")
    ax.set_xlabel("n_eff")
    ax.set_ylabel(f"KS statistic vs {ref}  (lower = flatter)")
    ax.set_title(f"Searching n_eff for the flattest $u_y$ distribution{region}")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_u_ecdf(u: np.ndarray, n_eff: float, out_path: Path) -> None:
    """Empirical CDF of u_y vs the Uniform(0,1) diagonal — the right flatness
    view when u is fundamentally discrete (few unique B(y) values from short
    -peptide identity fractions): a fixed-width histogram aliases against the
    atoms and looks like a comb, but a step-CDF is exact regardless."""
    x = np.sort(u)
    y = np.arange(1, len(x) + 1) / len(x)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(x, y, color="#2a78d6", linewidth=1.6, drawstyle="steps-post", label="empirical CDF of $u_y$")
    ax.plot([0, 1], [0, 1], color="#8a8a86", linestyle="--", linewidth=1.2, label="Uniform(0,1) CDF")

    ax.set_xlabel("u")
    ax.set_ylabel("cumulative probability")
    ax.set_title(f"ECDF of $u_y$ vs Uniform(0,1), n_eff = {n_eff:,.0f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_u_atoms(u: np.ndarray, n_eff: float, out_path: Path, max_atoms: int = 300) -> None:
    """Probability mass at each distinct observed u value, as a spike train.

    u is confined to as many atoms as B(y) has unique values (short peptides
    give coarse identity fractions), so this shows the true discrete
    distribution instead of a fixed-width histogram aliasing against it.
    """
    vals, counts = np.unique(u, return_counts=True)
    probs = counts / len(u)
    n_atoms = len(vals)
    if n_atoms > max_atoms:
        keep = np.sort(np.argsort(-probs)[:max_atoms])
        vals, probs = vals[keep], probs[keep]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.vlines(vals, 0, probs, color="#2a78d6", linewidth=1.5, label="observed mass at each atom")
    ax.axhline(1.0 / n_atoms, color="#8a8a86", linestyle="--", linewidth=1.0,
               label="equal mass per atom (naive reference)")

    ax.set_xlabel("u")
    ax.set_ylabel("probability mass")
    title = f"$u_y$ mass at each of {n_atoms:,} observed atoms, n_eff = {n_eff:,.0f}"
    if n_atoms > max_atoms:
        title += f"  (top {max_atoms} shown)"
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


BASELINE_COLOR = "#2a78d6"  # categorical slot 1 (blue)
SPIKED_COLOR   = "#eb6834"  # categorical slot 2 (orange)


def plot_B_comparison(B_a: np.ndarray, B_b: np.ndarray, label_a: str, label_b: str, out_path: Path) -> None:
    """Overlaid B(y) histograms, log-y so a thin near-zero spike from injected
    exact-duplicate train sequences is visible against the natural distribution."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    bins = np.linspace(0, max(B_a.max(), B_b.max()), 80)
    ax.hist(B_a, bins=bins, density=True, histtype="step", color=BASELINE_COLOR, linewidth=1.8, label=label_a)
    ax.hist(B_b, bins=bins, density=True, histtype="step", color=SPIKED_COLOR, linewidth=1.8, label=label_b)

    ax.set_yscale("log")
    ax.set_xlabel("B(y)")
    ax.set_ylabel("density (log scale)")
    ax.set_title("Nearest-train-neighbor distance B(y): baseline vs +injected train duplicates")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_u_ecdf_comparison(
    u_a: np.ndarray, u_b: np.ndarray, label_a: str, label_b: str, n_eff: float, out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for u, label, color in ((u_a, label_a, BASELINE_COLOR), (u_b, label_b, SPIKED_COLOR)):
        x = np.sort(u)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=color, linewidth=1.6, drawstyle="steps-post", label=label)
    ax.plot([0, 1], [0, 1], color="#8a8a86", linestyle="--", linewidth=1.0, label="Uniform(0,1)")

    ax.set_xlabel("u")
    ax.set_ylabel("cumulative probability")
    ax.set_title(f"ECDF of $u_y$: baseline vs +injected train duplicates, n_eff = {n_eff:,.0f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_u_atoms_comparison(
    u_a: np.ndarray, u_b: np.ndarray, label_a: str, label_b: str, n_eff: float, out_path: Path,
    max_atoms: int = 300,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), dpi=150, sharex=True)
    for ax, u, label, color in ((axes[0], u_a, label_a, BASELINE_COLOR), (axes[1], u_b, label_b, SPIKED_COLOR)):
        vals, counts = np.unique(u, return_counts=True)
        probs = counts / len(u)
        n_atoms = len(vals)
        if n_atoms > max_atoms:
            keep = np.sort(np.argsort(-probs)[:max_atoms])
            vals, probs = vals[keep], probs[keep]
        ax.vlines(vals, 0, probs, color=color, linewidth=1.3)
        ax.set_ylabel("probability mass")
        ax.set_title(f"{label}  ({n_atoms:,} atoms)", fontsize=10)
        ax.set_xlim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("u")
    fig.suptitle(f"$u_y$ atom mass: baseline vs +injected train duplicates, n_eff = {n_eff:,.0f}")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_c_fit_comparison(
    u_a: np.ndarray, u_b: np.ndarray, h: float, fit_a: dict, fit_b: dict,
    label_a: str, label_b: str, n_eff: float, out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for u, fit, label, color in ((u_a, fit_a, label_a, BASELINE_COLOR), (u_b, fit_b, label_b, SPIKED_COLOR)):
        x = np.sort(u)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=color, linewidth=1.1, alpha=0.55, drawstyle="steps-post")
        xs = np.array([h, 1.0])
        ys = fit["c"] + fit["slope"] * xs
        ax.plot(xs, ys, color=color, linewidth=2.2, label=f"{label}: c={fit['c']:.3f}")
    ax.axvline(h, color="#8a8a86", linestyle=":", linewidth=1.0, label=f"h = {h:g}")
    ax.plot([0, 1], [0, 1], color="#8a8a86", linestyle="--", linewidth=1.0, label="Uniform(0,1)  (c=0)")

    ax.set_xlabel("u")
    ax.set_ylabel("cumulative probability")
    ax.set_title(f"Collision-rate fit comparison, n_eff = {n_eff:,.0f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fit_c_from_ecdf(u: np.ndarray, h: float = 0.3) -> dict:
    """Estimate the collision rate c by OLS on the ECDF of u restricted to u > h.

    Per the theory note: F_u(tau) = c*F_coll(tau) + (1-c)*tau, and for tau > h
    the assumption F_coll(tau) ~= 1 (collisions cluster near u=0) makes this
    linear: F_u(tau) = c + (1-c)*tau. Fitting that line on the u > h tail gives
    the intercept as an estimate of c.
    """
    x_sorted = np.sort(u)
    y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    mask = x_sorted > h
    x_fit, y_fit = x_sorted[mask], y[mask]

    slope, intercept = np.polyfit(x_fit, y_fit, deg=1)
    y_pred = intercept + slope * x_fit
    ss_res = np.sum((y_fit - y_pred) ** 2)
    ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"c": float(intercept), "slope": float(slope), "n_points": int(mask.sum()), "r2": float(r2)}


def plot_c_fit(u: np.ndarray, h: float, fit: dict, n_eff: float, out_path: Path) -> None:
    x_sorted = np.sort(u)
    y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(x_sorted, y, color="#2a78d6", linewidth=1.4, drawstyle="steps-post", label="empirical CDF of $u_y$")
    ax.axvline(h, color="#8a8a86", linestyle=":", linewidth=1.2, label=f"h = {h:g}")

    xs = np.array([h, 1.0])
    ys = fit["c"] + fit["slope"] * xs
    ax.plot(xs, ys, color="#eb6834", linewidth=2.0,
            label=f"OLS fit (u>h): c={fit['c']:.3f}, slope={fit['slope']:.3f}")
    ax.plot([0, 1], [0, 1], color="#8a8a86", linestyle="--", linewidth=1.0, label="Uniform(0,1)  (c=0)")

    ax.set_xlabel("u")
    ax.set_ylabel("cumulative probability")
    ax.set_title(rf"Collision-rate fit: $F_u(\tau) = c + (1-c)\tau$ for $\tau > h$, n_eff = {n_eff:,.0f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _G_and_G0(B: np.ndarray, p0, n_eff: float, tau_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """G(tau): B(y)'s actual empirical CDF, measured directly (no assumptions).
    G0(tau) = 1 - (1-p0(tau))^n_eff: the null-model prediction, exactly
    `u_from_p0(p0(tau), n_eff)` since u_y = G0(B(y)) by construction."""
    sorted_B = np.sort(B)
    G_tau = np.searchsorted(sorted_B, tau_grid, side="right") / len(B)
    G0_tau = u_from_p0(p0(tau_grid), n_eff)
    return G_tau, G0_tau


def solve_tau_eq_c(
    B: np.ndarray, p0, n_eff: float, c: float, tau_grid: np.ndarray, g0_cutoff: float = 0.99,
) -> dict:
    """Directly solve eq_c, c = (G(tau)-G0(tau))/(1-G0(tau)), for tau.

    R(tau) := (G(tau)-G0(tau))/(1-G0(tau)) is the naive point-estimate of c
    you'd get by assuming F_coll(tau)=1 exactly at that tau (that assumption
    is literally how eq_c was derived from the mixture model in the first
    place). Substituting G(tau) = c*F_coll(tau) + (1-c)*G0(tau) back in:

        R(tau) = c * (F_coll(tau) - G0(tau)) / (1 - G0(tau))

    so R(tau) -> c exactly as F_coll(tau) -> 1. tau* is the smallest tau
    where R(tau) first reaches the fitted (already-known) c.

    Once G0(tau) -> 1 (a random pair matches almost surely by chance), the
    (1-G0(tau)) denominator -> 0 and amplifies noise in G(tau)-G0(tau) into
    wild swings — real instability, not signal. `g0_cutoff` restricts both
    the tau* search and the returned "valid" region to where G0(tau) is
    still comfortably below 1.
    """
    G_tau, G0_tau = _G_and_G0(B, p0, n_eff, tau_grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        R_tau = (G_tau - G0_tau) / (1.0 - G0_tau)
    valid = G0_tau < g0_cutoff

    tau_star = None
    if np.any(valid):
        valid_idx = np.flatnonzero(valid)
        hit = valid_idx[np.argmax(R_tau[valid_idx] >= c)]
        if R_tau[hit] >= c:
            tau_star = float(tau_grid[hit])
    return {"tau": tau_grid, "G": G_tau, "G0": G0_tau, "R": R_tau, "valid": valid, "tau_star": tau_star}


def plot_eq_c_solution(result: dict, c: float, n_eff: float, out_path: Path) -> None:
    valid = result["valid"]
    tau_v, R_v = result["tau"][valid], result["R"][valid]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(tau_v, R_v, color="#2a78d6", linewidth=2.0,
            label=r"$R(\tau)=\frac{G(\tau)-G_0(\tau)}{1-G_0(\tau)}$  (point estimate of c)")
    ax.axhline(c, color="#8a8a86", linestyle="--", linewidth=1.2, label=f"fitted c = {c:.4f}")
    if result["tau_star"] is not None:
        ax.axvline(result["tau_star"], color="#eb6834", linewidth=1.8, linestyle="--",
                   label=rf"$\tau^*$ = {result['tau_star']:.4f}")

    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$R(\tau)$")
    ax.set_title(rf"Solving $c=\frac{{G(\tau)-G_0(\tau)}}{{1-G_0(\tau)}}$ for $\tau$ (eq_c), n_eff={n_eff:,.0f}")
    ax.set_xlim(tau_v[0], tau_v[-1])
    pad = max(0.02, 0.15 * max(abs(R_v.min() - c), abs(R_v.max() - c)))
    ax.set_ylim(min(R_v.min(), c) - pad, max(R_v.max(), c) + pad)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def invert_F_coll(B: np.ndarray, p0, n_eff: float, c: float, tau_grid: np.ndarray) -> dict:
    """Given a fixed collision rate c (from `fit_c_from_ecdf`), invert the
    mixture equation G(tau) = c*F_coll(tau) + (1-c)*G0(tau) for F_coll(tau) at
    every tau in tau_grid — not just tau > h, where F_coll ~= 1 was only an
    *assumption* used to estimate c linearly in the first place. Related to,
    but distinct from, `solve_tau_eq_c`'s direct R(tau) solve above — see
    that function's docstring for the R(tau)/F_coll(tau) relationship.
    """
    G_tau, G0_tau = _G_and_G0(B, p0, n_eff, tau_grid)
    F_coll_raw = (G_tau - (1.0 - c) * G0_tau) / c
    return {
        "tau": tau_grid, "G": G_tau, "G0": G0_tau,
        "F_coll_raw": F_coll_raw, "F_coll": np.clip(F_coll_raw, 0.0, 1.0),
    }


def find_tau_star(tau_grid: np.ndarray, F_coll: np.ndarray, quantile: float = 0.95) -> float | None:
    """Smallest tau where the recovered F_coll crosses `quantile`: the point
    beyond which essentially all detectable train-graph collisions have
    already happened, so a larger tau only adds chance (null-model) matches.
    This is the theory's single-linkage proximity-threshold definition made
    concrete. Returns None if F_coll never reaches `quantile` on the grid.
    """
    idx = np.argmax(F_coll >= quantile)
    if F_coll[idx] < quantile:
        return None
    return float(tau_grid[idx])


def plot_tau_solution(
    result: dict, tau_star: float | None, quantile: float, c: float, n_eff: float, out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(result["tau"], result["G"], color="#8a8a86", linewidth=1.4, linestyle="--",
            label="$G(\\tau)$  (empirical, observed)")
    ax.plot(result["tau"], result["G0"], color="#4a3aa7", linewidth=1.4, linestyle=":",
            label="$G_0(\\tau)$  (null / chance)")
    ax.plot(result["tau"], result["F_coll"], color="#2a78d6", linewidth=2.2,
            label="$F_{coll}(\\tau)$  (recovered from c)")
    ax.axhline(quantile, color="#8a8a86", linewidth=0.8, linestyle="-", alpha=0.6)
    if tau_star is not None:
        ax.axvline(tau_star, color="#eb6834", linewidth=1.8, linestyle="--",
                   label=rf"$\tau^*$ = {tau_star:.4f}  ($F_{{coll}}$={quantile:g})")

    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("cumulative probability")
    ax.set_title(rf"Recovered collision CDF: $c$={c:.4f}, n_eff={n_eff:,.0f}")
    ax.set_xlim(result["tau"][0], result["tau"][-1])
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_u_histogram(
    u: np.ndarray, n_eff: float, out_path: Path,
    dataset_label: str = "PeptideAtlas (prod) vs DBAASP (train)",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.hist(u, bins=50, range=(0, 1), density=True,
            color="#2a78d6", edgecolor="white", linewidth=0.4, label="observed $u_y$")
    ax.axhline(1.0, color="#8a8a86", linestyle="--", linewidth=1.2, label="Uniform(0,1) — no collisions")

    ax.set_xlabel("u")
    ax.set_ylabel("density")
    ax.set_title(f"Distribution of $u_y$ — {dataset_label}, n_eff = {n_eff:,.0f}")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_injection_experiment(args: argparse.Namespace) -> None:
    """Held-out-duplicate control: sample `args.inject_n_train` deduplicated
    DBAASP sequences, REMOVE them from train, build the HNSW index on the
    reduced train set, then add those held-out sequences to a
    `args.prod_sample_size`-sample PeptideAtlas subset and compare B(y)/u_y/c
    against the un-injected baseline (both measured against the SAME reduced
    index, for a fair comparison).

    This is deliberately not "add train sequences to production while leaving
    them in train" — that version forces B(y)=0 by construction for every
    injected point, and since `fit_c_from_ecdf` excludes u < h from the
    regression entirely, the resulting "c shift" turns out to be forced by
    OLS-on-ECDF renormalization algebra (an injected count added to a set
    already ~uniform above h), not by anything the kernel/p0/n_eff machinery
    actually detected — see conversation.

    Holding sequences out instead makes the outcome genuinely uncertain: a
    held-out sequence's B(y) against the reduced train set now depends on
    whether it has surviving "cluster-mates" (other train sequences within
    DBAASP's own proximity threshold). One that had siblings should still
    land near u=0 (a real collision); a singleton should now look novel
    (large B(y), u > h) since its entire cluster is gone. So unlike the
    forced-duplicate version, the observed c shift isn't algebraically
    predictable ahead of time from n_inject alone.
    """
    cache = CacheStore()
    cfg = DATASETS[TRAIN_KEY]

    print(f"[bold][orange2]=== Held-out-duplicate control: {TRAIN_KEY} (train) vs "
          f"{PROD_KEY} (production) ===[/][/]")

    print("Loading datasets...")
    train_full, _ = load_dataset(TRAIN_KEY, cache)
    prod, _ = load_dataset(PROD_KEY, cache)
    train_dedup = sorted(set(train_full))
    print(f"  train ({TRAIN_KEY}): {len(train_full):,} total ({len(train_dedup):,} unique)   "
          f"production ({PROD_KEY}): {len(prod):,}")

    rng = np.random.default_rng(args.seed)
    prod_size = min(args.prod_sample_size, len(prod))
    baseline_idx = rng.choice(len(prod), size=prod_size, replace=False)
    prod_baseline = [prod[i] for i in baseline_idx]

    n_inject = min(args.inject_n_train, len(train_dedup))
    inject_idx = rng.choice(len(train_dedup), size=n_inject, replace=False)
    held_out = [train_dedup[i] for i in inject_idx]
    held_out_set = set(held_out)
    train_reduced = [s for s in train_full if s not in held_out_set]
    print(f"  holding out {n_inject:,} unique train sequences -> "
          f"train_reduced: {len(train_reduced):,} (removed {len(train_full) - len(train_reduced):,} "
          f"incl. duplicate copies)")

    prod_spiked = prod_baseline + held_out
    n_spiked = len(prod_spiked)
    naive_shift = n_inject / n_spiked
    print(f"  baseline production subsample: {prod_size:,}")
    print(f"  +{n_inject:,} held-out train sequences -> spiked set: {n_spiked:,}  "
          f"(naive 'if all were collisions' shift: {naive_shift:.4f} — NOT the real prediction here, "
          f"see below)")

    index_path = CACHE_DIR / f"hnsw_{TRAIN_KEY}_efc{args.ef_construction}_holdout{n_inject}_seed{args.seed}.hnsw"
    null_path  = (CACHE_DIR / f"null_{TRAIN_KEY}_n{args.n_null_samples}_seed{args.seed}"
                             f"_holdout{n_inject}.npy")
    B_base_path = (CACHE_DIR / f"B_ann_{PROD_KEY}_{TRAIN_KEY}_n{prod_size}_seed{args.seed}"
                              f"_efc{args.ef_construction}_ef{args.search_ef}_holdout{n_inject}.npy")
    B_spike_path = (CACHE_DIR / f"B_ann_{PROD_KEY}_{TRAIN_KEY}_n{prod_size}_seed{args.seed}"
                               f"_efc{args.ef_construction}_ef{args.search_ef}_holdout{n_inject}_spiked.npy")
    if args.recompute:
        index_path.unlink(missing_ok=True)
        null_path.unlink(missing_ok=True)
        B_base_path.unlink(missing_ok=True)
        B_spike_path.unlink(missing_ok=True)

    hnsw_holder: dict = {}

    def _get_hnsw() -> HNSWState:
        # Index is built on train_reduced (held-out sequences removed) — used for
        # BOTH baseline and spiked B(y), so the comparison is against the same reference set.
        if "hnsw" not in hnsw_holder:
            hnsw_holder["hnsw"] = build_or_load_hnsw(index_path, train_reduced, cfg, args.ef_construction)
        return hnsw_holder["hnsw"]

    print("Computing (or loading cached) B(y) for the baseline production subsample "
          "(vs train_reduced)...")
    B_baseline = _cached_array(B_base_path, lambda: compute_B(_get_hnsw(), prod_baseline, args.search_ef))
    print("Computing (or loading cached) B(y) for the +held-out production subsample "
          "(vs train_reduced)...")
    B_spiked = _cached_array(B_spike_path, lambda: compute_B(_get_hnsw(), prod_spiked, args.search_ef))

    print("Computing (or loading cached) permutation-null distances for p0 (on train_reduced, "
          "consistent with the reduced index used for B(y))...")
    null_scores = _cached_array(
        null_path,
        lambda: null_model_scores(train_reduced, cfg, n_samples=args.n_null_samples, seed=args.seed),
    )
    p0 = null_model_cdf(null_scores, tail_quantile=args.tail_quantile, min_tail_samples=args.min_tail_samples)
    p0_B_baseline = p0(B_baseline)

    if args.find_n_eff:
        # Calibrate n_eff on the BASELINE only — the spiked set has known injected
        # collisions, which would bias the "make u look uniform" search.
        n_eff_max = args.n_eff_max if args.n_eff_max is not None else len(train_reduced)
        n_eff_grid = np.unique(np.round(np.geomspace(args.n_eff_min, n_eff_max, args.n_eff_steps)))
        region = f", u > {args.ks_min_u:g} only" if args.ks_min_u > 0 else ""
        print(f"Searching n_eff in [{n_eff_grid[0]:,.0f}, {n_eff_grid[-1]:,.0f}] "
              f"({len(n_eff_grid)} points) on the baseline (vs train_reduced){region}...")
        ks_stats = find_best_n_eff(p0_B_baseline, n_eff_grid, min_u=args.ks_min_u)
        best_idx = int(np.nanargmin(ks_stats))
        n_eff = float(n_eff_grid[best_idx])
        print(f"  best n_eff = {n_eff:,.0f}  (KS statistic = {ks_stats[best_idx]:.4f})  "
              f"[was 67 for the full, un-reduced train set]")

        search_path = Path(__file__).parent / f"n_eff_search_holdout{n_inject}.png"
        plot_n_eff_search(n_eff_grid, ks_stats, n_eff, search_path, min_u=args.ks_min_u)
        print(f"[bold]Saved n_eff search curve to {search_path}[/]")
    else:
        n_eff = args.n_eff if args.n_eff is not None else len(train_reduced)

    print(f"Computing u_y with n_eff = {n_eff:,.0f}...")
    u_baseline = u_from_p0(p0_B_baseline, n_eff)
    u_spiked   = u_from_p0(p0(B_spiked), n_eff)

    # prod_spiked = prod_baseline + held_out, in that order, so the held-out items'
    # own B(y)/u_y are exactly the tail slice of B_spiked/u_spiked.
    B_heldout = B_spiked[-n_inject:]
    u_heldout = u_spiked[-n_inject:]
    frac_heldout_below_h = float(np.mean(u_heldout < args.h))
    refined_shift = frac_heldout_below_h * n_inject / n_spiked
    print(f"  held-out sequences' own B(y) vs train_reduced: min={B_heldout.min():.4f} "
          f"median={np.median(B_heldout):.4f} max={B_heldout.max():.4f}")
    print(f"  held-out sequences landing at u < h={args.h:g} (i.e. still 'siblinged' in train_reduced): "
          f"{frac_heldout_below_h:.4f}  ({int(round(frac_heldout_below_h * n_inject)):,}/{n_inject:,})")
    print(f"  refined expected c shift (only held-out items that actually land below h): "
          f"{refined_shift:.4f}")

    c_baseline = fit_c_from_ecdf(u_baseline, h=args.h)
    c_spiked   = fit_c_from_ecdf(u_spiked, h=args.h)
    observed_shift = c_spiked["c"] - c_baseline["c"]
    print(f"  baseline: n_zero_B={int(np.sum(B_baseline == 0)):,}  "
          f"c={c_baseline['c']:.4f}  R^2={c_baseline['r2']:.4f}")
    print(f"  spiked:   n_zero_B={int(np.sum(B_spiked == 0)):,}  "
          f"c={c_spiked['c']:.4f}  R^2={c_spiked['r2']:.4f}")
    print(f"  observed shift in c: {observed_shift:.4f}   naive shift: {naive_shift:.4f}   "
          f"refined shift: {refined_shift:.4f}   diff (observed - refined): "
          f"{observed_shift - refined_shift:+.4f}")

    label_a = f"baseline (n={prod_size:,})"
    label_b = f"+{n_inject:,} held-out train seqs (n={n_spiked:,})"
    stem = f"holdout{n_inject}_n{prod_size}_neff{n_eff:.0f}"
    out_dir = Path(__file__).parent

    plot_B_comparison(B_baseline, B_spiked, label_a, label_b, out_dir / f"B_comparison_{stem}.png")
    plot_u_ecdf_comparison(u_baseline, u_spiked, label_a, label_b, n_eff, out_dir / f"u_ecdf_comparison_{stem}.png")
    plot_u_atoms_comparison(u_baseline, u_spiked, label_a, label_b, n_eff, out_dir / f"u_atoms_comparison_{stem}.png")
    plot_c_fit_comparison(u_baseline, u_spiked, args.h, c_baseline, c_spiked, label_a, label_b, n_eff,
                           out_dir / f"c_fit_comparison_{stem}.png")
    print(f"[bold]Saved comparison plots to {out_dir}/*_{stem}.png[/]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prod-sample-size", type=int, default=125_000,
                        help="Number of PeptideAtlas production samples to subsample for B(y).")
    parser.add_argument("--inject-n-train", type=int, default=0,
                        help="If > 0, run the held-out-duplicate control instead of the normal single "
                             "run: sample this many deduplicated train sequences, REMOVE them from "
                             "train, add them to the --prod-sample-size production subsample, and "
                             "compare against the baseline (both measured against the reduced index).")
    parser.add_argument("--n-null-samples", type=int, default=10_000_000,
                        help="Number of shuffled-pair samples for the permutation null model.")
    parser.add_argument("--n-eff", type=int, default=None,
                        help="Effective independent train comparisons for u_y. Defaults to n_train. "
                             "Ignored if --find-n-eff is set.")
    parser.add_argument("--find-n-eff", action="store_true",
                        help="Search a log-spaced n_eff grid for the one minimizing the KS statistic "
                             "of u_y against Uniform(0,1), and use that instead of --n-eff.")
    parser.add_argument("--n-eff-min", type=float, default=1.0)
    parser.add_argument("--n-eff-max", type=float, default=None,
                        help="Defaults to n_train.")
    parser.add_argument("--n-eff-steps", type=int, default=60)
    parser.add_argument("--ks-min-u", type=float, default=0.0,
                        help="Restrict the n_eff-search KS test to u > this value, compared against "
                             "the conditional Uniform(ks_min_u, 1) — mirrors the theory note's h cutoff "
                             "(e.g. 0.3), where collisions are assumed to live near u=0.")
    parser.add_argument("--h", type=float, default=0.3,
                        help="Cutoff above which F_coll(tau) ~= 1 is assumed; the c regression "
                             "(F_u(tau) = c + (1-c)*tau) is fit on the u > h tail of the ECDF.")
    parser.add_argument("--tau-quantile", type=float, default=0.95,
                        help="tau* is the smallest tau where the recovered F_coll(tau) (inverted from "
                             "c) reaches this quantile — the theory's single-linkage threshold made concrete.")
    parser.add_argument("--tau-steps", type=int, default=400,
                        help="Number of tau grid points (0 to max observed B(y)) for the F_coll inversion.")
    parser.add_argument("--ef-construction", type=int, default=64,
                        help="HNSW ef_construction used to build the DBAASP index.")
    parser.add_argument("--search-ef", type=int, default=64,
                        help="HNSW ef used when searching the index for each production sample's NN.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tail-quantile", type=float, default=0.01)
    parser.add_argument("--min-tail-samples", type=int, default=1000)
    parser.add_argument("--recompute", action="store_true",
                        help="Ignore cached HNSW index/B(y)/null-score arrays and recompute from scratch.")
    args = parser.parse_args()

    if args.inject_n_train > 0:
        run_injection_experiment(args)
        return

    cache = CacheStore()
    cfg = DATASETS[TRAIN_KEY]

    print(f"[bold][orange2]=== Threshold theory test: {TRAIN_KEY} (train) vs {PROD_KEY} (production) ===[/][/]")

    print("Loading datasets...")
    train, _ = load_dataset(TRAIN_KEY, cache)
    prod, _ = load_dataset(PROD_KEY, cache)
    n_train = len(train)
    print(f"  train ({TRAIN_KEY}): {n_train:,}   production ({PROD_KEY}): {len(prod):,}")

    rng = np.random.default_rng(args.seed)
    prod_size = min(args.prod_sample_size, len(prod))
    prod_idx = rng.choice(len(prod), size=prod_size, replace=False)
    prod_subset = [prod[i] for i in prod_idx]
    print(f"  subsampled production set: {len(prod_subset):,}")

    index_path = CACHE_DIR / f"hnsw_{TRAIN_KEY}_efc{args.ef_construction}.hnsw"
    B_path     = (CACHE_DIR / f"B_ann_{PROD_KEY}_{TRAIN_KEY}_n{prod_size}_seed{args.seed}"
                             f"_efc{args.ef_construction}_ef{args.search_ef}.npy")
    null_path  = CACHE_DIR / f"null_{TRAIN_KEY}_n{args.n_null_samples}_seed{args.seed}.npy"
    if args.recompute:
        index_path.unlink(missing_ok=True)
        B_path.unlink(missing_ok=True)
        null_path.unlink(missing_ok=True)

    def _compute_B() -> np.ndarray:
        # Only build/load the HNSW index if B(y) actually needs (re)computing —
        # skip it entirely on a B(y) cache hit (the common case when just sweeping n_eff).
        hnsw = build_or_load_hnsw(index_path, train, cfg, args.ef_construction)
        return compute_B(hnsw, prod_subset, args.search_ef)

    print("Computing (or loading cached) approximate nearest-train-neighbor distances B(y)...")
    B = _cached_array(B_path, _compute_B)

    print("Computing (or loading cached) permutation-null distances for p0...")
    null_scores = _cached_array(
        null_path,
        lambda: null_model_scores(train, cfg, n_samples=args.n_null_samples, seed=args.seed),
    )

    print("Fitting p0 CDF...")
    p0 = null_model_cdf(null_scores, tail_quantile=args.tail_quantile, min_tail_samples=args.min_tail_samples)
    p0_B = p0(B)

    if args.find_n_eff:
        n_eff_max = args.n_eff_max if args.n_eff_max is not None else n_train
        n_eff_grid = np.unique(np.round(
            np.geomspace(args.n_eff_min, n_eff_max, args.n_eff_steps)
        ))
        region = f", u > {args.ks_min_u:g} only" if args.ks_min_u > 0 else ""
        print(f"Searching n_eff in [{n_eff_grid[0]:,.0f}, {n_eff_grid[-1]:,.0f}] "
              f"({len(n_eff_grid)} points) for the flattest u_y{region}...")
        ks_stats = find_best_n_eff(p0_B, n_eff_grid, min_u=args.ks_min_u)
        best_idx = int(np.nanargmin(ks_stats))
        n_eff = float(n_eff_grid[best_idx])
        print(f"  best n_eff = {n_eff:,.0f}  (KS statistic = {ks_stats[best_idx]:.4f})")

        search_path = Path(__file__).parent / f"n_eff_search{'_u_gt_' + str(args.ks_min_u) if args.ks_min_u > 0 else ''}.png"
        plot_n_eff_search(n_eff_grid, ks_stats, n_eff, search_path, min_u=args.ks_min_u)
        print(f"[bold]Saved n_eff search curve to {search_path}[/]")
    else:
        n_eff = args.n_eff if args.n_eff is not None else n_train

    print(f"Computing u_y with n_eff = {n_eff:,.0f}...")
    u = u_from_p0(p0_B, n_eff)

    print(f"  B(y):  min={B.min():.4f}  median={np.median(B):.4f}  max={B.max():.4f}")
    print(f"  p0(B): min={p0_B.min():.3e}  median={np.median(p0_B):.3e}  max={p0_B.max():.3e}")
    print(f"  u:     mean={u.mean():.4f}  frac(u<0.05)={np.mean(u < 0.05):.4f}  frac(u<0.01)={np.mean(u < 0.01):.4f}"
          f"  n_unique_B={len(np.unique(B)):,}  n_unique_u={len(np.unique(u)):,}")

    stem = f"n_eff_{n_eff:.0f}_n{prod_size}"
    hist_path  = Path(__file__).parent / f"u_histogram_{stem}.png"
    ecdf_path  = Path(__file__).parent / f"u_ecdf_{stem}.png"
    atoms_path = Path(__file__).parent / f"u_atoms_{stem}.png"
    plot_u_histogram(u, n_eff, hist_path)
    plot_u_ecdf(u, n_eff, ecdf_path)
    plot_u_atoms(u, n_eff, atoms_path)
    print(f"[bold]Saved histogram to {hist_path}[/]")
    print(f"[bold]Saved ECDF plot to {ecdf_path}[/]")
    print(f"[bold]Saved atom-mass plot to {atoms_path}[/]")

    c_fit = fit_c_from_ecdf(u, h=args.h)
    print(f"Fitting c on the u > {args.h:g} tail (F_u(tau) = c + (1-c)*tau)...")
    print(f"  c={c_fit['c']:.4f}  slope={c_fit['slope']:.4f} (expect 1-c={1 - c_fit['c']:.4f})  "
          f"n_points={c_fit['n_points']:,}  R^2={c_fit['r2']:.4f}")

    c_fit_path = Path(__file__).parent / f"c_fit_h{args.h:g}_{stem}.png"
    plot_c_fit(u, args.h, c_fit, n_eff, c_fit_path)
    print(f"[bold]Saved c-fit plot to {c_fit_path}[/]")

    tau_grid = np.linspace(0.0, B.max(), args.tau_steps)

    print(f"Solving eq_c directly for tau: c = [G(tau)-G0(tau)] / [1-G0(tau)], c={c_fit['c']:.4f}...")
    eq_c_solved = solve_tau_eq_c(B, p0, n_eff, c_fit["c"], tau_grid)
    if eq_c_solved["tau_star"] is not None:
        print(f"  tau* (smallest tau where R(tau) reaches c) = {eq_c_solved['tau_star']:.4f}   "
              f"[dataset's current proximity_threshold = {cfg.proximity_threshold:g}]")
    else:
        print(f"  R(tau) never reaches c={c_fit['c']:.4f} on [0, {B.max():.4f}].")
    eq_c_path = Path(__file__).parent / f"tau_eq_c_h{args.h:g}_{stem}.png"
    plot_eq_c_solution(eq_c_solved, c_fit["c"], n_eff, eq_c_path)
    print(f"[bold]Saved eq_c solution plot to {eq_c_path}[/]")

    solved = invert_F_coll(B, p0, n_eff, c_fit["c"], tau_grid)
    tau_star = find_tau_star(tau_grid, solved["F_coll"], quantile=args.tau_quantile)
    print(f"Cross-check: inverting for F_coll(tau) = [G(tau) - (1-c)*G0(tau)] / c...")
    if tau_star is not None:
        print(f"  tau* (smallest tau with F_coll >= {args.tau_quantile:g}) = {tau_star:.4f}")
    else:
        print(f"  F_coll never reaches {args.tau_quantile:g} on [0, {B.max():.4f}] — "
              f"either c is too small/noisy, or collisions aren't saturating in this range.")

    tau_path = Path(__file__).parent / f"tau_solution_h{args.h:g}_{stem}.png"
    plot_tau_solution(solved, tau_star, args.tau_quantile, c_fit["c"], n_eff, tau_path)
    print(f"[bold]Saved tau solution plot to {tau_path}[/]")


if __name__ == "__main__":
    main()
