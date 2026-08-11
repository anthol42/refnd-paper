"""Empirical test of the "Finding optimal threshold" theory (see CLAUDE.md /
the paper's methods note) on BELKA, at full scale — analog to
threshold/peptides.py.

Train: a random HALF of BELKA's ~98.4M unique molecules (DNA-tag stripped —
`[Dy]` is the DNA-conjugation attachment point, not part of the binding
pharmacophore, and its presence would add a spurious shared-similarity
contribution to every train/triazine-test comparison that the non-triazine
test portion and the null don't share). B(y) is each production molecule's
nearest-neighbor distance to this HNSW index.

Production: the REAL Kaggle BELKA test set in full (878,022 molecules), a
mix of triazine-scaffold molecules (sharing train's synthesis route — a true
positive control, ~44.6% of the set) and a non-DEL OOD population Leash Bio
mixed in (unrelated chemistry — a true negative control, ~55.4%).

p_0 (the null): NOT built from any real chemical population related to
BELKA. Every attempt that was — test-train pairs, train-train pairs, ZINC
(even filtered for redundancy, even density-matched to BELKA's fingerprint
popcount), and even novel recombinations of BELKA's own building blocks —
came out measurably biased, because each one carries some inherited
relatedness to train that the true OOD test portion doesn't share, which
overstates or understates collisions asymmetrically. The null that finally
gave a trustworthy, non-circular result (u genuinely flat above h on the
known-OOD subset, and a correctly recovered true mixture ratio when reused
unchanged on the labeled mix) is a population of PURELY RANDOM ATOM
molecules: linear SELFIES chains of valence>=2 atoms (C/N/O/S — halogens
excluded, they terminate the SELFIES derivation immediately), chain length
~Normal(50, 5) tokens, calibrated to land Morgan-fingerprint popcount at
~66 (between real train's 68.7 and test's 64.0 averages). Zero building-block
or fragment-level relationship to BELKA of any kind — matched only on coarse
size.

n_eff is fit via KS-minimization restricted to u > h=0.3 (the theory's own
"collisions cluster near u=0" assumption) — never by looking at c, which
would be circular (c is exactly the unknown quantity in real use).

Whether a molecule is triazine-scaffold or OOD is NOT used anywhere in this
script — it's the fully realistic scenario where you only have one unlabeled
production set.

Long steps (fingerprinting, HNSW build, HNSW search, the null) are all
cached to disk under /mnt/documents/refnd_cache/threshold_belka/, so a
re-run (e.g. after tweaking the plots) doesn't redo them.

Usage:
    uv run python -m threshold.belka
"""

from __future__ import annotations

import argparse
import gc
import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from refnd.core import HNSWState
from rich import print

from src.cache import CacheStore
from src.datasets import DATASETS, belka_unique_smiles
from src.fingerprints import (
    FP_N_BYTES_PACKED,
    compute_and_cache_fingerprints_to_disk,
    stream_bitfingerprints_from_disk,
)
from src.metrics import null_model_cdf, null_model_scores

from threshold.peptides import fit_c_from_ecdf, find_best_n_eff, plot_eq_c_solution, solve_tau_eq_c, u_from_p0

OUT_DIR = Path("/mnt/documents/refnd_cache/threshold_belka")
PLOT_DIR = Path(__file__).parent

SEED = 42
H = 0.3
DNA_TAG = "[Dy]"
EF_CONSTRUCTION = 64
SEARCH_EF = 64

# Null-model generation (see module docstring for the derivation).
N_NULL_MOLECULES = 100_000
NULL_MEAN_SIZE = 50.0
NULL_STD_SIZE = 5.0
NULL_ATOM_ALPHABET = ["[C]"] * 6 + ["[N]"] * 2 + ["[O]"] * 2 + ["[S]"] * 1

TRAIN_FP_PATH = OUT_DIR / f"belka_train_half_stripped_fp_seed{SEED}.bin"
TRAIN_SMILES_PATH = OUT_DIR / f"belka_train_half_stripped_smiles_seed{SEED}.pkl"
INDEX_PATH = OUT_DIR / f"belka_train_half_stripped_seed{SEED}.hnsw"
TEST_OOD_FP_PATH = OUT_DIR / "belka_test_ood_stripped_fp.bin"
TEST_TRIAZINE_FP_PATH = OUT_DIR / "belka_test_triazine_stripped_fp.bin"
B_PATH = OUT_DIR / f"B_belka_full_test_vs_halftrain_seed{SEED}.npy"
RANDOM_FP_PATH = OUT_DIR / f"random_atoms_fp_n{N_NULL_MOLECULES}_mean{NULL_MEAN_SIZE:g}_seed{SEED}.bin"
NULL_PATH = OUT_DIR / f"null_random_atoms_n{N_NULL_MOLECULES}_mean{NULL_MEAN_SIZE:g}_seed{SEED}.npy"

C_P0, C_G0, C_G = "#0072B2", "#E69F00", "#009E73"
GRID, TEXT, MUTED = "#e5e7eb", "#374151", "#6b7280"


def strip_dna_tag(smiles: str) -> str:
    return smiles.replace(DNA_TAG, "")


def mem_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1024 / 1024
    return float("nan")


def build_train_index(cfg) -> tuple[HNSWState, int]:
    """Random half of BELKA's unique train molecules, DNA-tag stripped,
    fingerprinted (checkpointed/resumable) and HNSW-indexed. Fully cached."""
    if INDEX_PATH.exists() and TRAIN_FP_PATH.exists():
        n_train = TRAIN_FP_PATH.stat().st_size // FP_N_BYTES_PACKED
        print(f"  Loading cached HNSW index ({INDEX_PATH.name}, {n_train:,} molecules)...")
        hnsw = HNSWState.load(
            cfg.modality, str(INDEX_PATH),
            stream_bitfingerprints_from_disk(str(TRAIN_FP_PATH), n_train, progress=True),
        )
        return hnsw, n_train

    # Fingerprinted to a .tmp path and atomically renamed to TRAIN_FP_PATH
    # only once fully complete -- TRAIN_FP_PATH existing is then a reliable
    # signal that fingerprinting is done, even if a later step (the index
    # build) gets interrupted. Without this, a restart could otherwise
    # overwrite (in "wb" mode) a genuinely complete fingerprint file in
    # place while re-fingerprinting, corrupting it if interrupted again.
    tmp_path = TRAIN_FP_PATH.with_suffix(".bin.tmp")
    checkpoint_path = tmp_path.with_name(tmp_path.name + ".checkpoint")

    if TRAIN_FP_PATH.exists():
        n_written = TRAIN_FP_PATH.stat().st_size // FP_N_BYTES_PACKED
        print(f"  Reusing complete fingerprint cache ({TRAIN_FP_PATH.name}): {n_written:,} molecules")
    else:
        resume_from = int(checkpoint_path.read_text()) if checkpoint_path.exists() else 0
        if resume_from == 0 and not TRAIN_SMILES_PATH.exists():
            cache = CacheStore()
            print("Loading full BELKA unique train SMILES...")
            train_smiles_full = belka_unique_smiles(cache)
            n_total = len(train_smiles_full)
            n_half = n_total // 2
            print(f"  {n_total:,} unique train molecules; using a random half ({n_half:,})")

            rng = np.random.default_rng(SEED)
            half_idx = rng.choice(n_total, size=n_half, replace=False)
            half_smiles = [strip_dna_tag(train_smiles_full[i]) for i in half_idx]
            del train_smiles_full
            gc.collect()
            with open(TRAIN_SMILES_PATH, "wb") as f:
                pickle.dump(half_smiles, f)
        else:
            print(f"  Loading cached half-train SMILES ({TRAIN_SMILES_PATH.name})...")
            with open(TRAIN_SMILES_PATH, "rb") as f:
                half_smiles = pickle.load(f)

        print(f"Fingerprinting {len(half_smiles):,} stripped train molecules "
              f"(resume_from={resume_from:,}, checkpointed every 2,000,000)...")
        t0 = time.time()
        n_written, n_missing = compute_and_cache_fingerprints_to_disk(
            half_smiles, str(tmp_path), progress=True,
            resume_from=resume_from, checkpoint_path=str(checkpoint_path), checkpoint_every=2_000_000,
        )
        if n_missing:
            print(f"  [yellow]Dropped {n_missing:,} SMILES RDKit couldn't parse[/]")
        del half_smiles
        gc.collect()
        tmp_path.rename(TRAIN_FP_PATH)
        checkpoint_path.unlink(missing_ok=True)
        print(f"  fingerprinting done in {time.time()-t0:.1f}s. MemAvailable: {mem_gb():.2f}GB")

    print(f"Building HNSW index on {n_written:,} train molecules...")
    t0 = time.time()
    hnsw = HNSWState(
        cfg.modality,
        stream_bitfingerprints_from_disk(str(TRAIN_FP_PATH), n_written, progress=True),
        proximity_threshold=cfg.proximity_threshold,
        ef_construction=EF_CONSTRUCTION,
        strict_ef=True,
        keep_all_edges=False,
        cache_capacity=0,
        **cfg.kernel_params,
    )
    hnsw.build(progress=True)
    print(f"  build done in {time.time()-t0:.1f}s. MemAvailable: {mem_gb():.2f}GB")
    hnsw.save(str(INDEX_PATH))
    return hnsw, n_written


def load_full_test_fps() -> list:
    """The REAL Kaggle test set, both triazine-scaffold and OOD portions
    combined in their true proportions -- both already fingerprinted
    (DNA-tag stripped) and cached from earlier in this investigation."""
    n_ood = TEST_OOD_FP_PATH.stat().st_size // FP_N_BYTES_PACKED
    n_tri = TEST_TRIAZINE_FP_PATH.stat().st_size // FP_N_BYTES_PACKED
    print(f"Loading full stripped test set: {n_ood:,} OOD + {n_tri:,} triazine = {n_ood+n_tri:,} total")
    ood_fps = list(stream_bitfingerprints_from_disk(str(TEST_OOD_FP_PATH), n_ood, progress=True))
    tri_fps = list(stream_bitfingerprints_from_disk(str(TEST_TRIAZINE_FP_PATH), n_tri, progress=True))
    return ood_fps + tri_fps


def generate_random_atom_fps(n: int, seed: int) -> list:
    """Purely random-atom molecules -- zero relation to BELKA's building
    blocks. See module docstring."""
    from refnd.utils import BitFingerprint
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    import selfies as sf

    rng = np.random.default_rng(seed)
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    n_invalid = 0
    t0 = time.time()
    while len(fps) < n:
        n_tok = max(10, int(round(rng.normal(NULL_MEAN_SIZE, NULL_STD_SIZE))))
        toks = rng.choice(NULL_ATOM_ALPHABET, size=n_tok)
        smi = sf.decoder("".join(toks))
        m = Chem.MolFromSmiles(smi)
        if m is None:
            n_invalid += 1
            continue
        fp = morgan_gen.GetFingerprint(m)
        fps.append(BitFingerprint.from_np(np.array(fp, dtype=bool)))
        if len(fps) % 20000 == 0:
            print(f"  {len(fps):,}/{n:,} generated ({time.time()-t0:.1f}s)")
    print(f"  done: {len(fps):,} valid, {n_invalid:,} invalid, {time.time()-t0:.1f}s total")
    return fps


def build_null_scores(cfg) -> np.ndarray:
    if NULL_PATH.exists():
        print(f"  Loading cached null ({NULL_PATH.name})...")
        return np.load(NULL_PATH)
    if RANDOM_FP_PATH.exists() and RANDOM_FP_PATH.stat().st_size // FP_N_BYTES_PACKED == N_NULL_MOLECULES:
        print(f"  Loading cached random-atom fingerprints ({RANDOM_FP_PATH.name})...")
        random_fps = list(stream_bitfingerprints_from_disk(str(RANDOM_FP_PATH), N_NULL_MOLECULES, progress=False))
    else:
        print(f"Generating {N_NULL_MOLECULES:,} random-atom molecules...")
        random_fps = generate_random_atom_fps(N_NULL_MOLECULES, SEED)
        with open(RANDOM_FP_PATH, "wb") as f:
            for fp in random_fps:
                f.write(np.packbits(np.asarray(fp.to_np(), dtype=np.uint8)).tobytes())
    print("Sampling random pairs within the random-atom molecules...")
    null_scores = null_model_scores(random_fps, cfg, n_samples=2_000_000, seed=SEED, shuffle=False)
    np.save(NULL_PATH, null_scores)
    return null_scores


def plot_p0_g0_g(p0, B: np.ndarray, n_eff: float, null_scores: np.ndarray, out_path: Path) -> None:
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
    ax.plot(tau_grid, p0_vals, color=C_P0, linewidth=1.6, linestyle=":", label=r"$p_0(\tau)$ — random-atom null")
    ax.plot(tau_grid, G0_vals, color=C_G0, linewidth=2.2, label=rf"$G_0(\tau)$, n_eff={n_eff:,.0f}")
    ax.plot(tau_grid, G_vals, color=C_G, linewidth=2.2, label=r"$G(\tau)$ — empirical (full test set)")
    ax.set_xlabel(r"$\tau$", fontsize=10, color=MUTED)
    ax.set_ylabel("cumulative probability", fontsize=10, color=MUTED)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("BELKA (full scale): $p_0$, $G_0$, $G$", fontsize=11, color=TEXT, loc="left")
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


def plot_u_pdf(u: np.ndarray, n_eff: float, c_ols: float, c_surv: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor("white")
    ax.hist(u, bins=50, range=(0, 1), density=True, color=C_G0, alpha=0.85, label="u density (full test set)")
    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle="--", label="Uniform(0,1)")
    ax.axhline(1 - c_ols, color=C_G, linewidth=1.4, linestyle="-.", label=rf"(1-c) = {1-c_ols:.3f}")
    ax.axvline(H, color=MUTED, linewidth=1, linestyle=":", label=f"h={H}")
    ax.set_xlabel(r"$u_y$", fontsize=11, color=MUTED)
    ax.set_ylabel("density", fontsize=11, color=MUTED)
    ax.set_title(
        f"u PDF — full BELKA test set (n={len(u):,}), n_eff={n_eff:,.0f}\n"
        f"c_ols={c_ols:+.3f}, c_surv={c_surv:+.3f}",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    cfg = DATASETS["belka"]
    print("[bold][orange2]=== BELKA: full-scale threshold theory test ===[/][/]")
    print(f"MemAvailable at start: {mem_gb():.2f}GB")

    hnsw, n_train = build_train_index(cfg)
    print(f"MemAvailable after index build: {mem_gb():.2f}GB")

    if B_PATH.exists():
        B = np.load(B_PATH)
        print(f"Loaded cached B(y) ({B_PATH.name}): n={len(B):,}")
    else:
        test_fps = load_full_test_fps()
        print(f"Searching {len(test_fps):,} test molecules against the {n_train:,}-molecule train index...")
        t0 = time.time()
        results = hnsw.search(test_fps, k=1, ef=SEARCH_EF, threads=0, progress=True)
        B = np.array([hits[0][1] for hits in results], dtype=np.float64)
        print(f"  search done in {time.time()-t0:.1f}s. MemAvailable: {mem_gb():.2f}GB")
        np.save(B_PATH, B)
        del test_fps, results
        gc.collect()
    print(f"B(y): n={len(B):,} min={B.min():.4f} median={np.median(B):.4f} max={B.max():.4f}")

    null_scores = build_null_scores(cfg)
    print(f"null (random-atom pairs): n={len(null_scores):,} min={null_scores.min():.4f} "
          f"median={np.median(null_scores):.4f} max={null_scores.max():.4f}")

    p0 = null_model_cdf(null_scores)
    p0_B = p0(B)
    n_eff_grid = np.unique(np.round(np.geomspace(1.0, float(n_train), 100)))
    print("Fitting n_eff (KS-minimized on u > h, non-circular -- no label information used)...")
    ks_stats = find_best_n_eff(p0_B, n_eff_grid, min_u=H)
    best_idx = int(np.nanargmin(ks_stats))
    n_eff = float(n_eff_grid[best_idx])
    u = u_from_p0(p0_B, n_eff)
    c_fit = fit_c_from_ecdf(u, h=H)
    frac_above = np.mean(u > H)
    c_surv = 1 - frac_above / (1 - H)

    print(f"\n[bold]n_eff={n_eff:,.0f}  KS={ks_stats[best_idx]:.4f}[/]")
    print(f"[bold]c_ols={c_fit['c']:+.4f} (R2={c_fit['r2']:.3f}, n_points={c_fit['n_points']})  "
          f"c_surv={c_surv:+.4f}[/]")
    print(f"u: mean={u.mean():.4f}  frac(u<0.05)={np.mean(u<0.05):.4f}  frac(u>0.95)={np.mean(u>0.95):.4f}")

    print("\nSolving eq_c for tau (given the fitted c)...")
    tau_grid = np.linspace(0.0, 1.0, 2000)
    tau_result = solve_tau_eq_c(B, p0, n_eff, c_fit["c"], tau_grid)
    if tau_result["tau_star"] is not None:
        print(f"[bold]tau* = {tau_result['tau_star']:.4f}[/]  (smallest tau where R(tau) reaches c={c_fit['c']:.4f})")
    else:
        print("[yellow]No tau in the grid reached the fitted c before G0(tau) saturated -- no solution found.[/]")
    plot_eq_c_solution(tau_result, c_fit["c"], n_eff, PLOT_DIR / "belka_tau_solution.png")
    print(f"Saved {PLOT_DIR / 'belka_tau_solution.png'}")

    plot_p0_g0_g(p0, B, n_eff, null_scores, PLOT_DIR / "belka_p0_g0_g.png")
    plot_u_pdf(u, n_eff, c_fit["c"], c_surv, PLOT_DIR / "belka_u_pdf.png")
    print(f"\nSaved {PLOT_DIR / 'belka_p0_g0_g.png'}")
    print(f"Saved {PLOT_DIR / 'belka_u_pdf.png'}")


if __name__ == "__main__":
    main()
