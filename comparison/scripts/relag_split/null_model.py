"""Null-model gamma for CPM Leiden.

Ported from refnd-paper/src/metrics.py. Estimates gamma = P(distance between two
random, element-shuffled samples <= proximity_threshold) via a GPD peaks-over-
threshold tail fit, then hands it to find_communities(..., objective=CPM) as the
CPM resolution. See relag-paper/src/metrics.py for the derivation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from refnd.utils import BitFingerprint
from refnd.kernels import zip_kernel


@dataclass
class NullCfg:
    """Minimal carrier for the fields null_model needs (mirrors DatasetConfig)."""
    modality: Any
    proximity_threshold: float
    kernel_params: dict = field(default_factory=dict)


def _shuffle_sample(sample: Any, rng: np.random.Generator) -> Any:
    """Return a copy of *sample* with its elements randomly permuted."""
    if isinstance(sample, str):
        chars = list(sample)
        rng.shuffle(chars)
        return "".join(chars)
    return BitFingerprint.random(len(sample), sample.count())


def fit_gpd_tail(scores: np.ndarray, tail_quantile: float = 0.01,
                 min_tail_samples: int = 1000):
    """Fit a Generalized Pareto Distribution to the lower tail of *scores*."""
    from scipy.stats import genpareto

    n = len(scores)
    k = max(min_tail_samples, int(n * tail_quantile))
    k = min(k, n)
    if k < min_tail_samples:
        return None

    sorted_scores = np.sort(scores)
    u = sorted_scores[k - 1]
    excesses = u - sorted_scores[:k]
    try:
        shape, _, scale = genpareto.fit(excesses, floc=0)
    except Exception:
        return None
    p_below_u = k / n
    return sorted_scores, float(u), float(shape), float(scale), p_below_u


def _gpd_tail_estimate(scores: np.ndarray, threshold: float,
                       tail_quantile: float = 0.01,
                       min_tail_samples: int = 1000):
    """Estimate P(score <= threshold) via the GPD tail, extrapolating below the
    sampled range. Returns None if the fit is infeasible or threshold isn't in the tail."""
    from scipy.stats import genpareto

    fit = fit_gpd_tail(scores, tail_quantile=tail_quantile, min_tail_samples=min_tail_samples)
    if fit is None:
        return None
    _sorted, u, shape, scale, p_below_u = fit
    if threshold >= u:
        return None
    excess_needed = u - threshold
    p_given_tail = genpareto.sf(excess_needed, shape, loc=0, scale=scale)
    return float(p_below_u * p_given_tail)


def null_model_scores(data: list, cfg: NullCfg, n_samples: int = 10_000_000,
                      seed: int = 42, shuffle: bool = True) -> np.ndarray:
    """Raw null distances: pair up random (element-shuffled) samples and score
    every pair with the dataset's kernel."""
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, len(data), size=n_samples)
    idx_b = rng.integers(0, len(data), size=n_samples)
    if shuffle:
        list_a = [_shuffle_sample(data[i], rng) for i in idx_a]
        list_b = [_shuffle_sample(data[i], rng) for i in idx_b]
    else:
        list_a = [data[i] for i in idx_a]
        list_b = [data[i] for i in idx_b]
    return np.asarray(
        zip_kernel(cfg.modality, list_a, list_b, n_threads=0, progress=True, **cfg.kernel_params),
        dtype=np.float64,
    )


def null_model(data: list, cfg: NullCfg, n_samples: int = 10_000_000, seed: int = 42,
               use_gpd_tail: bool = True, shuffle: bool = True) -> float:
    """Estimate gamma = P(distance <= proximity_threshold) under a null model.

    shuffle=True (sequences): element-permute each sampled item — a composition-
    preserving "no real structure" null. shuffle=False (molecules): pair up real,
    unshuffled items directly; permuting fingerprint bit positions destroys chemical
    structure and makes the null FAR more dissimilar than real pairs, so p0 underflows
    to 0 (see src/metrics.py BELKA note). Real-pair background similarity is the right
    null there."""
    scores = null_model_scores(data, cfg, n_samples=n_samples, seed=seed, shuffle=shuffle)
    if use_gpd_tail:
        gpd_p = _gpd_tail_estimate(scores, cfg.proximity_threshold)
        if gpd_p is not None:
            print(f"  null model (GPD tail): gamma={gpd_p:.3e}")
            return gpd_p
    n_hits = int(np.sum(scores <= cfg.proximity_threshold))
    # Jeffreys pseudocount avoids gamma=0 when no pair lands in the tail.
    gamma = (n_hits + 0.5) / (n_samples + 1) if n_hits == 0 else n_hits / n_samples
    print(f"  null model (empirical): gamma={gamma:.3e} ({n_hits} hits / {n_samples})")
    return gamma


# --- random-atom molecule null (ported from refnd-paper threshold/molecules.py) ---
# Linear SELFIES chains of bare, valence>=2 atoms, length ~Normal(mean, std) tokens.
# Zero relation to any real building block -- matched only on coarse size.
NULL_ATOM_ALPHABET = ["[C]"] * 6 + ["[N]"] * 2 + ["[O]"] * 2 + ["[S]"] * 1
NULL_MEAN_SIZE = 50.0
NULL_STD_SIZE = 5.0


def generate_random_molecule_fps(n: int, seed: int) -> list:
    """Purely random-atom molecules unrelated to any real building block: linear
    SELFIES chains of bare C/N/O/S atoms, length ~Normal(NULL_MEAN_SIZE, NULL_STD_SIZE),
    decoded to SMILES, kept only if RDKit accepts them. Morgan radius=2, fpSize=2048
    (matches run_molecule.py's fingerprints)."""
    from refnd.utils import BitFingerprint
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    import selfies as sf

    rng = np.random.default_rng(seed)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    n_invalid = 0
    while len(fps) < n:
        n_tok = max(10, int(round(rng.normal(NULL_MEAN_SIZE, NULL_STD_SIZE))))
        toks = rng.choice(NULL_ATOM_ALPHABET, size=n_tok)
        smi = sf.decoder("".join(toks))
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_invalid += 1
            continue
        fp = gen.GetFingerprint(mol)
        # DataStructs.ConvertToNumpyArray is RDKit's documented conversion path;
        # np.array(fp) via the buffer protocol segfaults on the cluster.
        arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(BitFingerprint.from_np(arr.astype(np.uint8)))
    print(f"  generated {len(fps):,} random-atom molecules ({n_invalid:,} invalid, discarded)")
    return fps


def random_molecule_null_gamma(cfg: NullCfg, n_molecules: int = 100_000,
                               n_pairs: int = 2_000_000, seed: int = 42,
                               cache_path=None) -> float:
    """gamma = P(distance <= proximity_threshold) under a random-atom-molecule null
    (ported from refnd-paper threshold/molecules.py:find_gamma_function). The null
    pool is synthetic, so gamma is dataset-independent: one value serves every
    molecule dataset at a given threshold. Caches the raw null scores (threshold-
    independent) to *cache_path* so re-runs / other datasets just reload them."""
    from pathlib import Path

    scores = None
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            print(f"  loading cached random-molecule null scores: {cache_path.name}")
            scores = np.load(cache_path)
    if scores is None:
        random_fps = generate_random_molecule_fps(n_molecules, seed)
        scores = null_model_scores(random_fps, cfg, n_samples=n_pairs, seed=seed, shuffle=False)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, scores)

    gpd_p = _gpd_tail_estimate(scores, cfg.proximity_threshold)
    if gpd_p is not None:
        print(f"  random-molecule null (GPD tail): gamma={gpd_p:.3e}")
        return gpd_p
    n_hits = int(np.sum(scores <= cfg.proximity_threshold))
    gamma = (n_hits + 0.5) / (n_pairs + 1) if n_hits == 0 else n_hits / n_pairs
    print(f"  random-molecule null (empirical): gamma={gamma:.3e} ({n_hits} hits / {n_pairs})")
    return gamma
