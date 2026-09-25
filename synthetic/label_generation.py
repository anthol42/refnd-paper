"""Mechanistic label model for the synthetic peptide forest.

The observed label of a peptide is

    y = signal(sequence) + N(0, sigma^2)

`signal` is a non-linear function of physicochemical properties computed from
the sequence alone with modlAMP (Müller et al. 2017).
"""

from __future__ import annotations

import numpy as np
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor


# ─────────────────────────── physicochemical properties ─────────────────────────

def properties(sequences: list[str]) -> dict[str, np.ndarray]:
    """modlAMP descriptors used by `signal`, one array per property.

    - charge: net charge at pH 7 (Henderson–Hasselbalch, free C-terminus).
    - hydrophobic_moment: max Eisenberg hydrophobic moment over 11-residue
      windows at 100° per residue (alpha-helix periodicity).
    - boman_index: Boman 2003 protein-binding potential (higher = binds more).
    """
    global_descriptor = GlobalDescriptor(sequences)
    global_descriptor.calculate_charge(ph=7.0, amide=False)
    charge = global_descriptor.descriptor[:, 0]
    global_descriptor.boman_index()
    boman = global_descriptor.descriptor[:, 0]

    moment_descriptor = PeptideDescriptor(sequences, "eisenberg")
    moment_descriptor.calculate_moment(window=11, angle=100, modality="max")
    moment = moment_descriptor.descriptor[:, 0]

    return {"charge": charge, "hydrophobic_moment": moment,
            "boman_index": boman}


# ──────────────────────────────── signal function ─────────────────────────────

# Reference scale of each property, measured on 5,000 PeptideAtlas-profile seeds
# (lengths 20-50). Fixed constants, so the annotated and production datasets
# share exactly the same `signal`. log is applied to the hydrophobic moment
# first: it is right-skewed (skew 0.48) and would otherwise skew `signal`.
PROPERTY_SCALE = {                  # name: (mean, std)
    "charge": (-1.655, 2.369),
    "log_hydrophobic_moment": (-0.937, 0.291),
    "boman_index": (1.648, 0.992),
}
SIGNAL_SCALE = (0.011, 1.461)       # (mean, std) of the raw combination, same seeds
SATURATION = 0.7                    # softness of the saturation in `saturate`


def standardized_properties(sequences: list[str]) -> dict[str, np.ndarray]:
    """`properties`, log-transforming the moment, z-scored with PROPERTY_SCALE."""
    props = properties(sequences)
    props["log_hydrophobic_moment"] = np.log(props.pop("hydrophobic_moment"))
    return {name: (props[name] - mean) / std
            for name, (mean, std) in PROPERTY_SCALE.items()}


def saturate(values: np.ndarray) -> np.ndarray:
    """Soft saturation: linear near 0, bounded at +-1/SATURATION in the tails."""
    return np.tanh(SATURATION * values) / SATURATION


def signal(sequences: list[str]) -> np.ndarray:
    """Mechanistic activity, standardized to ~N(0, 1) on atlas-profile seeds.

    Non-linear in composition, loosely modelled on membrane-active peptides:
    - saturating main effects of an amphipathic helix (hydrophobic moment) and
      positive charge;
    - interactions: a strong moment helps more when the peptide is also cationic,
      and less when it is already very protein-binding (Boman index);
    - a non-monotonic response to the Boman index (an optimum, not "more is
      better").
    Each term is roughly symmetric, so their sum is close to Gaussian (skew 0.13,
    excess kurtosis -0.09); a linear model on the three properties explains only
    ~62% of its variance.
    """
    z = standardized_properties(sequences)
    moment = saturate(z["log_hydrophobic_moment"])
    charge = saturate(z["charge"])
    boman = z["boman_index"]
    raw = (moment + 0.8 * charge
           + moment * charge
           - moment * saturate(boman)
           + 0.8 * np.sin(1.5 * boman))
    mean, std = SIGNAL_SCALE
    return (raw - mean) / std


# ────────────────────────────────── observed label ────────────────────────────

def meta_label(sequences: list[str], sigma: float, seed: int = 0) -> np.ndarray:
    """Observed label = signal + Gaussian noise.

    Args:
        sequences: Peptide sequences.
        sigma: Standard deviation of the per-peptide Gaussian noise.
        seed: Seeds the noise.
    """
    noise = np.random.default_rng(seed).normal(0.0, sigma, size=len(sequences))
    return signal(sequences) + noise
