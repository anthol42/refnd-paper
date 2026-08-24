#!/usr/bin/env python3
"""Per-dataset split-quality figures: distribution of each test sample's max
identity to any train sample, overlaid across splitting methods.

A leakage-free split pushes mass to the left (low max identity); a leaky split
(e.g. random) piles mass near 1.0 (near-duplicates across the train/test boundary).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import MOLECULE_DATASETS, DNA_DATASETS  # noqa: E402

PROTEIN_DATASETS = ["dbaasp_amp"]

BASE = Path(__file__).resolve().parent.parent  # comparison/ (was /scratch/jacobc/refnd_exp)
LEAK = BASE / "results" / "leakage"
FIGDIR = BASE / "results" / "figures_leakage"
THRESHOLD = 0.40  # leakage cutoff (Tanimoto for molecules, identity for DNA)

METHOD_COLOR = {
    "refnd": "#2a7fb8", "random": "#7f7f7f", "hestia": "#d1495b",
    "datasail": "#66a61e", "mmseqs2": "#e6ab02",
}
ORDER = ["refnd", "random", "hestia", "datasail", "mmseqs2"]


def plot_dataset(dtype, dataset, xlabel, threshold=THRESHOLD):
    ddir = LEAK / dtype / dataset
    files = {f.stem.split("_seed")[0]: f for f in sorted(ddir.glob("*_seed1.npy"))} if ddir.exists() else {}
    methods = [m for m in ORDER if m in files]
    if not methods:
        print(f"  skip {dataset}: no leakage arrays")
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bins = np.linspace(0, 1, 41)
    for m in methods:
        v = np.load(files[m])
        frac = float((v > threshold).mean())
        ax.hist(v, bins=bins, density=True, histtype="step", linewidth=2,
                color=METHOD_COLOR.get(m, "#444"),
                label=f"{m}  (mean={v.mean():.2f}, >{threshold:g}={frac:.0%})")

    ax.axvline(threshold, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(threshold + 0.01, ax.get_ylim()[1] * 0.95, f"leakage threshold {threshold:g}",
            fontsize=8, va="top")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, title="max test→train identity")
    ax.set_title(f"{dtype} / {dataset} — split quality (test→train max identity)")
    fig.tight_layout()

    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / f"{dtype}_{dataset}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    for ds in MOLECULE_DATASETS:
        plot_dataset("molecule", ds, "max Tanimoto to nearest train molecule")
    for ds in PROTEIN_DATASETS:
        # Peptide benchmark threshold is 50% identity (vs 40% for mol/DNA).
        plot_dataset("protein", ds, "max global identity to nearest train peptide", threshold=0.50)
    for ds in DNA_DATASETS:
        plot_dataset("dna", ds, "max identity to nearest train sequence")
    print(f"\nFigures in {FIGDIR}")


if __name__ == "__main__":
    main()
