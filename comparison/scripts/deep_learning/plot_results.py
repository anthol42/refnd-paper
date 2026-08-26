#!/usr/bin/env python3
"""Per-dataset overfitting figures: test vs train performance across methods.

For each dataset, one PNG with two panels (linear + MLP heads). Each panel shows,
per splitting method, grouped test/train bars (mean +/- std over seeds); the visible
gap between them is the overfitting. A larger gap = the split exposes more overfitting.
"""
import os
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import DATASET_CONFIG, PROTEIN_DATASETS, MOLECULE_DATASETS, DNA_DATASETS  # noqa: E402

BASE = Path(os.environ.get("REFND_EXP_BASE", str(Path(__file__).resolve().parent.parent)))
RESULTS = BASE / "results"
FIGDIR = RESULTS / "figures"
HEADS = ["linear", "mlp"]

# Consistent colors per method
METHOD_COLOR = {
    "refnd": "#2a7fb8", "random": "#7f7f7f", "hestia": "#d1495b",
    "datasail": "#66a61e", "mmseqs2": "#e6ab02",
}


def metric_label(dataset):
    return "PCC" if DATASET_CONFIG[dataset]["metric"] == "pcc" else "AUROC"


def plot_dataset(task_type, dataset):
    summ_path = RESULTS / task_type / dataset / "summary.json"
    if not summ_path.exists():
        print(f"  skip {dataset}: no summary.json")
        return
    summary = json.load(open(summ_path))
    methods = [m for m in summary if summary[m][HEADS[0]]["test"]["mean"] is not None]
    if not methods:
        print(f"  skip {dataset}: no results")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    x = np.arange(len(methods))
    w = 0.38

    for ax, head in zip(axes, HEADS):
        test_m = [summary[m][head]["test"]["mean"] for m in methods]
        test_s = [summary[m][head]["test"]["std"] for m in methods]
        train_m = [summary[m][head]["train"]["mean"] for m in methods]
        train_s = [summary[m][head]["train"]["std"] for m in methods]
        colors = [METHOD_COLOR.get(m, "#444") for m in methods]

        ax.bar(x - w / 2, test_m, w, yerr=test_s, capsize=3, color=colors,
               edgecolor="black", linewidth=0.6, label="test")
        ax.bar(x + w / 2, train_m, w, yerr=train_s, capsize=3, color=colors,
               alpha=0.45, hatch="//", edgecolor="black", linewidth=0.6, label="train")

        # Annotate the overfitting gap above each method
        for xi, m in zip(x, methods):
            gap = summary[m][head]["gap"]["mean"]
            top = max(summary[m][head]["test"]["mean"], summary[m][head]["train"]["mean"])
            ax.text(xi, top + 0.02, f"Δ{gap:.2f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

        ax.set_title(f"{head} head")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.6)

    metric = metric_label(dataset)
    axes[0].set_ylabel(metric)
    # Single legend (test vs train) using neutral swatches
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#888", edgecolor="black", label="test"),
               Patch(facecolor="#888", edgecolor="black", alpha=0.45, hatch="//", label="train")]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False)
    fig.suptitle(f"{task_type} / {dataset}  —  overfitting (Δ = train − test {metric})",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / f"{task_type}_{dataset}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    for ds in MOLECULE_DATASETS:
        plot_dataset("molecule", ds)
    for ds in PROTEIN_DATASETS:
        plot_dataset("protein", ds)
    for ds in DNA_DATASETS:
        plot_dataset("dna", ds)
    print(f"\nFigures in {FIGDIR}")


if __name__ == "__main__":
    main()
