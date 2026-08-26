#!/usr/bin/env python3
"""Aggregate per-seed result files into final JSON summaries + Wilcoxon table."""
import os
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

BASE = Path(os.environ.get("REFND_EXP_BASE", str(Path(__file__).resolve().parent.parent)))
RESULTS_DIR = BASE / "results"

PROTEIN_METHODS  = ["refnd", "random", "mmseqs2", "hestia", "datasail"]
MOLECULE_METHODS = ["refnd", "random", "hestia", "datasail"]
DNA_METHODS      = ["refnd", "random", "hestia"]

MOLECULE_DATASETS = [
    "cyp2c19_veith", "caco2_wang", "pgp_broccatelli", "ames", "lipophilicity", "sr_are",
]
DNA_DATASETS = [
    "gue_prom_core_all", "gue_prom_300_all", "gue_emp_h3", "gue_emp_h4", "gue_mouse_0",
    "deeppromoter",
]

HEADS = ["linear", "mlp"]
N_SEEDS = 10


def load_scores(task_type: str, dataset: str, method: str) -> dict:
    """Per head, return test scores, train scores, and per-seed overfitting
    gap (train - test)."""
    scores = {h: {"test": [], "train": [], "gap": []} for h in HEADS}
    method_dir = RESULTS_DIR / task_type / dataset / method
    for seed in range(1, N_SEEDS + 1):
        path = method_dir / f"seed_{seed}.json"
        d = json.load(open(path)) if path.exists() else {}
        for h in HEADS:
            test = d.get(h)
            train = d.get(f"{h}_train")
            scores[h]["test"].append(test)
            scores[h]["train"].append(train)
            scores[h]["gap"].append(
                train - test if (test is not None and train is not None) else None
            )
    return scores


def summarize(scores: list) -> dict:
    valid = [s for s in scores if s is not None]
    if not valid:
        return {"mean": None, "std": None, "n": 0, "values": scores}
    return {
        "mean": float(np.mean(valid)),
        "std":  float(np.std(valid)),
        "n":    len(valid),
        "values": scores,
    }


def wilcoxon_p(refnd_scores: list, other_scores: list) -> float:
    pairs = [(r, o) for r, o in zip(refnd_scores, other_scores)
             if r is not None and o is not None]
    if len(pairs) < 5:
        return None
    r_vals, o_vals = zip(*pairs)
    try:
        return float(wilcoxon(r_vals, o_vals).pvalue)
    except Exception:
        return None


def process_task(task_type: str, datasets: list, methods: list):
    all_wilcoxon = {}

    for dataset in datasets:
        out_path = RESULTS_DIR / task_type / dataset / "summary.json"
        summary = {}

        refnd = load_scores(task_type, dataset, "refnd")

        for method in methods:
            scores = load_scores(task_type, dataset, method)
            summary[method] = {
                h: {k: summarize(scores[h][k]) for k in ("test", "train", "gap")}
                for h in HEADS
            }
            if method != "refnd":
                for h in HEADS:
                    all_wilcoxon[f"{dataset}/{method}/{h}"] = wilcoxon_p(
                        refnd[h]["test"], scores[h]["test"])

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Overfitting-focused table: for each head, show test, train, and the
        # gap (train - test). A larger gap = the split exposes more overfitting.
        print(f"\n=== {task_type}/{dataset} ===")
        for h in HEADS:
            print(f"  [{h}]  {'method':<10} {'test':>8} {'train':>8} {'gap(tr-te)':>11}  n")
            for method in methods:
                s = summary[method][h]
                te, tr, gp = s["test"]["mean"], s["train"]["mean"], s["gap"]["mean"]
                if te is None:
                    print(f"        {method:<10} {'N/A':>8}")
                    continue
                print(f"        {method:<10} {te:8.4f} {tr:8.4f} {gp:11.4f}  {s['test']['n']}")

    return all_wilcoxon


def main():
    wilcoxon_table = {}

    print("\n" + "="*60)
    print("PROTEIN")
    wilcoxon_table.update(process_task("protein", ["dbaasp_amp", "enzyme_topt"], PROTEIN_METHODS))

    print("\n" + "="*60)
    print("MOLECULE")
    wilcoxon_table.update(process_task("molecule", MOLECULE_DATASETS, MOLECULE_METHODS))

    print("\n" + "="*60)
    print("DNA")
    wilcoxon_table.update(process_task("dna", DNA_DATASETS, DNA_METHODS))

    # Save Wilcoxon table
    wil_path = RESULTS_DIR / "wilcoxon_table.json"
    with open(wil_path, "w") as f:
        json.dump(wilcoxon_table, f, indent=2)

    print(f"\nWilcoxon table saved to {wil_path}")
    print("\nWilcoxon p-values (Refnd vs others):")
    for key, p in sorted(wilcoxon_table.items()):
        stars = ""
        if p is not None:
            if p < 0.001: stars = "***"
            elif p < 0.01: stars = "**"
            elif p < 0.05: stars = "*"
        p_str = f"{p:.4f}" if p is not None else "N/A"
        print(f"  {key:<50} p={p_str} {stars}")


if __name__ == "__main__":
    main()
