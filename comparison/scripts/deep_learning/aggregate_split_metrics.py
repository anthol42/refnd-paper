#!/usr/bin/env python3
"""Merge per-(method,dataset) split-cost records into results/split_metrics.csv.

Each split script writes results/split_metrics/{method}/{dataset}[_seed{s}].json
via split_utils.track_split. Build-once methods (relag/hestia/mmseqs/random) time
all 10 seeds in one process → one file per dataset. DataSAIL splits one seed per
process → 10 files per dataset; we sum their seconds (total compute to produce the
10 splits) and take the max peak RSS.
"""
import os
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(Path(os.environ.get("RELAG_EXP_BASE", str(Path(__file__).resolve().parent.parent))) / "results"))
    args = ap.parse_args()

    root = Path(args.results_dir) / "split_metrics"
    agg = defaultdict(lambda: {"seconds": 0.0, "peak_rss_mb": 0.0, "n": 0})
    for f in sorted(root.glob("*/*.json")):
        if f.name.endswith("_communities.json"):
            continue  # community stats, handled separately below
        rec = json.loads(f.read_text())
        key = (rec["method"], rec["dataset"])
        agg[key]["seconds"] += rec["seconds"]
        agg[key]["peak_rss_mb"] = max(agg[key]["peak_rss_mb"], rec["peak_rss_mb"])
        agg[key]["n"] += 1

    out = Path(args.results_dir) / "split_metrics.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "dataset", "total_seconds", "peak_rss_mb", "n_records"])
        for (method, dataset), v in sorted(agg.items()):
            w.writerow([method, dataset, round(v["seconds"], 2),
                        round(v["peak_rss_mb"], 1), v["n"]])
    print(f"Wrote {out} ({len(agg)} method×dataset rows)")

    # Community/cluster counts (relag, and any other method that writes them).
    comm_rows = []
    for f in sorted(root.glob("*/*_communities.json")):
        s = json.loads(f.read_text())
        per = s.get("per_seed", {})
        n_seeds = len(per) or 1
        mean_train = sum(v["train_communities"] for v in per.values()) / n_seeds if per else None
        mean_test = sum(v["test_communities"] for v in per.values()) / n_seeds if per else None
        comm_rows.append([
            s["method"], s["dataset"], s["n_nodes"], s["n_communities"],
            s.get("largest_community_frac"), s.get("n_components"),
            s.get("largest_component_frac"), s.get("singleton_communities"),
            round(mean_train, 1) if mean_train is not None else None,
            round(mean_test, 1) if mean_test is not None else None,
        ])
    if comm_rows:
        cout = Path(args.results_dir) / "community_stats.csv"
        with open(cout, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["method", "dataset", "n_nodes", "n_communities",
                        "largest_community_frac", "n_components",
                        "largest_component_frac", "singleton_communities",
                        "mean_train_communities", "mean_test_communities"])
            w.writerows(comm_rows)
        print(f"Wrote {cout} ({len(comm_rows)} rows)")


if __name__ == "__main__":
    main()
