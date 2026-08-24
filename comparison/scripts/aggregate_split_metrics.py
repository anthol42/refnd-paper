#!/usr/bin/env python3
"""Merge per-(method,dataset) split-cost records into results/split_metrics.csv.

Each split script writes results/split_metrics/{method}/{dataset}[_seed{s}].json
via split_utils.track_split. Build-once methods (refnd/hestia/mmseqs/random) time
all 10 seeds in one process → one file per dataset. DataSAIL splits one seed per
process → 10 files per dataset; we sum their seconds (total compute to produce the
10 splits) and take the max peak RSS.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="/scratch/jacobc/refnd_exp/results")
    args = ap.parse_args()

    root = Path(args.results_dir) / "split_metrics"
    agg = defaultdict(lambda: {"seconds": 0.0, "peak_rss_mb": 0.0, "n": 0})
    for f in sorted(root.glob("*/*.json")):
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


if __name__ == "__main__":
    main()
