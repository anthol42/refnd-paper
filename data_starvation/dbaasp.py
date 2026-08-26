"""Data-starvation experiment on dbaasp (PCC, log MIC from sequence).

Splits the dataset along communities into train/val/test, then downsamples
the train set at several drop fractions using two strategies: dropping whole
communities at random, vs. dropping the same number of samples uniformly at
random. Trains an MLP head on ESM-C embeddings (early stopping on the fixed
val set) and evaluates PCC on the fixed test set, for each (drop_frac,
method, repeat).

Usage:
    uv run python -m data_starvation.dbaasp
"""

import json
from pathlib import Path

import numpy as np
from rich import print

from src.cache import CacheStore
from src.datasets import DATASETS, load_dataset
from src.embeddings import compute_embeddings
from src.data_starvation import build_community_split, run_starvation_repeat

DROP_FRACTIONS = [0.0, 0.05, 0.10, 0.20, 0.40, 0.60]
N_REPEATS = 30
THRESHOLD = 0.3
GAMMA = 2.106840413550154e-11 # Computed from null model at threshold 0.3
OUT_PATH = Path("results/data_starvation/dbaasp.json")


def main() -> None:
    cfg = DATASETS["dbaasp"]
    cache = CacheStore()

    print("[bold][orange2]=== Data Starvation: dbaasp ===[/][/]")

    print("Loading dataset...")
    data, labels = load_dataset("dbaasp", cache)
    print(f"  {len(data):,} samples")

    print("Loading/computing embeddings...")
    embs = compute_embeddings("dbaasp", data, cfg, cache)

    print("Building community-based train/val/test split...")
    split = build_community_split(data, THRESHOLD, GAMMA, cfg, seed=7)
    communities = split["communities"]
    train_idx, val_idx, test_idx = split["train_idx"], split["val_idx"], split["test_idx"]
    print(f"  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    results: dict = {
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "drop_fractions": DROP_FRACTIONS,
        "n_repeats": N_REPEATS,
        "gamma": GAMMA,
        "threshold": THRESHOLD,
        "by_drop_fraction": {},
    }

    for drop_frac in DROP_FRACTIONS:
        print(f"\n[green]-- drop_frac={drop_frac:.0%} --[/]")
        community_scores, random_scores, n_dropped_list = [], [], []
        n_eff_community_list, n_eff_random_list = [], []
        for seed in range(N_REPEATS):
            r = run_starvation_repeat(
                embs, labels, cfg.metric, communities,
                train_idx, val_idx, test_idx, drop_frac, seed,
            )
            community_scores.append(r["community_score"])
            random_scores.append(r["random_score"])
            n_dropped_list.append(r["n_dropped"])
            n_eff_community_list.append(r["n_eff_community"])
            n_eff_random_list.append(r["n_eff_random"])
            print(f"  seed {seed}: dropped={r['n_dropped']}  "
                  f"community_pcc={r['community_score']:.4f} (n_eff={r['n_eff_community']})  "
                  f"random_pcc={r['random_score']:.4f} (n_eff={r['n_eff_random']})")

        community_arr = np.array(community_scores)
        random_arr = np.array(random_scores)
        results["by_drop_fraction"][f"{drop_frac}"] = {
            "n_dropped_mean": float(np.mean(n_dropped_list)),
            "n_eff_community_mean": float(np.mean(n_eff_community_list)),
            "n_eff_community_std": float(np.std(n_eff_community_list)),
            "n_eff_random_mean": float(np.mean(n_eff_random_list)),
            "n_eff_random_std": float(np.std(n_eff_random_list)),
            "community_score_mean": float(community_arr.mean()),
            "community_score_std": float(community_arr.std()),
            "random_score_mean": float(random_arr.mean()),
            "random_score_std": float(random_arr.std()),
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[bold]Results saved to {OUT_PATH}[/]")


if __name__ == "__main__":
    main()
