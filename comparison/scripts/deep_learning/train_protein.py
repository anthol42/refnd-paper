#!/usr/bin/env python3
"""Train Linear + MLP heads on DBAASP protein embeddings."""
import os
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    DATASET_CONFIG, PROTEIN_DATASETS, load_split,
    train_and_evaluate, save_seed_result,
)

BASE = Path(os.environ.get("RELAG_EXP_BASE", str(Path(__file__).resolve().parent.parent)))
PROTEIN_METHODS = ["refnd", "random", "mmseqs2", "hestia", "datasail"]
# Regression target column per protein dataset.
TARGET_COL = {"dbaasp_amp": "log_mic", "enzyme_topt": "label"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=PROTEIN_METHODS, required=True)
    parser.add_argument("--dataset", choices=PROTEIN_DATASETS, default="dbaasp_amp")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--splits-dir",  default=str(BASE / "splits"))
    parser.add_argument("--results-dir", default=str(BASE / "results"))
    parser.add_argument("--embeddings-dir", default=str(BASE / "embeddings"))
    parser.add_argument("--data-dir", default=str(BASE / "data"))
    args = parser.parse_args()

    import torch
    import pandas as pd
    import numpy as np

    cfg = DATASET_CONFIG[args.dataset]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load embeddings and targets
    X = torch.load(Path(args.embeddings_dir) / cfg["embed"], weights_only=True).numpy()
    df = pd.read_parquet(Path(args.data_dir) / "protein" / f"{args.dataset}.parquet")
    y = df[TARGET_COL[args.dataset]].values.astype(np.float32)

    # Load split (no subsampling: CPM/no-post-filter keeps relag's full train set,
    # so every method trains on its own full natural train set)
    tr_idx, val_idx, te_idx = load_split(args.splits_dir, args.method, args.dataset, args.seed)
    tr_idx = np.asarray(tr_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    te_idx = np.asarray(te_idx, dtype=np.int64)

    # Degenerate split (e.g. hestia giant-component collapse -> empty test): nothing
    # to evaluate. Record null and exit 0 so it aggregates as missing rather than
    # crashing the array task and cancelling finalize.
    if te_idx.size == 0 or tr_idx.size == 0:
        print(f"[{args.dataset}][{args.method}][seed={args.seed}] SKIP degenerate split "
              f"(train={tr_idx.size}, val={val_idx.size}, test={te_idx.size}) -> null result")
        save_seed_result(args.results_dir, "protein", args.dataset, args.method, args.seed,
                         {"linear": None, "linear_train": None, "mlp": None, "mlp_train": None})
        return

    embed_dim = X.shape[1]
    scores = train_and_evaluate(
        X[tr_idx], y[tr_idx],
        X[val_idx], y[val_idx],
        X[te_idx],  y[te_idx],
        task=cfg["task"], embed_dim=embed_dim, seed=args.seed, device=device,
    )
    save_seed_result(args.results_dir, "protein", args.dataset, args.method, args.seed, scores)
    print(f"[{args.method}][seed={args.seed}] linear={scores['linear']:.4f} mlp={scores['mlp']:.4f}")


if __name__ == "__main__":
    main()
