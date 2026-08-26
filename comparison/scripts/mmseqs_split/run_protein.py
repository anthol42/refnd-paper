#!/usr/bin/env python3
"""MMseqs2 cluster-based splits for DBAASP protein dataset."""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from split_utils import save_split, split_train_val, track_split, SEEDS, PROTEIN_DATASETS

IDENTITY = 0.40  # 40% identity (mid of Hestia's 0.3-0.5 protein range)
TEST_RATIO = 0.20
VAL_RATIO = 0.10


def write_fasta(sequences: list, ids: list, path: str):
    with open(path, "w") as f:
        for i, seq in zip(ids, sequences):
            f.write(f">{i}\n{seq}\n")


def run_mmseqs(fasta_path: str, tmp_dir: str, identity: float, threads: int) -> dict:
    """Run mmseqs easy-cluster; returns {member_id: cluster_rep_id}."""
    prefix = os.path.join(tmp_dir, "clust")
    mmseqs_tmp = os.path.join(tmp_dir, "mmseqs_tmp")
    os.makedirs(mmseqs_tmp, exist_ok=True)

    subprocess.run([
        "mmseqs", "easy-cluster",
        fasta_path, prefix, mmseqs_tmp,
        "--min-seq-id", str(identity),
        "--threads", str(threads),
        "-c", "0.8",
        "--cov-mode", "0",
    ], check=True, capture_output=False)

    tsv = prefix + "_cluster.tsv"
    member_to_rep = {}
    with open(tsv) as f:
        for line in f:
            rep, member = line.strip().split("\t")
            member_to_rep[member] = rep
    return member_to_rep


def clusters_from_mmseqs(member_to_rep: dict, ids: list) -> list:
    """Convert rep-dict to integer cluster labels aligned to ids list."""
    reps = sorted(set(member_to_rep.values()))
    rep_to_int = {r: i for i, r in enumerate(reps)}
    return [rep_to_int[member_to_rep[str(i)]] for i in ids]


def greedy_test_split(clusters: list, n: int, test_ratio: float, seed: int) -> tuple:
    import numpy as np
    rng = np.random.default_rng(seed)

    from collections import defaultdict
    cluster_members = defaultdict(list)
    for idx, c in enumerate(clusters):
        cluster_members[c].append(idx)

    cluster_ids = list(cluster_members.keys())
    rng.shuffle(cluster_ids)

    target_test = int(n * test_ratio)
    test_idx, train_val_idx = [], []
    for cid in cluster_ids:
        members = cluster_members[cid]
        if len(test_idx) < target_test:
            test_idx.extend(members)
        else:
            train_val_idx.extend(members)

    return train_val_idx, test_idx


def run(splits_dir: str, dataset: str, seed: int, threads: int):
    import pandas as pd
    import numpy as np

    df = pd.read_parquet(Path(splits_dir).parent / "data" / "protein" / f"{dataset}.parquet")
    sequences = df["sequence"].tolist()
    ids = list(range(len(sequences)))

    with tempfile.TemporaryDirectory(dir=os.environ.get("REFND_TMP", str(Path(__file__).resolve().parent.parent / "tmp"))) as tmp_dir:
        fasta_path = os.path.join(tmp_dir, "seqs.fasta")
        write_fasta(sequences, ids, fasta_path)
        member_to_rep = run_mmseqs(fasta_path, tmp_dir, IDENTITY, threads)

    clusters = clusters_from_mmseqs(member_to_rep, ids)
    train_val_idx, test_idx = greedy_test_split(clusters, len(sequences), TEST_RATIO, seed)
    train_idx, val_idx = split_train_val(train_val_idx, VAL_RATIO, seed)

    save_split(splits_dir, "mmseqs2", dataset, seed, train_idx, val_idx, test_idx)
    print(f"[seed={seed}] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default=str(Path(os.environ.get("REFND_EXP_BASE", str(Path(__file__).resolve().parent.parent))) / "splits"))
    parser.add_argument("--dataset", choices=PROTEIN_DATASETS, default="dbaasp_amp")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(os.environ.get("REFND_TMP", str(Path(__file__).resolve().parent.parent / "tmp")), exist_ok=True)
    with track_split("mmseqs2", args.dataset, args.splits_dir):
        for seed in args.seeds:
            run(args.splits_dir, args.dataset, seed, args.threads)


if __name__ == "__main__":
    main()
