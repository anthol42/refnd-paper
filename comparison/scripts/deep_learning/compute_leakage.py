#!/usr/bin/env python3
"""Per test sample, compute the max identity to any train sample (exact NN).

Identity = 1 - kernel_distance from refnd's exact_nearest_neighbors (brute force,
no threshold), so the full 0..1 distribution is captured. Saves one .npy array of
per-test-sample max identities to results/leakage/{dtype}/{dataset}/{method}_seed{seed}.npy
"""
import os
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE = Path(os.environ.get("REFND_EXP_BASE", str(Path(__file__).resolve().parent.parent)))


def build_fingerprints(smiles_list):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from refnd.utils import BitFingerprint
    gen = rdFingerprintGenerator.GetMorganGenerator(fpSize=2048, radius=2)
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
            continue
        arr = np.zeros(2048, dtype=np.uint8)
        for bit in gen.GetFingerprint(mol).GetOnBits():
            arr[bit] = 1
        fps.append(BitFingerprint.from_np(arr))
    return fps


def load_items(dtype, dataset):
    import pandas as pd
    if dtype == "molecule":
        df = pd.read_parquet(BASE / "data" / "molecule" / f"{dataset}.parquet")
        return build_fingerprints(df["smiles"].tolist())
    if dtype == "protein":
        df = pd.read_parquet(BASE / "data" / "protein" / f"{dataset}.parquet")
        return df["sequence"].tolist()
    df = pd.read_csv(BASE / "data" / "dna" / dataset / "pooled.csv")
    return df["sequence"].tolist()


def nn_call(dtype, queries, references, threads):
    from refnd import KernelVariant, exact_nearest_neighbors
    if dtype == "molecule":
        return exact_nearest_neighbors(
            KernelVariant.TanimotoBit, queries, references, 1,
            threads=threads, progress=False)
    if dtype == "protein":
        # Global alignment identity, matching the Refnd protein split.
        return exact_nearest_neighbors(
            KernelVariant.AlignmentGlobal, queries, references, 1,
            threads=threads, progress=False)
    from refnd.kernels.alignments import LocalIdentityMode, CoverageMode, ScoringMatrix
    return exact_nearest_neighbors(
        KernelVariant.AlignmentLocal, queries, references, 1,
        threads=threads, progress=False,
        identity_mode=LocalIdentityMode.MinSeqLength, cov_mode=CoverageMode.ShorterSeq,
        min_coverage=0.7, matrix=ScoringMatrix.Dnafull)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["molecule", "protein", "dna"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    out = BASE / "results" / "leakage" / args.dtype / args.dataset / f"{args.method}_seed{args.seed}.npy"
    if out.exists():
        print(f"exists, skip: {out}")
        return

    items = load_items(args.dtype, args.dataset)
    split = json.load(open(BASE / "splits" / args.method / args.dataset / f"{args.seed}.json"))

    # Keep only valid items (molecules may have None fingerprints)
    def valid(idx):
        return [i for i in idx if items[i] is not None]
    train_idx, test_idx = valid(split["train"]), valid(split["test"])
    train = [items[i] for i in train_idx]
    test = [items[i] for i in test_idx]
    print(f"[{args.dataset}/{args.method}/seed{args.seed}] test={len(test)} train={len(train)}")

    res = nn_call(args.dtype, test, train, args.threads)
    # res[q] = [(train_idx, distance)]; identity = 1 - distance
    max_ident = np.array([1.0 - r[0][1] for r in res], dtype=np.float32)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, max_ident)
    print(f"  saved {out}  mean_max_ident={max_ident.mean():.3f}  "
          f"frac>0.4={float((max_ident > 0.4).mean()):.3f}")


if __name__ == "__main__":
    main()
