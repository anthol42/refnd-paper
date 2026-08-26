#!/usr/bin/env python3
"""Refnd splits for TDC molecule datasets."""
import os
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from split_utils import (save_split, save_refnd_sizes, save_community_stats,
                         split_train_val, track_split, SEEDS, MOLECULE_DATASETS)

# refnd proximity_threshold is a DISTANCE (= 1 - similarity). To separate pairs
# at 0.40 Tanimoto similarity (matching Hestia / the benchmark), use 1 - 0.40.
THRESHOLD = 0.60
TEST_RATIO = 0.20
VAL_RATIO = 0.10
# Random-atom-molecule null (find_gamma_function defaults): dataset-independent,
# computed once and cached, reused across all molecule datasets.
NULL_N_MOLECULES = 100_000
NULL_N_PAIRS = 2_000_000


def build_fingerprints(smiles_list: list):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from refnd.utils import BitFingerprint
    import numpy as np

    gen = rdFingerprintGenerator.GetMorganGenerator(fpSize=2048, radius=2)
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
        else:
            arr = np.zeros(2048, dtype=np.uint8)
            for bit in gen.GetFingerprint(mol).GetOnBits():
                arr[bit] = 1
            fps.append(BitFingerprint.from_np(arr))
    return fps


def build_graph_and_communities(valid_fps: list, null_cache_path):
    """HNSW → distance graph → CPM communities (all seed-independent)."""
    from refnd.core import (HNSWState, INWeightType, LeidenObjective,
                            find_communities, find_components)
    from refnd import KernelVariant
    from null_model import random_molecule_null_gamma, NullCfg

    cfg = NullCfg(modality=KernelVariant.TanimotoBit, proximity_threshold=THRESHOLD)
    # Random-atom-molecule null (refnd-paper threshold/molecules.py:find_gamma_function):
    # gamma = p0(THRESHOLD) from a synthetic random-molecule pool, dataset-independent
    # and cached, so every molecule dataset shares the same universal null baseline.
    gamma = random_molecule_null_gamma(cfg, n_molecules=NULL_N_MOLECULES,
                                       n_pairs=NULL_N_PAIRS, cache_path=null_cache_path)

    hnsw = HNSWState(KernelVariant.TanimotoBit, valid_fps, proximity_threshold=THRESHOLD)
    hnsw.build(progress=True)
    graph = hnsw.edges().graph(inweight_type=INWeightType.Distance)
    # CPM Leiden with gamma = null-model P(random pair within threshold).
    communities = find_communities(graph, gamma=gamma, objective=LeidenObjective.CPM)
    components = find_components(graph)  # connected components: giant-component diagnostic
    return graph, communities, components


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default=str(Path(os.environ.get("REFND_EXP_BASE", str(Path(__file__).resolve().parent.parent))) / "splits"))
    parser.add_argument("--dataset", choices=MOLECULE_DATASETS, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    import pandas as pd

    df = pd.read_parquet(Path(args.splits_dir).parent / "data" / "molecule" / f"{args.dataset}.parquet")
    smiles = df["smiles"].tolist()
    fps = build_fingerprints(smiles)
    valid_idx = [i for i, fp in enumerate(fps) if fp is not None]
    valid_fps = [fps[i] for i in valid_idx]
    print(f"[{args.dataset}] {len(valid_fps)}/{len(smiles)} valid SMILES")

    from refnd.core import partition
    # Shared, dataset-independent random-molecule null: one cache file for all datasets.
    null_cache_path = (Path(args.splits_dir).parent / "cache" /
                       f"mol_null_random_n{NULL_N_MOLECULES}_pairs{NULL_N_PAIRS}_seed42.npy")
    sizes = {}
    per_seed_local = {}
    with track_split("refnd", args.dataset, args.splits_dir):
        graph, communities, components = build_graph_and_communities(valid_fps, null_cache_path)
        for seed in args.seeds:
            train_val_local, test_local = partition(
                communities, graph, test_ratio=TEST_RATIO, seed=seed, post_filtering=False)
            per_seed_local[seed] = (list(train_val_local), list(test_local))
            train_val_idx = [valid_idx[i] for i in train_val_local]
            test_idx = [valid_idx[i] for i in test_local]
            train_idx, val_idx = split_train_val(train_val_idx, VAL_RATIO, seed)
            save_split(args.splits_dir, "refnd", args.dataset, seed, train_idx, val_idx, test_idx)
            print(f"[{args.dataset}][seed={seed}] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
            sizes[str(seed)] = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}

    save_refnd_sizes(args.splits_dir, args.dataset, sizes)
    save_community_stats(args.splits_dir, "refnd", args.dataset,
                         communities, per_seed_local, components=components)
    print(f"Saved sizes.json for {args.dataset}")


if __name__ == "__main__":
    main()
