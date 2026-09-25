#!/usr/bin/env python3
"""REAL DataSAIL (library) splits — cold-cluster single-entity (C1e).

Runs INSIDE the datasail Apptainer container (see datasail.sif), which provides
the `datasail` package + its aligner binaries (ecfp/mash/cd-hit-est/mmseqs).

Unlike the old homemade datasail_split scripts (Tanimoto/edges -> SpectralClustering
-> stratify WITHIN clusters, which leaks because similar items land in both train and
test), C1e assigns WHOLE clusters to a single split, so it is genuinely leakage-aware.

Similarity is DataSAIL-native: ecfp (Morgan/Tanimoto) for molecules, cd-hit-est
(nucleotide clustering) for DNA. (MASH is DataSAIL's other genomic option but is a
whole-genome distance sketch, inappropriate for short FASTA-style sequences.)
Seed control: DataSAIL shuffles its input via the global numpy RNG
(reader/utils.permute -> np.random.permutation), so np.random.seed(seed) yields a
distinct split per seed.
"""
import os
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from split_utils import save_split, track_split  # noqa: E402

SPLITS = [0.7, 0.1, 0.2]
NAMES = ["train", "val", "test"]


def load_entities(splits_dir: str, dtype: str, dataset: str):
    import pandas as pd
    base = Path(splits_dir).parent / "data"
    if dtype == "molecule":
        df = pd.read_parquet(base / "molecule" / f"{dataset}.parquet")
        col = "smiles" if "smiles" in df.columns else "Drug"
        return df[col].tolist()
    if dtype == "protein":
        df = pd.read_parquet(base / "protein" / f"{dataset}.parquet")
        return df["sequence"].tolist()
    df = pd.read_csv(base / "dna" / dataset / "pooled.csv")
    return df["sequence"].tolist()


# DataSAIL-native similarity per data type.
E_TYPE = {"molecule": "M", "protein": "P", "dna": "G"}
E_SIM = {"molecule": "ecfp", "protein": "mmseqs", "dna": "cdhit_est"}


def split_entities(data: dict, e_type: str, e_sim: str, seed: int, max_sec: int,
                   threads: int, epsilon: float, e_clusters: int, cache_dir: Path):
    """DataSAIL C1e split of {id_str: entity}, with the infeasibility escalation.

    DataSAIL's C1e ILP can be infeasible when native clustering yields a
    lopsided cluster distribution. Empirically the effective lever is the
    NUMBER of clusters (finer granularity -> packable), not the size tolerance:
    peptides need e_clusters>=200 (they work then), molecules are fine at 50.
    So escalate e_clusters first (at base epsilon), then fall back to loosening
    epsilon at the finest granularity. Reseeding before each attempt keeps the
    input shuffle identical, so only the clustering config changes.

    Returns (train, val, test) sorted int ids and the (e_clusters, epsilon) used.
    """
    import numpy as np
    from datasail.sail import datasail

    n = len(data)
    ncl_ladder = sorted({c for c in [e_clusters, 200, 500, 1000, 2000] if c <= n})
    attempts = [(c, epsilon) for c in ncl_ladder]
    attempts += [(ncl_ladder[-1], round(min(epsilon + d, 0.7), 3)) for d in (0.2, 0.45, 0.65)]

    for ncl, eps in attempts:
        np.random.seed(seed)  # controls DataSAIL's internal input shuffle
        e_splits, _, _ = datasail(
            techniques=["C1e"], splits=SPLITS, names=NAMES, runs=1,
            solver="SCIP", e_type=e_type, e_data=data, e_sim=e_sim,
            max_sec=max_sec, verbose="E", threads=threads,
            epsilon=eps, e_clusters=ncl,
            cache=True, cache_dir=str(cache_dir),
        )
        if not e_splits.get("C1e"):
            print(f"[seed={seed}] e_clusters={ncl} epsilon={eps}: infeasible, escalating.")
            continue
        mapping = e_splits["C1e"][0]  # {id_str: split_name}
        buckets = {"train": [], "val": [], "test": []}
        for k, v in mapping.items():
            if v in buckets:
                buckets[v].append(int(k))
        tr, va, te = sorted(buckets["train"]), sorted(buckets["val"]), sorted(buckets["test"])
        if min(len(tr), len(va), len(te)) > 0:
            return tr, va, te, (ncl, eps)
        print(f"[seed={seed}] e_clusters={ncl} epsilon={eps}: degenerate, escalating.")

    raise RuntimeError(f"[seed={seed}] no feasible split (tried {attempts}).")


def run(splits_dir: str, dtype: str, dataset: str, seed: int, max_sec: int, threads: int,
        epsilon: float, e_clusters: int):
    split_path = Path(splits_dir) / "datasail" / dataset / f"{seed}.json"
    if split_path.exists():
        print(f"[{dataset}][seed={seed}] already computed, skipping.")
        return

    seqs = load_entities(splits_dir, dtype, dataset)
    e_type = E_TYPE[dtype]
    e_sim = E_SIM[dtype]
    data = {str(i): s for i, s in enumerate(seqs)}
    print(f"[{dataset}][seed={seed}] {len(data)} entities, e_type={e_type} e_sim={e_sim} "
          f"epsilon={epsilon} e_clusters={e_clusters}")

    # Per-dataset clustering cache. NOTE: DataSAIL re-clusters per seed anyway
    # (it hashes the shuffled input), so across seeds this mostly helps the
    # epsilon-escalation retries in split_entities, which reuse the same-seed clustering.
    cache_dir = Path(splits_dir).parent / "cache" / "datasail_cache" / dataset
    cache_dir.mkdir(parents=True, exist_ok=True)

    tr, va, te, used = split_entities(data, e_type, e_sim, seed, max_sec, threads,
                                      epsilon, e_clusters, cache_dir)
    save_split(splits_dir, "datasail", dataset, seed, tr, va, te)
    print(f"[{dataset}][seed={seed}] train={len(tr)} val={len(va)} test={len(te)} "
          f"(e_clusters={used[0]}, epsilon={used[1]})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir", default=str(Path(os.environ.get("REFND_EXP_BASE", str(Path(__file__).resolve().parent.parent))) / "splits"))
    p.add_argument("--dtype", choices=["molecule", "protein", "dna"], required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-sec", type=int, default=1000)
    p.add_argument("--threads", type=int, default=8)
    # DNA needs relaxed size tolerance + more clusters: cd-hit-est floors at 80%
    # identity, so short DNA barely clusters (near-singletons) and the default
    # 5% tolerance ILP is infeasible. Molecules cluster well and use the tighter
    # DataSAIL defaults (epsilon=0.05, e_clusters=50).
    p.add_argument("--epsilon", type=float, default=None)
    p.add_argument("--e-clusters", type=int, default=None)
    args = p.parse_args()
    # Defaults per type. Peptides need e_clusters>=200 to be feasible (mmseqs
    # yields a dominant cluster at 50); molecules are fine at 50; DNA needs more
    # clusters + looser tolerance (cd-hit-est 80% floor -> near-singletons).
    default_eps = {"molecule": 0.05, "protein": 0.05, "dna": 0.25}[args.dtype]
    default_ncl = {"molecule": 50, "protein": 200, "dna": 100}[args.dtype]
    epsilon = args.epsilon if args.epsilon is not None else default_eps
    e_clusters = args.e_clusters if args.e_clusters is not None else default_ncl
    with track_split("datasail", args.dataset, args.splits_dir, seed=args.seed):
        run(args.splits_dir, args.dtype, args.dataset, args.seed, args.max_sec,
            args.threads, epsilon, e_clusters)


if __name__ == "__main__":
    main()
