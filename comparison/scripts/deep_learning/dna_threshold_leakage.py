#!/usr/bin/env python3
"""Measure test->train max identity (exact AlignmentLocal NN, refnd) for the Hestia
threshold-sweep splits saved by hestia_threshold_split.py. Runs in the refnd venv so
the identity metric matches the refnd sweep. Also prints a random baseline."""
import os
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from refnd import KernelVariant, exact_nearest_neighbors
from refnd.kernels.alignments import LocalIdentityMode, CoverageMode, ScoringMatrix

BASE = Path(os.environ.get("REFND_EXP_BASE", str(Path(__file__).resolve().parent.parent)))
THREADS = 16
SEED = 1
ALN = dict(identity_mode=LocalIdentityMode.MinSeqLength, cov_mode=CoverageMode.ShorterSeq,
           min_coverage=0.7, matrix=ScoringMatrix.Dnafull)

df = pd.read_csv(BASE / "data" / "dna" / "deeppromoter" / "pooled.csv")
seqs = df["sequence"].tolist()
n = len(seqs)


def maxid(test_idx, train_idx):
    res = exact_nearest_neighbors(KernelVariant.AlignmentLocal,
                                  [seqs[i] for i in test_idx], [seqs[i] for i in train_idx],
                                  1, threads=THREADS, progress=False, **ALN)
    return np.array([1.0 - r[0][1] for r in res], dtype=np.float32)


# Random baseline
rng = np.random.default_rng(SEED)
perm = rng.permutation(n)
n_test = int(n * 0.20)
rv = maxid(perm[:n_test], perm[n_test:])
print(f"{'RANDOM':<10} train={n-n_test} test={n_test}  mean_max_id={rv.mean():.3f}  frac>0.4={(rv>0.4).mean():.0%}\n")

print(f"{'idn_thr':<9}{'train':<8}{'test':<7}{'mean_maxid':<12}{'>0.4':<7}{'>thr'}")
for f in sorted(glob.glob(str(BASE / "dna_thr_test" / "hestia_T*.json"))):
    d = json.load(open(f))
    T = d["threshold"]
    if not d["train"] or not d["test"]:
        print(f"{T:<9}{len(d['train']):<8}{len(d['test']):<7}DEGENERATE")
        continue
    v = maxid(d["test"], d["train"])
    print(f"{T:<9}{len(d['train']):<8}{len(d['test']):<7}{v.mean():<12.3f}{(v>0.4).mean():<7.0%}{(v>T).mean():.0%}")
