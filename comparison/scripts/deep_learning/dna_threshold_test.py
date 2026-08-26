#!/usr/bin/env python3
"""DNA threshold test (deeppromoter): does raising the identity threshold above the
~0.6 noise floor let Refnd carve out a test set that is genuinely dissimilar to train?

For each identity threshold T we build the Refnd proximity graph at
proximity_threshold = 1 - T  (Refnd's threshold is a DISTANCE), partition, then
measure the test->train max identity (exact NN). Compared against a random split.
"""
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from refnd import KernelVariant, HNSWState, find_communities, partition, exact_nearest_neighbors
from refnd.kernels.alignments import LocalIdentityMode, CoverageMode, ScoringMatrix

BASE = Path(os.environ.get("REFND_EXP_BASE", str(Path(__file__).resolve().parent.parent)))
SEED = 1
TEST_RATIO = 0.20
THREADS = 16
ALN = dict(identity_mode=LocalIdentityMode.MinSeqLength, cov_mode=CoverageMode.ShorterSeq,
           min_coverage=0.7, matrix=ScoringMatrix.Dnafull)


def max_ident(test_seqs, train_seqs):
    res = exact_nearest_neighbors(KernelVariant.AlignmentLocal, test_seqs, train_seqs, 1,
                                  threads=THREADS, progress=False, **ALN)
    return np.array([1.0 - r[0][1] for r in res], dtype=np.float32)


def main():
    df = pd.read_csv(BASE / "data" / "dna" / "deeppromoter" / "pooled.csv")
    seqs = df["sequence"].tolist()
    n = len(seqs)
    print(f"deeppromoter: {n} sequences\n")

    # Random baseline (threshold-independent)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_test = int(n * TEST_RATIO)
    r_test, r_train = perm[:n_test], perm[n_test:]
    rv = max_ident([seqs[i] for i in r_test], [seqs[i] for i in r_train])
    print(f"{'RANDOM':<10} test={len(r_test)} train={len(r_train)}  "
          f"mean_max_id={rv.mean():.3f}  frac>0.4={(rv>0.4).mean():.0%}\n")

    print(f"{'idn_thr':<8}{'prox=1-T':<9}{'edges':<9}{'n_comm':<8}{'max_comm':<9}"
          f"{'train':<7}{'test':<6}{'mean_maxid':<11}{'>0.4':<6}{'>thr'}")
    for T in [0.40, 0.60, 0.70, 0.80, 0.90]:
        hnsw = HNSWState(KernelVariant.AlignmentLocal, seqs,
                         proximity_threshold=round(1.0 - T, 3), n_threads=THREADS, **ALN)
        hnsw.build(progress=False)
        edges = hnsw.edges()
        g = edges.graph()
        n_edges = len(edges.edges())
        clusters = find_communities(g, gamma=1.0, n_iterations=2)
        try:
            sizes = Counter(list(clusters)); n_comm = len(sizes); max_comm = max(sizes.values())
        except Exception:
            n_comm = max_comm = -1
        tr_idx, te_idx = partition(clusters, g, test_ratio=TEST_RATIO, seed=SEED, post_filtering=True)
        if len(tr_idx) == 0 or len(te_idx) == 0:
            print(f"{T:<8}{1-T:<9}{n_edges:<9}{n_comm:<8}{max_comm:<9}"
                  f"{len(tr_idx):<7}{len(te_idx):<6}DEGENERATE")
            continue
        v = max_ident([seqs[i] for i in te_idx], [seqs[i] for i in tr_idx])
        print(f"{T:<8}{1-T:<9}{n_edges:<9}{n_comm:<8}{max_comm:<9}"
              f"{len(tr_idx):<7}{len(te_idx):<6}{v.mean():<11.3f}{(v>0.4).mean():<6.0%}{(v>T).mean():.0%}")


if __name__ == "__main__":
    main()
