#!/usr/bin/env python3
"""Hestia DNA threshold sweep (deeppromoter): produce a split at each identity
threshold and save the indices. Runs in the hestia venv. Leakage is measured
separately (refnd venv) so it's comparable to the refnd sweep.

Hestia's threshold is a plain similarity (no 1-T); a partition at threshold t means
no train/test pair exceeds t similarity."""
import os
import json
from pathlib import Path

import pandas as pd
from hestia.dataset_generator import HestiaGenerator, SimArguments

BASE = Path(os.environ.get("REFND_EXP_BASE", str(Path(__file__).resolve().parent.parent)))
OUT = BASE / "dna_thr_test"
OUT.mkdir(exist_ok=True)
SEED = 1

df = pd.read_csv(BASE / "data" / "dna" / "deeppromoter" / "pooled.csv").reset_index(drop=True)
df["idx"] = df.index
print(f"deeppromoter: {len(df)} sequences")

for T in [0.40, 0.60, 0.70, 0.80, 0.90]:
    gen = HestiaGenerator(df, verbose=False)
    gen.calculate_partitions(
        sim_args=SimArguments(data_type="dna sequence", field_name="sequence", min_threshold=T),
        test_size=0.20, valid_size=0.10, random_state=SEED, verbose=0,
    )
    pd_dict = gen.get_partitions(return_dict=True)

    def is_num(k):
        try:
            float(k); return True
        except (TypeError, ValueError):
            return False

    num_keys = [k for k in pd_dict.keys() if is_num(k)]  # exclude hestia's 'random' key
    key = min(num_keys, key=lambda k: abs(float(k) - T))
    parts = pd_dict[key]
    train = [int(x) for x in parts["train"]]
    test = [int(x) for x in parts["test"]]
    json.dump({"threshold": T, "key": float(key), "train": train, "test": test},
              open(OUT / f"hestia_T{T}.json", "w"))
    print(f"T={T} (key={float(key):.2f}): train={len(train)} test={len(test)}")
