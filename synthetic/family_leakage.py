"""Family-level leakage of each split method on the synthetic forest.

Complements the nearest-neighbour leakage of `synthetic.compare_splits` with a
ground-truth measure: the fraction of train samples whose TRUE family also has
at least one member in test (and the same fraction on the test side). A split
that keeps families whole scores 0; a random split scores close to 1.

compare_splits does not store split indices, so this script regenerates the
first N_REPEATS annotated datasets (same generator seeds) and re-splits them
with the same methods and settings. Random and oracle splits are identical to
compare_splits'; relag, Hestia and DataSAIL are fresh splits with identical
settings.

Usage:
    uv run python -m synthetic.family_leakage
"""

from __future__ import annotations

import dataclasses
import json
import traceback
from pathlib import Path

import numpy as np
from rich import print

from src.cache import CacheStore
from src.datasets import DATASETS, load_dataset
from src.metrics import null_model
from synthetic.compare_splits import (DATASET_KEY, METHODS, N_NULL_PAIRS, SIGMA,
                                      SPLIT_METHODS, TAU, build_context,
                                      dataset_seeds)
from synthetic.dataset import SyntheticConfig, build_dataset
from synthetic.generation import atlas_profile

N_REPEATS = 10
RESULTS_PATH = (Path(__file__).parent.parent / "results" / "synthetic"
                / f"family_leakage_tau{TAU:g}.json")


def shared_family_fractions(families: np.ndarray, train_idx: list[int],
                            test_idx: list[int]) -> dict:
    """Fraction of train (and test) samples whose family appears on both sides."""
    train_families = families[train_idx]
    test_families = families[test_idx]
    shared = np.intersect1d(train_families, test_families)
    return {
        "train_in_shared_family[%]": 100.0 * float(np.isin(train_families, shared).mean()),
        "test_in_shared_family[%]": 100.0 * float(np.isin(test_families, shared).mean()),
        "n_shared_families": int(len(shared)),
    }


def main() -> None:
    cache = CacheStore()
    config = SyntheticConfig(sigma=SIGMA)
    profile = atlas_profile(cache, len_range=config.len_range)

    print("[bold][orange2]=== Family-level leakage on the synthetic forest ===[/][/]")
    atlas_sequences, _ = load_dataset(DATASET_KEY, cache)
    null_cfg = dataclasses.replace(DATASETS[DATASET_KEY], proximity_threshold=TAU)
    gamma = null_model(atlas_sequences, null_cfg, n_samples=N_NULL_PAIRS)
    print(f"  gamma(tau={TAU}) = {gamma:.3e}")

    records: list[dict] = []
    for repeat in range(N_REPEATS):
        annotated_seed, _ = dataset_seeds(repeat)
        annotated = build_dataset(config, seed=annotated_seed, profile=profile)
        families = np.asarray(annotated.families)
        print(f"[bold]  -- repeat {repeat} (dataset {annotated_seed}) --[/]")
        for method in METHODS:
            base = {"method": method, "repeat": repeat, "annotated_seed": annotated_seed}
            try:
                context = build_context(method, annotated.sequences, families, gamma)
                train_idx, test_idx = SPLIT_METHODS[method](len(annotated), repeat, **context)
            except Exception as error:
                traceback.print_exc()
                records.append({**base, "error": f"{type(error).__name__}: {error}"})
                continue
            stats = shared_family_fractions(families, train_idx, test_idx)
            records.append({**base, **stats, "n_train": len(train_idx),
                            "n_test": len(test_idx), "error": None})
            print(f"    {method:<14} train in shared family="
                  f"{stats['train_in_shared_family[%]']:5.1f}%  "
                  f"test={stats['test_in_shared_family[%]']:5.1f}%")

        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps({
            "settings": {"tau": TAU, "n_repeats": N_REPEATS, "methods": METHODS,
                         "n_peptides": config.n_peptides, "n_families": config.n_families},
            "records": records,
        }))
    print(f"  Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
