"""Data-starvation experiment on dbaasp (PCC, log MIC from sequence).

Splits the dataset along communities into train/val/test, then downsamples
the train set at several drop fractions using two strategies: dropping whole
communities at random, vs. dropping the same number of samples uniformly at
random. Trains an MLP head on ESM-C embeddings (early stopping on the fixed
val set) and evaluates PCC on the fixed test set, for each (drop_frac,
method, repeat). See src/data_starvation.py.

Usage:
    uv run python -m data_starvation.dbaasp
"""

from pathlib import Path

from refnd.core import LeidenObjective
from src.data_starvation import run_starvation_experiment

THRESHOLD = 0.3
GAMMA = 1.0   # Modularity resolution parameter, not a null-model probability
OUT_PATH = Path("results/data_starvation/dbaasp.json")


def main() -> None:
    run_starvation_experiment("dbaasp", THRESHOLD, GAMMA, OUT_PATH, split_seed=7,
                              objective=LeidenObjective.Modularity)


if __name__ == "__main__":
    main()
