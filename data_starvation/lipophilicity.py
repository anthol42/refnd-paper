"""Data-starvation experiment on lipophilicity (PCC, lipophilicity from molecule).

Same protocol as dbaasp.py: community-based train/val/test split, then downsample
train two ways (whole communities vs. equal-count random) across drop fractions
and train an MLP head, scoring the fixed test set. See src/data_starvation.py.

Usage:
    uv run python -m data_starvation.lipophilicity
"""

from pathlib import Path

from refnd.core import LeidenObjective
from src.data_starvation import run_starvation_experiment

THRESHOLD = 0.4
GAMMA = 1.0   # Modularity resolution parameter, not a null-model probability
OUT_PATH = Path("results/data_starvation/lipophilicity.json")


def main() -> None:
    run_starvation_experiment("lipophilicity", THRESHOLD, GAMMA, OUT_PATH, split_seed=7,
                               objective=LeidenObjective.Modularity, singletons_are_com=True)


if __name__ == "__main__":
    main()
