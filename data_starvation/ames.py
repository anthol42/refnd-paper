"""Data-starvation experiment on ames (MCC, Ames mutagenicity from molecule).

Same protocol as dbaasp.py: community-based train/val/test split, then downsample
train two ways (whole communities vs. equal-count random) across drop fractions
and train an MLP head, scoring the fixed test set. See src/data_starvation.py.

Usage:
    uv run python -m data_starvation.ames
"""

from pathlib import Path

from refnd.core import LeidenObjective
from src.data_starvation import run_starvation_experiment


THRESHOLD = 0.7
GAMMA = 1.0
OUT_PATH = Path("results/data_starvation/ames.json")


def main() -> None:
    run_starvation_experiment("ames", THRESHOLD, GAMMA, OUT_PATH, split_seed=7,
                               objective=LeidenObjective.Modularity)


if __name__ == "__main__":
    main()
