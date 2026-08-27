"""Data-starvation experiment on enzyme_topt (PCC, enzyme optimal temperature from sequence).

Same protocol as dbaasp.py: community-based train/val/test split, then downsample
train two ways (whole communities vs. equal-count random) across drop fractions
and train an MLP head, scoring the fixed test set. See src/data_starvation.py.

Usage:
    uv run python -m data_starvation.enzyme_topt
"""

from pathlib import Path
from refnd import LeidenObjective
from src.data_starvation import run_starvation_experiment

THRESHOLD = 0.3                 # distance = 1 - similarity (0.7 identity/Tanimoto)
GAMMA = 1  # null-model P(random pair within threshold)
OUT_PATH = Path("results/data_starvation/enzyme_topt.json")


def main() -> None:
    run_starvation_experiment("enzyme_topt", THRESHOLD, GAMMA, OUT_PATH, split_seed=7, objective=LeidenObjective.Modularity)


if __name__ == "__main__":
    main()
