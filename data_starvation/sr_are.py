"""Data-starvation experiment on sr_are (AUROC, Tox21 SR-ARE stress response from molecule).

AUROC rather than MCC: SR-ARE is heavily class-imbalanced (~16% positive),
which makes MCC collapse to 0 (or go negative) whenever the model leans
toward the majority class -- see DATASETS["sr_are"] in src/datasets.py.

Same protocol as dbaasp.py: community-based train/val/test split, then downsample
train two ways (whole communities vs. equal-count random) across drop fractions
and train an MLP head, scoring the fixed test set. See src/data_starvation.py.

Usage:
    uv run python -m data_starvation.sr_are
"""

from pathlib import Path
from refnd import LeidenObjective
from src.data_starvation import run_starvation_experiment

THRESHOLD = 0.4
GAMMA = 1
OUT_PATH = Path("results/data_starvation/sr_are.json")


def main() -> None:
    run_starvation_experiment("sr_are", THRESHOLD, GAMMA, OUT_PATH, split_seed=7,
                              objective=LeidenObjective.Modularity, singletons_are_com=True)


if __name__ == "__main__":
    main()
