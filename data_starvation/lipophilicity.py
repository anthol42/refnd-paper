"""Data-starvation experiment on lipophilicity (PCC, lipophilicity from molecule).

Same protocol as dbaasp.py: community-based train/val/test split, then downsample
train two ways (whole communities vs. equal-count random) across drop fractions
and train an MLP head, scoring the fixed test set. See src/data_starvation.py.

Usage:
    uv run python -m data_starvation.lipophilicity
"""

from pathlib import Path

from src.data_starvation import run_starvation_experiment

THRESHOLD = 0.3                 # distance = 1 - similarity (0.7 identity/Tanimoto)
GAMMA = 3.3668e-07   # null-model P(random pair within threshold), @thr 0.3
OUT_PATH = Path("results/data_starvation/lipophilicity.json")


def main() -> None:
    run_starvation_experiment("lipophilicity", THRESHOLD, GAMMA, OUT_PATH, split_seed=7)


if __name__ == "__main__":
    main()
