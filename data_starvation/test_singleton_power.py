"""Singleton vs. non-singleton per-sample information content, across every
data_starvation dataset.

Community-based downsampling (min_community_size=1, the default used
throughout this experiment) never drops a singleton community -- only
random downsampling touches singletons, proportionally to the drop
fraction. So whether community-drop beats or loses to random-drop for a
given dataset should be explained by whether its singleton points or its
non-singleton-cluster points carry more per-sample generalization value.

For each dataset: train an MLP on ONLY the singleton subset of train, and
separately on a size-matched random subsample of the non-singleton subset,
evaluate both on the same fixed test set, across 5 splits x 6 repeats each.
Reports mean/std and a paired t-test per dataset, plus a combined summary
table. See src/data_starvation.py:run_singleton_power_experiment.

Usage:
    uv run python -m data_starvation.test_singleton_power
"""

import json
from pathlib import Path

from refnd.core import LeidenObjective

from src.data_starvation import run_singleton_power_experiment

OUT_DIR = Path("results/data_starvation/singleton_power")
SUMMARY_PATH = OUT_DIR / "summary.json"

# (threshold, gamma, objective) per dataset -- matches each dataset's own
# data_starvation/<name>.py script.
CONFIGS: dict[str, dict] = {
    "dbaasp":           dict(threshold=0.3, gamma=1.0, objective=LeidenObjective.Modularity),
    "lipophilicity":    dict(threshold=0.4, gamma=1.0, objective=LeidenObjective.Modularity),
    "caco2_wang":       dict(threshold=0.6, gamma=1.0, objective=LeidenObjective.Modularity),
    "cyp2c19_veith":    dict(threshold=0.5, gamma=1.0, objective=LeidenObjective.Modularity),
    "ames":             dict(threshold=0.7, gamma=1.0, objective=LeidenObjective.Modularity),
    "pgp_broccatelli":  dict(threshold=0.6, gamma=1.0, objective=LeidenObjective.Modularity),
    "sr_are":           dict(threshold=0.4, gamma=1.0, objective=LeidenObjective.Modularity),
    "enzyme_topt":      dict(threshold=0.8, gamma=1.0, objective=LeidenObjective.Modularity),
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, cfg in CONFIGS.items():
        out_path = OUT_DIR / f"{name}.json"
        result = run_singleton_power_experiment(
            name, cfg["threshold"], cfg["gamma"], out_path,
            split_seed=7, objective=cfg["objective"],
        )
        summary[name] = {
            k: result[k] for k in (
                "threshold", "objective", "metric",
                "n_singleton_train_mean", "n_nonsingleton_train_matched_mean",
                "singleton_score_mean", "singleton_score_std",
                "nonsingleton_score_mean", "nonsingleton_score_std",
                "diff_mean", "t_stat", "p_value", "significant",
            )
        }

    with open(SUMMARY_PATH, "w") as f:
        print(f"Stored at {SUMMARY_PATH}")
        json.dump(summary, f, indent=2)

    print("\n=== Singleton power summary (all datasets) ===")
    print(f"{'dataset':16s} {'singleton':>18s} {'non-singleton':>18s} {'diff':>9s} {'p':>8s}  sig")
    for name, s in summary.items():
        print(f"{name:16s} {s['singleton_score_mean']:.4f}+-{s['singleton_score_std']:.4f}  "
              f"{s['nonsingleton_score_mean']:.4f}+-{s['nonsingleton_score_std']:.4f}  "
              f"{s['diff_mean']:+.4f}  {s['p_value']:.4f}  {'*' if s['significant'] else ''}")
    print(f"\nSummary saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
