"""Build a labelled synthetic peptide dataset from the forest generator.

Two datasets drawn with different `seed`s are independent samples of the *same*
generative process: one serves as the annotated dataset (to be split into
train/val/test), the other as the production dataset (entirely unseen families).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.cache import CacheStore
from synthetic.generation import AtlasProfile, Forest, atlas_profile, grow_forest
from synthetic.label_generation import meta_label


@dataclass
class SyntheticConfig:
    """Generator and label settings."""
    n_peptides: int = 15_000        # dataset size (fixed)
    n_families: int = 1000          # number of seeds == number of families (fixed)
    sigma: float = 0.05             # per-peptide Gaussian noise
    # forest shape
    branching: float = 1.6
    edits_per_branch: float = 3.0
    max_depth_edits: int = 10
    temperature: float = 1.0
    len_range: tuple[int, int] = (20, 50)


@dataclass
class SyntheticDataset:
    sequences: list[str]
    families: np.ndarray
    labels: np.ndarray
    forest: Forest = field(repr=False)

    def __len__(self) -> int:
        return len(self.sequences)


def build_dataset(config: SyntheticConfig, seed: int,
                  profile: AtlasProfile | None = None,
                  cache: CacheStore | None = None) -> SyntheticDataset:
    """Grow a forest and label it. `seed` selects the independent sample."""
    if profile is None:
        profile = atlas_profile(cache or CacheStore(), len_range=config.len_range)

    forest = grow_forest(profile, config.n_families, config.n_peptides,
                         branching=config.branching,
                         edits_per_branch=config.edits_per_branch,
                         max_depth_edits=config.max_depth_edits,
                         temperature=config.temperature, seed=seed)
    sequences, families = forest.sequences, forest.families
    labels = meta_label(sequences, sigma=config.sigma, seed=seed)
    return SyntheticDataset(sequences, families, labels, forest)


def add_config_arguments(parser) -> None:
    """Attach the SyntheticConfig knobs to an argparse parser."""
    defaults = SyntheticConfig()
    parser.add_argument("--sigma", type=float, default=defaults.sigma,
                        help="std of the per-peptide Gaussian noise")
    parser.add_argument("--branching", type=float, default=defaults.branching)
    parser.add_argument("--edits-per-branch", type=float, default=defaults.edits_per_branch)
    parser.add_argument("--max-depth-edits", type=int, default=defaults.max_depth_edits)
    parser.add_argument("--temperature", type=float, default=defaults.temperature)


def config_from_args(args) -> SyntheticConfig:
    return SyntheticConfig(
        sigma=args.sigma,
        branching=args.branching, edits_per_branch=args.edits_per_branch,
        max_depth_edits=args.max_depth_edits, temperature=args.temperature,
    )
