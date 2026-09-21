"""Synthetic peptide forest generator.

Generates a forest of evolutionary trees whose *generative process is known*:

  1. Seeds are sampled from the PeptideAtlas length distribution and
     amino-acid background frequencies -- one seed per family.
  2. Each seed is mutated recursively, up to a maximum accumulated edit
     distance from the seed (`--max-depth-edits`), (Galton-Watson branching) with
     BLOSUM62-conditioned substitutions plus insertions/deletions. Every tree
     edge stores the number of edits applied on that branch, so the tree
     carries an additive edit metric.
  3. Each tree *is* a peptide family: no cut is applied, so the forest yields
     exactly two levels -- families and the peptides within them.

Usage:
    uv run python -m synthetic.generation --n-families 20 --n-peptides 3000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps

from refnd.kernels.alignments import GlobalAligner, ScoringMatrix

from src.cache import CacheStore
from src.datasets import load_dataset

AA = "ACDEFGHIKLMNPQRSTVWY"
OUT_DIR = Path(__file__).parent / "figs"


# ─────────────────────────────── atlas profile ───────────────────────────────

@dataclass
class AtlasProfile:
    """Empirical length distribution + amino-acid background of PeptideAtlas."""
    lengths: np.ndarray          # observed peptide lengths (sampled from directly)
    aa_freq: np.ndarray          # background frequencies, aligned with AA

    def sample_seed(self, rng: np.random.Generator) -> str:
        length = int(rng.choice(self.lengths))
        indices = rng.choice(len(AA), size=length, p=self.aa_freq)
        return "".join(AA[i] for i in indices)


def atlas_profile(cache: CacheStore, len_range: tuple[int, int] = (10, 100)) -> AtlasProfile:
    data, _ = load_dataset("peptide_atlas", cache)

    lengths, counts = [], np.zeros(len(AA))
    index_of = {aa: i for i, aa in enumerate(AA)}
    for sequence in data:
        if not (len_range[0] <= len(sequence) <= len_range[1]):
            continue
        lengths.append(len(sequence))
        for residue in sequence:
            index = index_of.get(residue)
            if index is not None:
                counts[index] += 1
    return AtlasProfile(np.asarray(lengths, dtype=int), counts / counts.sum())


# ───────────────────────── BLOSUM62 substitution model ───────────────────────

def blosum62_matrix() -> np.ndarray:
    """BLOSUM62 scores as a 20x20 array over AA (half-bit units)."""
    from Bio.Align import substitution_matrices
    matrix = substitution_matrices.load("BLOSUM62")
    return np.array([[float(matrix[a][b]) for b in AA] for a in AA])


def substitution_probs(scores: np.ndarray, background: np.ndarray,
                       temperature: float = 1.0) -> np.ndarray:
    """P(b | a, substitution happens), from BLOSUM62.

    BLOSUM scores are S(a,b) = (1/lambda) log2(q_ab / (p_a p_b)) in half bits,
    so the target conditional is q_{b|a} ∝ p_b * 2^(S(a,b)/2).

    Args:
        scores: 20x20 BLOSUM62 score matrix in half-bit units, rows and columns
            ordered as in `AA` (see `blosum62_matrix`).
        background: Amino-acid background frequencies, aligned with `AA`.
        temperature: Divides the score exponent. 1.0 reproduces the BLOSUM62
            conditional; > 1 flattens it towards the background (less
            conservative substitutions), < 1 sharpens it (more conservative).

    Returns:
        A 20x20 row-stochastic matrix; entry [a, b] is P(residue a -> b), with a
        zero diagonal since a substitution must change the residue.
    """
    weights = background[None, :] * np.power(2.0, scores / (2.0 * temperature))
    np.fill_diagonal(weights, 0.0)            # a substitution must change the residue
    return weights / weights.sum(axis=1, keepdims=True)


# ──────────────────────────────── the forest ─────────────────────────────────

@dataclass
class Node:
    index: int
    seq: str
    parent: int | None
    family: int               # family id == index of the tree this node belongs to
    edits: int                # edits on the branch from parent
    depth_edits: int          # accumulated edits from the root
    children: list[int] = field(default_factory=list)


@dataclass
class Forest:
    nodes: list[Node]

    @property
    def sequences(self) -> list[str]:
        return [node.seq for node in self.nodes]

    @property
    def families(self) -> np.ndarray:
        return np.array([node.family for node in self.nodes])


class Mutator:
    def __init__(self, profile: AtlasProfile, temperature: float = 1.0,
                 p_sub: float = 0.85, p_ins: float = 0.075,
                 min_len: int = 10, max_len: int = 100):
        self.background = profile.aa_freq
        self.sub_probs = substitution_probs(blosum62_matrix(), self.background, temperature)
        self.p_sub, self.p_ins = p_sub, p_ins
        self.min_len, self.max_len = min_len, max_len
        self.index_of = {aa: i for i, aa in enumerate(AA)}

    def _one_edit(self, residues: list[str], rng: np.random.Generator) -> list[str]:
        draw = rng.random()
        pos = int(rng.integers(len(residues)))
        if draw < self.p_sub:                                            # substitution
            current = self.index_of.get(residues[pos])
            if current is None:
                raise RuntimeError(
                    f"Received an invalid character: '{residues[pos]}' when trying to mutate.")
            residues[pos] = AA[int(rng.choice(len(AA), p=self.sub_probs[current]))]
        elif draw < self.p_sub + self.p_ins and len(residues) < self.max_len:  # insertion
            residues.insert(pos, AA[int(rng.choice(len(AA), p=self.background))])
        elif len(residues) > self.min_len:                               # deletion
            del residues[pos]
        return residues

    def mutate(self, seq: str, n_edits: int, rng: np.random.Generator) -> str:
        residues = list(seq)
        for _ in range(n_edits):
            residues = self._one_edit(residues, rng)
        return "".join(residues)


def grow_forest(profile: AtlasProfile, n_families: int, n_peptides: int,
                branching: float = 1.6, edits_per_branch: float = 3.0,
                max_depth_edits: int = 20, temperature: float = 1.0,
                seed: int = 0) -> Forest:
    """Galton-Watson growth: BFS over a queue of live nodes until the target
    population is reached. One tree per family.

    `max_depth_edits` caps the accumulated edit distance from the seed
    """
    rng = np.random.default_rng(seed)
    mutator = Mutator(profile, temperature=temperature)

    nodes: list[Node] = []
    queue: list[int] = []
    for family in range(n_families):
        nodes.append(Node(len(nodes), profile.sample_seed(rng), None, family, 0, 0))
        queue.append(nodes[-1].index)

    cursor = 0
    while len(nodes) < n_peptides:
        if cursor == len(queue):
            # every lineage is saturated (or died out): re-expand a node that
            # still has edit budget. Roots always qualify, so this terminates.
            queue.append(int(rng.choice(
                [node.index for node in nodes if node.depth_edits < max_depth_edits])))
        parent = nodes[queue[cursor]]
        cursor += 1
        budget = max_depth_edits - parent.depth_edits
        if budget <= 0:
            continue
        for _ in range(int(rng.poisson(branching))):
            if len(nodes) >= n_peptides:
                break
            n_edits = 1 + int(rng.poisson(max(edits_per_branch - 1.0, 0.0)))
            n_edits = min(n_edits, budget)
            child = Node(len(nodes), mutator.mutate(parent.seq, n_edits, rng),
                         parent.index, parent.family, n_edits,
                         parent.depth_edits + n_edits)
            nodes.append(child)
            parent.children.append(child.index)
            if child.depth_edits < max_depth_edits:
                queue.append(child.index)
    return Forest(nodes)


# ───────────────────────────── distance diagnostics ──────────────────────────

def sampled_distances(forest: Forest, n_pairs: int = 20_000,
                      seed: int = 0) -> dict[str, np.ndarray]:
    """Kernel distance (refnd GlobalAligner, 0 = identical) for random pairs,
    split by whether the two peptides share a family.

    Half of the pairs are drawn within a family, half across two different
    families -- sampling pairs uniformly would almost never hit the same family.
    """
    aligner = GlobalAligner(matrix=ScoringMatrix.Blosum62)
    rng = np.random.default_rng(seed)

    # Sequences grouped by family; singleton families can't give a within pair.
    family_sequences: dict[int, list[str]] = {}
    for node in forest.nodes:
        family_sequences.setdefault(node.family, []).append(node.seq)
    families = [group for group in family_sequences.values() if len(group) > 1]

    def pick_families(count: int) -> list[list[str]]:
        chosen = rng.choice(len(families), size=count, replace=False)
        return [families[index] for index in chosen]

    def pick_sequences(group: list[str], count: int) -> list[str]:
        chosen = rng.choice(len(group), size=count, replace=False)
        return [group[index] for index in chosen]

    within, between = [], []
    for _ in range(n_pairs // 2):
        (family,) = pick_families(1)
        first, second = pick_sequences(family, 2)
        within.append(aligner.call(first, second))

        family_a, family_b = pick_families(2)
        (first,) = pick_sequences(family_a, 1)
        (second,) = pick_sequences(family_b, 1)
        between.append(aligner.call(first, second))

    return {"within_family": np.asarray(within), "between_family": np.asarray(between)}


# ──────────────────────────────── plotting ───────────────────────────────────
def plot_distances(distances: dict[str, np.ndarray], threshold: float = 0.5,
                   path: Path = OUT_DIR / "generation_distances.png") -> None:
    labels = {"within_family": ("within family", "#2b7bba"),
              "between_family": ("between families", "#e8a33d")}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, (label, color) in labels.items():
        ax.hist(distances[key], bins=np.linspace(0, 1, 60), density=True, alpha=0.55,
                label=f"{label} (n={len(distances[key]):,})", color=color)
    ax.axvline(threshold, color="k", ls="--", lw=1,
               label=f"refnd threshold = {threshold}")
    ax.set_xlabel("kernel distance  (refnd GlobalAligner / BLOSUM62, 0 = identical)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


# ──────────────────────────────────── main ───────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-families", type=int, default=20)
    parser.add_argument("--n-peptides", type=int, default=3000)
    parser.add_argument("--branching", type=float, default=1.6)
    parser.add_argument("--edits-per-branch", type=float, default=3.0)
    parser.add_argument("--max-depth-edits", type=int, default=10,
                        help="max accumulated edit distance from the seed; bounds "
                             "how far a family can drift")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-len", type=int, default=20,
                        help="restrict the atlas length distribution seeds are drawn from")
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--pairs", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cache = CacheStore()
    print("Loading PeptideAtlas profile...")
    profile = atlas_profile(cache, len_range=(args.min_len, args.max_len))
    print(f"  lengths: median={np.median(profile.lengths):.0f} "
          f"[{profile.lengths.min()}-{profile.lengths.max()}], n={len(profile.lengths):,}")

    print("Growing forest...")
    forest = grow_forest(profile, args.n_families, args.n_peptides,
                         branching=args.branching,
                         edits_per_branch=args.edits_per_branch,
                         max_depth_edits=args.max_depth_edits,
                         temperature=args.temperature, seed=args.seed)
    sizes = np.bincount(forest.families)
    print(f"  {len(forest.nodes):,} peptides in {args.n_families} families "
          f"(size median={np.median(sizes):.0f} min={sizes.min()} max={sizes.max()}), "
          f"max depth_edits={max(node.depth_edits for node in forest.nodes)}")

    print("Sampling kernel distances...")
    distances = sampled_distances(forest, n_pairs=args.pairs, seed=args.seed)
    for name, values in distances.items():
        print(f"  {name:16s} mean={values.mean():.3f}  "
              f"p05={np.quantile(values, 0.05):.3f}  "
              f"frac<=0.5={(values <= 0.5).mean():.3f}")

    plot_distances(distances)


if __name__ == "__main__":
    main()
