"""Correlation between the ProtSpaM kernel and global alignment.

Random UniRef50 pairs are essentially all unrelated -- their global-alignment
distance piles up near 1.0 and ProtSpaM's mismatch rate saturates there, so a
correlation measured on such a pool says nothing about the 0.3-0.7 band where a
proximity threshold actually lives. To cover the full 0-1 alignment axis we
build a *mutation ladder*: real UniRef50 seed sequences are mutated at a sweep
of rates (BLOSUM62-weighted substitutions plus indels), and each (seed, mutant)
pair is scored with both kernels.

ProtSpaM uses exactly the parameters `build_uniref_benchmark_index.py` builds
the UniRef50 HNSW index with (n_patterns=1, weight=3, dont_care=20,
significance_threshold=-1e6, MismatchRate); global alignment uses the project
default (AlignmentGlobal + BLOSUM62).

Output: results/protspam_vs_alignment.json, whose "pairs" field is a list of
[global_alignment_distance, protspam_distance] tuples. Correlations and figures
are computed downstream in the notebook.

Usage:
    uv run python -m threshold.protspam_vs_alignment \\
        --uniref50-fasta /mnt/documents/refnd_cache/uniref50/uniref50.fasta.gz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from Bio.Align import substitution_matrices
from rich import print

from refnd.kernels import KernelVariant, zip_kernel
from refnd.kernels.alignments import ScoringMatrix
from refnd.kernels.protspam import ProtSpamDistance
from refnd.utils import SWPatternSet, SWSequence

from src.cache import CacheStore

from threshold.build_uniref_benchmark_index import stream_uniref50_fasta
from threshold.uniref import reservoir_sample

OUT_DIR = Path(__file__).parent.parent / "results"

SEED = 42
LENGTH_CAP = 2048           # same cap build_uniref_benchmark_index.py encodes with
SIGNIFICANCE_THRESHOLD = -1_000_000
AA = "ARNDCQEGHILKMFPSTWYV"  # the 20 standard residues we mutate among


def blosum62_substitution_table() -> dict[str, tuple[list[str], np.ndarray]]:
    """P(b | a, a substitution happened) ~ 2^S(a,b) over the 20 standard
    residues, excluding b == a. Returns {a: (residues, probabilities)}."""
    m = substitution_matrices.load("BLOSUM62")
    table = {}
    for a in AA:
        targets = [b for b in AA if b != a]
        w = np.array([2.0 ** float(m[a, b]) for b in targets])
        table[a] = (targets, w / w.sum())
    return table


def mutate(seq: str, rate: float, indel_fraction: float, mean_indel_len: float,
           sub_table: dict[str, tuple[list[str], np.ndarray]], rng: np.random.Generator) -> str:
    """Apply `rate` mutation events per residue. Each event is an indel with
    probability `indel_fraction` (insertion or deletion, equally likely, length
    geometric with mean `mean_indel_len`) and a BLOSUM62-weighted substitution
    otherwise.

    Rates above 1.0 are applied as repeated passes (multiple hits per site), so
    the ladder can reach the saturated end of the alignment axis -- a single
    pass caps out around 0.87 global-alignment distance."""
    while rate > 1.0:
        seq = _mutate_once(seq, 1.0, indel_fraction, mean_indel_len, sub_table, rng)
        rate -= 1.0
    return _mutate_once(seq, rate, indel_fraction, mean_indel_len, sub_table, rng)


def _mutate_once(seq: str, rate: float, indel_fraction: float, mean_indel_len: float,
                 sub_table: dict[str, tuple[list[str], np.ndarray]], rng: np.random.Generator) -> str:
    """One mutation pass with `rate` <= 1.0 events per residue."""
    chars = list(seq)
    if not chars:
        return ""
    n_events = rng.binomial(len(chars), min(rate, 1.0))
    if n_events == 0:
        return "".join(chars)

    positions = rng.choice(len(chars), size=n_events, replace=False)
    is_indel = rng.random(n_events) < indel_fraction

    # substitutions first: in-place, so positions stay valid
    for pos in positions[~is_indel]:
        a = chars[pos]
        if a not in sub_table:      # ambiguity codes (B/Z/X/J/*) are left alone
            continue
        targets, probs = sub_table[a]
        chars[pos] = targets[rng.choice(len(targets), p=probs)]

    # then indels, applied right-to-left so earlier positions stay valid
    p_geom = 1.0 / max(mean_indel_len, 1.0)
    for pos in sorted(positions[is_indel], reverse=True):
        length = int(rng.geometric(p_geom))
        if rng.random() < 0.5:
            del chars[pos:pos + length]
        else:
            chars[pos:pos] = list(rng.choice(list(AA), size=length))

    return "".join(chars)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uniref50-fasta", type=Path, required=True)
    parser.add_argument("--n-seeds", type=int, default=2_000,
                        help="Seed sequences reservoir-sampled from UniRef50.")
    parser.add_argument("--n-levels", type=int, default=20,
                        help="Mutation-rate levels spanning --rate-lo..--rate-hi.")
    parser.add_argument("--mutants-per-level", type=int, default=1)
    parser.add_argument("--round-digits", type=int, default=4,
                        help="Decimal places kept for the two distances in the JSON.")
    parser.add_argument("--rate-lo", type=float, default=0.0)
    parser.add_argument("--rate-hi", type=float, default=1.2,
                        help="Rates > 1.0 are applied as repeated mutation passes. Global-alignment\n"
                             "distance saturates near 0.87 (the unrelated-pair ceiling) by ~1.2, so\n"
                             "higher rates only re-sample the saturated end.")
    parser.add_argument("--indel-fraction", type=float, default=0.15,
                        help="Fraction of mutation events that are indels rather than substitutions.")
    parser.add_argument("--mean-indel-len", type=float, default=2.0)
    parser.add_argument("--min-seed-len", type=int, default=50,
                        help="Seeds shorter than this are skipped (alignment distance is too noisy).")
    parser.add_argument("--n-patterns", type=int, default=1)
    parser.add_argument("--weight", type=int, default=3)
    parser.add_argument("--dont-care", type=int, default=20)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "protspam_vs_alignment.json")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)

    print("[bold][orange2]=== ProtSpaM vs. global alignment: mutation ladder ===[/][/]")
    # the reservoir pass streams the whole 8.7GB gzipped FASTA, so cache it --
    # a re-run with the same --n-seeds reuses the pool instead of re-streaming.
    pool_path = CacheStore().root / f"uniref50_seed_pool_n{args.n_seeds}_cap{LENGTH_CAP}_seed{SEED}.json"
    if pool_path.exists():
        print(f"  [dim]Loading cached seed pool: {pool_path.name}[/]")
        pool = json.loads(pool_path.read_text())
    else:
        print(f"  Reservoir-sampling {args.n_seeds:,} UniRef50 seeds from {args.uniref50_fasta.name}...")
        pool = reservoir_sample(stream_uniref50_fasta(args.uniref50_fasta, LENGTH_CAP), args.n_seeds, SEED)
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        pool_path.write_text(json.dumps(pool))
    seeds = [s for s in pool if len(s) >= args.min_seed_len]
    print(f"  {len(seeds):,} seeds kept (>= {args.min_seed_len} residues)")

    sub_table = blosum62_substitution_table()
    rates = np.linspace(args.rate_lo, args.rate_hi, args.n_levels)

    print(f"  Generating {len(seeds) * args.n_levels * args.mutants_per_level:,} pairs "
          f"over {args.n_levels} rate levels...")
    left: list[str] = []
    right: list[str] = []
    pair_rate: list[float] = []
    for rate in rates:
        for seed in seeds:
            for _ in range(args.mutants_per_level):
                mutant = mutate(seed, float(rate), args.indel_fraction,
                                args.mean_indel_len, sub_table, rng)
                if not mutant:      # a high-rate ladder rung can delete everything
                    continue
                left.append(seed)
                right.append(mutant)
                pair_rate.append(float(rate))
    print(f"  {len(left):,} pairs")

    print("  Scoring global alignment (BLOSUM62, parallel)...")
    ga = np.asarray(zip_kernel(KernelVariant.AlignmentGlobal, left, right,
                               n_threads=0, progress=True,
                               matrix=ScoringMatrix.Blosum62), dtype=np.float64)

    print(f"  Encoding + scoring ProtSpaM (n_patterns={args.n_patterns}, "
          f"weight={args.weight}, dont_care={args.dont_care}, parallel)...")
    patterns = SWPatternSet(args.n_patterns, args.weight, args.dont_care)
    sw_left = [SWSequence(s, patterns) for s in left]
    sw_right = [SWSequence(s, patterns) for s in right]
    ps = np.asarray(zip_kernel(KernelVariant.ProtSpam, sw_left, sw_right,
                               n_threads=0, progress=True,
                               patterns=patterns,
                               significance_threshold=SIGNIFICANCE_THRESHOLD,
                               distance=ProtSpamDistance.MismatchRate), dtype=np.float64)

    rate_arr = np.asarray(pair_rate)
    per_level = [
        {"rate": float(r),
         "n": int((rate_arr == r).sum()),
         "ga_mean": float(ga[rate_arr == r].mean()),
         "ga_std": float(ga[rate_arr == r].std()),
         "protspam_mean": float(ps[rate_arr == r].mean()),
         "protspam_std": float(ps[rate_arr == r].std())}
        for r in rates
    ]

    # Distances are rounded to --round-digits and the per-pair mutation rate is
    # dropped (per_level carries it) -- full float repr made the file several MB,
    # too large to keep in git, and the extra digits are far below the kernels'
    # resolution.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "n_seeds_kept": len(seeds),
            "pairs": [[round(float(g), args.round_digits), round(float(p), args.round_digits)]
                      for g, p in zip(ga, ps)],
            "per_level": per_level,
        }, f)
    print(f"  Saved {args.out}  ({len(ga):,} pairs)")


if __name__ == "__main__":
    main()
