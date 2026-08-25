"""Build a ProtSpaM HNSW index over UniRef50, extend it with the protein
benchmark datasets (ProteinGym, PEER, CASP15 -- see src/protein_benchmarks.py),
and export the layer-0 edge graph with distances. Designed to run as a SLURM job:
every artifact path is a CLI argument, every stage is skipped if its output
already exists (so a requeued/restarted job resumes instead of recomputing),
and the UniRef50 sequences are streamed from the gzipped FASTA (never
materialized as a single Python-side list -- only one sequence exists in memory
at a time on the Python side; the only large in-memory allocation is refnd's
own Rust-owned copy).

Pipeline:
    1. Encode: stream uniref50.fasta.gz and the 3 benchmark datasets into
       SWSequence objects (all sharing one SWPatternSet) and write each to its
       own on-disk stream (individually pickled records, read back the same
       way -- no bulk list ever materialized). Also writes an `offsets.txt`
       file with each source's start offset into the final node order
       (uniref50 first, then proteingym/peer/casp15, matching the
       build-then-extend insertion order) -- segments are contiguous, so a
       per-node label array would be redundant.
    2. Build: HNSWState over the uniref50 stream alone, `use_heuristic=True`,
       saved to `--uniref-index-out`.
    3. Extend: `extend_build()` with the pooled benchmark stream, saved to
       `--extended-index-out`.
    4. Export: `get_layer(0, weights=True)` (real ProtSpaM mismatch-rate
       distances) from the extended graph, saved to `--edgestore-out`.

Sequences longer than `--length-cap` or containing characters outside
ProtSpaM's alphabet are dropped during encoding (verified to drop <2.5% everywhere).

Usage:
    uv run python -m threshold.build_uniref_benchmark_index \\
        --uniref50-fasta /mnt/documents/refnd_cache/uniref50/uniref50.fasta.gz \\
        --uniref-sequences-out /path/uniref50.swseq \\
        --benchmark-sequences-out /path/benchmarks.swseq \\
        --offsets-out /path/offsets.txt \\
        --benchmark-cache-dir /path/protein_benchmarks \\
        --patterns-out /path/patterns.bin \\
        --uniref-index-out /path/uniref50.hnsw \\
        --extended-index-out /path/unirefxbenchmark.hnsw \\
        --edgestore-out /path/unirefxbenchmark_layer0.edgestr
"""

from __future__ import annotations

import argparse
import gzip
import pickle
from collections.abc import Iterable, Iterator
from pathlib import Path
import itertools

from rich import print

from refnd.core import HNSWState
from refnd.kernels import KernelVariant
from refnd.kernels.protspam import ProtSpamDistance
from refnd.utils import SWPatternSet, SWSequence

from src.protein_benchmarks import (
    fetch_casp15_sequences,
    fetch_peer_sequences,
    fetch_proteingym_wildtypes,
)

VALID_ALPHABET = set("ARNDCQEGHILKMFPSTWYVBZXJ*")

# source labels, in insertion order (uniref50 first, then extend order)
SOURCE_LABELS = ["uniref50", "proteingym", "peer", "casp15"]


def _valid(seq: str, length_cap: int) -> bool:
    return len(seq) <= length_cap and set(seq.upper()) <= VALID_ALPHABET


def stream_uniref50_fasta(path: Path, length_cap: int) -> Iterator[str]:
    """Yields valid sequences one at a time -- never materializes the file."""
    with gzip.open(path, "rt") as f:
        chars: list[str] = []
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if chars:
                    seq = "".join(chars)
                    if _valid(seq, length_cap):
                        yield seq
                chars = []
            else:
                chars.append(line.strip())
        if chars:
            seq = "".join(chars)
            if _valid(seq, length_cap):
                yield seq

def stream_benchmark_sequences(length_cap: int, bench_dir: Path) -> Iterator[tuple[str, str]]:
    """Yields (source_label, sequence) for every valid benchmark sequence."""
    fetchers = [
        ("proteingym", fetch_proteingym_wildtypes),
        ("peer", fetch_peer_sequences),
        ("casp15", fetch_casp15_sequences),
    ]
    for label, fetch_fn in fetchers:
        print(f"  Fetching {label}...")
        for seq in fetch_fn(bench_dir):
            if _valid(seq, length_cap):
                yield label, seq


def encode_and_save(seqs: Iterable[str], patterns: SWPatternSet, out_path: Path) -> int:
    """Streams `seqs` (str iterable) through SWSequence(patterns) and writes
    each record individually-pickled to out_path. Returns the count written.

    Writes to a `.tmp` sibling and renames onto out_path only once the full
    stream is exhausted, so a crash/preemption mid-write never leaves a
    truncated file sitting at out_path that a resumed run would mistake for
    a completed stage."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    n = 0
    with open(tmp_path, "wb") as f:
        for seq in seqs:
            sw = SWSequence(seq, patterns)
            pickle.dump(sw, f, protocol=pickle.HIGHEST_PROTOCOL)
            n += 1
            if n % 500_000 == 0:
                print(f"    encoded {n:,}...")
    tmp_path.replace(out_path)
    return n


def atomic_save(save_fn, out_path: Path) -> None:
    """Calls save_fn(str(tmp_path)) then renames tmp_path onto out_path, so a
    crash mid-save can't leave a partial file at out_path that a resumed run
    would mistake for a completed stage.

    tmp_path keeps out_path's real extension (just prefixed with "tmp_")
    rather than appending ".tmp" -- some save_fns (e.g. EdgeStore.save)
    infer the save format from the path's extension, so a ".tmp" suffix
    made them fail with "unknown extension '.tmp'"."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f"tmp_{out_path.name}")
    save_fn(str(tmp_path))
    tmp_path.replace(out_path)


def load_sequences_stream(path: Path) -> Iterator[SWSequence]:
    """Generator reading back individually-pickled SWSequence records."""
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uniref50-fasta", type=Path, required=True)
    parser.add_argument("--uniref-sequences-out", type=Path, required=True)
    parser.add_argument("--benchmark-sequences-out", type=Path, required=True)
    parser.add_argument("--offsets-out", type=Path, required=True)
    parser.add_argument("--patterns-out", type=Path, required=True)
    parser.add_argument("--uniref-index-out", type=Path, required=True)
    parser.add_argument("--extended-index-out", type=Path, required=True)
    parser.add_argument("--edgestore-out", type=Path, required=True)
    parser.add_argument("--benchmark-cache-dir", type=Path, required=True,
                         help="Directory to cache downloaded ProteinGym/PEER/CASP15 files under "
                              "(passed straight through to src.protein_benchmarks' fetchers).")

    parser.add_argument("--n-patterns", type=int, default=1)
    parser.add_argument("--weight", type=int, default=3)
    parser.add_argument("--dont-care", type=int, default=20)
    parser.add_argument("--significance-threshold", type=int, default=-1_000_000)
    parser.add_argument("--length-cap", type=int, default=2048)
    parser.add_argument("--proximity-threshold", type=float, default=0.5)
    parser.add_argument("--ef-construction", type=int, default=64)
    parser.add_argument("--cache-capacity", type=int, default=2_000_000)
    parser.add_argument("--use-heuristic", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    # --- patterns ---------------------------------------------------------
    if args.patterns_out.exists():
        print(f"[dim]Loading cached patterns ({args.patterns_out})[/]")
        patterns = SWPatternSet.load(str(args.patterns_out))
    else:
        print(f"[bold][orange2]=== Building SWPatternSet(n={args.n_patterns}, weight={args.weight}, "
              f"dont_care={args.dont_care}) ===[/][/]")
        patterns = SWPatternSet(args.n_patterns, args.weight, args.dont_care)
        atomic_save(patterns.save, args.patterns_out)

    kernel_kwargs = dict(patterns=patterns, significance_threshold=args.significance_threshold,
                          distance=ProtSpamDistance.MismatchRate)
    hnsw_common = dict(proximity_threshold=args.proximity_threshold, ef_construction=args.ef_construction,
                        strict_ef=True, keep_all_edges=False, cache_capacity=args.cache_capacity)

    # --- stage 1: encode ----------------------------------------------------
    print("\n[bold][orange2]=== Stage 1: encode ===[/][/]")
    if args.uniref_sequences_out.exists():
        print(f"  [dim]{args.uniref_sequences_out} already exists, skipping uniref50 encode[/]")
    else:
        print(f"  Streaming + encoding {args.uniref50_fasta} (length_cap={args.length_cap})...")
        n = encode_and_save(stream_uniref50_fasta(args.uniref50_fasta, args.length_cap), patterns,
                             args.uniref_sequences_out)
        print(f"  {n:,} uniref50 sequences encoded -> {args.uniref_sequences_out}")

    if args.benchmark_sequences_out.exists() and args.offsets_out.exists():
        print(f"  [dim]{args.benchmark_sequences_out} already exists, skipping benchmark encode[/]")
    else:
        print("  Fetching + encoding benchmark datasets...")
        # stream_benchmark_sequences processes one fetcher (source label) to
        # completion before moving to the next, so each label's records are
        # contiguous in insertion order -- a per-node label array is
        # redundant; a per-segment start offset is enough to recover it.
        counts: dict[str, int] = {label: 0 for label in SOURCE_LABELS[1:]}

        def labelled_stream():
            for label, seq in stream_benchmark_sequences(args.length_cap, args.benchmark_cache_dir):
                counts[label] += 1
                yield seq

        n = encode_and_save(labelled_stream(), patterns, args.benchmark_sequences_out)
        print(f"  {n:,} benchmark sequences encoded -> {args.benchmark_sequences_out}")
        n_uniref = sum(1 for _ in load_sequences_stream(args.uniref_sequences_out))
        offset = n_uniref
        offsets = {"uniref50": 0}
        for label in SOURCE_LABELS[1:]:
            offsets[label] = offset
            offset += counts[label]
        args.offsets_out.parent.mkdir(parents=True, exist_ok=True)
        tmp_offsets_out = args.offsets_out.with_name(args.offsets_out.name + ".tmp")
        with open(tmp_offsets_out, "w") as f:
            for label in SOURCE_LABELS:
                f.write(f"{label}: {offsets[label]}\n")
        tmp_offsets_out.replace(args.offsets_out)
        print(f"  offsets ({n_uniref:,} uniref50 + {n:,} benchmark) -> {args.offsets_out}")

    # --- stage 2: build uniref50-only index ---------------------------------
    print("\n[bold][orange2]=== Stage 2: build UniRef50 index (use_heuristic="
          f"{args.use_heuristic}) ===[/][/]")
    if args.uniref_index_out.exists():
        print(f"  [dim]Loading cached {args.uniref_index_out}[/]")
        hnsw = HNSWState.load(KernelVariant.ProtSpam, str(args.uniref_index_out),
                               load_sequences_stream(args.uniref_sequences_out), **kernel_kwargs)
    else:
        hnsw = HNSWState(KernelVariant.ProtSpam, load_sequences_stream(args.uniref_sequences_out),
                          **hnsw_common, use_heuristic=args.use_heuristic, **kernel_kwargs)
        hnsw.build(progress=True)
        atomic_save(hnsw.save, args.uniref_index_out)
        print(f"  saved -> {args.uniref_index_out}")

    # --- stage 3: extend with benchmarks -------------------------------------
    print("\n[bold][orange2]=== Stage 3: extend with benchmark datasets ===[/][/]")
    if args.extended_index_out.exists():
        print(f"  [dim]Loading cached {args.extended_index_out}[/]")
        del hnsw  # drop the stage-2 (uniref-only) index before loading the
                  # extended one -- otherwise both full graphs are resident
                  # in memory at once during the load call below.
        all_seqs = itertools.chain(load_sequences_stream(args.uniref_sequences_out),
                                    load_sequences_stream(args.benchmark_sequences_out))
        hnsw = HNSWState.load(KernelVariant.ProtSpam, str(args.extended_index_out), all_seqs, **kernel_kwargs)
    else:
        hnsw.extend_build(load_sequences_stream(args.benchmark_sequences_out), progress=True)
        atomic_save(hnsw.save, args.extended_index_out)
        print(f"  saved -> {args.extended_index_out}")

    # --- stage 4: export layer-0 edges with distances ------------------------
    print("\n[bold][orange2]=== Stage 4: export layer-0 EdgeStore (weighted) ===[/][/]")
    if args.edgestore_out.exists():
        print(f"  [dim]{args.edgestore_out} already exists, skipping[/]")
    else:
        es = hnsw.get_layer(0, directed=False, weights=True, progress=True)
        atomic_save(es.save, args.edgestore_out)
        print(f"  saved -> {args.edgestore_out}")

    print("\n[bold][green]Done.[/][/]")


if __name__ == "__main__":
    main()
