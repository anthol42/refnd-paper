"""Threshold sweep for BELKA molecules: full training set (train) vs. the
non-triazine (OOD) subset of the real Kaggle test set (production).

Both molecule pools are streamed off disk (pyarrow batches, one column) to
avoid ever materializing the raw multi-column parquet in memory, and
fingerprints are computed once and cached to a packed binary file so a
re-run just streams them back in. At BELKA's scale (~98M unique train
molecules), everything downstream of "list of SMILES" stays disk-backed:
fingerprinting writes straight to disk (never accumulates an in-memory
array), and the combined HNSW is built from a chained on-disk fingerprint
stream, never a materialized combined list.

Usage:
    uv run python -m thresholdv2.molecules
"""

from __future__ import annotations

import argparse
import faulthandler
import itertools
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from rich import print

from refnd.core import EdgeStore, HNSWState, INWeightType

from src.cache import CacheStore
from src.datasets import DATASETS, belka_download
from src.fingerprints import (
    FP_N_BYTES_PACKED,
    compute_and_cache_fingerprints_to_disk,
    stream_bitfingerprints_from_disk,
)
from src.metrics import null_model_cdf, null_model_scores

from threshold.fit import sweep_thresholds

# DEBUG: last SMILES handed to RDKit, module-level so it survives a segfault
# inside generate_random_molecule_fps -- combined with the periodic
# _debug_stamp() calls below (every 10k molecules), this narrows a crash to
# within 10k molecules of a known-good point even though a segfault gives us
# no exception/traceback of its own.
# _LAST_RDKIT_SMILES: str | None = None
# _LAST_RDKIT_STAGE: str = ""

# DEBUG: dump a Python-level (all-threads) traceback to stderr on a fatal
# signal (SIGSEGV/SIGFPE/SIGABRT/SIGBUS/SIGILL) before the process dies.
# Won't show C/Rust frames, but pinpoints which Python line each thread was
# executing -- e.g. distinguishes "stuck in the RDKit call on the main
# thread" from "died inside a rayon worker thread in zip_kernel".
# faulthandler.enable(all_threads=True)


# def _debug_stamp(msg: str) -> None:
#     """Flushed, timestamped progress marker -- so the sbatch log shows
#     exactly which stage was reached even if the process is killed before
#     Python's normal (buffered) stdout would otherwise flush."""
#     ts = time.strftime("%H:%M:%S")
#     print(f"  [dim][DEBUG {ts}] {msg}[/]")
#     sys.stdout.flush()


OUT_DIR = Path(__file__).parent.parent / "results" / "thresholds"
GRAPH_CACHE_KEY = "threshold_sweep_belka"
SEED = 42  # only used to seed the (optional) train fraction sub-sample

DNA_TAG = "[Dy]"
TRIAZINE_SMARTS = "c1ncncn1"  # bare 1,3,5-triazine aromatic ring, any substitution

# Random-molecule null model (see belka.py's generate_random_atom_fps,
# the earlier version of this same strategy): linear SELFIES chains of bare,
# valence>=2 atoms, chain length ~Normal(mean, std) tokens. Zero relation to
# BELKA's real building blocks -- matched only on coarse size.
NULL_ATOM_ALPHABET = ["[C]"] * 6 + ["[N]"] * 2 + ["[O]"] * 2 + ["[S]"] * 1
NULL_MEAN_SIZE = 50.0
NULL_STD_SIZE = 5.0


def strip_dna_tag(smiles: str) -> str:
    return smiles.replace(DNA_TAG, "")


def _fp_count(path: Path) -> int:
    return path.stat().st_size // FP_N_BYTES_PACKED


def stream_unique_smiles(
    path: Path, column: str = "molecule_smiles", batch_size: int = 1_000_000,
    strip_tag: str | None = None,
) -> list[str]:
    """Deduped SMILES from one column of a (possibly huge) parquet/csv file,
    read in batches -- never loads the full multi-column table, and never
    holds more than one batch of raw column data at a time. The dedup set
    itself (unavoidably, one entry per unique molecule) is the only thing
    that grows over the whole pass.
    """
    from tqdm import tqdm

    seen: dict[str, None] = {}
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(str(path))
        n_batches = -(-pf.metadata.num_rows // batch_size)
        batches: Iterable = pf.iter_batches(batch_size=batch_size, columns=[column])
        for batch in tqdm(batches, total=n_batches, desc=f"Streaming {path.name}"):
            for smi in batch.column(column).to_pylist():
                seen[strip_tag and smi.replace(strip_tag, "") or smi] = None
    else:
        import pandas as pd

        for chunk in tqdm(pd.read_csv(path, usecols=[column], chunksize=batch_size), desc=f"Streaming {path.name}"):
            for smi in chunk[column]:
                seen[strip_tag and smi.replace(strip_tag, "") or smi] = None
    return list(seen.keys())


def filter_non_triazine(smiles: list[str]) -> list[str]:
    """Keep only molecules that do NOT match the 1,3,5-triazine ring SMARTS
    (BELKA's DEL synthesis scaffold) -- the OOD portion of the test set."""
    from rdkit import Chem
    from tqdm import tqdm

    patt = Chem.MolFromSmarts(TRIAZINE_SMARTS)
    out = []
    for smi in tqdm(smiles, desc="Filtering non-triazine"):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None and not mol.HasSubstructMatch(patt):
            out.append(smi)
    return out


def get_or_build_fingerprints(name: str, build_smiles: Callable[[], list[str]], cache: CacheStore) -> tuple[Path, int]:
    """Returns (fp_path, n_written) for the packed Morgan fingerprints of the
    molecule pool `name`, computing + caching them if not already on disk.

    build_smiles() is only invoked when the fingerprint file doesn't already
    exist; its result is itself cached (pickled) so an interrupted
    fingerprinting pass can resume without re-deriving the SMILES list.
    """
    fp_path = cache.root / f"{name}_fp.bin"
    if fp_path.exists():
        n = _fp_count(fp_path)
        print(f"  [dim]Loading cached fingerprints: {fp_path.name} (n={n:,})[/]")
        return fp_path, n

    smiles_path = cache.root / f"{name}_smiles.pkl"
    tmp_path = fp_path.with_suffix(".bin.tmp")
    checkpoint_path = tmp_path.with_name(tmp_path.name + ".checkpoint")

    if smiles_path.exists():
        print(f"  [dim]Loading cached SMILES: {smiles_path.name}[/]")
        with open(smiles_path, "rb") as f:
            smiles = pickle.load(f)
    else:
        smiles = build_smiles()
        with open(smiles_path, "wb") as f:
            pickle.dump(smiles, f)
        print(f"  {len(smiles):,} unique molecules -> {smiles_path.name}")

    resume_from = int(checkpoint_path.read_text()) if checkpoint_path.exists() else 0
    print(f"  Fingerprinting {len(smiles):,} molecules ({name}, resume_from={resume_from:,})...")
    n_written, n_missing = compute_and_cache_fingerprints_to_disk(
        smiles, str(tmp_path), progress=True,
        resume_from=resume_from, checkpoint_path=str(checkpoint_path), checkpoint_every=2_000_000,
    )
    del smiles
    if n_missing:
        print(f"  [yellow]Dropped {n_missing:,} SMILES RDKit couldn't parse[/]")
    tmp_path.rename(fp_path)
    checkpoint_path.unlink(missing_ok=True)
    return fp_path, n_written


def frac_suffix(frac: float) -> str:
    return "" if frac >= 1.0 else f"_frac{frac:g}"


def build_train_smiles(frac: float = 1.0) -> list[str]:
    train_path = belka_download()
    smiles = stream_unique_smiles(train_path, strip_tag=DNA_TAG)
    if frac < 1.0:
        rng = np.random.default_rng(SEED)
        n_keep = int(round(frac * len(smiles)))
        idx = rng.choice(len(smiles), size=n_keep, replace=False)
        smiles = [smiles[i] for i in idx]
        print(f"  Sub-sampled train to frac={frac:g}: {len(smiles):,} molecules")
    return smiles


def build_prod_smiles() -> list[str]:
    train_path = belka_download()
    test_path = train_path.parent / "test.parquet"
    if not test_path.exists():
        test_path = train_path.parent / "test.csv"
    test_smiles = stream_unique_smiles(test_path, strip_tag=DNA_TAG)
    print(f"  {len(test_smiles):,} unique test molecules; filtering out triazine-scaffold ones...")
    return filter_non_triazine(test_smiles)


class _ChainedFPStream:
    """Concatenates multiple on-disk fingerprint streams into one iterable,
    exposing `__length_hint__` (but not `__len__`) so `HNSWState(...)` stays
    on its unsized streaming path instead of materializing a combined list.
    """

    def __init__(self, streams: list, total: int):
        self._chain = itertools.chain(*streams)
        self._total = total

    def __iter__(self):
        return self

    def __length_hint__(self) -> int:
        return self._total

    def __next__(self):
        return next(self._chain)


def build_hnsw(
    train_fp_path: Path, n_train: int, prod_fp_path: Path, n_prod: int,
    cache: CacheStore, graph_cache_key: str = GRAPH_CACHE_KEY, ef_construction: int = 64,
) -> EdgeStore:
    """Build (or load, if cached) the combined train+prod HNSW's layer-0 graph,
    streamed straight from the on-disk fingerprint files -- train leads prod,
    so node ids [0, n_train) are train and [n_train, n_train+n_prod) are prod.
    Never materializes the combined fingerprint list in memory.
    """
    cached = cache.get_edges(graph_cache_key)
    if cached is not None:
        print(f"  [dim]Loading cached layer-0 graph: {graph_cache_key}[/]")
        return cached

    cfg = DATASETS["belka"]
    index_path = cache.root / f"{graph_cache_key}.hnsw"

    def combined_stream() -> _ChainedFPStream:
        return _ChainedFPStream(
            [
                stream_bitfingerprints_from_disk(str(train_fp_path), n_train, progress=True),
                stream_bitfingerprints_from_disk(str(prod_fp_path), n_prod, progress=True),
            ],
            n_train + n_prod,
        )

    if index_path.exists():
        # Built (possibly on a prior run that then died in the risky
        # get_layer(0, weights=True) step below) -- skip re-paying the
        # ~20-30min build cost and just re-attach the same on-disk data.
        print(f"  [dim]Loading cached built index: {index_path.name}[/]")
        hnsw = HNSWState.load(cfg.modality, str(index_path), combined_stream(), **cfg.kernel_params)
    else:
        print(f"  Building combined HNSW (n_train={n_train:,}, n_prod={n_prod:,})...")
        hnsw = HNSWState(
            cfg.modality, combined_stream(),
            proximity_threshold=0.0,  # no-op: keep_all_edges=False, strict_ef=True
            ef_construction=ef_construction,
            keep_all_edges=False,
            cache_capacity=0,
            strict_ef=True,
            **cfg.kernel_params,
        )
        hnsw.build(progress=True)
        tmp_index_path = index_path.with_name(f"tmp_{index_path.name}")
        hnsw.save(str(tmp_index_path))
        tmp_index_path.rename(index_path)
        print(f"  Saved built index: {index_path.name}")

    # _debug_stamp("build_hnsw: about to call hnsw.get_layer(0, weights=True)")
    es = hnsw.get_layer(0, weights=True, progress=True)
    # _debug_stamp(f"build_hnsw: get_layer(0) returned, n_edges={len(es):,}")
    print(f"  layer0: n_edges={len(es):,}")
    cache.store_edges(graph_cache_key, es)
    return es


def generate_random_molecule_fps(n: int, seed: int) -> list:
    """Purely random-atom molecules -- zero relation to BELKA's building
    blocks. Linear SELFIES chains of bare atoms (C/N/O/S), chain length
    ~Normal(NULL_MEAN_SIZE, NULL_STD_SIZE) tokens, decoded to SMILES and
    kept only if RDKit accepts them."""
    from refnd.utils import BitFingerprint
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    import selfies as sf

    # global _LAST_RDKIT_SMILES, _LAST_RDKIT_STAGE

    rng = np.random.default_rng(seed)
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    n_invalid = 0
    # _debug_stamp(f"generate_random_molecule_fps: starting, n={n:,} seed={seed}")
    while len(fps) < n:
        n_tok = max(10, int(round(rng.normal(NULL_MEAN_SIZE, NULL_STD_SIZE))))
        toks = rng.choice(NULL_ATOM_ALPHABET, size=n_tok)

        # _LAST_RDKIT_STAGE = "selfies.decoder"
        smi = sf.decoder("".join(toks))
        # _LAST_RDKIT_SMILES = smi

        # _LAST_RDKIT_STAGE = "Chem.MolFromSmiles"
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_invalid += 1
            continue

        # _LAST_RDKIT_STAGE = "morgan_gen.GetFingerprint"
        fp = morgan_gen.GetFingerprint(mol)
        # np.array(fp, dtype=bool) relies on ExplicitBitVect's buffer/sequence
        # protocol, which segfaults on the cluster (see src/fingerprints.py's
        # belka_fp_worker for the same crash, hit and fixed there already).
        # DataStructs.ConvertToNumpyArray is RDKit's documented conversion path.
        arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(BitFingerprint.from_np(arr.astype(bool)))

        # if len(fps) % 10_000 == 0:
        #     _debug_stamp(
        #         f"generate_random_molecule_fps: {len(fps):,}/{n:,} "
        #         f"(invalid so far={n_invalid:,}, last smiles={smi!r})"
        #     )
    print(f"  generated {len(fps):,} random-atom molecules ({n_invalid:,} invalid, discarded)")
    return fps


def find_gamma_function(
    cache: CacheStore, n_molecules: int = 100_000, n_pairs: int = 2_000_000, seed: int = 42,
) -> Callable[[np.ndarray], np.ndarray]:
    """Null model p0(t) = P(distance <= t) between two unrelated random-atom
    molecules (see generate_random_molecule_fps). Cached end-to-end."""
    cfg = DATASETS["belka"]
    null_path = cache.root / f"null_random_molecules_n{n_molecules}_pairs{n_pairs}_seed{seed}.npy"
    if null_path.exists():
        print(f"  [dim]Loading cached null scores: {null_path.name}[/]")
        null_scores = np.load(null_path)
    else:
        random_fps = generate_random_molecule_fps(n_molecules, seed)
        # _debug_stamp(
        #     f"find_gamma_function: about to call null_model_scores/zip_kernel "
        #     f"(n_pairs={n_pairs:,}, RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS')})"
        # )
        null_scores = null_model_scores(random_fps, cfg, n_samples=n_pairs, seed=seed, shuffle=False)
        # _debug_stamp("find_gamma_function: null_model_scores/zip_kernel returned")
        np.save(null_path, null_scores)
    return null_model_cdf(null_scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frac", type=float, default=1.0,
                        help="Fraction of the train set to sub-sample (uniform, seed=%d). "
                             "Included in cache keys when < 1.0, so different fractions never "
                             "clobber each other's cached fingerprints/index/graph." % SEED)
    parser.add_argument("--thresh-lo", type=float, default=0.2)
    parser.add_argument("--thresh-hi", type=float, default=0.4)
    parser.add_argument("--n-sweep", type=int, default=5)
    parser.add_argument("--ef-construction", type=int, default=64)
    parser.add_argument("--n-null-molecules", type=int, default=100_000)
    parser.add_argument("--n-null-pairs", type=int, default=2_000_000)
    args = parser.parse_args()

    cache = CacheStore()
    suffix = frac_suffix(args.frac)

    print("[bold][orange2]=== Threshold sweep: BELKA train vs. non-triazine test (prod) ===[/][/]")
    train_fp_path, n_train = get_or_build_fingerprints(
        f"belka_train_stripped{suffix}", lambda: build_train_smiles(args.frac), cache,
    )
    prod_fp_path, n_prod = get_or_build_fingerprints("belka_test_ood_stripped", build_prod_smiles, cache)
    print(f"  train={n_train:,}  prod(non-triazine test)={n_prod:,}")

    # _debug_stamp("main: about to call build_hnsw")
    hnsw_graph = build_hnsw(
        train_fp_path, n_train, prod_fp_path, n_prod, cache,
        graph_cache_key=f"{GRAPH_CACHE_KEY}{suffix}", ef_construction=args.ef_construction,
    )
    # _debug_stamp("main: build_hnsw returned")
    train_nodes = np.arange(n_train, dtype=np.int32)
    prod_nodes = np.arange(n_train, n_train + n_prod, dtype=np.int32)

    print("  Computing gamma(tau) from a random-molecule null...")
    gamma_cdf = find_gamma_function(cache, n_molecules=args.n_null_molecules, n_pairs=args.n_null_pairs)
    # _debug_stamp("main: find_gamma_function returned")

    # _debug_stamp("main: about to call sweep_thresholds")
    thresholds, results = sweep_thresholds(
        hnsw_graph, train_nodes, prod_nodes, gamma_cdf,
        args.thresh_lo, args.thresh_hi, args.n_sweep,
        inweight_type=INWeightType.SimilarityComplement,
    )
    # _debug_stamp("main: sweep_thresholds returned")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"molecules{suffix}.json"
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
