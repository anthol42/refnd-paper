from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# os.cpu_count() reports the node's total physical CPUs, not the cpuset a
# Slurm job is actually confined to via --cpus-per-task -- on a shared
# cluster node that oversubscribes ProcessPoolExecutor's default worker
# count far past the job's --mem budget, and a worker getting OOM-killed by
# the cgroup surfaces as BrokenProcessPool. sched_getaffinity respects the
# cpuset Slurm sets, so this stays within what was actually allocated.
_MAX_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()


def belka_fp_worker(smiles: str) -> np.ndarray | None:
    """Top-level so it's picklable for ProcessPoolExecutor."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = morgan_gen.GetFingerprint(mol)
    return np.array(fp, dtype=bool)


def compute_fingerprints(
    smiles: list[str], *, chunksize: int = 256, progress: bool = False,
) -> list[np.ndarray | None]:
    """Morgan fingerprints for a list of SMILES, computed in parallel across
    all available CPU cores. Entries are None where RDKit couldn't parse the
    SMILES.
    """
    with ProcessPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        results = executor.map(belka_fp_worker, smiles, chunksize=chunksize)
        if progress:
            from tqdm import tqdm
            results = tqdm(results, total=len(smiles))
        return list(results)


def compute_bitfingerprints(
    smiles: list[str], *, chunksize: int = 256, progress: bool = False,
):
    """Like `compute_fingerprints`, but converts each worker result straight
    to a packed `BitFingerprint` and drops the unpacked np.bool_ array as it
    arrives, instead of materializing the full unpacked list/array first.
    Peak memory holds one packed copy (~256 B/fp) instead of an unpacked
    (~2048 B/fp) and packed copy at once — the difference matters at BELKA
    scale (~98M molecules).

    Returns (fingerprints, n_missing) where n_missing counts SMILES RDKit
    couldn't parse (skipped, not included in `fingerprints`).
    """
    from refnd.utils import BitFingerprint

    with ProcessPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        results = executor.map(belka_fp_worker, smiles, chunksize=chunksize)
        if progress:
            from tqdm import tqdm
            results = tqdm(results, total=len(smiles))

        fingerprints = []
        n_missing = 0
        for fp in results:
            if fp is None:
                n_missing += 1
                continue
            fingerprints.append(BitFingerprint.from_np(fp))
    return fingerprints, n_missing


FP_N_BITS = 2048
FP_N_BYTES_PACKED = FP_N_BITS // 8  # 256


def compute_and_cache_fingerprints_to_disk(
    smiles: list[str], out_path: str, *, chunksize: int = 256, progress: bool = False,
    resume_from: int = 0, checkpoint_path: str | None = None, checkpoint_every: int = 500_000,
) -> tuple[int, int]:
    """Computes Morgan fingerprints in parallel (same workers as `compute_fingerprints`),
    packing and writing each one to `out_path` as it arrives instead of accumulating any
    in-memory list/array. Peak memory during this phase is just `smiles` + the
    ProcessPoolExecutor/RDKit worker overhead — no fingerprint data held in RAM at all — so
    this phase's memory cost can be measured in isolation from `HNSWState`'s construction
    (which streams back from this file via `stream_bitfingerprints_from_disk`).

    Returns (n_written, n_missing): the file holds `n_written` packed fingerprints total
    (including any from a resumed prior run) — `FP_N_BYTES_PACKED` bytes each, sequential,
    no header — for the SMILES RDKit could parse; `n_missing` (this call only, not
    cumulative) counts the ones it couldn't (skipped, not written).

    resume_from: skip this many leading `smiles` (already processed by an interrupted prior
    run whose partial output is still at `out_path`) and append instead of overwriting. The
    caller is responsible for knowing this is a safe resume point — `out_path`'s byte size
    alone only tells you how many fingerprints were *written*, not how many SMILES were
    *consumed* to produce them, since RDKit parse failures are skipped rather than written
    (the two only coincide if there were zero failures in the already-processed prefix).

    checkpoint_path: if given, the number of SMILES consumed so far (not `n_written` — see
    above) is periodically written here as plain text, so a future interruption can resume
    exactly via `resume_from=int(Path(checkpoint_path).read_text())`, no guessing required.
    """
    mode = "ab" if resume_from > 0 else "wb"
    remaining = smiles[resume_from:]
    n_written = 0
    n_missing = 0
    consumed = resume_from
    with open(out_path, mode) as out_f, ProcessPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        results = executor.map(belka_fp_worker, remaining, chunksize=chunksize)
        if progress:
            from tqdm import tqdm
            results = tqdm(results, total=len(remaining))

        for fp in results:
            if fp is not None:
                out_f.write(np.packbits(fp).tobytes())
                n_written += 1
            else:
                n_missing += 1
            consumed += 1
            if checkpoint_path is not None and consumed % checkpoint_every == 0:
                out_f.flush()
                Path(checkpoint_path).write_text(str(consumed))
    if checkpoint_path is not None:
        Path(checkpoint_path).write_text(str(consumed))

    total_written = Path(out_path).stat().st_size // FP_N_BYTES_PACKED
    return total_written, n_missing


class _DiskFingerprintStream:
    """Iterator over packed `BitFingerprint`s read from a file written by
    `compute_and_cache_fingerprints_to_disk`, one at a time. No RDKit or
    ProcessPoolExecutor involved here.

    Plain sequential buffered reads (`file.read()`), not `np.memmap`: memmap'd pages,
    once touched, are counted in *this process's* RSS for as long as they stay resident
    (confirmed empirically: touching 2M rows / 512MB via memmap grew RSS by ~488MB;
    reading the identical data via plain `file.read()` grew RSS by 0MB — the OS's page
    cache for a regular read lives system-wide, not attributed to the reading process).
    At BELKA's ~98.4M-row scale, that's tens of GB of RSS inflation from the source file
    alone, on top of the fingerprint data actually being retained. We only ever read this
    file forward once per index build, so there's no reason to want random-access mmap
    semantics here.

    Exposes `__length_hint__` but deliberately not `__len__`: `HNSWState(...)` checks
    `len()` to decide between its sized bulk-extract path (which would need this fully
    materialized into a Python-side list first) and its unsized streaming path (drains one
    item at a time). `__length_hint__` is a *separate* protocol `len()` doesn't consult, so
    this stays on the streaming path while still letting the Rust side pre-size its
    internal Vec correctly from the hint — avoiding both the double-copy `__len__` would
    reintroduce and the reallocation waste an unhinted `Vec::new()` would otherwise pay
    growing to a ~98M-item Vec.
    """

    def __init__(self, path: str, n_items: int, *, progress: bool = False):
        from refnd.utils import BitFingerprint

        self._BitFingerprint = BitFingerprint
        self._f = open(path, "rb", buffering=1024 * 1024)
        self._n_items = n_items
        self._i = 0
        self._pbar = None
        if progress:
            from tqdm import tqdm
            self._pbar = tqdm(total=n_items)

    def __iter__(self):
        return self

    def __length_hint__(self) -> int:
        return self._n_items - self._i

    def __next__(self):
        if self._i >= self._n_items:
            self._f.close()
            if self._pbar is not None:
                self._pbar.close()
            raise StopIteration
        packed = self._f.read(FP_N_BYTES_PACKED)
        row = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))
        self._i += 1
        if self._pbar is not None:
            self._pbar.update(1)
        return self._BitFingerprint.from_np(row)


def stream_bitfingerprints_from_disk(path: str, n_items: int, *, progress: bool = False):
    """See `_DiskFingerprintStream`. Meant to be passed directly as `data` to
    `HNSWState(...)` (refnd>=0.0.3 accepts any Python iterable, not just a sized sequence),
    so its construction-time memory profile is just the growing Rust-owned copy, isolated
    from fingerprint-computation cost.
    """
    return _DiskFingerprintStream(path, n_items, progress=progress)
