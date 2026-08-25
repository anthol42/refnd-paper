"""Fetch scripts for standard protein benchmark datasets, each returning a flat
``list[str]`` of protein sequences (downloading and caching from the official
source on first use, matching the ``_load_peptide_atlas``-style cache-then-fetch
pattern in ``src/datasets.py``).

- **ProteinGym**: one wildtype sequence per DMS assay -- NOT the ~1.5M mutant
  rows ("unique DMS wild-type sequences ... NOT the ... single/multi-mutant variant
  rows, which would violate the i.i.d. assumption"). Fetched from the official
  HuggingFace dataset ``OATML-Markslab/ProteinGym_v1``
  (config=``DMS_substitutions``), deduplicated by ``DMS_id``.
- **PEER**: pooled, deduplicated sequences from every *single-protein* task in
  the PEER benchmark (github.com/DeepGraphLearning/PEER_Benchmark, NeurIPS
  2022) -- 9 tasks hosted as LMDB archives (fluorescence, stability,
  solubility, beta_lactamase, subcellular_localization, binary_localization,
  fold/remote_homology, secondary_structure, proteinnet/contact) plus 3
  FLIP-sourced CSV tasks (aav, gb1, thermostability), each read with PEER's own
  default split (verified against ``peer/flip.py`` and each torchdrug dataset
  class -- e.g. AAV defaults to split="two_vs_many"). Deliberately excludes the
  3 PPI tasks (human_ppi, yeast_ppi, ppi_affinity -- sequence *pairs*, not
  single sequences) and the 2 protein-ligand tasks (bindingdb, pdbbind --
  protein+ligand pairs), since those don't fit a flat list-of-sequences return
  type.
- **CASP15**: protein target sequences (ID prefix "T"), fetched from
  predictioncenter.org's official ``sequences/casp15.seq.txt`` and filtered to
  exclude RNA targets (ID prefix "R") -- the file mixes both; CASP15 introduced
  RNA structure prediction alongside the traditional protein categories.

None of this requires torchdrug/TorchProtein -- only ``lmdb`` (added as a
dependency) to read the LMDB archives PEER's underlying tasks are stored in,
plus plain ``requests``/``zipfile``/``tarfile`` for the FLIP CSV tasks and
CASP15's flat FASTA-like file.

Usage:
    uv run python -m src.protein_benchmarks /scratch/me/protein_benchmarks --benchmark proteingym
    uv run python -m src.protein_benchmarks /scratch/me/protein_benchmarks --benchmark peer
    uv run python -m src.protein_benchmarks /scratch/me/protein_benchmarks --benchmark casp15
    uv run python -m src.protein_benchmarks /scratch/me/protein_benchmarks --benchmark all
"""

from __future__ import annotations

import argparse
import csv
import pickle
import tarfile
import zipfile
from pathlib import Path

from rich import print

CASP15_SEQ_URL = "https://predictioncenter.org/download_area/CASP15/sequences/casp15.seq.txt"

# (task_name, url, lmdb_dirname, splits) -- lmdb files live at
# {extract_dir}/{lmdb_dirname}/{lmdb_dirname}_{split}.lmdb, verified against
# each torchdrug/datasets/*.py source file.
PEER_LMDB_TASKS = [
    ("fluorescence", "http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/fluorescence.tar.gz",
     "fluorescence", ["train", "valid", "test"]),
    ("stability", "http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/stability.tar.gz",
     "stability", ["train", "valid", "test"]),
    ("solubility", "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/solubility.tar.gz",
     "solubility", ["train", "valid", "test"]),
    ("beta_lactamase", "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/beta_lactamase.tar.gz",
     "beta_lactamase", ["train", "valid", "test"]),
    ("subcellular_localization",
     "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/subcellular_localization.tar.gz",
     "subcellular_localization", ["train", "valid", "test"]),
    ("binary_localization",
     "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/subcellular_localization_2.tar.gz",
     "subcellular_localization_2", ["train", "valid", "test"]),
    ("fold", "http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/remote_homology.tar.gz",
     "remote_homology", ["train", "valid", "test_fold_holdout", "test_family_holdout", "test_superfamily_holdout"]),
    ("secondary_structure", "http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/secondary_structure.tar.gz",
     "secondary_structure", ["train", "valid", "casp12", "ts115", "cb513"]),
    ("contact", "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/data/proteinnet.tar.gz",
     "proteinnet", ["train", "valid", "test"]),
]

# (task_name, url, split_name) -- split_name matches each FLIP dataset class's
# own default in peer/flip.py (AAV="two_vs_many", GB1="two_vs_rest",
# Thermostability="human_cell"), so this reproduces exactly what PEER's own
# configs load (they don't override split).
PEER_FLIP_TASKS = [
    ("aav", "https://github.com/J-SNACKKB/FLIP/raw/d5c35cc716ca93c3c74a0b43eef5b60cbf88521f/splits/aav/splits.zip",
     "two_vs_many"),
    ("gb1", "https://github.com/J-SNACKKB/FLIP/raw/d5c35cc716ca93c3c74a0b43eef5b60cbf88521f/splits/gb1/splits.zip",
     "two_vs_rest"),
    ("thermostability",
     "https://github.com/J-SNACKKB/FLIP/raw/d5c35cc716ca93c3c74a0b43eef5b60cbf88521f/splits/meltome/splits.zip",
     "human_cell"),
]


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [dim]Downloading {url} -> {dest.name}[/]")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    return dest


def _extract(archive: Path, dest_dir: Path) -> Path:
    if dest_dir.exists() and any(dest_dir.iterdir()):
        return dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".tar.gz") or archive.suffix == ".tgz":
        with tarfile.open(archive) as tf:
            tf.extractall(dest_dir)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)
    else:
        raise ValueError(f"Don't know how to extract {archive}")
    return dest_dir


def _read_lmdb_sequences(lmdb_path: Path, sequence_field: str = "primary") -> list[str]:
    import lmdb
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False)
    sequences = []
    with env.begin(write=False) as txn:
        num = pickle.loads(txn.get(b"num_examples"))
        for i in range(num):
            item = pickle.loads(txn.get(str(i).encode()))
            sequences.append(item[sequence_field])
    env.close()
    return sequences


def _parse_fasta_records(path: Path) -> list[tuple[str, str]]:
    """(header_without_'>' , sequence) pairs, sequence lines concatenated."""
    records = []
    header, seq_lines = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_lines)))
                header, seq_lines = line[1:], []
            elif header is not None:
                seq_lines.append(line.strip())
        if header is not None:
            records.append((header, "".join(seq_lines)))
    return records


# ---------------------------------------------------------------------------
# ProteinGym
# ---------------------------------------------------------------------------

def fetch_proteingym_wildtypes(bench_dir: Path) -> list[str]:
    """One wildtype sequence per DMS assay (~87-217 unique proteins depending
    on ProteinGym version), NOT the full mutant-row set."""
    cache_path = bench_dir / "proteingym_wildtypes.csv"
    if cache_path.exists():
        print(f"  [dim]Loading cached {cache_path.name}[/]")
        with open(cache_path) as f:
            rows = list(csv.DictReader(f))
        return [r["target_seq"] for r in rows]

    from datasets import load_dataset
    print("  Fetching ProteinGym DMS_substitutions from HuggingFace (OATML-Markslab/ProteinGym_v1)...")
    # name="DMS_substitutions" (config-name lookup) 404s: the repo's README
    # YAML declares this config's files at glob "data/DMS_substitutions-*",
    # but the actual files live at "DMS_substitutions/train-*.parquet" (no
    # "data/" prefix) -- a mismatch on HF's side, not ours. Pointing
    # data_files at the real glob directly sidesteps the broken config
    # metadata entirely (verified: yields the expected 2,465,767 rows).
    ds = load_dataset("OATML-Markslab/ProteinGym_v1", data_files="DMS_substitutions/train-*.parquet",
                       split="train")

    wildtypes: dict[str, str] = {}
    for row in ds:
        wildtypes.setdefault(row["DMS_id"], row["target_seq"])

    bench_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["DMS_id", "target_seq"])
        for dms_id, seq in wildtypes.items():
            writer.writerow([dms_id, seq])
    print(f"  {len(wildtypes)} unique wildtype sequences cached to {cache_path}")
    return list(wildtypes.values())


# ---------------------------------------------------------------------------
# CASP15
# ---------------------------------------------------------------------------

def fetch_casp15_sequences(bench_dir: Path) -> list[str]:
    """Protein target sequences only (RNA targets, ID prefix 'R', excluded)."""
    raw_cache = bench_dir / "casp15.seq.txt"
    protein_cache = bench_dir / "casp15_protein_only.fasta"
    if protein_cache.exists():
        print(f"  [dim]Loading cached {protein_cache.name}[/]")
        return [seq for _, seq in _parse_fasta_records(protein_cache)]

    _download(CASP15_SEQ_URL, raw_cache)
    records = _parse_fasta_records(raw_cache)
    protein_records = [(hdr, seq) for hdr, seq in records if hdr.split()[0].startswith("T")]

    with open(protein_cache, "w") as f:
        for hdr, seq in protein_records:
            f.write(f">{hdr}\n{seq}\n")
    print(f"  {len(protein_records)} protein targets (of {len(records)} total, "
          f"{len(records) - len(protein_records)} RNA excluded) cached to {protein_cache}")
    return [seq for _, seq in protein_records]


# ---------------------------------------------------------------------------
# PEER
# ---------------------------------------------------------------------------

def _fetch_peer_lmdb_task(task_name: str, url: str, lmdb_dirname: str, splits: list[str],
                           peer_dir: Path) -> list[str]:
    archive = peer_dir / f"{task_name}.tar.gz"
    _download(url, archive)
    extract_dir = _extract(archive, peer_dir / task_name)
    sequences = []
    for split in splits:
        lmdb_path = extract_dir / lmdb_dirname / f"{lmdb_dirname}_{split}.lmdb"
        sequences.extend(_read_lmdb_sequences(lmdb_path))
    return sequences


def _fetch_peer_flip_task(task_name: str, url: str, split_name: str, peer_dir: Path) -> list[str]:
    archive = peer_dir / f"{task_name}_splits.zip"
    _download(url, archive)
    extract_dir = _extract(archive, peer_dir / task_name)
    csv_path = extract_dir / "splits" / f"{split_name}.csv"
    sequences = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequences.append(row["sequence"])
    return sequences


def fetch_peer_sequences(bench_dir: Path) -> list[str]:
    """Pooled, deduplicated sequences from every single-protein PEER task (see
    module docstring for exactly which tasks and why)."""
    peer_dir = bench_dir / "peer"
    cache_path = bench_dir / "peer_sequences.txt"
    if cache_path.exists():
        print(f"  [dim]Loading cached {cache_path.name}[/]")
        with open(cache_path) as f:
            return [line.rstrip("\n") for line in f]

    all_sequences: set[str] = set()
    for task_name, url, lmdb_dirname, splits in PEER_LMDB_TASKS:
        print(f"  PEER task '{task_name}' (LMDB)...")
        seqs = _fetch_peer_lmdb_task(task_name, url, lmdb_dirname, splits, peer_dir)
        print(f"    {len(seqs):,} sequences ({len(set(seqs)):,} unique)")
        all_sequences.update(seqs)
    for task_name, url, split_name in PEER_FLIP_TASKS:
        print(f"  PEER task '{task_name}' (FLIP CSV, split={split_name})...")
        seqs = _fetch_peer_flip_task(task_name, url, split_name, peer_dir)
        print(f"    {len(seqs):,} sequences ({len(set(seqs)):,} unique)")
        all_sequences.update(seqs)

    sequences = sorted(all_sequences)
    bench_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        for seq in sequences:
            f.write(seq + "\n")
    n_tasks = len(PEER_LMDB_TASKS) + len(PEER_FLIP_TASKS)
    print(f"  {len(sequences):,} unique pooled sequences across {n_tasks} PEER tasks, cached to {cache_path}")
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_path", type=Path,
                         help="Directory to cache downloaded/derived benchmark files under. Filenames are chosen by each fetcher.")
    parser.add_argument("--benchmark", choices=["proteingym", "peer", "casp15", "all"], default="all")

    args = parser.parse_args()

    fetchers = {
        "proteingym": ("ProteinGym", fetch_proteingym_wildtypes),
        "peer": ("PEER", fetch_peer_sequences),
        "casp15": ("CASP15", fetch_casp15_sequences),
    }
    targets = fetchers.items() if args.benchmark == "all" else [(args.benchmark, fetchers[args.benchmark])]

    for key, (label, fetch_fn) in targets:
        print(f"\n[bold][orange2]=== {label} ===[/][/]")
        sequences = fetch_fn(args.dataset_path)
        print(f"  [green]{key}: {len(sequences):,} sequences[/]")


if __name__ == "__main__":
    main()
