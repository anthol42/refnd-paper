from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from refnd.kernels import KernelVariant
from refnd.kernels.alignments import ScoringMatrix, CoverageMode, LocalIdentityMode

from .cache import CacheStore


@dataclass
class DatasetConfig:
    modality: KernelVariant
    metric: str | None          # "pcc" | "mcc" | None (no supervised task)
    encoder: str | None         # HuggingFace model id | None (no embeddings)
    proximity_threshold: float
    kernel_params: dict         # forwarded to kernel constructor as **kwargs


DATASETS: dict[str, DatasetConfig] = {
    "dbaasp": DatasetConfig(
        modality=KernelVariant.AlignmentGlobal,
        metric="pcc",
        encoder="EvolutionaryScale/esmc-300m",
        proximity_threshold=0.5,
        kernel_params={"matrix": ScoringMatrix.Blosum62},
    ),
    "ld50_zhu": DatasetConfig(
        modality=KernelVariant.TanimotoBit,
        metric="pcc",
        encoder="seyonec/ChemBERTa-zinc-base-v1",
        proximity_threshold=0.4,
        kernel_params={},
    ),
    "prom_core_all": DatasetConfig(
        modality=KernelVariant.AlignmentLocal,
        metric="mcc",
        encoder="zhihan1996/DNABERT-2-117M",
        proximity_threshold=0.35,
        kernel_params={"matrix": ScoringMatrix.Dnafull,
                       "identity_mode": LocalIdentityMode.MinSeqLength,
                       "cov_mode": CoverageMode.ShorterSeq,
                       "min_coverage": 0.7},
    ),
    "belka": DatasetConfig(
        modality=KernelVariant.TanimotoBit,
        metric="mcc-multilabel",
        encoder="seyonec/ChemBERTa-zinc-base-v1",
        proximity_threshold=0.1,
        kernel_params={},
    ),
    "peptide_atlas": DatasetConfig(
        modality=KernelVariant.AlignmentGlobal,
        metric=None,
        encoder=None,
        proximity_threshold=0.5,
        kernel_params={"matrix": ScoringMatrix.Blosum62},
    ),
}


# Maps the scaling-benchmark's --dataset CLI key to a DATASETS config entry.
SCALING_DATASET_KEY: dict[str, str] = {
    "atlas": "peptide_atlas",
    "belka": "belka",
}


def prepare_hnsw_input(dataset_key: str, items: list[str]) -> list[Any]:
    """Convert raw subset items (sequences or SMILES) into what HNSWState expects.

    For "belka" this computes Morgan fingerprints in parallel across all cores —
    intentionally timed as part of the caller's run, since Hestia's own
    `molecular_similarity` incurs the same fingerprinting cost internally.
    """
    if dataset_key == "belka":
        from refnd.utils import BitFingerprint
        from .fingerprints import compute_fingerprints

        fps = compute_fingerprints(items)
        missing = sum(1 for fp in fps if fp is None)
        if missing:
            raise ValueError(f"{missing} SMILES in subset could not be parsed by RDKit")
        return [BitFingerprint.from_np(fp) for fp in fps]
    return items


def load_dataset(name: str, cache: CacheStore) -> tuple[list[Any], np.ndarray]:
    cached = cache.get_dataset(name)
    if cached is not None:
        data, labels = cached
        if name in ("ld50_zhu", "belka"):
            from refnd.utils import BitFingerprint
            data = [BitFingerprint.from_np(row) for row in data]
        return data, labels

    if name == "peptide_atlas":
        data, labels = _load_peptide_atlas()
        cache.store_dataset(name, data, labels)
        return data, labels
    elif name == "dbaasp":
        data, labels = _load_dbaasp()
    elif name == "ld50_zhu":
        fp_arrays, labels, smiles = _load_ld50_zhu()
        cache.store_dataset(name, fp_arrays, labels)
        cache.store_dataset(f"{name}_smiles", smiles, np.array([]))
        from refnd.utils import BitFingerprint
        return [BitFingerprint.from_np(row) for row in fp_arrays], labels
    elif name == "belka":
        fp_arrays, labels, smiles = _load_belka()
        cache.store_dataset(name, fp_arrays, labels)
        cache.store_dataset(f"{name}_smiles", smiles, np.array([]))
        from refnd.utils import BitFingerprint
        return [BitFingerprint.from_np(row) for row in fp_arrays], labels
    elif name == "prom_core_all":
        data, labels = _load_prom_core_all()
    else:
        raise ValueError(f"Unknown dataset: {name!r}")

    cache.store_dataset(name, data, labels)
    return data, labels  # type: ignore[return-value]


def _load_peptide_atlas() -> tuple[list[str], np.ndarray]:
    fasta_path = Path(".cache/peptide_atlas.fasta")
    if not fasta_path.exists():
        import re
        import sys
        from glob import glob
        from warnings import warn
        import requests
        from bs4 import BeautifulSoup
        from tqdm import tqdm

        cache_files = Path(".cache/files")
        cache_files.mkdir(parents=True, exist_ok=True)

        print("Fetching PeptideAtlas build list...")
        resp = requests.get("https://peptideatlas.org/builds/", timeout=60)
        resp.raise_for_status()
        soup  = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"id": "bdtable"})
        if table is None:
            raise ValueError("PeptideAtlas build table not found")

        thead      = table.find("thead")
        header_row = (thead or table).find("tr")
        headers    = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]
        tbody      = table.find("tbody")
        data_rows  = (tbody or table).find_all("tr")
        if not thead:
            data_rows = data_rows[1:]

        rows = []
        for row in data_rows:
            cells    = row.find_all(["td", "th"])
            row_data = []
            for cell in cells:
                link = cell.find("a")
                if link and link.get("href") and link.get_text(strip=True).endswith(".fasta"):
                    row_data.append(f"[{link.get_text(strip=True)}]({link['href']})")
                else:
                    row_data.append(cell.get_text(strip=True))
            if row_data:
                rows.append(row_data)

        import pandas as pd
        max_cols = len(headers)
        rows     = [r[:max_cols] + [""] * (max_cols - len(r)) for r in rows]
        df       = pd.DataFrame(rows, columns=headers)
        df       = df.loc[df["Build Name"] != ""]

        entries = [s for s in df["Peptide Sequences"].tolist() if s]
        parsed  = [re.findall(r"(\[.*?])(\(.*?\))", s)[0] for s in entries]
        names   = [e[0][1:-1] for e in parsed]
        urls    = [e[1][1:-1] for e in parsed]

        print(f"Downloading {len(names)} FASTA files...")
        for name, url in tqdm(zip(names, urls), total=len(names)):
            local = cache_files / name
            if local.exists():
                continue
            r = requests.get(f"https://peptideatlas.org/builds/{url}", timeout=120)
            if not r.ok:
                warn(Warning(f"Failed to fetch {name} (HTTP {r.status_code})"))
                continue
            local.write_text(r.text)

        print("Merging and deduplicating sequences...")
        all_sequences: set[str] = set()
        for file in glob(str(cache_files / "*.fasta")):
            with open(file) as f:
                current_id = None
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        current_id = line[1:]
                    elif current_id is not None and line:
                        all_sequences.add(line)
        all_sequences = {seq for seq in all_sequences if len(seq) <= 100}
        with open(fasta_path, "w") as f:
            for i, seq in enumerate(all_sequences):
                f.write(f">seq_{i}\n{seq}\n")
        print(f"  Wrote {len(all_sequences):,} sequences to {fasta_path}")

    sequences = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith(">") and line:
                sequences.append(line)
    return sequences, np.zeros(len(sequences), dtype=np.float32)


def _load_dbaasp() -> tuple[list[str], np.ndarray]:
    from qmap import DBAASPDataset
    import pandas as pd

    print("Downloading DBAASP from HuggingFace...")
    ds = (
        DBAASPDataset()
        .with_l_aa_only()
        .with_canonical_only()
    )
    df = ds.tabular(["sequence", "Escherichia coli"])
    df = df.dropna(subset=["Escherichia coli"])
    # MIC in µg/mL — log-transform for regression
    sequences = df["sequence"].tolist()
    labels = np.log10(df["Escherichia coli"].astype(float).values)
    return sequences, labels


def _load_ld50_zhu() -> tuple[Any, np.ndarray, list[str]]:
    import io
    import requests
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from refnd.utils import BitFingerprint

    url = "https://huggingface.co/datasets/scikit-fingerprints/TDC_ld50_zhu/resolve/main/tdc_ld50_zhu.csv"
    print("Downloading LD50 (Zhu) dataset...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    import pandas as pd
    df = pd.read_csv(io.StringIO(resp.text))
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp_arrays, labels, smiles = [], [], []
    for _, row in df.iterrows():
        y = float(row["Y"])
        if y <= 0:
            continue
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            continue
        rdkit_fp = morgan_gen.GetFingerprint(mol)
        fp_arrays.append(np.array(rdkit_fp, dtype=bool))
        labels.append(y)
        smiles.append(row["SMILES"])

    # Store as numpy array so it's picklable; reconstruct BitFingerprint on load
    return np.array(fp_arrays, dtype=bool), np.log10(np.array(labels, dtype=np.float32)), smiles


_BELKA_PROTEINS = ["BRD4", "HSA", "sEH"]  # fixed bit order for the 3-bit multi-label vector


def belka_download() -> Path:
    """Download (or reuse cached) BELKA competition files via kagglehub, return train file path."""
    import os

    import kagglehub
    from dotenv import load_dotenv
    from kagglehub.config import get_kaggle_credentials

    load_dotenv()
    if get_kaggle_credentials() is None:
        raise RuntimeError(
            "No Kaggle credentials found. Set KAGGLE_API_TOKEN, or KAGGLE_USERNAME "
            "and KAGGLE_KEY, in a .env file "
            "(see https://www.kaggle.com/settings -> API -> Create New Token)."
        )

    print("Downloading BELKA competition data via kagglehub...")
    path = Path(kagglehub.competition_download("leash-BELKA"))

    train_path = path / "train.parquet"
    if not train_path.exists():
        train_path = path / "train.csv"
    return train_path


def belka_unique_smiles(cache: "CacheStore | None" = None) -> list[str]:
    """Unique molecule SMILES only (no fingerprints/labels) — cheap, used for scaling subsets."""
    if cache is not None:
        cached = cache.get_dataset("belka_unique_smiles")
        if cached is not None:
            return cached[0]

    import pandas as pd

    train_path = belka_download()
    print(f"  Loading {train_path} (molecule_smiles only)...")
    df = pd.read_parquet(train_path, columns=["molecule_smiles"]) \
        if train_path.suffix == ".parquet" \
        else pd.read_csv(train_path, usecols=["molecule_smiles"])
    smiles = df["molecule_smiles"].drop_duplicates().tolist()
    print(f"  {len(smiles):,} unique molecules")

    if cache is not None:
        cache.store_dataset("belka_unique_smiles", smiles, np.array([]))
    return smiles


def belka_test_unique_smiles(cache: "CacheStore | None" = None) -> list[str]:
    """Kaggle BELKA competition's held-out TEST set, unique molecule SMILES only
    (no fingerprints/labels). A genuine production set for the train/production
    threshold theory: Leash Bio built it from building-block combinations not
    present in train, by design."""
    if cache is not None:
        cached = cache.get_dataset("belka_test_unique_smiles")
        if cached is not None:
            return cached[0]

    import pandas as pd

    train_path = belka_download()
    test_path = train_path.parent / "test.parquet"
    if not test_path.exists():
        test_path = train_path.parent / "test.csv"
    print(f"  Loading {test_path} (molecule_smiles only)...")
    df = pd.read_parquet(test_path, columns=["molecule_smiles"]) \
        if test_path.suffix == ".parquet" \
        else pd.read_csv(test_path, usecols=["molecule_smiles"])
    smiles = df["molecule_smiles"].drop_duplicates().tolist()
    print(f"  {len(smiles):,} unique test molecules")

    if cache is not None:
        cache.store_dataset("belka_test_unique_smiles", smiles, np.array([]))
    return smiles


def _load_belka() -> tuple[np.ndarray, np.ndarray, list[str]]:
    import os

    import pandas as pd

    from .fingerprints import compute_fingerprints

    train_path = belka_download()
    print(f"  Loading {train_path}...")
    df = pd.read_parquet(train_path, columns=["molecule_smiles", "protein_name", "binds"]) \
        if train_path.suffix == ".parquet" \
        else pd.read_csv(train_path, usecols=["molecule_smiles", "protein_name", "binds"])

    print("Pivoting to one row per unique molecule with a 3-bit protein-binder label...")
    pivot = df.pivot_table(
        index="molecule_smiles", columns="protein_name", values="binds", aggfunc="max"
    )
    missing = pivot[_BELKA_PROTEINS].isna().any(axis=1).sum()
    if missing:
        print(f"  [warn] {missing:,} molecules missing an assay for >=1 protein; "
              f"filling missing bits with 0")
    pivot = pivot.reindex(columns=_BELKA_PROTEINS).fillna(0).astype(np.int64)

    smiles_list = pivot.index.tolist()
    labels = pivot.to_numpy()  # shape (n, 3), one column per protein

    print(f"  {len(smiles_list):,} unique molecules; computing Morgan fingerprints "
          f"across {os.cpu_count()} cores...")
    fp_arrays = compute_fingerprints(smiles_list, progress=True)

    keep = [i for i, fp in enumerate(fp_arrays) if fp is not None]
    if len(keep) < len(fp_arrays):
        print(f"  Dropping {len(fp_arrays) - len(keep):,} molecules RDKit couldn't parse")

    fps    = np.stack([fp_arrays[i] for i in keep])
    labels = labels[keep]
    smiles = [smiles_list[i] for i in keep]
    return fps, labels, smiles


def mnist_download(cache: "CacheStore | None" = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """MNIST train/test images as flattened float32 vectors in [0, 1], shape
    (N, 784) -- ready for the Cosine kernel. Downloaded via torchvision;
    additionally cached as plain arrays through `cache` so a cache hit needs
    neither torchvision nor a re-parse of the raw files.

    Returns (train_vectors, train_labels, test_vectors, test_labels).
    """
    if cache is not None:
        cached_train = cache.get_dataset("mnist_train")
        cached_test = cache.get_dataset("mnist_test")
        if cached_train is not None and cached_test is not None:
            return cached_train[0], cached_train[1], cached_test[0], cached_test[1]

    from torchvision.datasets import MNIST

    root = str((cache.root if cache is not None else Path(".cache")) / "mnist_raw")
    print("Downloading MNIST via torchvision...")
    train_ds = MNIST(root=root, train=True, download=True)
    test_ds = MNIST(root=root, train=False, download=True)

    train_vectors = train_ds.data.numpy().reshape(len(train_ds), -1).astype(np.float32) / 255.0
    train_labels = train_ds.targets.numpy().astype(np.int64)
    test_vectors = test_ds.data.numpy().reshape(len(test_ds), -1).astype(np.float32) / 255.0
    test_labels = test_ds.targets.numpy().astype(np.int64)

    if cache is not None:
        cache.store_dataset("mnist_train", train_vectors, train_labels)
        cache.store_dataset("mnist_test", test_vectors, test_labels)
    return train_vectors, train_labels, test_vectors, test_labels


def tiny_imagenet_download() -> Path:
    """Download (or reuse cached) the akash2sharma/tiny-imagenet Kaggle
    dataset (kaggle.com/datasets/akash2sharma/tiny-imagenet) -- a mirror of
    Stanford's tiny-imagenet-200 (64x64 JPEGs, 200 classes). Used only as a
    real-image null-model population for thresholdv2/mnist.py, downsized +
    grayscaled to look structurally like MNIST.
    """
    import kagglehub
    from dotenv import load_dotenv
    from kagglehub.config import get_kaggle_credentials

    load_dotenv()
    if get_kaggle_credentials() is None:
        raise RuntimeError(
            "No Kaggle credentials found. Set KAGGLE_API_TOKEN, or KAGGLE_USERNAME "
            "and KAGGLE_KEY, in a .env file "
            "(see https://www.kaggle.com/settings -> API -> Create New Token)."
        )

    print("Downloading tiny-imagenet (akash2sharma/tiny-imagenet) via kagglehub...")
    return Path(kagglehub.dataset_download("akash2sharma/tiny-imagenet"))


def tiny_imagenet_grayscale_vectors(
    cache: "CacheStore | None" = None, n: int = 100_000, size: int = 28, seed: int = 42,
) -> np.ndarray:
    """Random sample of `n` tiny-imagenet TRAIN images (the only split with
    per-image class labels, needed so callers can build cross-class null
    pairs), grayscaled and resized to (size, size) -- 28x28 by default, to
    match MNIST -- flattened to float32 vectors in [0, 1]. Cached as a plain
    (vectors, labels) pair, keyed by (size, n, seed).

    Returns (vectors, labels): labels are integer-encoded wnid class ids
    (folder names under .../tiny-imagenet-200/train/), same length as
    vectors.
    """
    cache_name = f"tiny_imagenet_gray{size}_n{n}_seed{seed}"
    if cache is not None:
        cached = cache.get_dataset(cache_name)
        if cached is not None:
            return cached[0], cached[1]

    from PIL import Image

    root = tiny_imagenet_download()
    train_dir = root / "tiny-imagenet-200" / "train"
    if not train_dir.exists():
        # some kagglehub layouts nest an extra tiny-imagenet-200/ level
        candidates = list(root.rglob("train"))
        train_dir = next((d for d in candidates if (d.parent / "val").exists()), candidates[0])
    paths = sorted(train_dir.glob("*/images/*.JPEG"))
    if not paths:
        raise RuntimeError(f"No labeled train images found under {train_dir}")
    wnids = [p.parent.parent.name for p in paths]
    class_to_id = {w: i for i, w in enumerate(sorted(set(wnids)))}
    all_labels = np.array([class_to_id[w] for w in wnids], dtype=np.int64)
    print(f"  Found {len(paths):,} labeled tiny-imagenet train images ({len(class_to_id)} classes); sampling {n:,}...")

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=min(n, len(paths)), replace=False)

    vectors = np.empty((len(idx), size * size), dtype=np.float32)
    for i, j in enumerate(idx):
        img = Image.open(paths[j]).convert("L").resize((size, size))
        vectors[i] = np.asarray(img, dtype=np.float32).reshape(-1) / 255.0
    labels = all_labels[idx]

    if cache is not None:
        cache.store_dataset(cache_name, vectors, labels)
    return vectors, labels


def _load_prom_core_all() -> tuple[list[str], np.ndarray]:
    import io
    import requests
    import pandas as pd

    url = "https://huggingface.co/datasets/leannmlindsey/GUE/resolve/main/GUE/prom_core_all/train.csv"
    print("Downloading GUE prom_core_all dataset...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    return df["sequence"].tolist(), np.array(df["label"].values, dtype=np.int64)
