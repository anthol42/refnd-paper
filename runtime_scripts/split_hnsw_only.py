"""HNSW → Leiden → partition. Usage: uv run python -m runtime_scripts.split_hnsw_only <atlas|belka> <subset_file>"""
import sys
from pathlib import Path

from src.datasets import DATASETS, SCALING_DATASET_KEY, prepare_hnsw_input
from refnd.core import HNSWState

dataset_key = sys.argv[1]
subset_path = Path(sys.argv[2])
items = [line.strip() for line in subset_path.read_text().splitlines()
         if line.strip() and not line.startswith(">")]

cfg  = DATASETS[SCALING_DATASET_KEY[dataset_key]]
data = prepare_hnsw_input(dataset_key, items)
hnsw = HNSWState(cfg.modality, data, proximity_threshold=cfg.proximity_threshold, **cfg.kernel_params)
hnsw.build(progress=True)
