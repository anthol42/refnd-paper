"""HNSW → Leiden → partition. Usage: uv run python -m runtime_scripts.split_refnd <atlas|belka> <subset_file>"""
import sys
from pathlib import Path

from src.datasets import DATASETS, SCALING_DATASET_KEY, prepare_hnsw_input
from refnd.core import HNSWState, INWeightType, LeidenObjective, find_communities, partition
from refnd import KernelVariant

dataset_key = sys.argv[1]
subset_path = Path(sys.argv[2])
items = [line.strip() for line in subset_path.read_text().splitlines()
         if line.strip() and not line.startswith(">")]

cfg   = DATASETS[SCALING_DATASET_KEY[dataset_key]]
data  = prepare_hnsw_input(dataset_key, items)
print(f"Cache capacity: {0 if cfg.modality == KernelVariant.TanimotoBit else 2_000_000}")
hnsw  = HNSWState(cfg.modality, data, proximity_threshold=cfg.proximity_threshold, **cfg.kernel_params,
                  keep_all_edges=cfg.modality != KernelVariant.TanimotoBit, cache_capacity=0 if cfg.modality == KernelVariant.TanimotoBit else 2_000_000)
hnsw.build(progress=True)
es    = hnsw.edges()
graph = es.graph(inweight_type=INWeightType.Distance)
coms  = find_communities(graph, gamma=1.0, objective=LeidenObjective.Modularity)
partition(coms, graph, test_ratio=0.2, post_filtering=True)
