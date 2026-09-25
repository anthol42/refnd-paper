"""Compare split strategies on the synthetic forest, at a fixed threshold TAU.

For one annotated dataset (split into train/val/test) and one *production*
dataset -- an independent draw of the same generative process, so every family in
it is unseen -- each split method is scored on:

  leakage:    % of test samples whose nearest train neighbour is closer than TAU
  test:       MLP score on the annotated test split
  production: the SAME trained model, scored on the production dataset

The validation set (early stopping) is a random 10% of train, as in the
real-data benchmark (comparison/scripts); DataSAIL re-runs its own split.

Datasets have a fixed size and family count (SyntheticConfig), and labels carry
no family effect: label = signal(sequence) + noise. Each of the N_REPEATS repeats draws a FRESH
annotated and production pair, so the spread over repeats reflects dataset
noise as well as split noise.

A random split should show high leakage, a high test score and a large
test-production gap, because it lets the model pick up near-duplicate signal it
cannot transfer. Similarity-aware splits (refnd, hestia, datasail) should show
little leakage and a test score close to production.

Every run is appended to RESULTS_PATH as one record, and the file is rewritten
after each repeat so partial progress survives an interruption.
A method that fails to split (e.g. DataSAIL without its Apptainer image) is
recorded with None scores and its error, and the sweep carries on.

Usage:
    uv run python -m synthetic.compare_splits
"""

from __future__ import annotations

import dataclasses
import json
import traceback
from pathlib import Path

import numpy as np
import torch
from rich import print

from refnd.core import (HNSWState, INWeightType, LeidenObjective, find_communities,
                        partition)

from src.cache import CacheStore
from src.datasets import DATASETS, load_dataset
from src.metrics import null_model
from src.mlp import train_eval_mlp
from synthetic.dataset import SyntheticConfig, build_dataset
from synthetic.generation import atlas_profile

# ─────────────────────────────── experiment knobs ─────────────────────────────

TAU = 0.5                              # proximity threshold, from synthetic.find_threshold
SIGMA = 0.05
METHODS = ["random", "refnd", "hestia", "datasail", "family_oracle"]
N_REPEATS = 100                        # each repeat draws a fresh annotated + production pair
N_NULL_PAIRS = 10_000_000
DATASAIL_SIF = None                    # else $DATASAIL_SIF or $REFND_EXP_BASE/datasail.sif

RESULTS_PATH = Path(__file__).parent.parent / "results" / "synthetic" / f"compare_splits_tau{TAU:g}.json"
DATASET_KEY = "peptide_atlas"          # shares the peptide kernel/modality
TEST_RATIO, VAL_RATIO = 0.20, 0.10      # as comparison/scripts (VAL_RATIO of train)
METRIC = "pcc"
LEARNING_RATE = 1e-3                   # as comparison/scripts/deep_learning/common.py
EMBED_BATCH = 64


def _embed_esmc(sequences: list[str], device: str) -> torch.Tensor:
    """ESM-C 300M, mean-pooled over ALL tokens (BOS/EOS included, padding excluded).

    Matches comparison/scripts/compute_embeddings.sh, which pools
    `model.logits(...).embeddings.mean(dim=0)` per sequence. Batched here; the
    padding mask makes it equal to that unbatched per-sequence mean.
    """
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein

    model = ESMC.from_pretrained("esmc_300m").to(device).eval()
    pad_id = model.tokenizer.pad_token_id
    pooled: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(sequences), EMBED_BATCH):
            batch = sequences[start:start + EMBED_BATCH]
            tokens = torch.nn.utils.rnn.pad_sequence(
                [model.encode(ESMProtein(sequence=seq)).sequence for seq in batch],
                batch_first=True, padding_value=pad_id).to(device)
            mask = (tokens != pad_id).unsqueeze(-1).float()
            embeddings = model(tokens).embeddings.float()
            pooled.append(((embeddings * mask).sum(1) / mask.sum(1)).cpu())
    return torch.cat(pooled).to(torch.float32)


def embed(sequences: list[str], cache: CacheStore, cache_key: str) -> torch.Tensor:
    cached = cache.get_embs(cache_key)
    if cached is not None:
        return cached
    embeddings = _embed_esmc(sequences, "cuda" if torch.cuda.is_available() else "cpu")
    cache.store_embs(cache_key, embeddings)
    return embeddings


# ─────────────────────────────── split methods ───────────────────────────────
# Each method returns (train_idx, test_idx) over the annotated dataset. Shared,
# seed-independent structures come in through `context` (see build_context).

def random_split(n_samples: int, seed: int, ratio: float = TEST_RATIO,
                 **_) -> tuple[list[int], list[int]]:
    """Train/test split ignoring similarity entirely."""
    order = np.random.default_rng(seed).permutation(n_samples)
    n_test = int(ratio * n_samples)
    return order[n_test:].tolist(), order[:n_test].tolist()


def refnd_split(n_samples: int, seed: int, ratio: float = TEST_RATIO,
                communities=None, graph=None, **_) -> tuple[list[int], list[int]]:
    """Community-based partition of the HNSW graph (CPM Leiden, null-model gamma)."""
    train_idx, test_idx = partition(communities, graph, test_ratio=ratio,
                                    seed=seed, post_filtering=False)
    return list(train_idx), list(test_idx)


def hestia_split(n_samples: int, seed: int, ratio: float = TEST_RATIO,
                 sequences=None, sim_df=None, **_) -> tuple[list[int], list[int]]:
    """Hestia ccpart_random: whole connected components assigned to one side.

    Hestia thresholds on *identity*, refnd on distance, so the identity
    threshold matching TAU is 1 - TAU.
    """
    import pandas as pd
    from hestia.partition import ccpart_random

    frame = pd.DataFrame({"sequence": sequences})
    train_idx, test_idx, _clusters = ccpart_random(
        frame, label_name=None, test_size=ratio, threshold=1.0 - TAU,
        n_bins=10, sim_df=sim_df, seed=seed, verbose=0)
    return [int(i) for i in train_idx], [int(i) for i in test_idx]


def datasail_split(n_samples: int, seed: int, ratio: float = TEST_RATIO,
                   sequences=None, **_) -> tuple[list[int], list[int]]:
    """DataSAIL C1e (cold-cluster) split, run inside its Apptainer image.

    DataSAIL clusters with its own mmseqs similarity and has no distance
    threshold, so it does not use TAU.
    """
    from synthetic.datasail import datasail_split as run_datasail, find_sif
    return run_datasail(sequences, seed, find_sif(DATASAIL_SIF))


def family_oracle_split(n_samples: int, seed: int, ratio: float = TEST_RATIO,
                        families=None, **_) -> tuple[list[int], list[int]]:
    """Oracle: whole TRUE generated families go to test until TEST_RATIO is reached.
    The upper bound on what a similarity-aware splitter can achieve."""
    test_families, n_test = [], 0
    for family in np.random.default_rng(seed).permutation(np.unique(families)):
        if n_test >= ratio * n_samples:
            break
        test_families.append(family)
        n_test += int((families == family).sum())
    in_test = np.isin(families, test_families)
    return np.nonzero(~in_test)[0].tolist(), np.nonzero(in_test)[0].tolist()


SPLIT_METHODS = {"random": random_split, "refnd": refnd_split,
                 "hestia": hestia_split, "datasail": datasail_split,
                 "family_oracle": family_oracle_split}


def build_context(method: str, sequences: list[str], families: np.ndarray,
                  gamma: float) -> dict:
    """Seed-independent structures a method needs, built once per dataset."""
    context = {"sequences": sequences, "families": families}
    if method == "refnd":
        cfg = DATASETS[DATASET_KEY]
        hnsw = HNSWState(cfg.modality, sequences, proximity_threshold=TAU,
                         **cfg.kernel_params)
        hnsw.build(progress=True)
        graph = hnsw.edges().graph(inweight_type=INWeightType.Distance)
        communities = find_communities(graph, gamma=gamma, objective=LeidenObjective.CPM)
        print(f"    refnd: {len(set(communities)):,} communities")
        context.update(graph=graph, communities=communities)
        # How well the communities recover the true families (annotated dataset).
        from sklearn.metrics import adjusted_rand_score
        context["ari"] = float(adjusted_rand_score(families, communities))
        print(f"    refnd: ARI vs true families = {context['ari']:.3f}")
    elif method == "hestia":
        import pandas as pd
        from hestia.dataset_generator import HestiaGenerator, SimArguments
        generator = HestiaGenerator(pd.DataFrame({"sequence": sequences}), verbose=False)
        # denominator="longest" normalises identity by the longer sequence, as
        # refnd's kernel does (GlobalIdentityMode.MaxLength). Hestia's default
        # normalises by the aligned length, so a short local match between two
        # unrelated peptides scores ~0.7 and chains every family into a single
        # connected component, leaving nothing to split.
        context["sim_df"] = generator.calculate_similarity(
            SimArguments(data_type="sequence", field_name="sequence",
                         min_threshold=1.0 - TAU, denominator="longest"))
    elif method == "datasail":
        from synthetic.datasail import find_runtime, find_sif
        find_runtime()                     # fail fast, before any split is attempted
        find_sif(DATASAIL_SIF)
    return context


# ──────────────────────────────── evaluation ─────────────────────────────────

def nearest_train_distance(train_sequences: list[str], query_sequences: list[str]) -> np.ndarray:
    """Distance from each query to its nearest train sequence (exact, brute force).

    Exact like comparison/scripts/deep_learning/compute_leakage.py: an approximate
    search can miss the true nearest neighbour and so under-report leakage.
    """
    from refnd import exact_nearest_neighbors
    cfg = DATASETS[DATASET_KEY]
    hits = exact_nearest_neighbors(cfg.modality, query_sequences, train_sequences, 1,
                                   threads=0, progress=False, **cfg.kernel_params)
    return np.array([hit[0][1] for hit in hits], dtype=float)


def split_val(method: str, train_idx: list[int], seed: int,
              context: dict) -> tuple[list[int], list[int]]:
    """Carve a validation set out of train. Returns (inner_train_idx, val_idx).

    Random VAL_RATIO of train, as comparison/scripts/split_utils.split_train_val
    does for the real-data benchmark. DataSAIL is the exception there too: it
    re-runs its own C1e split on the train rows.
    """
    if method == "datasail":
        sequences = [context["sequences"][i] for i in train_idx]
        inner_local, val_local = datasail_split(len(train_idx), seed, sequences=sequences)
        return [train_idx[i] for i in inner_local], [train_idx[i] for i in val_local]
    order = np.array(train_idx)
    np.random.default_rng(seed).shuffle(order)
    n_val = max(1, int(len(order) * VAL_RATIO))
    return order[n_val:].tolist(), order[:n_val].tolist()


def pcc(predictions: np.ndarray, truth: np.ndarray) -> float:
    from scipy.stats import pearsonr
    return float(pearsonr(predictions, truth)[0])


def score_model(embeddings: torch.Tensor, labels: np.ndarray, inner_train_idx: list[int],
                val_idx: list[int], test_idx: list[int], production_idx: list[int],
                seed: int) -> dict:
    """Test and production scores of one model, from a SINGLE fit.

    train_eval_mlp is given test_idx + production_idx as one evaluation set and
    returns per-sample predictions, which are sliced back apart here. (Fitting
    twice -- once per set -- would retrain the identical model, since training
    depends only on train_idx, val_idx and seed.)
    """
    result = train_eval_mlp(embeddings, labels, inner_train_idx, val_idx,
                            list(test_idx) + list(production_idx), metric=METRIC,
                            lr=LEARNING_RATE, seed=seed)
    predictions = np.asarray(result["predictions"])
    return {"test_pcc": pcc(predictions[:len(test_idx)], labels[test_idx]),
            "production_pcc": pcc(predictions[len(test_idx):], labels[production_idx])}


def failed_record(error: str) -> dict:
    return {"leakage[%]": None, "median_nn_distance": None, "n_train": None,
            "n_test": None, "test_pcc": None, "production_pcc": None, "error": error}


def save(records: list[dict], distances: list[dict]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config = SyntheticConfig()
    settings = {"tau": TAU, "n_peptides": config.n_peptides,
                "n_families": config.n_families, "sigma": SIGMA,
                "methods": METHODS, "n_repeats": N_REPEATS}
    RESULTS_PATH.write_text(json.dumps({
        "settings": settings,
        "records": records,            # one per (method, repeat)
        "nn_distances": distances,     # test -> nearest train, per (method, repeat)
    }))


def dataset_seeds(repeat: int) -> tuple[int, int]:
    """Generator seeds for one repeat's annotated and production datasets.

    Every repeat draws its own pair, so the spread across repeats includes the
    variation between datasets, not only between splits of one dataset.
    """
    return 2 * repeat, 2 * repeat + 1


# ──────────────────────────────────── main ───────────────────────────────────

def run_repeat(repeat: int, gamma: float, profile, cache: CacheStore,
               records: list[dict], distances: list[dict]) -> None:
    """One repeat: fresh annotated + production datasets, every method scored on them."""
    annotated_seed, production_seed = dataset_seeds(repeat)
    config = SyntheticConfig(sigma=SIGMA)
    annotated = build_dataset(config, seed=annotated_seed, profile=profile)
    production = build_dataset(config, seed=production_seed, profile=profile)
    print(f"[bold]  -- repeat {repeat} (datasets {annotated_seed}/{production_seed}) --[/]")

    name = f"synthetic_{config.n_peptides}_{config.n_families}"
    embeddings = torch.cat([
        embed(annotated.sequences, cache, f"{name}_annotated{annotated_seed}"),
        embed(production.sequences, cache, f"{name}_production{production_seed}"),
    ])
    production_idx = list(range(len(annotated), len(annotated) + len(production)))
    labels = np.concatenate([annotated.labels, production.labels])

    for method in METHODS:
        base = {"method": method, "repeat": repeat,
                "annotated_seed": annotated_seed, "production_seed": production_seed}
        try:
            context = build_context(method, annotated.sequences, annotated.families, gamma)
            train_idx, test_idx = SPLIT_METHODS[method](len(annotated), repeat, **context)
            if min(len(train_idx), len(test_idx)) < 2:
                raise ValueError(f"degenerate split: train={len(train_idx)} "
                                 f"test={len(test_idx)}")
            inner_train_idx, val_idx = split_val(method, train_idx, repeat, context)
            if min(len(inner_train_idx), len(val_idx)) < 2:
                raise ValueError(f"degenerate val split: train={len(inner_train_idx)} "
                                 f"val={len(val_idx)}")
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            print(f"    [red]{method} failed: {message}[/]")
            records.append({**base, **failed_record(message)})
            continue

        distances_to_train = nearest_train_distance(
            [annotated.sequences[i] for i in train_idx],
            [annotated.sequences[i] for i in test_idx])
        distances.append({**base, "values": np.round(distances_to_train, 4).tolist()})
        split_stats = {"n_train": len(train_idx), "n_test": len(test_idx),
                       "leakage[%]": 100.0 * float((distances_to_train < TAU).mean()),
                       "median_nn_distance": float(np.median(distances_to_train))}

        scores = score_model(embeddings, labels, inner_train_idx, val_idx,
                             test_idx, production_idx, repeat)
        records.append({**base, **split_stats, **scores,
                        "ari": context.get("ari"), "error": None})
        print(f"    {method:<14} leakage={split_stats['leakage[%]']:5.1f}%  "
              f"test_pcc={scores['test_pcc']:.3f}  prod_pcc={scores['production_pcc']:.3f}")


def main() -> None:
    cache = CacheStore()
    profile = atlas_profile(cache, len_range=SyntheticConfig().len_range)

    print("[bold][orange2]=== Split comparison on the synthetic forest ===[/][/]")
    atlas_sequences, _ = load_dataset(DATASET_KEY, cache)
    null_cfg = dataclasses.replace(DATASETS[DATASET_KEY], proximity_threshold=TAU)
    gamma = null_model(atlas_sequences, null_cfg, n_samples=N_NULL_PAIRS)
    print(f"  gamma(tau={TAU}) = {gamma:.3e}")

    records: list[dict] = []
    distances: list[dict] = []
    for repeat in range(N_REPEATS):
        run_repeat(repeat, gamma, profile, cache, records, distances)
        save(records, distances)
    print(f"  Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
