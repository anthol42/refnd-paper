# Comparison benchmark

Outputs **and full pipeline** for the graph-based dataset-splitting comparison
(Refnd vs Hestia / MMseqs2 / DataSAIL / Random) across molecule, protein, and
DNA datasets. Generated from `refnd_exp/` (refnd 0.0.3, CPM + null-model splits,
no post-filtering, no cross-method subsampling). All results are from the same
run (training `summary.json` files and leakage arrays are mutually consistent).

`scripts/` is a **verbatim mirror** of the cluster experiment code, so every
result here is reproducible end-to-end (data → embeddings → splits → training →
leakage → figures).

## Layout

```
results/                            committed outputs (analysis reads these)
  molecule/{dataset}/summary.json   test/train AUROC|PCC per method x head(linear,mlp),
  dna/{dataset}/summary.json          mean/std/n=10 + full per-seed values[] + gap
  protein/{dbaasp_amp,enzyme_topt}/summary.json
  leakage/{dtype}/{dataset}/{method}_seed1.npy
                                    1-D float32: each test sample's MAX identity
                                    (Tanimoto / seq-identity) to the train set.
  split_metrics/{method}/*.json     per-split wall-time + peak RSS (+ *_communities.json)
  split_metrics.csv                 aggregated split cost (time, peak_rss_mb)
  community_stats.csv               per-split community/component counts
  wilcoxon_table.json               paired Wilcoxon refnd-vs-baselines
  figures/                          per-dataset overfitting plots (test vs train)
  figures_leakage/                  per-dataset max-identity histograms
scripts/                            VERBATIM cluster pipeline (see below)
  split_utils.py                    shared split helpers (save_split, community stats, ...)
  refnd_split/    run_{protein,molecule,dna}.py + null_model.py + submit*.sh
  hestia_split/   run_{protein,molecule,dna}.py + submit*.sh
  mmseqs_split/   run_protein.py + submit.sh
  random_split/   run_all.py + submit.sh
  datasail_split/ run_datasail.py + submit.sh
  deep_learning/  train_{protein,molecule,dna}.py, common.py, aggregate_*.py,
                  compute_leakage.py, plot_{results,leakage}.py, submit_*.sh
  setup.sh  download_data.sh  compute_embeddings.sh  run_full.sh
```

## Reproduce everything (cluster)

The scripts are **path-relocatable** — no hardcoded absolute paths. Each script
resolves an experiment root `BASE` from its own location (the parent of the
`{method}/` dir), overridable with `REFND_EXP_BASE`. Data, embeddings, splits,
results, logs, and caches all live under `$BASE`. Venvs (`$HOME/venvs/...`),
`module load`, and SLURM still assume the Alliance cluster. To reproduce:

```
export REFND_EXP_BASE=/path/to/experiment   # optional; else derived from script location
bash scripts/setup.sh              # create per-method venvs + logs/ dirs (login node, once)
bash scripts/download_data.sh      # datasets + cache HF models under $BASE (needs internet)
bash scripts/compute_embeddings.sh # frozen embeddings $BASE/embeddings/*.pt (GPU session)
bash scripts/run_full.sh           # splits -> train -> leakage -> finalize (SLURM chain)
```

`#SBATCH --output=` paths are relative (`logs/...`), so submit from `$BASE`
(`run_full.sh` and `setup.sh` create the `logs/` subdirs). Overridable env:
`REFND_EXP_BASE` (root), `HF_HOME`/`TDC_DATA_PATH` (default `$BASE/{hf_cache,tdc_data}`),
`REFND_TMP` (mmseqs scratch, default `$BASE/tmp`).

`run_full.sh` submits all split jobs, then training + leakage (chained
`afterok`), then finalize (aggregation + figures). Each split method runs in its
own venv to avoid dependency conflicts (see the per-dir `submit.sh`). Splitting
and training are decoupled: splits write `splits/{method}/{dataset}/{seed}.json`;
training reads those and writes `results/`.

**Thresholds** (as run): protein/peptide 40 % identity, molecules 40 % Tanimoto,
DNA 60 % identity. refnd's `proximity_threshold` is a *distance* (= 1 − identity),
so those map to refnd 0.60 / 0.60 / 0.40 respectively.

**Molecule null model** uses the random-atom-molecule null
(`refnd_split/null_model.py:random_molecule_null_gamma`, ported from the paper's
`threshold/molecules.py:find_gamma_function`): a synthetic, dataset-independent
gamma cached under `cache/mol_null_random_*.npy` and shared across all molecule
datasets. `run_full.sh` clears that cache so each run recomputes it.

## Regenerate figures only (locally, from committed results)

The plotting scripts derive `BASE` from `REFND_EXP_BASE` (falling back to their
location). Point it at this `comparison/` dir so they read `results/` here:

```
REFND_EXP_BASE=comparison uv run python comparison/scripts/deep_learning/plot_results.py
REFND_EXP_BASE=comparison uv run python comparison/scripts/deep_learning/plot_leakage.py
```

`compute_leakage.py` and `aggregate_results.py` additionally need the embeddings /
splits / raw per-seed files that are produced on the cluster (not committed here).

## Max-identity histogram

Each `leakage/.../{method}_seed1.npy` is one max-identity value per test sample.
Overlay methods for a dataset:

```python
import numpy as np, matplotlib.pyplot as plt
for m in ["random", "hestia", "refnd"]:
    a = np.load(f"results/leakage/molecule/ames/{m}_seed1.npy")
    plt.hist(a, bins=30, range=(0, 1), alpha=0.5, density=True, label=m)
plt.axvline(0.40, ls="--", c="k")  # leakage threshold
plt.xlabel("max test->train identity"); plt.legend(); plt.show()
```

Leakage was computed for seed 1 only (single-seed by design) — no error bars.

## Not included

`graphpart_split/` (GraphPart is not in the final method set), `archive/`, and
`standalone_experiments/` (DeepPromoter, SR-ARE, AMES one-offs) are omitted; they
live in `refnd_exp/` on the cluster. Ask if you want them mirrored here too.
