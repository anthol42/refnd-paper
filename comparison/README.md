# Comparison benchmark

Outputs and analysis scripts for the graph-based dataset-splitting comparison
(Refnd vs Hestia / MMseqs2 / DataSAIL / Random) across molecule, protein, and
DNA datasets. Generated from `refnd_exp/` (refnd 0.0.3, CPM + null-model splits,
no post-filtering, no cross-method subsampling). All results are from the same
run (training `summary.json` files and leakage arrays are mutually consistent).

## Layout

```
results/
  molecule/{dataset}/summary.json   test/train AUROC|PCC per method x head(linear,mlp),
  dna/{dataset}/summary.json          mean/std/n=10 + full per-seed values[] + gap
  protein/dbaasp_amp/summary.json
  leakage/{dtype}/{dataset}/{method}_seed1.npy
                                    1-D float32: each test sample's MAX identity
                                    (Tanimoto / seq-identity) to the train set.
                                    -> source data for the max-identity histograms.
  split_metrics/{method}/*.json     per-split wall-time + peak RSS
  split_metrics.csv                 aggregated split cost (time, peak_rss_mb)
  wilcoxon_table.json               paired Wilcoxon refnd-vs-baselines
  figures/                          per-dataset overfitting plots (test vs train)
  figures_leakage/                  per-dataset max-identity histograms
scripts/
  plot_leakage.py       max-identity histograms  -> figures_leakage/
  plot_results.py       overfitting figures       -> figures/
  aggregate_results.py  per-seed raw -> summary.json  (needs raw per-seed files; reference)
  aggregate_split_metrics.py  split_metrics/ -> split_metrics.csv
  compute_leakage.py    generates leakage .npy     (needs embeddings+splits; reference)
  common.py             shared dataset config (imported by the plot scripts)
```

## Regenerate figures

```
uv run python comparison/scripts/plot_leakage.py
uv run python comparison/scripts/plot_results.py
```

`BASE` in each script resolves to this `comparison/` directory, so the scripts
read `results/` and write figures back in place. Only `plot_leakage.py` and
`plot_results.py` run standalone from the copied data; `compute_leakage.py` and
`aggregate_results.py` also require the embeddings / splits / raw per-seed files
that live in `refnd_exp/` and are kept here for reference only.

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
