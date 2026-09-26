# Ablation study


## Hyperparameter sweeping test
> In `hyperparameters.py`

The HNSW parameters we will toggle are:
  - ef_construction
  - ef_init
  - extend_candidates
  - keep_pruned_connections
  - use_heuristic
  - strict_ef

  - Leiden objective: Modularity / CPM
  - Gamma

  We will extract two types of graph from the built HNSW: from `.edges()`.

  The results will evaluate:
  - Runtime (Time to build the HNSW graph, and find communities, two values not aggregated)
  - The percentage of edges recovered (From the exact computation)
  - The distribution of weight of missed edges. (Report the mean, std, median, 25, 75, 10, 90 percentiles)
  - Make a community detection on the components of the exact proximity graph, and observe the percentage of missing
  edge that are infact inter-community edges. The community detection will default to Modularity, but can be changed
  to CPM if we know the gamma (null model probability)
  - Size of the three largest component, and their number of communities within them (Again, with Modularity by
  default, or CPM if we know gamma)
  - Split the dataset community-wise, and observe the maximal similarity to the nearest train neighbor for each test
  sample, and count the number that violates the threshold. Split with and without post-filtering, and report both
  values
  - Train a MLP model (2 layers, relu in-between and 2x neurons in the hidden layer) on train set, on top of the
  embeddings of a foundational model (Will be cached), then evaluate on the test set, and report test performances. Do
  this with both, post-filtering and no post-filtering

## Scaling experiment
> By running `scaling_benchmark.py`
> You can also run `debug_scaling_relag.py` to see the time each step of the relag pipeline take.

Explore experimental scaling in speed and memory pressure over dataset size. Test for peptide Atlas and Belka dataset.
Compares with Hestia's split.


## Effect of dataset size on edge recall
> By running `edge_recall_scaling.py`

Compute the exact edges (and cached) for multiple sizes: 5K, 25K, 125K, and with hnsw with the same default 
parameters to see how scaling the dataset size affects the edge recall. 

Then, test if increasing the ef_construction proportional to log(n) compared to the 5K run allows to keep recall 
constant.

## In-distribution test
We will use a split method to create a 50/50 train / test split, and assign a label to each sample corresponding to 
their subset. Then shuffle, and do again a split 80/20, then train a linear regression on top of embeddings to 
predict the label (In which split the sample was). If the split are in distribution, the balance accuracy should be 
50%. 