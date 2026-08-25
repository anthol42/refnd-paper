# Simulation of synthethic data
Generate synthetic data, and try to find the threshold that recovers the partition. We can see that KS is a good 
estimator as it converges to a partition with 100% ARI compared to the true parititon. 

## How to run
```shell
# From repo root:
uv run python -m simulation.generate_data.py
uv run python -m http.server -d simulation/
```