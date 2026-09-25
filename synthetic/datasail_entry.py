"""Runs INSIDE the DataSAIL Apptainer image; called by `synthetic.datasail.datasail_split`.

Reads {"sequences": [...]} from <input.json>, runs DataSAIL's C1e split with the
same escalation ladder as comparison/scripts/datasail_split/run_datasail.py, and
writes {"train": [...], "val": [...], "test": [...], "e_clusters", "epsilon"} to
<output.json>. Only needs the container's own python + datasail.
"""

import argparse
import json
import sys
from pathlib import Path

# Python puts this script's dir (synthetic/) first on sys.path, where
# synthetic/datasail.py would shadow the real `datasail` package.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "comparison" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "comparison" / "scripts" / "datasail_split"))
from run_datasail import split_entities  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--max-sec", type=int, default=1000)
parser.add_argument("--epsilon", type=float, default=0.05)
parser.add_argument("--e-clusters", type=int, default=200)   # peptides need >= 200
args = parser.parse_args()

sequences = json.loads(Path(args.input).read_text())["sequences"]
data = {str(i): sequence for i, sequence in enumerate(sequences)}
cache_dir = Path(args.output).parent / "datasail_cache"
cache_dir.mkdir(parents=True, exist_ok=True)

train, val, test, (e_clusters, epsilon) = split_entities(
    data, "P", "mmseqs", args.seed, args.max_sec, args.threads,
    args.epsilon, args.e_clusters, cache_dir)
Path(args.output).write_text(json.dumps(
    {"train": train, "val": val, "test": test, "e_clusters": e_clusters, "epsilon": epsilon}))
