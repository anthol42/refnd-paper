"""Host-side DataSAIL split: runs the DataSAIL Apptainer image as a subprocess,
so it plugs into `synthetic.compare_splits` like any in-process split method.

The image is located via --datasail-sif / $DATASAIL_SIF, falling back to
$RELAG_EXP_BASE/datasail.sif (where comparison/scripts/datasail_split/submit.sh
expects it). The runtime is `apptainer`, or `singularity` if that is what exists.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRY = REPO_ROOT / "synthetic" / "datasail_entry.py"


def find_sif(explicit: str | None = None) -> Path:
    candidates = [explicit, os.environ.get("DATASAIL_SIF")]
    if os.environ.get("RELAG_EXP_BASE"):
        candidates.append(str(Path(os.environ["RELAG_EXP_BASE"]) / "datasail.sif"))
    # Where comparison/scripts/datasail_split/build_sif.sh puts it.
    candidates.append(str(REPO_ROOT / "comparison" / "scripts" / "datasail.sif"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError(
        "DataSAIL image not found: pass --datasail-sif, or set $DATASAIL_SIF or "
        "$RELAG_EXP_BASE (containing datasail.sif).")


def find_runtime() -> str:
    for runtime in ("apptainer", "singularity"):
        if shutil.which(runtime):
            return runtime
    raise FileNotFoundError(
        "Neither apptainer nor singularity is on PATH (on the cluster: "
        "`module load apptainer/1.4.5`).")


def datasail_split(sequences: list[str], seed: int, sif: Path, threads: int = 8
                   ) -> tuple[list[int], list[int]]:
    """DataSAIL C1e split (mmseqs similarity). Returns (train, test); DataSAIL's
    own val bucket is folded into train, since compare_splits draws its own val."""
    runtime = find_runtime()
    # The working dir must sit under the bound repo so the container can see it.
    work_root = REPO_ROOT / "tmp"
    work_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="datasail_", dir=work_root) as work_dir:
        work = Path(work_dir)
        (work / "input.json").write_text(json.dumps({"sequences": sequences}))
        subprocess.run(
            [runtime, "exec", "--bind", str(REPO_ROOT), "--pwd", str(work), str(sif),
             "python", str(ENTRY), str(work / "input.json"), str(work / "output.json"),
             "--seed", str(seed), "--threads", str(threads)],
            check=True,
        )
        result = json.loads((work / "output.json").read_text())
    return result["train"] + result["val"], result["test"]
