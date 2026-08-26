#!/bin/bash
# Run on login node (needs internet). Downloads all datasets and caches HF models.
# Also pre-downloads ChemBERTa and DNABERT-2 model weights for offline use in jobs.
set -euo pipefail

BASE="${REFND_EXP_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
HF_CACHE="$BASE/hf_cache"
TDC_CACHE="$BASE/tdc_data"

mkdir -p "$BASE/data/protein" "$BASE/data/molecule" "$BASE/data/dna"
mkdir -p "$HF_CACHE" "$TDC_CACHE"

module load python/3.13.2 rdkit/2024.09.6

# Use deep_learning venv for data downloads (has tdc, transformers)
source "$HOME/venvs/refnd_exp/deep_learning/bin/activate"

export HF_HOME="$HF_CACHE"
export TDC_DATA_PATH="$TDC_CACHE"

echo "=== Downloading DBAASP (protein) ==="
python - <<'EOF'
import os, sys
sys.path.insert(0, "/home/jacobc/links/projects/def-sgobeil/jacobc/notebook")
os.environ.setdefault("HF_HOME", os.environ["HF_HOME"])
from qmap import DBAASPDataset
import pandas as pd

ds = DBAASPDataset()
df = ds.tabular(["id", "sequence", "Escherichia coli"])
df = df.loc[df["Escherichia coli"].notna()].reset_index(drop=True)
df = df.rename(columns={"Escherichia coli": "mic"})
df["log_mic"] = __import__("numpy").log10(df["mic"].astype(float))
df = df[["id", "sequence", "log_mic"]].dropna().reset_index(drop=True)
df.to_parquet("$BASE/data/protein/dbaasp_amp.parquet", index=False)
print(f"Saved {len(df)} protein sequences")
EOF

echo "=== Downloading enzyme optimal-temperature (proteinglm/optimal_temperature) ==="
python - <<'EOF'
import pandas as pd
from huggingface_hub import snapshot_download

# DeepET-sourced Topt regression, the version xTrimoPGLM/Hestia used.
# Pool the published train+test and re-split downstream (matches the GUE convention).
d = snapshot_download(repo_id="proteinglm/optimal_temperature", repo_type="dataset")
frames = [pd.read_parquet(f"{d}/data/{s}-00000-of-00001.parquet") for s in ("train", "test")]
df = pd.concat(frames, ignore_index=True)
df = df.rename(columns={"seq": "sequence"})[["sequence", "label"]].dropna()
df = df.drop_duplicates(subset="sequence").reset_index(drop=True)
df.to_parquet("$BASE/data/protein/enzyme_topt.parquet", index=False)
print(f"Saved {len(df)} enzyme sequences (Topt regression)")
EOF

echo "=== Downloading TDC molecule datasets ==="
python - <<'EOF'
import os
os.environ["TDC_DATA_PATH"] = os.environ["TDC_DATA_PATH"]
import numpy as np
import pandas as pd
from tdc.single_pred import ADME, Tox

tasks = {
    "cyp2c19_veith":      ("ADME",  "CYP2C19_Veith"),
    "caco2_wang":         ("ADME",  "Caco2_Wang"),
    "pgp_broccatelli":    ("ADME",  "Pgp_Broccatelli"),
    "ames":               ("Tox",   "AMES"),
    "lipophilicity":      ("ADME",  "Lipophilicity_AstraZeneca"),
}
module_map = {"ADME": ADME, "Tox": Tox}

for name, (module_name, tdc_name) in tasks.items():
    cls = module_map[module_name]
    data = cls(name=tdc_name)
    df = data.get_data()
    # TDC returns Drug (SMILES) and Y (label)
    df = df[["Drug", "Y"]].dropna().reset_index(drop=True)
    df.columns = ["smiles", "label"]
    out = f"$BASE/data/molecule/{name}.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {len(df)} molecules for {name}")
EOF

echo "=== Downloading GUE DNA datasets ==="
# GUE files are distributed via Google Drive (gdrive ID: 1uOrwlf07qGQuruXqGXWMpPn8avBoW7T-)
# Download the full archive once, then extract the 5 needed subdirectories.
GUE_GDRIVE_ID="1uOrwlf07qGQuruXqGXWMpPn8avBoW7T-"
GUE_ZIP="$BASE/data/dna/GUE.zip"

if [ ! -f "$GUE_ZIP" ]; then
    echo "Downloading GUE archive from Google Drive..."
    # gdown is available in the deep_learning venv (pip install gdown)
    python -m gdown "$GUE_GDRIVE_ID" -O "$GUE_ZIP"
fi

echo "Extracting GUE datasets..."
cd "$BASE/data/dna"
unzip -qo "$GUE_ZIP" -d gue_extracted/

# Map our names to GUE archive paths
declare -A GUE_DATASETS=(
    ["gue_prom_core_all"]="GUE/prom/prom_core_all"
    ["gue_prom_300_all"]="GUE/prom/prom_300_all"
    ["gue_emp_h3"]="GUE/EMP/H3"
    ["gue_emp_h4"]="GUE/EMP/H4"
    ["gue_mouse_0"]="GUE/mouse/0"
)

for name in "${!GUE_DATASETS[@]}"; do
    src_path="$BASE/data/dna/gue_extracted/${GUE_DATASETS[$name]}"
    dst="$BASE/data/dna/$name"
    mkdir -p "$dst"
    cp "$src_path"/train.csv "$dst/"
    cp "$src_path"/test.csv  "$dst/"
    # dev.csv if present (some GUE tasks have it)
    [ -f "$src_path/dev.csv" ] && cp "$src_path/dev.csv" "$dst/"

    # Pool all splits into pooled.csv (we re-split entirely)
    python - <<PYEOF
import pandas as pd, os
name = "$name"
base = "$BASE/data/dna/$name"
parts = []
for fn in ["train.csv", "dev.csv", "test.csv"]:
    p = f"{base}/{fn}"
    if os.path.exists(p):
        parts.append(pd.read_csv(p))
pooled = pd.concat(parts, ignore_index=True)
pooled.to_csv(f"{base}/pooled.csv", index=False)
print(f"  {name}: {len(pooled)} sequences pooled")
PYEOF
done

cd "$BASE"

echo "=== Patching DNABERT-2 cached code for PyTorch compatibility ==="
python - <<'PATCHEOF'
import glob, os
for base in [os.path.expanduser("~/.cache/huggingface"), "$BASE/hf_cache"]:
  for path in glob.glob(os.path.join(base,
        "modules/transformers_modules/zhihan1996/DNABERT*/**/bert_layers.py"),
        recursive=True):
    with open(path) as f:
        content = f.read()
    old = (
        "        context_position = torch.arange(size, device=device)[:, None]\n"
        "        memory_position = torch.arange(size, device=device)[None, :]\n"
        "        relative_position = torch.abs(memory_position - context_position)\n"
        "        # [n_heads, max_token_length, max_token_length]\n"
        "        relative_position = relative_position.unsqueeze(0).expand(\n"
        "            n_heads, -1, -1)\n"
        "        slopes = torch.Tensor(_get_alibi_head_slopes(n_heads)).to(device)\n"
        "        alibi = slopes.unsqueeze(1).unsqueeze(1).cpu() * -relative_position.cpu()"
    )
    new = (
        "        context_position = torch.arange(size, device='cpu')[:, None]\n"
        "        memory_position = torch.arange(size, device='cpu')[None, :]\n"
        "        relative_position = torch.abs(memory_position - context_position)\n"
        "        # [n_heads, max_token_length, max_token_length]\n"
        "        relative_position = relative_position.unsqueeze(0).expand(\n"
        "            n_heads, -1, -1)\n"
        "        slopes = torch.Tensor(_get_alibi_head_slopes(n_heads))\n"
        "        alibi = slopes.unsqueeze(1).unsqueeze(1) * -relative_position"
    )
    if old in content:
        with open(path, "w") as f:
            f.write(content.replace(old, new))
        print(f"Patched: {path}")
    else:
        print(f"Already patched or not found: {path}")
PATCHEOF

echo "=== Pre-caching HuggingFace models ==="
python - <<'EOF'
from transformers import AutoTokenizer, AutoModel, AutoConfig
import importlib

print("Downloading ESM-C 300M weights...")
from huggingface_hub import snapshot_download
snapshot_download(repo_id="EvolutionaryScale/esmc-300m-2024-12")
print("ESM-C 300M cached.")

print("Downloading ChemBERTa...")
AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

print("Downloading DNABERT-2...")
AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
# Bypass AutoModel registration conflict via config's auto_map
config = AutoConfig.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
config.pad_token_id = 0
base_module = type(config).__module__.rsplit(".", 1)[0]
model_file, model_cls_name = config.auto_map["AutoModel"].split(".")
model_file = model_file.split("--")[-1]
ModelClass = getattr(importlib.import_module(f"{base_module}.{model_file}"), model_cls_name)
ModelClass.from_pretrained("zhihan1996/DNABERT-2-117M", config=config,
    attn_implementation="eager", low_cpu_mem_usage=False)

print("HuggingFace models cached.")
EOF

deactivate
echo ""
echo "All data downloaded. Next: bash compute_embeddings.sh"
