#!/bin/bash
# Compute all frozen embeddings. Run on login node or GPU interactive session.
# Request GPU interactive: salloc --gres=gpu:1 --mem=32G --time=3:00:00 --account=def-sgobeil
set -euo pipefail

BASE="${RELAG_EXP_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
HF_CACHE="$BASE/hf_cache"

module load python/3.13.2

source "$HOME/venvs/refnd_exp/deep_learning/bin/activate"

export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE=1      # all models must be pre-cached by download_data.sh
export TRANSFORMERS_OFFLINE=1

echo "=== ESM-C 300M protein embeddings ==="
if [ -f "$BASE/embeddings/protein_esmc.pt" ]; then
    echo "Already computed, skipping."
else
python - <<'EOF'
import torch, pandas as pd, numpy as np
from tqdm import tqdm
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

df = pd.read_parquet("$BASE/data/protein/dbaasp_amp.parquet")

# Filter: valid sequence string, non-empty, only standard AAs, finite log_mic
mask = (
    df["sequence"].notna() &
    df["log_mic"].notna() &
    df["log_mic"].apply(np.isfinite) &
    df["sequence"].apply(lambda s: isinstance(s, str) and len(s) > 0 and set(s.upper()).issubset(VALID_AA))
)
df = df[mask].reset_index(drop=True)
sequences = df["sequence"].str.upper().tolist()
print(f"Sequences after filtering: {len(sequences)}")

# Save filtered dataset so splits use same indices
df.to_parquet("$BASE/data/protein/dbaasp_amp.parquet", index=False)

model = ESMC.from_pretrained("esmc_300m")
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

cfg = LogitsConfig(return_embeddings=True)
BATCH = 32
all_embeds = []
with torch.no_grad():
    for i in tqdm(range(0, len(sequences), BATCH), desc="ESM-C protein", unit="batch"):
        batch = sequences[i:i+BATCH]
        proteins = [ESMProtein(sequence=s) for s in batch]
        tensors = [model.encode(p) for p in proteins]
        # embeddings shape: [seq_len, dim] — mean pool over sequence length
        embeds = torch.stack([model.logits(t, cfg).embeddings.squeeze(0).mean(dim=0) for t in tensors])
        all_embeds.append(embeds.cpu().float())

embeddings = torch.cat(all_embeds, dim=0)
torch.save(embeddings, "$BASE/embeddings/protein_esmc.pt")
print(f"Saved protein embeddings: {embeddings.shape}")
EOF
fi

echo "=== ESM-C 300M enzyme_topt embeddings ==="
if [ -f "$BASE/embeddings/protein_enzyme_topt.pt" ]; then
    echo "Already computed, skipping."
else
python - <<'EOF'
import torch, pandas as pd, numpy as np
from tqdm import tqdm
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

# ESM-C tokenizes the 20 standard AAs plus X (unknown); drop anything else (rare
# B/U/Z/O). Enzymes are long (up to ~2000 residues), so embed one at a time.
VALID_AA = set("ACDEFGHIKLMNPQRSTVWYX")

df = pd.read_parquet("$BASE/data/protein/enzyme_topt.parquet")
mask = (
    df["sequence"].notna() &
    df["label"].notna() &
    df["label"].apply(np.isfinite) &
    df["sequence"].apply(lambda s: isinstance(s, str) and len(s) > 0 and set(s.upper()).issubset(VALID_AA))
)
df = df[mask].reset_index(drop=True)
sequences = df["sequence"].str.upper().tolist()
print(f"Sequences after filtering: {len(sequences)}")

# Rewrite filtered dataset so splits use the same indices as the embeddings.
df.to_parquet("$BASE/data/protein/enzyme_topt.parquet", index=False)

model = ESMC.from_pretrained("esmc_300m")
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

cfg = LogitsConfig(return_embeddings=True)
all_embeds = []
with torch.no_grad():
    for s in tqdm(sequences, desc="ESM-C enzyme_topt", unit="seq"):
        t = model.encode(ESMProtein(sequence=s))
        emb = model.logits(t, cfg).embeddings.squeeze(0).mean(dim=0)
        all_embeds.append(emb.cpu().float())

embeddings = torch.stack(all_embeds, dim=0)
torch.save(embeddings, "$BASE/embeddings/protein_enzyme_topt.pt")
print(f"Saved enzyme_topt embeddings: {embeddings.shape}")
EOF
fi

echo "=== ChemBERTa molecule embeddings ==="
python - <<'EOF'
import os, torch, pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

tok = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

name_map = {
    "cyp2c19_veith": "mol_cyp2c19",
    "caco2_wang":     "mol_caco2",
    "pgp_broccatelli":"mol_pgp",
    "ames":           "mol_ames",
    "lipophilicity":  "mol_lipophilicity",
    "sr_are":         "mol_sr_are",
}

BATCH = 64
for name, out_name in name_map.items():
    out_path = f"$BASE/embeddings/{out_name}.pt"
    if os.path.exists(out_path):
        print(f"Skipping {name}, already computed.")
        continue
    # Load from parquet or CSV
    parquet_path = f"$BASE/data/molecule/{name}.parquet"
    csv_path = f"$BASE/data/molecule/{name}.csv"
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        df = df[df["smiles"].notna() & df["label"].notna()].reset_index(drop=True)
        df.to_parquet(parquet_path, index=False)
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df["Drug"].notna() & df["Y"].notna()].reset_index(drop=True)
        df = df.rename(columns={"Drug": "smiles", "Y": "label"})
        df.to_csv(csv_path, index=False)
    else:
        print(f"ERROR: No data found for {name}")
        continue
    smiles = df["smiles"].tolist()
    all_embeds = []
    batches = range(0, len(smiles), BATCH)
    with torch.no_grad():
        for i in tqdm(batches, desc=f"ChemBERTa {name}", unit="batch"):
            batch = smiles[i:i+BATCH]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            embed = out.last_hidden_state[:, 0, :].cpu().float()
            all_embeds.append(embed)
    embeddings = torch.cat(all_embeds, dim=0)
    torch.save(embeddings, f"$BASE/embeddings/{out_name}.pt")
    print(f"Saved {name}: {embeddings.shape}")
EOF

echo "=== Patching DNABERT-2 cached code for PyTorch compatibility ==="
python - <<'PATCHEOF'
import glob, os, re

PATCH_DIRS = [os.path.expanduser("~/.cache/huggingface"), "$BASE/hf_cache"]

CLEAN_FLASH = "flash_attn_qkvpacked_func = None\n"

OLD_ALIBI = (
    "        context_position = torch.arange(size, device=device)[:, None]\n"
    "        memory_position = torch.arange(size, device=device)[None, :]\n"
    "        relative_position = torch.abs(memory_position - context_position)\n"
    "        # [n_heads, max_token_length, max_token_length]\n"
    "        relative_position = relative_position.unsqueeze(0).expand(\n"
    "            n_heads, -1, -1)\n"
    "        slopes = torch.Tensor(_get_alibi_head_slopes(n_heads)).to(device)\n"
    "        alibi = slopes.unsqueeze(1).unsqueeze(1) * -relative_position"
)
NEW_ALIBI = (
    "        context_position = torch.arange(size, device='cpu')[:, None]\n"
    "        memory_position = torch.arange(size, device='cpu')[None, :]\n"
    "        relative_position = torch.abs(memory_position - context_position)\n"
    "        # [n_heads, max_token_length, max_token_length]\n"
    "        relative_position = relative_position.unsqueeze(0).expand(\n"
    "            n_heads, -1, -1)\n"
    "        slopes = torch.Tensor(_get_alibi_head_slopes(n_heads))\n"
    "        alibi = slopes.unsqueeze(1).unsqueeze(1) * -relative_position"
)

for base in PATCH_DIRS:
    for path in glob.glob(os.path.join(base,
            "modules/transformers_modules/zhihan1996/DNABERT*/**/bert_layers.py"),
            recursive=True):
        with open(path) as f:
            content = f.read()
        changed = False
        # Disable flash_attn entirely (triton dot() API is incompatible at runtime)
        fixed = re.sub(
            r'(?:try:\n(?:    )*from \.flash_attn_triton import flash_attn_qkvpacked_func\n(?:except \w.*:\n    flash_attn_qkvpacked_func = None\n)+|flash_attn_qkvpacked_func = None\n)',
            CLEAN_FLASH, content)
        # Also catch original import line
        fixed2 = re.sub(
            r'from \.flash_attn_triton import flash_attn_qkvpacked_func\n',
            CLEAN_FLASH, fixed)
        if fixed2 != content:
            content = fixed2
            changed = True
        # Fix alibi tensor meta device issue
        if OLD_ALIBI in content:
            content = content.replace(OLD_ALIBI, NEW_ALIBI)
            changed = True
        if changed:
            with open(path, "w") as f:
                f.write(content)
            print(f"Patched: {path}")
        else:
            print(f"Already patched: {path}")
PATCHEOF

echo "=== DNABERT-2 embeddings ==="
ALL_DNA_DONE=true
for dna_name in gue_prom_core_all gue_prom_300_all gue_emp_h3 gue_emp_h4 gue_mouse_0 deeppromoter; do
    [ -f "$BASE/embeddings/dna_${dna_name}.pt" ] || { ALL_DNA_DONE=false; break; }
done
if $ALL_DNA_DONE; then
    echo "All DNA embeddings already computed, skipping."
else
python - <<'EOF'
import os, importlib, torch, pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

tok = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

# Load config via trust_remote_code; bypass AutoModel registration check by
# resolving the model class from the config's auto_map and calling from_pretrained directly.
config = AutoConfig.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
config.pad_token_id = 0

config_module = type(config).__module__          # e.g. transformers_modules.zhihan1996.DNABERT-2-117M.<hash>.configuration_bert
base_module   = config_module.rsplit(".", 1)[0]  # strip .configuration_bert
model_file, model_cls_name = config.auto_map["AutoModel"].split(".")
model_file = model_file.split("--")[-1]  # strip "zhihan1996/DNABERT-2-117M--" prefix
ModelClass = getattr(importlib.import_module(f"{base_module}.{model_file}"), model_cls_name)

model = ModelClass.from_pretrained("zhihan1996/DNABERT-2-117M", config=config,
                                   attn_implementation="eager", low_cpu_mem_usage=False)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

datasets = ["gue_prom_core_all", "gue_prom_300_all", "gue_emp_h3", "gue_emp_h4", "gue_mouse_0", "deeppromoter"]

BATCH = 32
for name in datasets:
    out_path = f"$BASE/embeddings/dna_{name}.pt"
    if os.path.exists(out_path):
        print(f"Skipping {name}, already computed.")
        continue
    df = pd.read_csv(f"$BASE/data/dna/{name}/pooled.csv")
    df = df[df["sequence"].notna() & df["label"].notna()].reset_index(drop=True)
    df.to_csv(f"$BASE/data/dna/{name}/pooled.csv", index=False)
    sequences = df["sequence"].tolist()
    all_embeds = []
    batches = range(0, len(sequences), BATCH)
    with torch.no_grad():
        for i in tqdm(batches, desc=f"DNABERT-2 {name}", unit="batch"):
            batch = sequences[i:i+BATCH]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state
            embed = hidden[:, 0, :].cpu().float()
            all_embeds.append(embed)
    embeddings = torch.cat(all_embeds, dim=0)
    torch.save(embeddings, f"$BASE/embeddings/dna_{name}.pt")
    print(f"Saved {name}: {embeddings.shape}")
EOF
fi

deactivate
echo ""
echo "All embeddings computed. Ready to run: bash run_all.sh"
