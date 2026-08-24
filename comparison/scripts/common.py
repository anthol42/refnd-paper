"""Shared utilities for deep learning training."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr


# ── Constants ────────────────────────────────────────────────────────────────

SEEDS = list(range(1, 11))

MOLECULE_DATASETS = [
    "cyp2c19_veith",
    "caco2_wang",
    "pgp_broccatelli",
    "ames",
    "lipophilicity",
    "sr_are",
]

DNA_DATASETS = [
    "gue_prom_core_all",   # core promoter regions
    "gue_prom_300_all",    # larger (300 bp) promoter regions
    "gue_emp_h3",          # epigenetic markers H3
    "gue_emp_h4",          # epigenetic markers H4
    "gue_mouse_0",         # mouse TF-binding regions
    "deeppromoter",        # E. coli strong vs. weak promoters (khanhlee/deepPromoter)
]

DATASET_CONFIG = {
    "dbaasp_amp":          {"embed": "protein_esmc.pt",           "metric": "pcc",   "task": "regression"},
    "cyp2c19_veith":       {"embed": "mol_cyp2c19.pt",            "metric": "auroc", "task": "classification"},
    "caco2_wang":          {"embed": "mol_caco2.pt",               "metric": "pcc",   "task": "regression"},
    "pgp_broccatelli":     {"embed": "mol_pgp.pt",                 "metric": "auroc", "task": "classification"},
    "ames":                {"embed": "mol_ames.pt",                 "metric": "auroc", "task": "classification"},
    "lipophilicity":       {"embed": "mol_lipophilicity.pt",       "metric": "pcc",   "task": "regression"},
    "sr_are":              {"embed": "mol_sr_are.pt",              "metric": "auroc", "task": "classification"},
    "gue_prom_core_all":   {"embed": "dna_gue_prom_core_all.pt",   "metric": "auroc", "task": "classification"},
    "gue_prom_300_all":    {"embed": "dna_gue_prom_300_all.pt",    "metric": "auroc", "task": "classification"},
    "gue_emp_h3":          {"embed": "dna_gue_emp_h3.pt",          "metric": "auroc", "task": "classification"},
    "gue_emp_h4":          {"embed": "dna_gue_emp_h4.pt",          "metric": "auroc", "task": "classification"},
    "gue_mouse_0":         {"embed": "dna_gue_mouse_0.pt",         "metric": "auroc", "task": "classification"},
    "deeppromoter":        {"embed": "dna_deeppromoter.pt",        "metric": "auroc", "task": "classification"},
}


# ── Model heads ──────────────────────────────────────────────────────────────

class LinearHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2 * in_dim),
            nn.ReLU(),
            nn.Linear(2 * in_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Training ─────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(model, X_train, y_train, X_val, y_val,
                task: str, max_epochs: int, patience: int,
                lr: float = 1e-3, batch_size: int = 64,
                device: str = "cpu") -> nn.Module:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()

    Xtr = torch.tensor(X_train, dtype=torch.float32).to(device)
    ytr = torch.tensor(y_train, dtype=torch.float32).to(device)
    Xva = torch.tensor(X_val, dtype=torch.float32).to(device)
    yva = torch.tensor(y_val, dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(model(Xb), yb).backward()
            optimizer.step()

        if patience > 0:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(Xva), yva).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate(model, X_test, y_test, task: str, device: str = "cpu") -> float:
    model.eval()
    Xte = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(Xte).cpu().numpy()
    if task == "regression":
        return float(pearsonr(preds, y_test)[0])
    else:  # classification -> auroc
        return float(roc_auc_score(y_test, preds))


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test,
                       task: str, embed_dim: int, seed: int,
                       device: str = "cpu") -> dict:
    set_seed(seed)
    results = {}

    for name, Model, epochs, patience in [
        ("linear", LinearHead, 100, 0),
        ("mlp",    MLPHead,    300, 15),
    ]:
        model = Model(embed_dim)
        train_model(model, X_train, y_train, X_val, y_val,
                    task=task, max_epochs=epochs, patience=patience, device=device)
        # Score test (primary) and train (to measure the overfitting gap =
        # train - test). Leakage-free splits should show a larger, more honest
        # gap than random splits, which inflate test via near-duplicates.
        results[name] = evaluate(model, X_test, y_test, task, device)
        results[f"{name}_train"] = evaluate(model, X_train, y_train, task, device)

    return results


# ── I/O helpers ──────────────────────────────────────────────────────────────

def load_split(splits_dir: str, method: str, dataset: str, seed: int):
    path = Path(splits_dir) / method / dataset / f"{seed}.json"
    with open(path) as f:
        d = json.load(f)
    return np.array(d["train"]), np.array(d["val"]), np.array(d["test"])


def load_refnd_sizes(splits_dir: str, dataset: str) -> dict:
    path = Path(splits_dir) / "refnd" / dataset / "sizes.json"
    if not path.exists():
        raise FileNotFoundError(f"sizes.json missing for {dataset}. Run refnd_split first.")
    with open(path) as f:
        d = json.load(f)
    if len(d) < 10:
        raise ValueError(f"sizes.json for {dataset} has {len(d)}/10 seeds — incomplete.")
    return d


def subsample(idx: np.ndarray, target_n: int, seed: int) -> np.ndarray:
    if len(idx) <= target_n:
        return idx
    rng = np.random.default_rng(seed)
    return rng.choice(idx, size=target_n, replace=False)


def save_seed_result(results_dir: str, task_type: str, dataset: str,
                     method: str, seed: int, scores: dict):
    """Each seed writes its own file — no locking needed."""
    out_dir = Path(results_dir) / task_type / dataset / method
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"seed_{seed}.json"
    with open(path, "w") as f:
        json.dump(scores, f)
