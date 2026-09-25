"""Two-layer MLP with early stopping, supporting PCC (regression), MCC or AUROC
(binary/multiclass classification) and MCC-multilabel (independent binary
classification per label, e.g. BELKA's 3 protein-binder bits)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_eval_mlp(
    embs: torch.Tensor,
    labels: np.ndarray,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    metric: str,                # "pcc", "mcc", "auroc", or "mcc-multilabel"
    epochs: int = 300,
    lr: float = 0.003,
    patience: int = 15,
    batch_size: int = 64,
    seed: int = 42,
) -> dict:
    """Train MLP on train_idx, use val_idx for early stopping, evaluate on test_idx."""
    import time
    from rich import print as rprint
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    is_classification = metric in ("mcc", "auroc")
    is_multilabel     = metric == "mcc-multilabel"
    n_classes = int(np.max(labels) + 1) if is_classification else 1
    n_labels  = labels.shape[1] if is_multilabel else 1

    # --- build tensors ---
    X_all = embs.float()
    if is_classification:
        y_all = torch.tensor(labels, dtype=torch.long)
    else:
        # regression targets and multilabel binary targets are both float
        y_all = torch.tensor(labels, dtype=torch.float32)

    tr_idx  = np.array(train_idx)

    val_idx_arr = np.array(val_idx)

    X_tr,  y_tr  = X_all[tr_idx].to(device),       y_all[tr_idx].to(device)
    X_val, y_val = X_all[val_idx_arr].to(device),  y_all[val_idx_arr].to(device)
    X_te         = X_all[test_idx].to(device)

    # --- model ---
    in_dim = X_all.shape[1]
    out_dim = n_classes if is_classification else (n_labels if is_multilabel else 1)
    model = MLP(in_dim, out_dim).to(device)

    if is_classification:
        criterion = nn.CrossEntropyLoss()
    elif is_multilabel:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True
    )

    def _loss(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if is_classification:
            return criterion(out, y)
        if is_multilabel:
            return criterion(out, y)
        return criterion(out.squeeze(-1), y)

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = _loss(out, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = _loss(val_out, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- evaluate ---
    model.eval()
    with torch.no_grad():
        test_out = model(X_te).cpu()

    y_te = y_all[test_idx].numpy()

    if is_classification:
        if metric == "auroc":
            # AUROC needs a ranking score, not a hard prediction -- softmax
            # probability of the positive class.
            probs = torch.softmax(test_out, dim=-1)[:, 1].numpy()
            score = float(roc_auc_score(y_te, probs))
            preds_out = probs
        else:
            preds = test_out.argmax(dim=-1).numpy()
            score = float(matthews_corrcoef(y_te, preds))
            preds_out = preds
        per_label_scores = None
    elif is_multilabel:
        preds = (torch.sigmoid(test_out) > 0.5).numpy().astype(np.int64)
        preds_out = preds
        per_label_scores = []
        for j in range(n_labels):
            y_col, p_col = y_te[:, j], preds[:, j]
            # MCC is undefined if a label has a single class in the test fold
            if len(np.unique(y_col)) < 2:
                per_label_scores.append(float("nan"))
            else:
                per_label_scores.append(float(matthews_corrcoef(y_col, p_col)))
        score = float(np.nanmean(per_label_scores))
    else:
        preds = test_out.squeeze(-1).numpy()
        preds_out = preds
        score, _ = pearsonr(preds, y_te)
        score = float(score)
        r2 = float(1.0 - np.mean((preds - y_te) ** 2) / np.var(y_te))
        per_label_scores = None

    rprint(f"[dim]    MLP done in {time.perf_counter() - t0:.1f}s — {metric}={score:.4f}[/]")
    # Raw per-sample predictions for test_idx, in the same order. Lets a caller
    # score several disjoint sets from ONE fit: pass their union as test_idx and
    # slice this, instead of refitting the identical model per set.
    result = {"metric": metric, "score": score, "predictions": preds_out.tolist()}
    if metric == "pcc":
        result["r2"] = r2
    if per_label_scores is not None:
        result["per_label_score"] = per_label_scores
    return result
