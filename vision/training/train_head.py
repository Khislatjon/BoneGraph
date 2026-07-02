"""
vision/training/train_head.py
=============================
Train the classifier head on cached BiomedCLIP features and report accuracy on
MURA's held-out valid split.

This is the cheap half of the hybrid. BiomedCLIP is frozen; we only train a
small head (a linear layer by default, or a 1-hidden-layer MLP with
``--hidden``) that maps a 512-d feature to a bone label. On cached features this
is seconds of compute — the classic "linear probe". If the probe separates the
classes well, the features are good and a full fine-tune is worth trying on the
uni server; if it doesn't, fine-tuning won't rescue it (cheap signal, early).

Targets:
  * ``region``   — 7-way body-region classification (the default; clean labels).
  * ``abnormal`` — binary normal/abnormal. NOTE: framing only; the Vision tab is
                   explicitly NOT a diagnostic tool. Kept for completeness.

Validation is carved out of MURA's *train* pool **by patient** (no leakage) for
early stopping; the final numbers are reported on MURA's official *valid* split,
which the model never sees during training. Class imbalance (wrist ≫ humerus) is
handled with inverse-frequency loss weights.

Usage::

    python -m vision.training.train_head --target region
    python -m vision.training.train_head --target region --hidden 256 --epochs 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config.settings import DATA_DIR
from vision.training.cache_features import DEFAULT_OUT_DIR
from vision.training.mura_dataset import REGIONS

DEFAULT_MODEL_DIR = DATA_DIR / "models"


def _load(features_dir: Path, split: str):
    npz = features_dir / f"features_{split}.npz"
    if not npz.exists():
        raise FileNotFoundError(
            f"{npz} not found — run vision.training.cache_features --split {split} first."
        )
    d = np.load(npz, allow_pickle=True)
    return d["X"], d["region_idx"], d["abnormal"], d["patient"].astype(str)


def _patient_val_split(patients: np.ndarray, val_frac: float, seed: int):
    """Boolean mask selecting a patient-disjoint validation slice."""
    uniq = np.array(sorted(set(patients)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_frac))
    val_patients = set(uniq[:n_val].tolist())
    return np.array([p in val_patients for p in patients])


def _report(name: str, y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> float:
    """Print overall + per-class accuracy; return overall accuracy."""
    overall = float((y_true == y_pred).mean())
    print(f"\n{name}: overall accuracy {overall:.3f}  (n={len(y_true)})")
    print(f"    {'class':10s} {'acc':>6s} {'n':>6s}")
    for i, c in enumerate(classes):
        mask = y_true == i
        n = int(mask.sum())
        acc = float((y_pred[mask] == i).mean()) if n else float("nan")
        print(f"    {c:10s} {acc:6.3f} {n:6d}")
    try:
        from sklearn.metrics import f1_score
        print(f"    macro-F1 {f1_score(y_true, y_pred, average='macro'):.3f}")
    except Exception:
        pass
    return overall


def train(features_dir: Path, target: str, *, hidden: int, epochs: int, lr: float,
          weight_decay: float, batch_size: int, val_frac: float, seed: int,
          out_dir: Path) -> None:
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_key = 0 if target == "region" else 1  # index into the tuple below
    classes = list(REGIONS) if target == "region" else ["normal", "abnormal"]

    Xtr_all, reg_tr, abn_tr, pat_tr = _load(features_dir, "train")
    Xte, reg_te, abn_te, _ = _load(features_dir, "valid")
    ytr_all = (reg_tr if target == "region" else abn_tr).astype(np.int64)
    yte = (reg_te if target == "region" else abn_te).astype(np.int64)

    val_mask = _patient_val_split(pat_tr, val_frac, seed)
    Xtr, ytr = Xtr_all[~val_mask], ytr_all[~val_mask]
    Xval, yval = Xtr_all[val_mask], ytr_all[val_mask]
    print(f"target={target}  train={len(Xtr)}  val={len(Xval)}  test(MURA valid)={len(Xte)}")

    dim = Xtr.shape[1]
    n_classes = len(classes)
    # Inverse-frequency class weights so rare regions (humerus) aren't ignored.
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    weights = torch.tensor((counts.sum() / (n_classes * np.maximum(counts, 1))),
                           dtype=torch.float32, device=device)

    if hidden > 0:
        model = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                              nn.Dropout(0.2), nn.Linear(hidden, n_classes))
    else:
        model = nn.Linear(dim, n_classes)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xval_t = torch.tensor(Xval, dtype=torch.float32, device=device)

    best_val, best_state = -1.0, None
    n = len(Xtr_t)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(Xval_t).argmax(1).cpu().numpy()
        val_acc = float((val_pred == yval).mean())
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train-loss {loss.item():.3f}  val-acc {val_acc:.3f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(torch.tensor(Xte, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
    print(f"\nbest val-acc {best_val:.3f}")
    _report("TEST (MURA valid, unseen)", yte, test_pred, classes)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"vision_{target}_head.pt"
    import torch as _torch
    _torch.save({"state_dict": best_state, "target": target, "classes": classes,
                 "hidden": hidden, "dim": dim, "encoder": "BiomedCLIP"}, out)
    (out_dir / f"vision_{target}_head.json").write_text(json.dumps(
        {"target": target, "classes": classes, "best_val_acc": best_val,
         "hidden": hidden}, indent=2))
    print(f"saved → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Vision classifier head.")
    ap.add_argument("--features-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--target", choices=["region", "abnormal"], default="region")
    ap.add_argument("--hidden", type=int, default=0, help="0 = linear probe; >0 = MLP hidden units")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = ap.parse_args()
    train(args.features_dir, args.target, hidden=args.hidden, epochs=args.epochs,
          lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
          val_frac=args.val_frac, seed=args.seed, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
