# trainer.py: the training engine. It knows nothing about Streamlit; it calls you back after every epoch.
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEFAULTS = {"size": 256, "dropout": 0.0, "decay": 0.0, "lr": 0.001, "patience": 0, "scale": True, "epochs": 300}


def load_data(scale=True, seed=0):
    """569 patients, 30 measurements, malignant = 1. Split 60 / 20 / 20, stratified."""
    d = load_breast_cancer()
    X, y = d.data.astype("float32"), (d.target == 0).astype("int64")      # sklearn has 0 = malignant; flip so malignant = 1
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=seed)
    if scale:
        sc = StandardScaler().fit(X_tr)                                    # fit on TRAINING rows only
        X_tr, X_va, X_te = (sc.transform(a).astype("float32") for a in (X_tr, X_va, X_te))
    t = lambda a, dt: torch.tensor(a, dtype=dt)
    return {"X_tr": t(X_tr, torch.float32), "y_tr": t(y_tr, torch.long),
            "X_va": t(X_va, torch.float32), "y_va": t(y_va, torch.long),
            "X_te": t(X_te, torch.float32), "y_te": t(y_te, torch.long),
            "names": list(d.feature_names)}


def build_model(cfg):
    h, p = cfg["size"], cfg["dropout"]
    return nn.Sequential(nn.Linear(30, h), nn.ReLU(), nn.Dropout(p),
                         nn.Linear(h, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, 2))


def train(cfg, data, on_epoch=None):
    cfg = {**DEFAULTS, **cfg}
    torch.manual_seed(0)
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["decay"])
    loss_fn = nn.CrossEntropyLoss()
    hist = {"train": [], "val": []}
    best, wait, best_state = float("inf"), 0, None
    for epoch in range(cfg["epochs"]):
        model.train()
        loss = loss_fn(model(data["X_tr"]), data["y_tr"])
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            hist["train"].append(loss_fn(model(data["X_tr"]), data["y_tr"]).item())
            hist["val"].append(loss_fn(model(data["X_va"]), data["y_va"]).item())
        if hist["val"][-1] < best:
            best, wait = hist["val"][-1], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if on_epoch:
            on_epoch(epoch, hist)                          # blank: on_epoch  (the page redraws here)
        if cfg["patience"] and wait >= cfg["patience"]:
            break
    if cfg["patience"] and best_state is not None:
        model.load_state_dict(best_state)                  # blank: load_state_dict(best_state)
    model.eval()
    return model, hist


def diagnose(hist):
    """Training loss over 0.3 = underfitting; validation ending 30% above its best = overfitting; else healthy."""
    if hist["train"][-1] > 0.3:
        return "underfitting"
    best = min(hist["val"])                                # blank: min
    if hist["val"][-1] > 1.3 * best:
        return "overfitting"
    return "healthy"


def counts(model, X, y):
    """Cancers found (true positives) and false alarms, at the usual 0.5 threshold."""
    with torch.no_grad():
        pred = model(X).argmax(1)
    found = int(((pred == 1) & (y == 1)).sum())
    return {"cancers_found": found, "cancers_total": int((y == 1).sum()),
            "false_alarms": int(((pred == 1) & (y == 0)).sum()),
            "accuracy": round(float((pred == y).float().mean()), 3)}
