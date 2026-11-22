# Project 1 · Courier company · Little data, big model
# Run:  python p1_train.py     (about a minute on a laptop)  ->  curves.png, results.csv, results.md, best_*.pt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
np.random.seed(0)

# ---- Steps 1 and 2: load, scale to 0-1, split 200 / 798 / 799 with a fixed seed ----
digits = load_digits()
X = (digits.data / 16.0).astype("float32")          # 8x8 pixels, every pixel scaled to 0-1
y = digits.target
X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, train_size=200, random_state=0, stratify=y)
X_va, X_te, y_va, y_te = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0, stratify=y_rest)
X_tr, X_va, X_te = map(torch.tensor, (X_tr, X_va, X_te))
y_tr, y_va, y_te = (torch.tensor(a, dtype=torch.long) for a in (y_tr, y_va, y_te))
print("train / val / test:", len(X_tr), len(X_va), len(X_te))

loss_fn = nn.CrossEntropyLoss()


# ---- Step 3: ONE training loop that records both losses every epoch ----
def train(name, model, opt, epochs=300, patience=None):
    hist, best, wait = {"train": [], "val": []}, float("inf"), 0
    for epoch in range(epochs):
        model.train(); loss = loss_fn(model(X_tr), y_tr); opt.zero_grad(); loss.backward(); opt.step()   # blank: backward
        model.eval()                                   # blank: eval (switch dropout off)
        with torch.no_grad():
            hist["train"].append(loss_fn(model(X_tr), y_tr).item())
            hist["val"].append(loss_fn(model(X_va), y_va).item())
        if hist["val"][-1] < best:                     # keep the best checkpoint
            best, wait = hist["val"][-1], 0
            torch.save(model.state_dict(), f"best_{name}.pt")
        else:
            wait += 1
        if patience and wait >= patience:              # early stopping
            break
    return hist


def accuracy(model):
    model.eval()
    with torch.no_grad():
        return (model(X_te).argmax(1) == y_te).float().mean().item()


# ---- Step 4: three models, 300 epochs ----
torch.manual_seed(0); small = nn.Linear(64, 10)         # underfits (a fixed seed before every model)
torch.manual_seed(0)
big = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
torch.manual_seed(0)
reg = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Dropout(0.3),        # blank: nn.Dropout(0.3)
                    nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 10))

runs = {
    "Small": (small, torch.optim.SGD(small.parameters(), lr=0.01), None, "small"),
    "Big": (big, torch.optim.Adam(big.parameters(), lr=0.001), None, "big"),
    "Big + dropout + decay": (reg, torch.optim.AdamW(reg.parameters(), lr=0.001, weight_decay=0.05), 30, "reg"),  # blank: patience=30
}
hists = {}
for label, (model, opt, patience, key) in runs.items():
    hists[label] = train(key, model, opt, epochs=300, patience=patience)
    print(f"trained {label}: {len(hists[label]['train'])} epochs")

# ---- Step 5: the third model is tested with its best checkpoint, not its last weights ----
reg.load_state_dict(torch.load("best_reg.pt"))

# ---- Step 6: plot the three runs side by side ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ylims = {"Small": 2.5, "Big": 1.0, "Big + dropout + decay": 1.0}
for ax, (label, h) in zip(axes, hists.items()):
    best_ep = int(np.argmin(h["val"]))
    ax.plot(h["train"], color="#E8710A", lw=2, label="training loss")
    ax.plot(h["val"], color="#1A73E8", lw=2, ls="--", label="validation loss")
    ax.axvline(best_ep, color="grey", ls=":", lw=1.5)
    ax.set_ylim(0, ylims[label]); ax.set_xlim(0, 300)
    ax.set_title(label, fontweight="bold"); ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.legend(loc="upper right"); ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Project 1: 200 training images. Orange = training loss, blue dashed = validation loss, dotted line = best epoch",
             fontsize=11)
plt.tight_layout()
plt.savefig("curves.png", dpi=110)

# ---- Step 7: the results table ----
rows = []
for label, h in hists.items():
    best_ep = int(np.argmin(h["val"]))
    model = {"Small": small, "Big": big, "Big + dropout + decay": reg}[label]
    rows.append({"Model": label,
                 "Final training loss": round(h["train"][-1], 3),
                 "Best validation loss": f'{min(h["val"]):.3f} at epoch {best_ep + 1}',
                 "Epochs run": len(h["train"]),
                 "Test accuracy": round(accuracy(model), 3)})
table = pd.DataFrame(rows)
table.to_csv("results.csv", index=False)
print(table.to_string(index=False))

# ---- Step 8: one sentence: which model ships, and what would help most next ----
stopped_early = 300 - table.loc[2, "Epochs run"]
sentence = (f"Ship the regularised model (big + dropout + decay): it has the lowest validation loss "
            f"({table.loc[2, 'Best validation loss']}) and it stopped itself {stopped_early} epochs early. "
            f"Its test accuracy is only {round((table.loc[2, 'Test accuracy'] - table.loc[1, 'Test accuracy']) * 100, 1)} points "
            f"better than the plain big model, so the most help now is more labelled data: 200 images is the real limit.")
print("\n" + sentence)

with open("results.md", "w") as f:
    f.write("# Project 1 results: which model ships?\n\n")
    try:
        f.write(table.to_markdown(index=False))        # needs the tabulate package
    except ImportError:
        f.write(table.to_string(index=False))          # plain text if tabulate is not installed
    f.write("\n\n![curves](curves.png)\n\n**Decision:** " + sentence + "\n")