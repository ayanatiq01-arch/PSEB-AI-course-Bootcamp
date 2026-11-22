import torch
import torch.nn as nn
from sklearn.datasets import load_digits

# Har .pt file ke liye wahi banawat likhni zaroori hai jo p1_train.py mein thi
models = {
    "best_small.pt": nn.Linear(64, 10),
    "best_big.pt": nn.Sequential(nn.Linear(64, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10)),
    "best_reg.pt": nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 10)),
}

d = load_digits()
x = torch.tensor(d.data[0] / 16.0, dtype=torch.float32).unsqueeze(0)     # image 0, scaled 0-1

for file, model in models.items():
    state = torch.load(file)
    print(f"\n=== {file} ===")
    for name, tensor in state.items():
        print(" ", name, tuple(tensor.shape))
    print("  total numbers saved:", sum(t.numel() for t in state.values()))
    model.load_state_dict(state)
    model.eval()
    probs = torch.softmax(model(x), 1)[0]
    print(f"  image 0 -> predicted {probs.argmax().item()} (confidence {probs.max().item():.2f}), true {d.target[0]}")
