# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from scipy.ndimage import shift, rotate, center_of_mass
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

# NOTE: put `if __name__ == "__main__":` around the code in Part_A.py
# so that importing it does not re-run all of Part A/B/C.
from Part_A import handwritten_images, handwritten_labels


# Reproducibility
S = 1187
np.random.seed(S)
torch.manual_seed(S)

F = 4 + (S % 5)
print("Number of filters:", F)


# %%
# PART 1: CNN ARCHITECTURE

class HandwritingCNN(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(1, F, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(F, 2 * F, kernel_size=3, stride=1, padding=0)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # Dropout has no parameters, so the parameter count is unchanged.
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * F * 2 * 2, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


torch.manual_seed(S)
model = HandwritingCNN()
print(model)

total_parameters = sum(p.numel() for p in model.parameters())
print("Total Parameters (PyTorch):", total_parameters)


# %%
# PART 1: BY-HAND CALCULATION USING (N - K + 2P) / S + 1

def out_size(n, k, p, s):
    return (n - k + 2 * p) // s + 1


N = 8

n_conv1 = out_size(N, 3, 1, 1)           # conv1: K=3, P=1, S=1
n_pool = out_size(n_conv1, 2, 0, 2)      # pool : K=2, P=0, S=2
n_conv2 = out_size(n_pool, 3, 0, 1)      # conv2: K=3, P=0, S=1
flat = 2 * F * n_conv2 * n_conv2

print("\n--- Output sizes ---")
print(f"Input:   1 x {N} x {N}")
print(f"Conv1:   ({N} - 3 + 2*1) / 1 + 1 = {n_conv1}  -> {F} x {n_conv1} x {n_conv1}")
print(f"ReLU:    {F} x {n_conv1} x {n_conv1}")
print(f"MaxPool: ({n_conv1} - 2 + 2*0) / 2 + 1 = {n_pool}  -> {F} x {n_pool} x {n_pool}")
print(f"Conv2:   ({n_pool} - 3 + 2*0) / 1 + 1 = {n_conv2}  -> {2*F} x {n_conv2} x {n_conv2}")
print(f"ReLU:    {2*F} x {n_conv2} x {n_conv2}")
print(f"Flatten: {2*F} * {n_conv2} * {n_conv2} = {flat}")
print(f"Dense:   {flat} -> 10")

conv1_params = (3 * 3 * 1 + 1) * F
conv2_params = (3 * 3 * F + 1) * (2 * F)
fc_params = flat * 10 + 10
manual_total = conv1_params + conv2_params + fc_params

print("\n--- Parameters ---")
print(f"Conv1: (3*3*1 + 1) * {F} = {conv1_params}")
print(f"Conv2: (3*3*{F} + 1) * {2*F} = {conv2_params}")
print(f"Dense: {flat}*10 + 10 = {fc_params}")
print("Pooling / ReLU / Flatten: 0")
print("Manual total:", manual_total)
print("PyTorch total:", total_parameters)
print("Totals match:", manual_total == total_parameters)
assert manual_total == total_parameters


# %%
# PART 2: LOAD AND SPLIT SKLEARN DATASET

digits = load_digits()

X = digits.data / 16.0
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=S,
    stratify=y
)

X_train = X_train.reshape(-1, 1, 8, 8)
X_test = X_test.reshape(-1, 1, 8, 8)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

print("Training shape:", X_train_t.shape)
print("Testing shape:", X_test_t.shape)

criterion = nn.CrossEntropyLoss()


def train_model(net, X_t, y_t, epochs, lr=0.001, weight_decay=0.0,
                batch_size=32, log_every=5, tag="Model"):
    """Train a network and return the list of average epoch losses."""
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []

    for epoch in range(epochs):
        net.train()
        total = 0.0
        for bx, by in loader:
            opt.zero_grad()
            loss = criterion(net(bx), by)
            loss.backward()
            opt.step()
            total += loss.item() * len(by)

        avg = total / len(X_t)
        losses.append(avg)
        if (epoch + 1) % log_every == 0:
            print(f"{tag} | Epoch {epoch+1}, Loss: {avg:.4f}")
    return losses


def predict(net, X_t):
    net.eval()
    with torch.no_grad():
        return net(X_t).argmax(dim=1).numpy()


# %%
# PART 2: TRAIN CNN (BASELINE)

torch.manual_seed(S)
model = HandwritingCNN()
training_losses = train_model(model, X_train_t, y_train_t, epochs=30, tag="Baseline")

plt.plot(training_losses)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("CNN Training Loss")
plt.show()


# %%
# PART 2: LOAD THE 50 HANDWRITTEN DIGITS + BASELINE ACCURACIES

handwritten_labels = np.array(handwritten_labels)
raw_hw = np.array(handwritten_images, dtype=np.float32).reshape(-1, 8, 8)

assert len(raw_hw) == 50 and len(handwritten_labels) == 50, \
    f"Expected 50 digits, got {len(raw_hw)} images / {len(handwritten_labels)} labels"

# Baseline input: plain /16 scaling (same as sklearn training data)
hw_raw_t = torch.tensor((raw_hw / 16.0).reshape(-1, 1, 8, 8), dtype=torch.float32)

sk_pred = predict(model, X_test_t)
sklearn_test_acc = accuracy_score(y_test, sk_pred) * 100
hw_pred_raw = predict(model, hw_raw_t)
baseline_hw_acc = accuracy_score(handwritten_labels, hw_pred_raw) * 100

print(f"Sklearn Test Accuracy (baseline CNN):      {sklearn_test_acc:.2f} %")
print(f"Handwritten (50 digits) Accuracy (baseline): {baseline_hw_acc:.2f} %")

print("\nBaseline predictions on the 50 digits:")
for i in range(50):
    print(f"Image {i+1}: Actual = {handwritten_labels[i]} | Predicted = {hw_pred_raw[i]}")


# %%
# PART 3: DISPLAY THE F LEARNED FILTERS

filters = model.conv1.weight.detach().cpu().numpy()   # shape (F, 1, 3, 3)

fig, axes = plt.subplots(1, F, figsize=(2 * F, 2.6))
for i in range(F):
    axes[i].imshow(filters[i, 0], cmap="gray")
    axes[i].set_title(f"Filter {i+1}")
    axes[i].axis("off")
plt.suptitle(f"{F} Learned 3x3 Filters of Conv Layer 1")
plt.tight_layout()
plt.show()


# %%
# PART 3: FEATURE MAPS FOR ONE OF MY OWN DIGITS

DIGIT_INDEX = 4   # change this to look at another digit

digit_tensor = hw_raw_t[DIGIT_INDEX:DIGIT_INDEX + 1]
model.eval()
with torch.no_grad():
    feature_maps = torch.relu(model.conv1(digit_tensor)).squeeze(0).numpy()

fig, axes = plt.subplots(1, F + 1, figsize=(2 * (F + 1), 2.6))
axes[0].imshow(hw_raw_t[DIGIT_INDEX, 0].numpy(), cmap="gray")
axes[0].set_title(f"Original ({handwritten_labels[DIGIT_INDEX]})")
axes[0].axis("off")
for i in range(F):
    axes[i + 1].imshow(feature_maps[i], cmap="gray")
    axes[i + 1].set_title(f"Feature Map {i+1}")
    axes[i + 1].axis("off")
plt.tight_layout()
plt.show()


# %%
# PART 3: WHAT DOES ONE FILTER DETECT? (data-driven)
# Compare each learned 3x3 kernel with simple edge / blob templates
# using absolute cosine similarity (sign-insensitive).

templates = {
    "horizontal edges": np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=float),
    "vertical edges": np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=float),
    "diagonal edges (/)": np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]], dtype=float),
    "diagonal edges (\\)": np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]], dtype=float),
    "small spots / stroke centres": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float),
}


def cosine(a, b):
    a = a.flatten() - a.mean()
    b = b.flatten() - b.mean()
    return abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


best_filter, best_name, best_score = None, None, -1
for i in range(F):
    scores = {name: cosine(filters[i, 0], t) for name, t in templates.items()}
    name = max(scores, key=scores.get)
    print(f"Filter {i+1}: closest to {name} (similarity {scores[name]:.2f})")
    if scores[name] > best_score:
        best_filter, best_name, best_score = i, name, scores[name]

print(f"""
Filter explanation:
Filter {best_filter + 1} looks most like a detector of {best_name}
(similarity {best_score:.2f} with a standard 3x3 template). In its feature map
the bright regions are the places where the digit's strokes run in that
direction, while other regions of the image stay dark. So this filter helps
the network find that kind of stroke segment before later layers combine
such segments into whole digits.
""")
# TIP: look at the filter and feature-map plots above and make sure this
# sentence agrees with what you actually see before submitting.


# %%
# PART 4: IMPROVEMENT 1 - BETTER PREPROCESSING
# The handwritten pixels only reach 13 (sklearn reaches 16) and the digits
# are not centred. Fix both: stretch contrast to the full range and move
# each digit's centre of mass to the middle of the 8x8 grid.

def preprocess_digit(img):
    img = img.astype(np.float32).copy()
    if img.max() > 0:
        img = img / img.max()                       # stretch to 0-1 (== 0-16)
    cy, cx = center_of_mass(img)
    dy, dx = int(round(3.5 - cy)), int(round(3.5 - cx))
    img = shift(img, (dy, dx), order=0, mode="constant", cval=0)
    return np.clip(img, 0, 1)


hw_proc = np.array([preprocess_digit(im) for im in raw_hw])
hw_proc_t = torch.tensor(hw_proc.reshape(-1, 1, 8, 8), dtype=torch.float32)

print("Mean pixel (sklearn train):        ", round(float(X_train.mean()), 3))
print("Mean pixel (handwritten, raw /16): ", round(float((raw_hw / 16.0).mean()), 3))
print("Mean pixel (handwritten, processed):", round(float(hw_proc.mean()), 3))

# Effect of preprocessing alone (same baseline model, no retraining)
hw_pred_proc = predict(model, hw_proc_t)
baseline_proc_acc = accuracy_score(handwritten_labels, hw_pred_proc) * 100
print(f"\nBaseline CNN on preprocessed digits: {baseline_proc_acc:.2f} %")

fig, axes = plt.subplots(2, 10, figsize=(14, 3.2))
for j in range(10):
    axes[0, j].imshow(raw_hw[j], cmap="gray"); axes[0, j].axis("off")
    axes[1, j].imshow(hw_proc[j], cmap="gray"); axes[1, j].axis("off")
axes[0, 0].set_title("Raw", loc="left")
axes[1, 0].set_title("Preprocessed", loc="left")
plt.tight_layout()
plt.show()


# %%
# PART 4: IMPROVEMENT 2 - DATA AUGMENTATION (TRAINING IMAGES ONLY)
# Random small shifts and rotations. The 50 handwritten digits are NOT used.

rng = np.random.RandomState(S)
AUG_PER_IMAGE = 4

aug_images = [im[0] for im in X_train]
aug_labels = list(y_train)

for img, label in zip(X_train, y_train):
    base = img[0]
    for _ in range(AUG_PER_IMAGE):
        angle = rng.uniform(-15, 15)
        dy, dx = rng.uniform(-1, 1, size=2)
        new = rotate(base, angle=angle, reshape=False, order=1, mode="constant", cval=0)
        new = shift(new, shift=(dy, dx), order=1, mode="constant", cval=0)
        aug_images.append(np.clip(new, 0, 1))
        aug_labels.append(label)

aug_X = torch.tensor(np.array(aug_images).reshape(-1, 1, 8, 8), dtype=torch.float32)
aug_y = torch.tensor(np.array(aug_labels), dtype=torch.long)

print("Original training images :", len(X_train))
print("Augmented training images:", len(aug_X))


# %%
# PART 4: IMPROVEMENT 3 + 4 - DROPOUT AND WEIGHT DECAY
# Fresh CNN trained on augmented data with dropout=0.25 and weight_decay=1e-4.

torch.manual_seed(S)
improved_model = HandwritingCNN(dropout=0.25)
assert sum(p.numel() for p in improved_model.parameters()) == manual_total

improved_losses = train_model(
    improved_model, aug_X, aug_y,
    epochs=40, lr=0.001, weight_decay=1e-4, tag="Improved"
)


# %%
# PART 4: COMPARE ORIGINAL AND IMPROVED MODEL

improved_sk_acc = accuracy_score(y_test, predict(improved_model, X_test_t)) * 100
improved_hw_pred = predict(improved_model, hw_proc_t)
improved_hw_acc = accuracy_score(handwritten_labels, improved_hw_pred) * 100

# Same improved model but on raw (unprocessed) digits, to show what preprocessing adds
improved_raw_acc = accuracy_score(handwritten_labels, predict(improved_model, hw_raw_t)) * 100

print("\n================ RESULTS ================")
print(f"{'Setting':50s} {'Sklearn test':>13s} {'50 digits':>10s}")
print(f"{'Baseline CNN, raw digits':50s} {sklearn_test_acc:12.2f}% {baseline_hw_acc:9.2f}%")
print(f"{'Baseline CNN, preprocessed digits':50s} {sklearn_test_acc:12.2f}% {baseline_proc_acc:9.2f}%")
print(f"{'Improved CNN (aug+dropout+wd), raw digits':50s} {improved_sk_acc:12.2f}% {improved_raw_acc:9.2f}%")
print(f"{'Improved CNN (aug+dropout+wd), preprocessed':50s} {improved_sk_acc:12.2f}% {improved_hw_acc:9.2f}%")

print("\nImproved model predictions on the 50 digits:")
for i in range(50):
    print(f"Image {i+1}: Actual = {handwritten_labels[i]} | Predicted = {improved_hw_pred[i]}")

print("""
Improvements used (none of them trains on the 50 test digits):
1. Preprocessing: contrast stretched to the full 0-16 range and each digit
   centred by its centre of mass, so my handwriting looks like sklearn digits.
2. Data augmentation: random shifts (+-1 px) and rotations (+-15 deg) of the
   sklearn TRAINING images, giving 5x more training data.
3. Dropout (0.25) before the dense layer to reduce co-adaptation.
4. Weight decay (1e-4) in Adam to keep the weights small.
""")


# %%
# PART 5: OVERFITTING EXPERIMENT
# Only 100 images from the sklearn training set; 80 train / 20 validation.
# The sklearn test set and the 50 handwritten digits are not used here.

idx = np.arange(100)
tr_idx, va_idx = train_test_split(
    idx, test_size=0.2, random_state=S, stratify=y_train[:100]
)

of_train_X, of_train_y = X_train_t[tr_idx], y_train_t[tr_idx]
of_val_X, of_val_y = X_train_t[va_idx], y_train_t[va_idx]

torch.manual_seed(S)
overfit_model = HandwritingCNN()
loader = DataLoader(TensorDataset(of_train_X, of_train_y), batch_size=16, shuffle=True)
opt = optim.Adam(overfit_model.parameters(), lr=0.001)

OVERFIT_EPOCHS = 300
train_losses, val_losses = [], []

for epoch in range(OVERFIT_EPOCHS):
    overfit_model.train()
    total = 0.0
    for bx, by in loader:
        opt.zero_grad()
        loss = criterion(overfit_model(bx), by)
        loss.backward()
        opt.step()
        total += loss.item() * len(by)
    train_losses.append(total / len(of_train_X))

    overfit_model.eval()
    with torch.no_grad():
        val_losses.append(criterion(overfit_model(of_val_X), of_val_y).item())

# Overfitting starts after the epoch with the lowest validation loss
best_epoch = int(np.argmin(val_losses)) + 1

plt.figure(figsize=(9, 5))
plt.plot(range(1, OVERFIT_EPOCHS + 1), train_losses, label="Training Loss")
plt.plot(range(1, OVERFIT_EPOCHS + 1), val_losses, label="Validation Loss")
plt.axvline(best_epoch, linestyle="--", color="red",
            label=f"Overfitting starts after epoch {best_epoch}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Overfitting Experiment (100 Training Images)")
plt.legend()
plt.show()

print(f"Lowest validation loss at epoch {best_epoch} "
      f"({val_losses[best_epoch-1]:.4f}). After this the training loss keeps "
      f"falling ({train_losses[-1]:.4f} at the end) while the validation loss rises "
      f"({val_losses[-1]:.4f}), which is overfitting.")