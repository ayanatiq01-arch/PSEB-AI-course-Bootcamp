# %%
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
np.random.seed(1187)
torch.manual_seed(1187)

digits = load_digits()  # If not imported, import load_digits first
X = digits.data / 16.0
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=1187,
    stratify=y
)
# %%
class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(64, 32)
        self.output = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x
# %%
# %%
def initialize_weights(seed=1187):

    np.random.seed(seed)

    W1 = np.random.randn(64, 32) * np.sqrt(2/64)
    b1 = np.zeros((1, 32))

    W2 = np.random.randn(32, 10) * np.sqrt(2/32)
    b2 = np.zeros((1, 10))

    return W1, b1, W2, b2
# %%
W1, b1, W2, b2 = initialize_weights(seed=1187)

model = NeuralNetwork().double()

with torch.no_grad():
    model.hidden.weight.copy_(torch.tensor(W1.T))
    model.hidden.bias.copy_(torch.tensor(b1.reshape(-1)))

    model.output.weight.copy_(torch.tensor(W2.T))
    model.output.bias.copy_(torch.tensor(b2.reshape(-1)))
# %%
def forward(X, W1, b1, W2, b2):

    Z1 = np.dot(X, W1) + b1
    A1 = np.maximum(0, Z1)

    Z2 = np.dot(A1, W2) + b2

    exp_values = np.exp(
        Z2 - np.max(Z2, axis=1, keepdims=True)
    )

    A2 = exp_values / np.sum(
        exp_values, axis=1, keepdims=True
    )

    return A1, A2
def loss(y_true, y_pred):

    n = len(y_true)

    correct_probabilities = y_pred[
        np.arange(n), y_true
    ]

    loss_value = -np.mean(
        np.log(correct_probabilities + 1e-9)
    )

    return loss_value
# %%
X_check = X_train[:20]
y_check = y_train[:20]

A1_np, A2_np = forward(X_check, W1, b1, W2, b2)
numpy_loss = loss(y_check, A2_np)

X_torch = torch.tensor(X_check, dtype=torch.float64)
y_torch = torch.tensor(y_check, dtype=torch.long)

criterion = nn.CrossEntropyLoss()

outputs = model(X_torch)
torch_loss = criterion(outputs, y_torch)

print("NumPy Loss:", numpy_loss)
print("PyTorch Loss:", torch_loss.item())

print("Loss Match:", round(numpy_loss, 4) == round(torch_loss.item(), 4))
def backward(X, y_true, A1, A2, W2):
    n = X.shape[0]

    dZ2 = A2.copy()
    dZ2[np.arange(n), y_true] -= 1
    dZ2 = dZ2 / n

    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (A1 > 0)

    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2

# %%
model.zero_grad()

outputs = model(X_torch)
loss_value = criterion(outputs, y_torch)

loss_value.backward()

# NumPy gradients
A1_np, A2_np = forward(X_check, W1, b1, W2, b2)
dW1_np, db1_np, dW2_np, db2_np = backward(
    X_check, y_check, A1_np, A2_np, W2
)

# Compare 5 weights
weights_to_check = [
    ("W1", 5, 10),
    ("W1", 20, 15),
    ("W1", 36, 12),
    ("W2", 0, 0),
    ("W2", 15, 5)
]

print("Parameter | NumPy Gradient | PyTorch Gradient | Difference")

for name, i, j in weights_to_check:

    if name == "W1":
        numpy_grad = dW1_np[i, j]
        torch_grad = model.hidden.weight.grad[j, i].item()

    else:
        numpy_grad = dW2_np[i, j]
        torch_grad = model.output.weight.grad[j, i].item()

    difference = abs(numpy_grad - torch_grad)

    print(
        name, (i, j),
        "|", numpy_grad,
        "|", torch_grad,
        "|", difference
    )

# %%
X_train_part, X_val, y_train_part, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=1187,
    stratify=y_train
)

train_dataset = TensorDataset(
    torch.tensor(X_train_part, dtype=torch.float64),
    torch.tensor(y_train_part, dtype=torch.long)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

X_val_torch = torch.tensor(X_val, dtype=torch.float64)

# %%
learning_rates = [0.0001, 0.001, 0.01, 0.1, 1, 10]

epochs = 30

loss_curves = {}
validation_results = {}

for lr in learning_rates:

    # Same initial weights for fair comparison
    W1, b1, W2, b2 = initialize_weights(seed=1187)

    model = NeuralNetwork().double()

    with torch.no_grad():
        model.hidden.weight.copy_(torch.tensor(W1.T))
        model.hidden.bias.copy_(torch.tensor(b1.reshape(-1)))
        model.output.weight.copy_(torch.tensor(W2.T))
        model.output.bias.copy_(torch.tensor(b2.reshape(-1)))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    epoch_losses = []

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:

            optimizer.zero_grad()

            outputs = model(batch_X)

            loss_value = criterion(outputs, batch_y)

            loss_value.backward()

            optimizer.step()

            total_loss += loss_value.item() * len(batch_y)

        average_loss = total_loss / len(train_dataset)
        epoch_losses.append(average_loss)

    model.eval()

    with torch.no_grad():
        predictions = model(X_val_torch).argmax(dim=1).numpy()

    accuracy = accuracy_score(y_val, predictions) * 100

    loss_curves[lr] = epoch_losses
    validation_results[lr] = accuracy

    print("Learning Rate:", lr)
    print("Validation Accuracy:", accuracy)

# %%
plt.figure(figsize=(10, 6))

for lr in learning_rates:
    plt.plot(
        range(1, epochs + 1),
        loss_curves[lr],
        label=f"LR = {lr}"
    )

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Learning Rate Comparison")
plt.legend()
plt.grid()
plt.show()

# %%
print("\nLearning Rate | Validation Accuracy")

for lr in learning_rates:
    print(lr, "|", round(validation_results[lr], 2), "%")




