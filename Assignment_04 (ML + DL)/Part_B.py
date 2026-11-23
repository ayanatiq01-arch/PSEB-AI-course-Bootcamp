# %%
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(1187)

digits = load_digits()

X = digits.data / 16.0
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1187,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
# %%
# %%
def initialize_weights(seed=1187):

    np.random.seed(seed)

    W1 = np.random.randn(64, 32) * np.sqrt(2 / 64)
    b1 = np.zeros((1, 32))

    W2 = np.random.randn(32, 10) * np.sqrt(2 / 32)
    b2 = np.zeros((1, 10))

    return W1, b1, W2, b2


W1, b1, W2, b2 = initialize_weights()

print("W1 shape:", W1.shape)
print("b1 shape:", b1.shape)
print("W2 shape:", W2.shape)
print("b2 shape:", b2.shape)
# %%
# %%
def forward(X, W1, b1, W2, b2):

    Z1 = np.dot(X, W1) + b1

    A1 = np.maximum(0, Z1)

    Z2 = np.dot(A1, W2) + b2

    exp_values = np.exp(Z2 - np.max(Z2, axis=1, keepdims=True))

    A2 = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    return A1, A2
# %%
# %%
A1, A2 = forward(X_train, W1, b1, W2, b2)

print("Hidden output:", A1.shape)
print("Final output:", A2.shape)

print("First image probabilities:")
print(A2[0])

print("Predicted digit:", np.argmax(A2[0]))
print("Actual digit:", y_train[0])
# %%
# %%
def loss(y_true, y_pred):

    n = len(y_true)

    correct_probabilities = y_pred[np.arange(n), y_true]

    loss_value = -np.mean(np.log(correct_probabilities + 1e-9))

    return loss_value


current_loss = loss(y_train, A2)

print("Initial Loss:", current_loss)
# %%
# %%
def backward(X, y_true, A1, A2, W2):

    n = X.shape[0]

    # Output layer gradient
    dZ2 = A2.copy()
    dZ2[np.arange(n), y_true] -= 1
    dZ2 = dZ2 / n

    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    # Hidden layer gradient
    dA1 = np.dot(dZ2, W2.T)

    dZ1 = dA1 * (A1 > 0)

    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2
# %%
# %%
dW1, db1, dW2, db2 = backward(
    X_train, y_train, A1, A2, W2
)

print("dW1:", dW1.shape)
print("db1:", db1.shape)
print("dW2:", dW2.shape)
print("db2:", db2.shape)
# %%
# %%
def update(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    return W1, b1, W2, b2
# %%
# %%
learning_rate = 0.5
epochs = 100

train_losses = []

for epoch in range(epochs):

    A1, A2 = forward(X_train, W1, b1, W2, b2)

    current_loss = loss(y_train, A2)

    dW1, db1, dW2, db2 = backward(
        X_train, y_train, A1, A2, W2
    )

    W1, b1, W2, b2 = update(
        W1, b1, W2, b2,
        dW1, db1, dW2, db2,
        learning_rate
    )

    train_losses.append(current_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {current_loss:.4f}")
# %%
# %%
_, test_predictions = forward(X_test, W1, b1, W2, b2)

y_pred = np.argmax(test_predictions, axis=1)

accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy:", accuracy * 100, "%")
# %%
# %%
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Neural Network Training Loss")
plt.grid()
plt.show()
# %%
# %%
def numerical_gradient(X, y, W1, b1, W2, b2, param, index, h=1e-5):

    if param == "W1":
        weights = W1
    else:
        weights = W2

    original_value = weights[index]

    weights[index] = original_value + h
    _, pred_plus = forward(X, W1, b1, W2, b2)
    loss_plus = loss(y, pred_plus)

    weights[index] = original_value - h
    _, pred_minus = forward(X, W1, b1, W2, b2)
    loss_minus = loss(y, pred_minus)

    weights[index] = original_value

    return (loss_plus - loss_minus) / (2 * h)


# Use a small batch for gradient checking
X_check = X_train[:10]
y_check = y_train[:10]

A1_check, A2_check = forward(X_check, W1, b1, W2, b2)

dW1_check, db1_check, dW2_check, db2_check = backward(
    X_check, y_check, A1_check, A2_check, W2
)

checks = [
    ("W1", (0, 0), dW1_check[0, 0]),
    ("W1", (5, 10), dW1_check[5, 10]),
    ("W1", (20, 15), dW1_check[20, 15]),
    ("W2", (0, 0), dW2_check[0, 0]),
    ("W2", (15, 5), dW2_check[15, 5])
]

print("Parameter | Backprop Gradient | Numerical Gradient | Difference")

for param, index, backprop_grad in checks:

    numerical_grad = numerical_gradient(
        X_check, y_check, W1, b1, W2, b2, param, index
    )

    difference = abs(backprop_grad - numerical_grad)

    print(
        f"{param}{index} | {backprop_grad:.10f} | "
        f"{numerical_grad:.10f} | {difference:.10e}"
    )
# %%
# %%
W1_zero = np.zeros((64, 32))
b1_zero = np.zeros((1, 32))

W2_zero = np.zeros((32, 10))
b2_zero = np.zeros((1, 10))

A1_zero, A2_zero = forward(
    X_train[:5], W1_zero, b1_zero, W2_zero, b2_zero
)

print("First image hidden neurons:")
print(A1_zero[0])

print("\nAre all hidden neurons identical?")
print(np.all(A1_zero[0] == A1_zero[0][0]))

print("\nFirst row of W1:")
print(W1_zero[0])

print("\nFirst row of W2:")
print(W2_zero[0])
# %%
# %%
def forward_no_relu(X, W1, b1, W2, b2):

    Z1 = np.dot(X, W1) + b1

    A1 = Z1  # ReLU removed

    Z2 = np.dot(A1, W2) + b2

    exp_values = np.exp(
        Z2 - np.max(Z2, axis=1, keepdims=True)
    )

    A2 = exp_values / np.sum(
        exp_values, axis=1, keepdims=True
    )

    return A1, A2
# %%
# %%
def backward_no_relu(X, y_true, A1, A2, W2):

    n = X.shape[0]

    dZ2 = A2.copy()
    dZ2[np.arange(n), y_true] -= 1
    dZ2 /= n

    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)

    dZ1 = dA1  # No ReLU derivative

    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2
# %%
# %%
W1_nr, b1_nr, W2_nr, b2_nr = initialize_weights(seed=1187)

losses_no_relu = []

for epoch in range(100):

    A1_nr, A2_nr = forward_no_relu(
        X_train, W1_nr, b1_nr, W2_nr, b2_nr
    )

    current_loss = loss(y_train, A2_nr)

    dW1_nr, db1_nr, dW2_nr, db2_nr = backward_no_relu(
        X_train, y_train, A1_nr, A2_nr, W2_nr
    )

    W1_nr, b1_nr, W2_nr, b2_nr = update(
        W1_nr, b1_nr, W2_nr, b2_nr,
        dW1_nr, db1_nr, dW2_nr, db2_nr,
        0.5
    )

    losses_no_relu.append(current_loss)

_, test_probs_nr = forward_no_relu(
    X_test, W1_nr, b1_nr, W2_nr, b2_nr
)

pred_nr = np.argmax(test_probs_nr, axis=1)

print("No ReLU Test Accuracy:", accuracy_score(y_test, pred_nr) * 100)
print("Final Training Loss:", losses_no_relu[-1])
# %%
# %%
W1_big, b1_big, W2_big, b2_big = initialize_weights(seed=1187)

big_lr = 50
losses_big_lr = []

for epoch in range(100):

    A1_big, A2_big = forward(
        X_train, W1_big, b1_big, W2_big, b2_big
    )

    current_loss = loss(y_train, A2_big)

    dW1_big, db1_big, dW2_big, db2_big = backward(
        X_train, y_train, A1_big, A2_big, W2_big
    )

    W1_big, b1_big, W2_big, b2_big = update(
        W1_big, b1_big, W2_big, b2_big,
        dW1_big, db1_big, dW2_big, db2_big,
        big_lr
    )

    losses_big_lr.append(current_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {current_loss:.4f}")

_, big_predictions = forward(
    X_test, W1_big, b1_big, W2_big, b2_big
)

big_pred = np.argmax(big_predictions, axis=1)

print(
    "Big Learning Rate Accuracy:",
    accuracy_score(y_test, big_pred) * 100
)
# %%
# %%
def backward_bug(X, y_true, A1, A2, W2):

    n = X.shape[0]

    dZ2 = A2.copy()
    dZ2[np.arange(n), y_true] -= 1
    dZ2 /= n

    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)

    #dZ1 = dA1  # Bug ReLU derivative missing
    dZ1 = dA1 * (A1 > 0)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2
# %%
# %%
A1_check, A2_check = forward(
    X_check, W1, b1, W2, b2
)

bug_dW1, bug_db1, bug_dW2, bug_db2 = backward_bug(
    X_check, y_check, A1_check, A2_check, W2
)

correct_grad = numerical_gradient(
    X_check, y_check, W1, b1, W2, b2,
    "W1", (5, 10)
)

bug_difference = abs(bug_dW1[5, 10] - correct_grad)

print("Buggy Backprop Gradient:", bug_dW1[5, 10])
print("Numerical Gradient:", correct_grad)
print("Difference:", bug_difference)

print("Gradient Check:", "FAIL" if bug_difference >= 1e-6 else "PASS")
# %%
# %%
A1_check, A2_check = forward(
    X_check, W1, b1, W2, b2
)

bug_dW1, _, _, _ = backward_bug(
    X_check, y_check, A1_check, A2_check, W2
)

best_difference = 0
best_index = None
best_numerical = 0

for i in range(64):
    for j in range(32):

        numerical_grad = numerical_gradient(
            X_check, y_check,
            W1, b1, W2, b2,
            "W1", (i, j)
        )

        difference = abs(bug_dW1[i, j] - numerical_grad)

        if difference > best_difference:
            best_difference = difference
            best_index = (i, j)
            best_numerical = numerical_grad

print("Selected Weight:", best_index)
print("Buggy Gradient:", bug_dW1[best_index])
print("Numerical Gradient:", best_numerical)
print("Difference:", best_difference)

print(
    "Gradient Check:",
    "FAIL" if best_difference >= 1e-6 else "PASS"
)
# %%
