import numpy as np

S = 1187

digits = [int(d) for d in str(S)]

x1 = np.array(digits[:2], dtype=float) / 5
x2 = np.array(digits[2:], dtype=float) / 5
x3 = np.array([1.0, 1.0])

# Same vectors are used as Query, Key and Value
X = np.array([x1, x2, x3])

# Dot-product scores divided by sqrt(2)
scores = X @ X.T / np.sqrt(2)

# Softmax row-wise
exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
attention = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

# Attention output
output = attention @ X

print("Token vectors:")
print(X)

print("\nAttention matrix:")
print(np.round(attention, 4))

print("\nOutput vectors:")
print(np.round(output, 4))

print("\nToken 1 attention weights:")
print(np.round(attention[0], 4))

print("\nToken 1 output:")
print(np.round(output[0], 4))