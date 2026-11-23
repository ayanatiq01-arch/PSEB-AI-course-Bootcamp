# import torch, torch.nn as nn
# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# S = 1234  
# torch.manual_seed(S)
# X, y = load_digits(return_X_y=True)
# X = StandardScaler().fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=S, stratify=y)
# X_train = torch.tensor(X_train, dtype=torch.float32)
# X_test = torch.tensor(X_test, dtype=torch.float32)
# y_train = torch.tensor(y_train)
# y_test = torch.tensor(y_test)

# model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5),
#     nn.Linear(32, 10), nn.Softmax(dim=1))

# loss_fn = nn.CrossEntropyLoss()
# opt = torch.optim.SGD(model.parameters(), lr=0.1)

# for epoch in range(100):
#     loss = loss_fn(model(X_train), y_train)
#     loss.backward()
#     opt.step()

# pred = model(X_train).argmax(dim=1)
# acc = (pred == y_train).float().mean().item()
# print("test accuracy:", round(acc, 3))
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

S = 1187
torch.manual_seed(S)

X, y = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=S, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

model = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(32, 10)
)

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    model.train()

    opt.zero_grad()
    loss = loss_fn(model(X_train), y_train)
    loss.backward()
    opt.step()

model.eval()

with torch.no_grad():
    pred = model(X_test).argmax(dim=1)
    acc = (pred == y_test).float().mean().item()

print("True test accuracy:", round(acc, 3))