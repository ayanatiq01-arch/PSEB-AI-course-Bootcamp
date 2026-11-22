import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import make_scorer, precision_score

NUM = ["monthly_bill", "tenure_months", "support_calls_90d", "outage_hours_90d", "age"]
CAT = ["region", "plan", "tv_bundle", "payment_method", "auto_pay", "paid_late_last_bill"]

df = pd.read_csv("data/citynet_churn.csv")
X, y = df[NUM + CAT], df["churned"]
X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=42)

def make_preprocess(scale=True):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    numeric = Pipeline(steps)
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUM), ("cat", categorical, CAT)])

pipe = Pipeline([("prep", make_preprocess()),
                 ("model", SVC(kernel="linear", class_weight="balanced"))])
pipe.fit(X_train, y_train)

# ===== CUSTOMERS WHO ALREADY LEFT (churned = 1) =====
left = df[df["churned"] == 1].drop(columns=["churned"]).sort_values("customer_id")
print("Customers who left:", len(left))
print(left.head(10).to_string(index=False))
left.to_csv("churned_customers.csv", index=False)

# ===== PREPROCESSING REPORT: kis column mein kya hua =====
prep = pipe[0]
num_pipe = prep.named_transformers_["num"]
cat_pipe = prep.named_transformers_["cat"]

print("\n=== 1. FILL GAPS · numeric columns (median) ===")
for col, miss, med in zip(NUM, X_train[NUM].isna().sum(), num_pipe["impute"].statistics_):
    print(f"{col:20s} missing: {miss:4d} | median: {med:8.2f} | " +
          ("gaps bhare gaye" if miss > 0 else "koi gap nahi"))

print("\n=== 2. FILL GAPS · text columns (most_frequent) ===")
for col, miss, val in zip(CAT, X_train[CAT].isna().sum(), cat_pipe["impute"].statistics_):
    print(f"{col:20s} missing: {miss:4d} | most frequent: {val:14s} | " +
          ("gaps bhare gaye" if miss > 0 else "koi gap nahi"))

print("\n=== 3. SCALE · numeric columns (StandardScaler: (x - mean) / std) ===")
for col, m, sd in zip(NUM, num_pipe["scale"].mean_, num_pipe["scale"].scale_):
    print(f"{col:20s} mean: {m:8.2f} | std: {sd:8.2f}")

print("\n=== 4. ONE-HOT · text columns ===")
cats = cat_pipe["onehot"].categories_
for col, c in zip(CAT, cats):
    print(f"{col:20s} -> {len(c)} columns: {list(c)}")

print("\nTotal columns: 5 numeric + ", sum(len(c) for c in cats), "one-hot =",
      len(prep.get_feature_names_out()))

# ===== SCORE, TOP 100 aur REASONS =====
score = pipe.decision_function(X_val)
top = np.argsort(-score)[:100]
print("\nTop-100 churners:", y_val.to_numpy()[top].sum(),
      "| random 100 would find about", round(100 * y_val.mean()))

# ===== TOP-100 CALL LIST (CSV) =====
call_list = df.loc[X_val.index[top], ["customer_id"] + NUM + CAT].copy()
call_list["svm_score"] = score[top].round(3)
call_list["actually_left"] = y_val.to_numpy()[top]      # 1 = sach mein chala gaya
call_list.to_csv("top100_call_list.csv", index=False)
print(call_list.head(10).to_string(index=False))

w = pd.Series(pipe[-1].coef_[0], index=pipe[0].get_feature_names_out())
print("\nReasons to LEAVE:\n", w.nlargest(3).round(2))
print("\nReasons to STAY:\n", w.nsmallest(3).round(2))

# ===== PR-AUC: scaling ke saath aur bina =====
def test(name, prep):
    p = Pipeline([("prep", prep), ("model", SVC(kernel="linear", class_weight="balanced"))])
    t0 = time.time()
    cv = cross_validate(p, X_train, y_train, cv=5, scoring={
        "pr_auc": "average_precision", "recall": "recall",
        "precision": make_scorer(precision_score, zero_division=0)})
    print(name, {m: round(cv[f"test_{m}"].mean(), 3) for m in ["pr_auc", "recall", "precision"]},
          f"{time.time() - t0:.1f}s")

print()
test("SVM linear", make_preprocess(scale=True))
test("SVM linear, no scaling", make_preprocess(scale=False))

# ===== PLOTS: kon kon chor chuke (sab se aakhir mein, kyunki plt.show() window band hone tak rukta hai) =====
base = df.churned.mean() * 100                      # overall churn rate (17.9%)
teal, grey, orange = "#0E7C86", "#9AA5B1", "#F28C28"

fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
fig.suptitle(f"Who left CityNet? {df.churned.sum()} of {len(df):,} customers ({base:.1f}%)",
             fontsize=18, fontweight="bold", color="#0B3C49")

def bar(a, s, title, xlabel, color=teal):           # churn % bar chart
    b = a.bar([str(i) for i in s.index], s.values, color=color)
    a.axhline(base, color="#C8553D", ls="--", lw=1.2)   # red line = overall churn
    a.set_title(title, fontsize=13, fontweight="bold")
    a.set_ylabel("% who left"); a.set_xlabel(xlabel); a.set_ylim(0, max(s.values) * 1.2)
    for r, v in zip(b, s.values):
        a.text(r.get_x() + r.get_width() / 2, v + 0.8, f"{v:.0f}%", ha="center", fontsize=11)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)

# 1 · Left vs stayed (count)
a = ax[0, 0]
cnt = df.churned.value_counts().sort_index()
b = a.bar(["Stayed", "Left"], cnt.values, color=[grey, teal])
for r, v in zip(b, cnt.values):
    a.text(r.get_x() + r.get_width() / 2, v + 60, f"{v:,}", ha="center", fontsize=12, fontweight="bold")
a.set_title("Customers who left vs stayed", fontsize=13, fontweight="bold")
a.set_ylabel("customers"); a.set_ylim(0, 5000)
for sp in ("top", "right"):
    a.spines[sp].set_visible(False)

# 2 · Tenure bands
t = pd.cut(df.tenure_months, [0, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"])
bar(ax[0, 1], df.groupby(t, observed=True).churned.mean() * 100, "By tenure (months)", "tenure months")

# 3 · Support calls (5 aur zyada ek group mein)
sc = df.groupby(df.support_calls_90d.clip(upper=5)).churned.mean() * 100
sc.index = [str(i) if i < 5 else "5+" for i in sc.index]
bar(ax[0, 2], sc, "By support calls (last 90 days)", "support calls")

# 4 · Auto-pay
bar(ax[1, 0], df.groupby("auto_pay").churned.mean() * 100, "By auto-pay", "auto-pay on?", orange)

# 5 · Paid last bill late
bar(ax[1, 1], df.groupby("paid_late_last_bill").churned.mean() * 100,
    "By late payment of last bill", "paid last bill late?", orange)

# 6 · Leavers by region (count)
a = ax[1, 2]
reg = df[df.churned == 1].region.value_counts()
b = a.barh(reg.index[::-1], reg.values[::-1], color=teal)
for rr, v in zip(b, reg.values[::-1]):
    a.text(v + 4, rr.get_y() + rr.get_height() / 2, str(v), va="center", fontsize=11)
a.set_title("Customers who left, by region", fontsize=13, fontweight="bold")
a.set_xlabel("customers who left")
for sp in ("top", "right"):
    a.spines[sp].set_visible(False)

fig.text(0.5, 0.01, f"Red dashed line = overall churn rate ({base:.1f}%)", ha="center", fontsize=11, color="#C8553D")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("who_left_plots.png", dpi=110)        
plt.show()                                          