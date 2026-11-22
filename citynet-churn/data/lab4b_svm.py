import time
import numpy as np
import pandas as pd
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

score = pipe.decision_function(X_val)             

top = np.argsort(-score)[:100]                      
print("Top-100 churners:", y_val.to_numpy()[top].sum(),
      "| random 100 would find about", round(100 * y_val.mean()))   

w = pd.Series(pipe[-1].coef_[0], index=pipe[0].get_feature_names_out())
print("\nReasons to LEAVE:\n", w.nlargest(3).round(2))              
print("\nReasons to STAY:\n", w.nsmallest(3).round(2))

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