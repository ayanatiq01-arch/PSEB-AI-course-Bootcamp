import pandas as pd
df = pd.read_csv("data/citynet_churn.csv")
print(df.shape)
print(df.dtypes.value_counts().to_dict())
missing = df.isna().sum()
print(missing[missing > 0].to_dict())
print(df["churned"].value_counts(normalize=True).round(3).to_dict())
print(df[["tenure_months", "support_calls_90d", "outage_hours_90d"]].describe().loc[["mean", "min", "max"]].round(1))
print(df.groupby("support_calls_90d")["churned"].mean().round(2).head(6).to_dict())
print(df.groupby("cancel_request_logged")["churned"].mean().round(3).to_dict())

# TODO 1: Which column is missing values, and how many?
# TODO 2: What share of customers churned? Is the data balanced?
# TODO 3: One column predicts churn almost perfectly. Which one, and why must we NOT use it?
# ANSWER 1: age is missing 411 values and payment_method is missing 149 (5,000 rows)
# ANSWER 2: 17.9% churned (0.179). Not balanced, so accuracy would mislead; use PR-AUC
# ANSWER 3: cancel_request_logged (95.8% churn when 1 vs 3.3% when 0). It is logged AFTER the customer
#           decides to leave, so it leaks the answer and is not known at prediction time.