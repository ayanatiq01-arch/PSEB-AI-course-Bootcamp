# CityNet churn: weekly call list

**User story:** As the retention manager at CityNet, I want a weekly list of the 100 customers most likely to cancel next month, so my team can call them first.

## How to run (from the project folder)
```
python -m venv .venv          # then activate it
pip install -r requirements.txt
python src/lab1_inspect.py    # look at the data
python src/lab2_split.py      # 60/20/20 stratified split
python src/lab3_pipeline.py   # preprocessing pipeline
python src/lab4_candidates.py # baseline and candidates (5-fold CV)
python src/lab5_evaluate.py   # validation metrics and top-100
python src/lab6_errors.py     # error analysis, test score (once)
python src/lab7_save.py       # save models/churn_pipeline.joblib
python src/predict.py data/new_customers.csv   # score new customers -> reports/predictions.csv
```
Everything uses seed 42, so results repeat on every laptop.

## Results
| Model (5-fold, training rows) | PR-AUC | Recall | Precision |
|---|---|---|---|
| Baseline (always "stays") | 0.179 | 0.000 | 0.000 |
| **Logistic regression (chosen)** | **0.407** | 0.669 | 0.306 |
| Random forest | 0.390 | 0.544 | 0.341 |
| Hist. gradient boosting (extra) | 0.361 | 0.399 | 0.339 |

- Validation: PR-AUC 0.396; top-100 holds **49 churners** vs about 18 for a random 100 (2.7 times).
- **Test (run once): PR-AUC 0.361**; top-100 holds 51 churners vs about 18.
- Leak experiment: adding `cancel_request_logged` gives PR-AUC 0.909. That is leakage, so it is not used.
- Decision: logistic regression goes forward (best PR-AUC, simpler to explain).

## Acceptance criteria
- [x] 1. Top-100 holds at least twice as many real churners as a random 100 (49 vs 18 on validation, 51 vs 18 on test)
- [x] 2. Test-set PR-AUC reported (0.361) and beats the baseline (0.179)
- [x] 3. One command scores new customer rows (`python src/predict.py data/new_customers.csv`)
- [x] 4. No leaking column used (`cancel_request_logged` removed, see DATA.md)
- [x] 5. Data, assumptions, intended use and risks written down (DATA.md and this file)
- [x] 6. Everything is in Git and reruns with seed 42

## Intended use
Rank customers for a weekly retention call list. A human decides who to call and what to offer.

## Not for
Changing prices or contracts automatically. Judging individual customers without a call.

## Assumptions
- The last 90 days predict next month.
- New customers look like the training data.
- Data is a snapshot; retrain every quarter.

## Risks and limitations
- **Misses quiet long-time customers:** churners caught had tenure 18.9 months; churners missed had 43.2 months.
- **East region:** only 50% of East churners are caught (Central 76%, North 73%, West 71%, South 68%).
- The threshold depends on team capacity: at 0.3 the model catches 90% of churners (recall 0.899) but precision falls to 0.234, so many more wrong calls.
- Weights show association, not proof of cause.

## What drives churn (permutation importance)
tenure 0.13 · auto_pay 0.044 · support calls 0.043 · paid late 0.026
