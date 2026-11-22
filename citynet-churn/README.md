# CityNet churn baseline

## User story
As the retention manager, I want a weekly list of the 100 customers most likely to cancel next month, so my team can call them first.

## How to run
    pip install -r requirements.txt
    python src/lab7_save.py                       # train and save the pipeline
    python src/predict.py data/new_customers.csv  # writes reports/predictions.csv

## Results (validation, then test once)
| Metric | Baseline | Our model |
|---|---|---|
| PR-AUC | ____ | ____ |
| Churners in top 100 | ____ | ____ |

## Intended use
____

## Not for
____

## Assumptions
____


## Risks and limitations
- Misses quiet long-time customers (missed churners had 43.2 months of tenure vs 18.9 for caught ones).
- Not fair across regions: East churners are caught only 50% of the time (Central 76%).
- Test PR-AUC is 0.361 (validation 0.396).
