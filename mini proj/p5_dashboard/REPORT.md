# Report for City Hospital: screening model dashboard

## Experiments (validation set, one change at a time)
| Run | Diagnosis | Cancers found | False alarms |
|---|---|---|---|
| 256 neurons, no fixes | overfitting | 41 of 43 | 3 |
| + dropout 0.3, decay 0.05 | overfitting | 40 of 43 | 2 |
| + early stopping (20) | healthy | 40 of 43 | 3 |
| 16 neurons + all fixes | healthy | 40 of 43 | 1 |
| same, not scaled | underfitting (curves jump around) | 33 of 43 | 2 |

The big model overfits after epoch 23: training loss goes to zero while validation loss rises (best 0.125 at epoch 23). Dropout and weight decay alone barely changed the curve; early stopping plus a smaller model made it healthy. Without scaling the 30 measurements (0.001 to about 2,500) make the optimiser overshoot and the losses jump around.

## Recommendation
16 neurons, dropout 0.3, weight decay 0.05, early stopping with patience 20, scaling on (the scaler is fitted on the training rows only).

## Final test (run once, chosen settings)
Accuracy 0.956 · 41 of 42 cancers found · 4 false alarms.

## Three reasons it is not ready for real patients
1. One public dataset of 569 patients: no proof it works on our hospital's scans or patients.
2. It still missed 1 cancer in the final test; a missed cancer costs far more than a false alarm, so the 0.5 threshold should be lowered and checked with doctors.
3. It gives no explanation for a result: a doctor must review every case, and the model must be checked again whenever the scanner or patient group changes.
