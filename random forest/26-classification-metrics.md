# 26 - Classification Metrics for Random Forest

How to evaluate a Random Forest classifier honestly.

## Never Use Accuracy Alone

Accuracy is misleading with class imbalance and is not threshold-friendly.
Use a metric aligned with the business goal.

## Core Metrics

### Precision
Of predicted positives, how many are correct? `TP / (TP + FP)`

### Recall (Sensitivity)
Of actual positives, how many were caught? `TP / (TP + FN)`

### F1-Score
Harmonic mean: `2·(P·R)/(P+R)` — balances precision & recall.

### ROC-AUC
Probability that the model ranks a random positive above a random negative.
Threshold-agnostic.

### PR-AUC
Better than ROC-AUC for **imbalanced** data.

## In sklearn

```python
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
print(average_precision_score(y_test, clf.predict_proba(X_test)[:, 1]))
```

## Choosing the Right Metric (Pareto)

| Goal | Primary Metric |
|------|----------------|
| Balanced classes | Accuracy / F1 |
| Catch all fraud | Recall |
| Avoid false alarms | Precision |
| Ranking / Threshold-free | ROC-AUC |
| Rare positive class | PR-AUC |

## Next Steps

- 19 - Class Imbalance
- 27 - Cross-Validation
- 29 - Regression Metrics