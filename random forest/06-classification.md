# 06 - Random Forest for Classification

Random Forest handles classification by aggregating tree votes.

## How Predictions are Made

1. Each tree classifies the sample.
2. Trees "vote" for a class.
3. The class with the **most votes** (hard voting) is the prediction.
4. Optionally, class probabilities are averaged (soft voting).

## Class Probabilities

sklearn's `predict_proba` returns the mean of per-tree predicted class
fractions:

```
p(class c) = (1/T) · Σ_tree [fraction of c in tree's leaf]
```

This is useful for ranking confidence and for ROC curves.

## Key Concepts

- **Hard voting**: majority class label.
- **Soft voting**: average of probabilities (smoother, uses confidence).
- **Decision boundary**: ensemble boundaries are smoother than single trees.

## Weighted Classes (Imbalance)

Set `class_weight='balanced'` to automatically weight classes inversely to
their frequency, helping minority classes.

## Evaluation Metrics

- Accuracy
- Precision / Recall / F1
- ROC-AUC
- Confusion matrix

## Example Sketch (sklearn)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
```

## Next Steps

- 07 - Regression
- 09 - Sklearn Basics
- 19 - Class Weighting & Imbalance
