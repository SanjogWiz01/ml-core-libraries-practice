# 19 - Class Imbalance & Class Weighting

Random Forest struggles with imbalanced classes unless handled correctly.

## The Problem

If class A = 95% and class B = 5%, majority voting and the default
`min_samples*` behavior push the forest toward predicting the majority class
almost always — high accuracy but useless for the minority.

## Solution 1: class_weight

```python
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
```

- `'balanced'` weights each class inversely to its frequency.
- Or pass a dict: `{0: 1.0, 1: 10.0}`.
- Applies to Gini computations, biasing splits toward minorities.

## Solution 2: Resampling

- **Oversample** minority (SMOTE, random over-sampling).
- **Undersample** majority.
- Do resampling inside CV folds to avoid leakage.

## Solution 3: Sample Weights

```python
rf.fit(X_train, y_train, sample_weight=weights)
```

- Manual control per-sample.

## Evaluation Matters

Never evaluate imbalanced models with accuracy alone. Use:

- Precision / Recall / F1 (especially for the minority class)
- ROC-AUC / PR-AUC
- Confusion matrix

## Practical Advice (Pareto)

1. Start with `class_weight='balanced'` — one line, big effect.
2. Tune threshold on predicted probabilities (soft voting) to favor minority.
3. Add SMOTE only if needed.

## Next Steps

- 20 - Categorical & Scaling
- 26 - Classification Metrics
