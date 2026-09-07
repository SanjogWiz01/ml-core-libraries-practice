# 34 - Common Pitfalls & How to Avoid Them

Practical mistakes data scientists make with Random Forest.

## 1. Over-Tuning Everything

**Problem**: Grid-searching 10+ hyperparameters overfits the validation set.
**Fix**: Tune the top 3–4 knobs (15,16,17). Random Forest is robust.

## 2. Trusting Inner Importance Blindly

**Problem**: MDI favors high-cardinality features (13).
**Fix**: Confirm rankings with permutation importance (22).

## 3. Ignoring Class Imbalance

**Problem**: High accuracy but zero minority recall (19).
**Fix**: Use `class_weight='balanced'`, right metrics (26).

## 4. Data Leakage in Preprocessing

**Problem**: Fitting imputation/encoding on full data before CV bakes in
leakage (27).
**Fix**: Fit everything inside a Pipeline per fold.

## 5. Assuming Extrapolation

**Problem**: Expecting predictions beyond the training target range (07).
**Fix**: Understand that RF predicts inside the observed envelope.

## 6. Memory Blow-Up

**Problem**: 10,000 deeply-grown trees on wide data eat RAM.
**Fix**: Cap `max_depth`/`max_leaf_nodes`, tune `n_estimators` (16), use
`n_jobs` wisely.

## 7. Missing Values Silently Dropped

**Problem**: Dropping NaN rows silently biases results (18).
**Fix**: Impute explicitly; add missingness flags.

## 8. Miscalibrated Probabilities

**Problem**: Using raw `predict_proba` for business thresholds (25).
**Fix**: Calibrate, then threshold-tune.

## Pareto Reminder

Roughly 20% of mistakes cause 80% of damage: **leakage, imbalance, and
misinterpreted importance** top that list.

## Next Steps

- 35 - Cheat Sheet & Recap