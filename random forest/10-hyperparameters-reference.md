# 10 - Hyperparameters Reference

A complete reference of Random Forest hyperparameters you will tune.

## Key Hyperparameters

| Parameter | Meaning | Default | Typical Range |
|-----------|---------|---------|---------------|
| `n_estimators` | number of trees | 100 | 100–1000 |
| `max_depth` | max tree depth | None | 5–30 |
| `min_samples_split` | samples to split | 2 | 2–20 |
| `min_samples_leaf` | samples per leaf | 1 | 1–10 |
| `max_features` | features per split | sqrt | 'sqrt','log2',0.1–1.0 |
| `max_leaf_nodes` | cap on leaves | None | 10–1000 |
| `max_samples` | bootstrap sample size | None | 0.5–1.0 |
| `bootstrap` | use bootstrap | True | True/False |
| `class_weight` | class balancing | None | 'balanced' |
| `criterion` | split quality | 'gini' | 'gini','entropy','mse' |
| `n_jobs` | parallel cores | None | -1 |
| `oob_score` | use OOB eval | False | True/False |

## Which Matter Most (Pareto 80/20)

1. `n_estimators` — more trees, more stable; diminishing returns.
2. `max_features` — biggest effect on decorrelation.
3. `max_depth` / `min_samples_leaf` — control overfitting.
4. `class_weight` — critical for imbalanced data.

## Don't Over-Tune

Random Forest is fairly robust. Tune the top 3–4 parameters; leave the rest at
defaults. Over-tuning invites overfitting to the validation set.

## Pareto Reasoning

Roughly 20% of the parameters drive 80% of performance gains. Focus effort
there first (see 15, 16, 17).

## Next Steps

- 15 - Max Features Tuning
- 16 - Number of Trees (Estimators)
- 17 - Depth & Leaf Constraints
