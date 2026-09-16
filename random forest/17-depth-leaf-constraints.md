# 17 - Depth & Leaf Constraints

These parameters control tree complexity and overfitting.

## Parameters

### max_depth
- Maximum number of levels from root to leaf.
- `None` → trees grow until pure or until other constraints.
- Smaller depth → simpler trees → less overfitting.

### min_samples_split
- Minimum samples required to split a node.
- Larger → fewer splits → simpler forest.

### min_samples_leaf
- Minimum samples a leaf must contain.
- Larger → smoother, more regularized predictions.

### max_leaf_nodes
- Caps the total number of leaves per tree.

## Overfitting in Random Forest

Random Forest overfits **far less** than a single tree (because of averaging),
but it still can — especially with deep, unconstrained trees on small data.

## When to Regularize

- Very small datasets.
- Noisy features / label noise.
- When validation error is much worse than training error.

## Practical Starting Point

- Keep `max_depth=None` as a baseline; Random Forest rarely needs deep pruning.
- Use `min_samples_leaf` (e.g. 1–10) to reduce variance with little bias cost.
- Let `max_features` do the heavy lifting (see 15).

## Best Practice (Pareto)

Don't exhaustively grid-search all three. Set `min_samples_leaf` and
`min_samples_split` sensibly, and only constrain `max_depth` if you observe
overfitting.

## Next Steps

- 18 - Imputation & Missing Values
- 15 - Max Features Tuning
