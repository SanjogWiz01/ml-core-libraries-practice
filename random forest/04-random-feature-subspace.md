# 04 - Random Feature Subspace

This is the second source of randomness in Random Forest (and its key
differentiator from plain bagging).

## The Idea

At each split, instead of considering **all** features, the algorithm picks a
**random subset of features** and only searches for the best split among them.

## Why It Matters

- With bootstrap sampling only, the strongest features tend to dominate every
  tree → trees stay correlated → variance reduction is limited.
- By forcing trees to consider different subsets, trees become **more
  decorrelated**, improving the ensemble's generalization.

## How Many Features? (max_features)

Conventionally:
- **Classification**: `max_features = sqrt(p)`
- **Regression**: `max_features = p/3`

Practical options in sklearn:
- `"sqrt"`, `"log2"`, or an integer / float fraction.
- Good default: `"sqrt"` for classification.

## Trade-off

- Too few features → individual trees very noisy (high bias).
- Too many features → trees too correlated (less variance reduction).
- A sweet spot (usually sqrt or log2 for classification) works best.

## Two Randomnesses Summary

| Layer | Randomness |
|-------|------------|
| Data   | Bootstrap sampling (03) |
| Features | Random subspace at each split (04) |

## Intuition

Feature subsampling is like having each expert only look at their specialty
instead of every expert scrutinizing the same details — the collective
decision becomes more robust.

## Next Steps

- 05 - Ensemble & Bagging
- 15 - Max Features Tuning
