# 01 - Introduction to Random Forest

Random Forest is a supervised ensemble learning algorithm used for both
classification and regression. It builds a large number of decision trees and
combines their outputs to produce a more robust and accurate prediction.

## Key Idea

- Build many decision trees on **bootstrapped** samples of the data.
- At each split, only a **random subset of features** is considered.
- For classification: each tree votes, the majority class wins.
- For regression: predictions of trees are averaged.

## Why It Works

The core insight is the **wisdom of crowds**: many weak learners (trees) that
are individually noisy can, when averaged, produce a strong learner. The
randomness injected at two levels (data sampling + feature subsampling)
**decorrelates** the trees, which reduces variance without a large increase in
bias.

## Mental Model

```
             ┌── Tree 1 ──┐
Input ───────┼── Tree 2 ──┼── Combine (vote / average) ── Output
             └── Tree N ──┘
```

## When to Use

- Structured / tabular data
- Needs to handle non-linear relationships
- Robust to outliers and irrelevant features out of the box

## References in this series

- 03 - Bootstrap Sampling
- 04 - Random Feature Subspace
- 05 - Ensemble & Bagging
