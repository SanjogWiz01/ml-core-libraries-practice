# 03 - Bootstrap Sampling (Bagging Basics)

Bagging = **B**ootstrap **Agg**regat**ing**. It is the foundation of Random
Forest's data-level randomness.

## What is Bootstrap?

A resampling technique where new datasets are created by **sampling with
replacement** from the original data, each of the same size as the original.

## How it Works in Random Forest

1. Start with N training samples.
2. For each tree, draw N samples **with replacement** from the training set.
3. Each tree trains on its own bootstrap sample (~63.2% unique rows).
4. The leftover samples (~36.8%) form the **out-of-bag (OOB)** set.

## Why Replacement Matters

- Without replacement, all trees would see identical data (no diversity).
- With replacement, different trees see different subsets, decorrelating them.

## The ~63.2% Rule

The probability that a given sample is *not* selected in one draw is
`(1 - 1/N)^N`, which approaches `e⁻¹ ≈ 0.368`. So ~63.2% appear at least once.

## Benefits

- Provides free **OOB validation** (see 12 - OOB Score).
- Injects variance among trees → reduces overall variance of the ensemble.
- Lets each tree treat an `in_bag` vs `out_of_bag` split without a holdout set.

## Relationship to Bagging

Bagging averages many models trained on bootstrap samples to reduce variance.
Random Forest extends bagging by also randomizing features (see 04).

## Next Steps

- 04 - Random Feature Subspace
- 05 - Ensemble & Bagging
