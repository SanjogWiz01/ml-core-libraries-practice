# 02 - Decision Trees: The Building Blocks

Random Forest is a forest of decision trees. Understanding decision trees is
essential before studying the ensemble.

## What is a Decision Tree?

A tree that recursively partitions the feature space into regions. Each node
tests a feature, each branch is an outcome, and each leaf holds a prediction.

## Splitting Criteria

### Classification
- **Gini Impurity**: `1 - Σ p_i²`
- **Entropy**: `- Σ p_i log(p_i)`
- Lower value = purer node

### Regression
- **Mean Squared Error (MSE)**: `Σ(y - ŷ)² / n`
- **Mean Absolute Error (MAE)**

## How a Split is Chosen

For each candidate feature, the algorithm evaluates every threshold and picks
the one that most reduces impurity (or variance for regression).

## Tree Growth Parameters

- `max_depth` - max levels
- `min_samples_split` - min samples to split a node
- `min_samples_leaf` - min samples per leaf
- `max_features` - features considered per split
- `max_leaf_nodes` - cap on leaf count

## Greedy Nature

Splits are chosen greedily (locally optimal). This makes individual trees
highly variable and prone to **overfitting**, which is exactly what the Random
Forest ensemble corrects.

## Pitfall: Instability

Small changes in data can produce very different trees. This high variance is
the weakness the ensemble averaging exploits.

## Next Steps

- 03 - Bootstrap Sampling
- 04 - Random Feature Subspace
