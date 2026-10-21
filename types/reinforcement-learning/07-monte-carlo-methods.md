# 7. Monte Carlo methods

Monte Carlo (MC) methods estimate values from sampled returns. They do not need a known transition model and do not bootstrap from a value estimate, but typically need complete episodes for straightforward updates.

## First-visit state-value estimation

For each episode, record the return following each visit to state `s`. First-visit MC averages returns after the first occurrence of `s` in each episode. Every-visit MC averages after all occurrences. With enough representative episodes and finite returns, sample averages approach expected values.

An incremental mean avoids storing every return:

`V(s) ← V(s) + (G - V(s))/N(s)`

## Control with exploring starts and epsilon-greedy

To improve a policy, estimate `Q(s,a)` for state-action pairs, then choose higher-value actions. Every relevant pair must be sampled enough. Exploring starts assume episodes can begin from any pair, which is often unrealistic; epsilon-greedy behavior is a more practical exploration strategy in many small problems.

## Off-policy MC and importance sampling

MC can estimate a target policy using data from a different behavior policy when action probabilities are known and coverage is sufficient. For a trajectory, an importance ratio multiplies target-policy probabilities divided by behavior-policy probabilities. Ratios can have enormous variance, especially over long horizons. Weighted variants can reduce variance but introduce different statistical properties.

## Strengths and limits

- Strength: conceptually simple, unbiased return targets under suitable sampling.
- Strength: no value bootstrapping, so no bootstrap target error.
- Limit: high variance, especially with long episodes or sparse rewards.
- Limit: complete returns may be unavailable until an episode ends.
- Limit: coverage depends on the data-generating policy.

## Data-science lesson

An offline trajectory is not automatically informative about every action. If the behavior policy almost never selected an action in a state, MC cannot reliably estimate what would have happened under that action. This is a data coverage problem, not a coding bug.

## Practice

On a small episodic environment, compare MC value estimates with exact DP values. Increase the number of episodes and observe variability. Then shorten the horizon or add sparse rewards and note how sample efficiency changes.
