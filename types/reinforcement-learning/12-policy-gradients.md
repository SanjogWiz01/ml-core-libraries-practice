# 12. Policy gradients

Value-based methods estimate action scores and choose actions from them. Policy-gradient methods represent a policy directly and adjust its parameters to increase expected return. A stochastic policy can naturally represent uncertainty and continuous actions.

## The policy-gradient idea

For a parameterized policy `πθ(a|s)`, the objective is expected return `J(θ)`. The likelihood-ratio identity gives a gradient estimator of the form:

`∇θ J(θ) = E[∇θ log πθ(A_t|S_t) G_t]`

Intuitively, increase the probability of actions followed by better-than-expected outcomes and decrease it after worse outcomes. The expectation is estimated from sampled trajectories, so gradients can be noisy.

## REINFORCE

REINFORCE uses complete returns to update policy parameters. It is unbiased under standard assumptions but can have high variance, particularly for long episodes. A baseline `b(s)` that does not depend on the chosen action can reduce variance without changing the expected policy gradient:

`∇θ log πθ(a|s) [G_t - b(s)]`

## Entropy regularization

An entropy term encourages a policy to retain action diversity. It can aid exploration, but too much entropy can prevent useful specialization. Its coefficient is a tuning choice, not a safety guarantee.

## Continuous actions

Policies may output parameters of a distribution, such as a Gaussian. Bound actions to the valid range and account for any transformation when computing probabilities and gradients. Unconstrained Gaussian outputs can violate physical or business constraints unless the action parameterization handles them.

## Strengths and limits

Direct optimization handles stochastic and continuous policies. However, on-policy gradients can require many environment interactions, are sensitive to reward scale and step size, and optimize the defined objective—including any flawed proxy. Report variability and constraint outcomes alongside mean return.

## Practice

Implement REINFORCE with and without a learned baseline on a tiny episodic task. Compare gradient variance and learning curves across seeds. Verify that sampled action log probabilities correspond to the actions actually taken.
