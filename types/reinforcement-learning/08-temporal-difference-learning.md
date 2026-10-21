# 8. Temporal-difference learning

Temporal-difference (TD) learning updates a value estimate after each transition by combining the observed reward with a bootstrapped estimate of the next value.

## TD(0) prediction

For a transition `(S_t, A_t, R_{t+1}, S_{t+1})`:

`δ_t = R_{t+1} + γ V(S_{t+1}) - V(S_t)`

`V(S_t) ← V(S_t) + α δ_t`

At a true terminal transition, use zero continuation value. TD can learn before the episode ends and often has lower variance than MC, but it can be biased because it bootstraps from an imperfect estimate.

## TD versus Monte Carlo

MC learns from the full observed return: less bootstrap bias, potentially high variance, and usually waits for episode completion. TD learns from a one-step target: more frequent updates and often lower variance, but depends on the current value estimate. Neither method dominates universally.

## n-step returns and eligibility traces

An n-step return uses the next `n` rewards, then bootstraps. One-step TD is the short end; MC is the full-episode end. Eligibility traces let recent states or features share credit over multiple steps. The parameter `λ` controls the blend in TD(λ): values near zero emphasize one-step TD; values near one approach MC-like credit assignment under common formulations.

## Why off-policy learning is delicate

TD methods can learn from a behavior policy different from the target policy. In tabular settings, some off-policy algorithms have convergence guarantees under required coverage and step-size conditions. With function approximation, the combination of off-policy data, bootstrapping, and approximation can diverge. Monitor learning and use algorithms designed for the setting.

## Practice

Implement TD(0) and MC prediction for the same small environment. Use the same episode set where possible, compare learning curves across repeated seeds, and explain differences in update timing and target variance.
