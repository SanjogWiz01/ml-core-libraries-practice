# 6. Dynamic programming

Dynamic programming (DP) solves planning problems when the transition and reward model is known and the state/action space is manageable. It is also the conceptual foundation for many model-free methods.

## Policy evaluation

For a fixed policy, repeatedly update each state toward the expected Bellman backup:

`V(s) ← Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) [r + γ V(s')]`

This is **iterative policy evaluation**. Stop when values change less than a chosen tolerance, while recognizing that a numerical stopping threshold is not evidence that the model is correct.

## Policy improvement

Given values, improve the policy by choosing an action with the largest expected one-step return:

`π_new(s) ∈ argmax_a Σ_{s',r} p(s',r|s,a) [r + γ V(s')]`

Alternating evaluation and improvement is **policy iteration**. **Value iteration** combines them by repeatedly applying the optimality backup. In finite discounted MDPs, these algorithms converge under standard assumptions.

## Computational considerations

Exact sweeps cost roughly the number of state-action transitions considered per iteration. Large or continuous state spaces make full enumeration impossible. Sparse transitions, prioritized updates, approximate value functions, and sampled planning can reduce cost but introduce their own assumptions.

## Why data scientists should learn DP

DP makes the role of a model explicit. It provides a reference solution for small problems, helps validate an environment implementation, and separates planning error from learning error. If a simulator is available, compare a learned policy against a DP solution on a tiny version of the problem.

## Model accuracy matters

A planner can exploit small model errors. If the model incorrectly predicts a cheap shortcut or ignores rare failures, optimization may discover and repeatedly use that flaw. Validate transitions and rewards in high-impact states, not just average prediction accuracy.

## Practice

Create a small grid world with a terminal goal and a cost per step. Implement iterative policy evaluation and value iteration. Check that the policy avoids walls and that changing the step cost changes path preferences as expected. Keep the state space small enough to inspect all transitions by hand.
