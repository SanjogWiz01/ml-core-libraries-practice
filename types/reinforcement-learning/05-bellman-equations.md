# 5. Bellman equations

Bellman equations express a value as immediate reward plus discounted value of what follows. They are the bridge between sequential objectives and practical algorithms.

For a fixed policy `π`:

`Vπ(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) [r + γ Vπ(s')]`

The corresponding action value is:

`Qπ(s,a) = Σ_{s',r} p(s',r|s,a) [r + γ Σ_{a'} π(a'|s') Qπ(s',a')]`

The expectation averages over actions chosen by the policy and uncertain environment outcomes.

## Optimality equations

For an optimal policy:

`V*(s) = max_a Σ_{s',r} p(s',r|s,a) [r + γ V*(s')]`

`Q*(s,a) = Σ_{s',r} p(s',r|s,a) [r + γ max_{a'} Q*(s',a')]`

The maximum makes the equation a control problem. If the model is known and the state/action spaces are small, these equations can be solved iteratively. If the model is unknown, sample transitions can estimate the targets.

## Bellman operator intuition

Applying the policy Bellman operator replaces a value estimate with one-step lookahead under the policy. For finite discounted problems, repeated application converges under standard conditions. In approximate learning, however, function approximation, bootstrapping, and off-policy data can interact unstably; the tabular convergence intuition does not automatically carry over.

## Bellman error

For an estimated value function, a one-step temporal-difference residual is:

`δ_t = R_{t+1} + γ V(S_{t+1}) - V(S_t)`

For action values in Q-learning:

`δ_t = R_{t+1} + γ max_a Q(S_{t+1},a) - Q(S_t,A_t)`

The residual is a learning signal. A small average residual does not by itself prove good decisions: errors can cancel, the data may omit important states, and model or reward misspecification can remain.

## Terminal states

At a true terminal state, there is no future value to bootstrap from; its continuation value is zero. Do not confuse termination with a time-limit truncation. If an episode was cut off by a data collection limit but the task continues, discarding the continuation value biases learning.

## Practice

Derive the one-step value update from the policy Bellman equation. Then explain why the max in the optimality equation can make noisy Q estimates optimistic.
