# 9. Control, exploration, and off-policy learning

**Prediction** estimates values for a fixed policy. **Control** seeks a policy with better decisions. Control must balance acting on current knowledge with collecting information that can improve future decisions.

## SARSA and Q-learning

SARSA is on-policy: it learns the value of the actions the current behavior policy actually takes.

`Q(S_t,A_t) ← Q(S_t,A_t) + α [R_{t+1} + γ Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]`

Q-learning is off-policy: it targets the greedy action value while behavior may explore.

`Q(S_t,A_t) ← Q(S_t,A_t) + α [R_{t+1} + γ max_a Q(S_{t+1},a) - Q(S_t,A_t)]`

The distinction matters when exploration itself has a cost. SARSA evaluates the exploratory behavior; Q-learning aims at the greedy target policy.

## Exploration approaches

- **Epsilon-greedy:** simple random exploration, but ignores uncertainty and can take obviously poor actions.
- **Optimism/UCB:** add an uncertainty bonus; useful when uncertainty estimates are meaningful.
- **Thompson sampling:** sample a plausible model or value and act optimally for it.
- **Entropy or parameter noise:** encourage stochastic behavior in larger problems.
- **Intrinsic reward:** reward novelty or prediction error; can be misaligned and should be monitored.

Exploration must be constrained by safety, user consent, and the cost of bad actions. In high-stakes settings, exploration may need to happen only in simulation, a sandbox, or a carefully governed experiment.

## Coverage and importance weighting

Off-policy data support a target policy only where the behavior policy provides coverage. If `π(a|s)>0` where the behavior policy has zero probability, outcomes for that action cannot be recovered from those logs without extra assumptions. Importance sampling corrects distribution mismatch when propensities are known, but high weights can make estimates unstable.

## Regret and sample efficiency

Regret compares reward collected during learning with a benchmark. It includes the cost of exploration. In a deployed system, distinguish learning-period reward from final-policy evaluation. A method that eventually learns well may still be unacceptable if learning harms users.

## Practice

In a small environment with a risky shortcut and a safe route, compare SARSA and Q-learning under epsilon-greedy behavior. Explain which value each estimates and how exploration affects the observed return.
