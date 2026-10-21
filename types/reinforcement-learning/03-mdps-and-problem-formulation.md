# 3. Markov decision processes and problem formulation

A Markov decision process (MDP) is a model for sequential decisions under uncertainty. A standard MDP is described by `(S, A, P, R, γ)`:

- `S`: states;
- `A`: actions available in each state;
- `P(s'|s,a)`: transition distribution;
- `R(s,a,s')` or an equivalent reward distribution;
- `γ`: discount factor for future rewards.

The **Markov property** says the current state contains enough information to predict the distribution of the next state and reward given the action. It does not say the world has no history; it says the chosen state representation summarizes relevant history.

## Specify the problem before the algorithm

Write down:

1. Decision timing and episode boundaries.
2. What is observed and what is hidden.
3. State variables available at decision time, without future leakage.
4. Feasible actions and any hard constraints.
5. Reward components, units, delays, and measurement process.
6. Horizon and discounting convention.
7. Initial-state distribution and which environments/users are in scope.
8. Whether the transition dynamics are known, simulated, or inferred from logs.

## State design is a modeling decision

Too little state can violate the Markov assumption. Too much state can increase data needs, leak sensitive details, or make learning unstable. Include only information available at the moment of action and relevant to future outcomes. If the state must include history, summarize it with a justified sufficient statistic or use a partially observable model (chapter 15).

## Reward design

Reward is the optimization target, not a neutral description of value. Keep raw outcomes and reward components separately so stakeholders can inspect tradeoffs. Avoid reward terms that encourage loopholes, such as rewarding a proxy that can improve while the real outcome worsens. Express penalties and benefits in interpretable units when possible.

## Finite versus continuing tasks

An episodic task has a terminal condition and often a finite horizon. A continuing task has no natural end; discounting or average reward makes the objective well-defined. If the horizon matters, include remaining time in state when that affects optimal action.

## Example: inventory replenishment

- State: current inventory, recent demand summary, supplier lead time, and remaining horizon if finite.
- Action: order quantity in allowed units.
- Transition: inventory changes with order arrivals and uncertain demand.
- Reward: sales margin minus holding, ordering, and shortage costs.
- Constraints: capacity, cash, and service-level requirements.

This formulation immediately exposes data questions: are stockouts recorded, is demand censored when stock is zero, and does the simulator represent lead-time variability?

## Practice

Formulate a sequential problem from a familiar domain. For each state feature, explain why it is available before action and how omitting it could affect the Markov property. Identify one reward-hacking risk and one hard constraint.
