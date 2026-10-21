# Reinforcement learning for data scientists: roadmap

This folder is a progressive course in reinforcement learning (RL), from decision-problem framing to reliable evaluation and deployment. It focuses on the ideas a data scientist needs to choose, implement, and critique RL methods. It is not a claim that one course can exhaust every research topic.

## How to use this course

Read in order. For each chapter, derive the equations once, implement a small example, and write down the assumptions that make the method valid. Use a tiny finite environment before scaling up. Keep a notebook with state, action, reward, transition, and episode definitions for every problem.

| # | File | Milestone |
|---|---|---|
| 1 | [Prerequisites and notation](01-prerequisites-and-notation.md) | Read probability, optimization, and learning notation used throughout |
| 2 | [Decision problems and bandits](02-decision-problems-and-bandits.md) | Distinguish contextual decisions from sequential control |
| 3 | [MDPs and problem formulation](03-mdps-and-problem-formulation.md) | Specify a Markov decision process carefully |
| 4 | [Returns and value functions](04-returns-and-value-functions.md) | Define the objective and quantify delayed reward |
| 5 | [Bellman equations](05-bellman-equations.md) | Relate values across time and understand optimality |
| 6 | [Dynamic programming](06-dynamic-programming.md) | Solve small known models by planning |
| 7 | [Monte Carlo methods](07-monte-carlo-methods.md) | Learn from complete sampled episodes |
| 8 | [Temporal-difference learning](08-temporal-difference-learning.md) | Bootstrap from partial experience |
| 9 | [Control, exploration, and off-policy learning](09-control-exploration-and-off-policy.md) | Learn action choices and account for exploration |
| 10 | [Function approximation](10-function-approximation.md) | Generalize values and policies beyond tabular states |
| 11 | [Deep Q-networks](11-deep-q-networks.md) | Use neural networks for value-based control |
| 12 | [Policy gradients](12-policy-gradients.md) | Optimize stochastic policies directly |
| 13 | [Actor-critic methods](13-actor-critic-methods.md) | Combine policy optimization and value estimation |
| 14 | [Model-based and planning methods](14-model-based-and-planning.md) | Learn or use a transition model to plan |
| 15 | [Special settings and modern practice](15-special-settings-and-modern-practice.md) | Handle partial observability, hierarchy, multiple agents, and offline data |
| 16 | [Evaluation, safety, and deployment](16-evaluation-safety-and-deployment.md) | Assess evidence and manage real-world risk |

## The core learning loop

An agent observes information about the environment, chooses an action, receives a reward, and sees a new observation. The learning problem is to choose a policy that maximizes a clearly stated long-run objective while respecting what the agent can know and what actions it can safely take.

```text
formulate → build a baseline → collect/inspect experience → train → evaluate
    ↑                                                            ↓
    └──────────── revise assumptions and problem definition ←────┘
```

## Working conventions

- State whether an episode ends, whether rewards are discounted, and which policy generated each dataset.
- Separate training data, validation decisions, and final evaluation. Do not repeatedly tune against the final test environments.
- Compare with simple policies, supervised or optimization baselines, and the behavior policy where applicable.
- Report uncertainty across independent seeds or scenarios, not just the best run.
- RL is not automatically the right tool. If actions do not affect future data or outcomes, a contextual bandit or ordinary prediction/optimization method may be simpler and safer.

## Suggested capstone

Choose a small simulated sequential decision problem. Define its state, action, reward, transition, horizon, constraints, and evaluation metric before training. Implement a random or heuristic baseline, a tabular method when possible, then one function-approximation method. Compare performance, variability, constraint violations, and sensitivity to assumptions. Finish with a short decision memo explaining whether RL is justified.
