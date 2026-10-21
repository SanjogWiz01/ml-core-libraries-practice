# 1. Prerequisites and notation

RL combines probability, optimization, sequential decision-making, and machine learning. You do not need advanced measure theory to begin, but you should be comfortable with conditional probability, expectations, vectors, and iterative algorithms.

## Probability essentials

- A random variable represents an uncertain quantity; a distribution describes its possible values.
- Conditional probability and expectation describe uncertainty after observing information: `P(X | Y)` and `E[X | Y]`.
- The law of total expectation lets us average over possible next states: `E[X] = E[E[X | Y]]`.
- A trajectory is a sequence of random observations, actions, rewards, and states.
- An estimator computed from a finite trajectory has sampling noise. Repeated independent runs help quantify it.

## Optimization essentials

An agent often chooses a policy parameter `θ` to maximize an objective `J(θ)`. Gradient ascent updates `θ ← θ + α ∇θ J(θ)`, where `α` is a learning rate. In practice, the gradient is estimated from samples, so step size, variance, and data distribution matter.

## Notation used here

| Symbol | Meaning |
|---|---|
| `S`, `A` | State and action spaces |
| `s`, `a` | A particular state and action |
| `R_{t+1}` | Reward received after action at time `t` |
| `S_t`, `A_t` | State and action at time `t` |
| `π(a|s)` | Policy: probability of action `a` in state `s` |
| `P(s'|s,a)` | Probability of next state `s'` given current state and action |
| `γ` | Discount factor, usually between zero and one |
| `G_t` | Return accumulated from time `t` onward |
| `Vπ(s)` | Expected return from state `s` following policy `π` |
| `Qπ(s,a)` | Expected return after taking `a` in `s`, then following `π` |
| `α` | Learning rate or step size; context determines its use |
| `θ`, `w` | Parameters of a policy or value approximator |

## Sequential data differs from ordinary i.i.d. data

Actions affect what happens next and which data become observable. Nearby transitions in a trajectory are correlated. A policy change can change the state distribution, so a model trained on yesterday's data may be unreliable in states visited by tomorrow's policy. Randomly splitting individual transitions can leak information; split by episode, time, user, or environment as appropriate.

## Before starting

Be able to compute a mean and standard error, explain conditional probability, fit a basic predictive model, and implement a loop with a state update. Learn enough linear algebra to read a gradient and enough Python to represent a table, sample a categorical distribution, and plot learning curves.

## Check yourself

1. Why are rewards from one trajectory not independent observations?
2. If a policy changes, why might the distribution of training states change?
3. What does an expectation over trajectories mean operationally?
