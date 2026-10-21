# 13. Actor-critic methods

Actor-critic methods combine a policy (the **actor**) with a learned value function (the **critic**). The critic supplies a lower-variance learning signal; the actor changes action probabilities.

## Advantage estimates

The advantage function measures whether an action was better than the policy's average action in a state:

`Aπ(s,a) = Qπ(s,a) - Vπ(s)`

A common one-step estimate is the TD residual `δ_t = r + γV(s') - V(s)`. The actor update is proportional to `∇ log π(a|s) δ_t`, while the critic learns to predict returns or satisfy a TD objective.

## n-step and generalized advantage estimates

n-step returns trade bias for variance by choosing how far to sample before bootstrapping. Generalized advantage estimation (GAE) combines TD residuals with a decay parameter `λ`; lower `λ` uses shorter-horizon estimates, while higher `λ` uses longer traces. These are practical tuning choices, not universal defaults.

## On-policy and off-policy families

Methods such as A2C and PPO commonly use on-policy or near-on-policy data. DDPG, TD3, and SAC are actor-critic approaches designed for continuous actions and replay-based, off-policy learning, with differing mechanisms for stability, exploration, and entropy. Their assumptions, action bounds, and data needs differ; choose based on the problem rather than name recognition.

## PPO in brief

Proximal Policy Optimization clips a probability ratio objective to discourage overly large policy updates on a batch. Clipping often improves practical stability, but does not guarantee a globally safe update or good final behavior. Its critic, advantage estimates, entropy, and data collection still require care.

## Failure modes

- Critic error gives the actor a misleading direction.
- Policy changes make old on-policy data stale.
- Reward scaling can destabilize both actor and critic.
- Evaluation on training rollouts overstates performance.
- Continuous actions can saturate at bounds or exploit simulator quirks.

## Practice

Trace one actor update and one critic update from a single transition. Then describe how a biased critic can alter the actor's behavior even if policy-gradient code is correct.
