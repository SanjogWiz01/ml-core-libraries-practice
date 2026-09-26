# Reinforcement Learning

Reinforcement learning (RL) is learning from **interaction**. An agent acts in
an environment, receives rewards, and must learn a policy that maximises expected
cumulative reward.

There is no dataset. The agent *generates* its own experience, which is why RL is
separate from both supervised and unsupervised learning.

## The Loop

```
        observe state s_t
              |
              v
   +---> choose action a_t  <---- policy pi(a | s)
   |              |
   |              v
   |     environment steps
   |              |
   |              v
   +----- reward r_t + next state s_{t+1}
```

The agent's goal is the **return** `G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...`
- discounted future reward. `gamma` (the discount factor) trades long-term reward
against immediate reward.

## Vocabulary

| Term | Meaning |
|------|---------|
| **Agent** | the decision maker (the model being trained) |
| **Environment** | everything the agent interacts with |
| **State `s`** | full description of the situation |
| **Action `a`** | choice available to the agent |
| **Reward `r`** | scalar feedback signal, higher is better |
| **Policy `pi(a | s)`** | the agent's strategy for picking actions |
| **Value `V(s)`** | expected return starting from state `s` |
| **Q-value `Q(s, a)`** | expected return from taking `a` in `s`, then acting optimally |
| **Epsilon `eps`** | exploration probability in epsilon-greedy |

## The Two Families

- **Model-based** - the agent knows/learns the transition dynamics, then plans.
- **Model-free** - learns directly from sampled experience, no transition model.
  - **Value-based**: Q-Learning, SARSA, DQN
  - **Policy-based**: REINFORCE, Actor-Critic, PPO

## The Exploration/Exploitation Tradeoff

Greedy action selection exploits known rewards. Exploration tries unknown
actions that might be better. Epsilon-greedy: with probability `eps` act randomly,
otherwise act greedily. Too little exploration = stuck in a local optimum; too
much = noisy, unstable learning.

## Practical Libraries

| Library | Use |
|---------|-----|
| `gymnasium` | standard environments (CartPole, LunarLander) |
| `stable-baselines3` | reliable PPO / DQN / SAC implementations |

```bash
pip install gymnasium stable-baselines3
```

## Real Applications

Game playing, robotics control, recommendation with long-term reward, LLM
fine-tuning with RLHF, traffic signal control, portfolio trading.

## Current Contents

Nothing yet - notebooks and scripts will be added here.

| Topic | Status |
|-------|--------|
| `q-learning-gridworld/` | planned |
| `sarsa-vs-qlearning/` | planned |
| `dqn-cartpole/` | planned |

## Related

- [`../supervised/`](../supervised/)
- [`../unsupervised/`](../unsupervised/)
