"""
Reinforcement Learning 5: A Simple Policy-Gradient Bandit
---------------------------------------------------------
A small REINFORCE-style example for a one-step bandit problem.

The agent learns a stochastic policy over three actions. It uses a
softmax distribution and updates action preferences in the direction
of reward-weighted log-probability gradients.

This example is intentionally small and uses only the Python standard
library; it demonstrates the idea rather than a production-grade agent.

Run:
    python 5_policy_gradient_bandit.py
"""

import math
import random

random.seed(21)

# Unknown reward probabilities belonging to the environment.
REWARD_PROBABILITIES = [0.15, 0.45, 0.80]
PREFERENCE = [0.0, 0.0, 0.0]
LEARNING_RATE = 0.03
EPISODES = 12_000


def softmax(values):
    """Convert preferences into stable action probabilities."""
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    total = sum(exps)
    return [x / total for x in exps]


def sample_action(probabilities):
    """Sample an action according to its probability."""
    x = random.random()
    cumulative = 0.0
    for action, probability in enumerate(probabilities):
        cumulative += probability
        if x <= cumulative:
            return action
    return len(probabilities) - 1


total_reward = 0
for episode in range(1, EPISODES + 1):
    probabilities = softmax(PREFERENCE)
    action = sample_action(probabilities)

    # Bandit environment returns a binary reward.
    reward = 1 if random.random() < REWARD_PROBABILITIES[action] else 0
    total_reward += reward

    # REINFORCE-style preference update:
    # for selected action i: lr * reward * (1 - p_i)
    # for other actions j:   lr * reward * (0 - p_j)
    # A baseline is omitted here to keep the core idea visible.
    for i in range(len(PREFERENCE)):
        gradient = (1.0 if i == action else 0.0) - probabilities[i]
        PREFERENCE[i] += LEARNING_RATE * reward * gradient

    if episode % 2_000 == 0:
        print(
            f"Episode {episode:5d} | "
            f"average reward so far = {total_reward / episode:.3f} | "
            f"policy = {[round(p, 3) for p in softmax(PREFERENCE)]}"
        )

print("\\nFinal policy probabilities:")
for action, probability in enumerate(softmax(PREFERENCE)):
    print(f"Action {action}: {probability:.3f}")

print("\\nKey idea: policy-gradient methods directly adjust the policy's preferences.")
