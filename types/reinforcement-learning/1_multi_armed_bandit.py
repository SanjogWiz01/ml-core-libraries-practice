"""
Reinforcement Learning 1: Multi-Armed Bandit
--------------------------------------------
Learn the exploration vs. exploitation trade-off using epsilon-greedy.

Run:
    python 1_multi_armed_bandit.py

No third-party libraries are required.
"""

import random

random.seed(7)

# Each arm has an unknown probability of giving reward 1.
TRUE_REWARD_PROBABILITIES = [0.10, 0.25, 0.55, 0.75, 0.40]
STEPS = 2_000
EPSILON = 0.10  # Explore randomly 10% of the time.

estimated_values = [0.0] * len(TRUE_REWARD_PROBABILITIES)
action_counts = [0] * len(TRUE_REWARD_PROBABILITIES)
total_reward = 0

for step in range(STEPS):
    # Exploration: try a random arm.
    # Exploitation: choose the arm with the best estimated reward.
    if random.random() < EPSILON:
        action = random.randrange(len(TRUE_REWARD_PROBABILITIES))
    else:
        best_value = max(estimated_values)
        best_actions = [
            i for i, value in enumerate(estimated_values)
            if value == best_value
        ]
        action = random.choice(best_actions)

    # Environment returns a reward based on the chosen arm.
    reward = 1 if random.random() < TRUE_REWARD_PROBABILITIES[action] else 0

    # Incremental sample-average update.
    action_counts[action] += 1
    n = action_counts[action]
    estimated_values[action] += (reward - estimated_values[action]) / n
    total_reward += reward

print("Multi-Armed Bandit Results")
print("-" * 32)
for i, (true_p, estimate, count) in enumerate(
    zip(TRUE_REWARD_PROBABILITIES, estimated_values, action_counts)
):
    print(
        f"Arm {i}: true probability={true_p:.2f}, "
        f"estimated value={estimate:.3f}, selected {count} times"
    )

print(f"\\nAverage reward: {total_reward / STEPS:.3f}")
print("\\nKey idea: exploration gathers information; exploitation uses what was learned.")
