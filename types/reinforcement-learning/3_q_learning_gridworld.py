"""
Reinforcement Learning 3: Q-Learning
-------------------------------------
Model-free, off-policy learning in a small grid world.

The agent learns a route from the top-left start to the bottom-right goal.
Q-learning update:
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

Run:
    python 3_q_learning_gridworld.py

No third-party libraries are required.
"""

import random

random.seed(7)

ROWS, COLS = 4, 4
START = (0, 0)
GOAL = (3, 3)
ACTIONS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

ALPHA = 0.15
GAMMA = 0.95
EPSILON = 0.20
EPISODES = 2_500
MAX_STEPS = 60

Q = {
    (r, c): {action: 0.0 for action in ACTIONS}
    for r in range(ROWS)
    for c in range(COLS)
}


def step(state, action):
    """Apply an action and return (next_state, reward, done)."""
    dr, dc = ACTIONS[action]
    r, c = state
    nr = max(0, min(ROWS - 1, r + dr))
    nc = max(0, min(COLS - 1, c + dc))
    following = (nr, nc)

    if following == GOAL:
        return following, 10.0, True
    return following, -1.0, False


def choose_action(state, epsilon):
    """Epsilon-greedy action selection."""
    if random.random() < epsilon:
        return random.choice(list(ACTIONS))
    best = max(Q[state].values())
    candidates = [a for a, value in Q[state].items() if value == best]
    return random.choice(candidates)


episode_returns = []
for episode in range(EPISODES):
    state = START
    total_return = 0.0

    for _ in range(MAX_STEPS):
        action = choose_action(state, EPSILON)
        following, reward, done = step(state, action)

        # Terminal states have no future reward.
        future = 0.0 if done else max(Q[following].values())
        target = reward + GAMMA * future
        Q[state][action] += ALPHA * (target - Q[state][action])

        state = following
        total_return += reward
        if done:
            break

    episode_returns.append(total_return)

print("Training finished.")
print(f"Average return over final 100 episodes: {sum(episode_returns[-100:]) / 100:.2f}")

# Display the greedy policy learned from Q-values.
print("\\nLearned policy (S=start, G=goal):")
for r in range(ROWS):
    row = []
    for c in range(COLS):
        state = (r, c)
        if state == START:
            row.append(" S ")
        elif state == GOAL:
            row.append(" G ")
        else:
            best_action = max(Q[state], key=Q[state].get)
            row.append(f" {best_action} ")
    print(" ".join(row))

# Demonstrate one greedy route.
state = START
route = [state]
for _ in range(ROWS * COLS * 2):
    if state == GOAL:
        break
    action = max(Q[state], key=Q[state].get)
    state, _, _ = step(state, action)
    route.append(state)

print("\\nGreedy route:", " -> ".join(map(str, route)))
print("\\nKey idea: Q-learning learns from rewards without needing a model of the environment.")
