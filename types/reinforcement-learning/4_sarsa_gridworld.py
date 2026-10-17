"""
Reinforcement Learning 4: SARSA
-------------------------------
SARSA is an on-policy temporal-difference control method.

Update:
Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]

Unlike Q-learning, SARSA uses the next action actually selected by the
epsilon-greedy policy when calculating its target.

Run:
    python 4_sarsa_gridworld.py

No third-party libraries are required.
"""

import random

random.seed(11)

ROWS, COLS = 4, 4
START = (0, 0)
GOAL = (3, 3)
ACTIONS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
ALPHA, GAMMA, EPSILON = 0.15, 0.95, 0.15
EPISODES, MAX_STEPS = 2_500, 60

Q = {
    (r, c): {action: 0.0 for action in ACTIONS}
    for r in range(ROWS)
    for c in range(COLS)
}


def step(state, action):
    dr, dc = ACTIONS[action]
    r, c = state
    following = (
        max(0, min(ROWS - 1, r + dr)),
        max(0, min(COLS - 1, c + dc)),
    )
    if following == GOAL:
        return following, 10.0, True
    return following, -1.0, False


def choose_action(state):
    if random.random() < EPSILON:
        return random.choice(list(ACTIONS))
    highest = max(Q[state].values())
    return random.choice([a for a, v in Q[state].items() if v == highest])


returns = []
for _ in range(EPISODES):
    state = START
    action = choose_action(state)
    total_return = 0.0

    for _ in range(MAX_STEPS):
        following, reward, done = step(state, action)
        total_return += reward

        if done:
            target = reward
            Q[state][action] += ALPHA * (target - Q[state][action])
            break

        next_action = choose_action(following)
        target = reward + GAMMA * Q[following][next_action]
        Q[state][action] += ALPHA * (target - Q[state][action])

        state, action = following, next_action

    returns.append(total_return)

print("SARSA training finished.")
print(f"Average return over final 100 episodes: {sum(returns[-100:]) / 100:.2f}")

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
            row.append(f" {max(Q[state], key=Q[state].get)} ")
    print(" ".join(row))

print("\\nKey difference: SARSA learns using the action chosen by its current policy.")
