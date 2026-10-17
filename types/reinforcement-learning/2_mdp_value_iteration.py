"""
Reinforcement Learning 2: Markov Decision Process + Value Iteration
--------------------------------------------------------------------
A small deterministic grid world solved with value iteration.

States are grid cells. The agent can move up, down, left, or right.
The goal cell gives +10; each ordinary move costs -1.

Run:
    python 2_mdp_value_iteration.py

No third-party libraries are required.
"""

ROWS, COLS = 4, 4
GOAL = (3, 3)
ACTIONS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
GAMMA = 0.90
THETA = 1e-8


def next_state(state, action):
    """Return the next cell, keeping the agent inside the grid."""
    dr, dc = ACTIONS[action]
    r, c = state
    nr = max(0, min(ROWS - 1, r + dr))
    nc = max(0, min(COLS - 1, c + dc))
    return nr, nc


def reward(state, following_state):
    """Reward for a transition."""
    return 10.0 if following_state == GOAL else -1.0


states = [(r, c) for r in range(ROWS) for c in range(COLS)]
values = {state: 0.0 for state in states}

# Bellman optimality update:
# V(s) = max_a [ R(s,a,s') + gamma * V(s') ]
iteration = 0
while True:
    biggest_change = 0.0
    new_values = values.copy()

    for state in states:
        if state == GOAL:
            new_values[state] = 0.0
            continue

        action_returns = []
        for action in ACTIONS:
            following = next_state(state, action)
            action_returns.append(
                reward(state, following) + GAMMA * values[following]
            )

        new_values[state] = max(action_returns)
        biggest_change = max(
            biggest_change, abs(new_values[state] - values[state])
        )

    values = new_values
    iteration += 1
    if biggest_change < THETA:
        break

print(f"Value iteration converged after {iteration} iterations.\\n")
print("State-value grid (higher values are generally closer to the goal):")
for r in range(ROWS):
    print("  ".join(f"{values[(r, c)]:7.2f}" for c in range(COLS)))

print("\\nGreedy policy (G = goal):")
for r in range(ROWS):
    symbols = []
    for c in range(COLS):
        state = (r, c)
        if state == GOAL:
            symbols.append(" G ")
        else:
            best_action = max(
                ACTIONS,
                key=lambda a: reward(state, next_state(state, a))
                + GAMMA * values[next_state(state, a)]
            )
            symbols.append(f" {best_action} ")
    print(" ".join(symbols))

print("\\nKey idea: value iteration repeatedly applies the Bellman optimality equation.")
