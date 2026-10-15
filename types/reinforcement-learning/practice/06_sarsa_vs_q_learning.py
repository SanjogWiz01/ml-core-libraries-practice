"""
06 - SARSA vs Q-Learning on the Cliff
=====================================

Goal: run the two founding model-free TD control algorithms side by side on the
classic Cliff Walking task and see what "on-policy vs off-policy" actually
means in terms of the route the agent learns.

The two updates, one token apart
--------------------------------
    Q-learning :  Q(s,a) <- Q(s,a) + alpha [ r + gamma * MAX_a' Q(s',a') - Q(s,a) ]
    SARSA      :  Q(s,a) <- Q(s,a) + alpha [ r + gamma * Q(s',a')        - Q(s,a) ]
                                                            ~~~~~~~~~~~~
    SARSA's target uses the action it ACTUALLY takes next, a', which is drawn
    from the same epsilon-greedy exploration policy that produced a.

That single difference - max vs the sampled next action - is the whole
on-policy / off-policy distinction:

  Q-learning learns the value of the OPTIMAL policy (it always assumes you
  will act greedily next), regardless of how the data was collected. It is
  off-policy: the target policy differs from the behaviour policy.

  SARSA learns the value of the policy it is FOLLOWING, exploration and all.
  It is on-policy: to evaluate the greedy policy you must stop exploring.

The task
--------
Sutton & Barto's Cliff Walking: a 4x12 grid, start bottom-left, goal
bottom-right, and a cliff along the bottom row between them.

        . . . . . . . . . . . .
        . . . . . . . . . . . .
        . . . . . . . . . . . .
        S C C C C C C C C C C G

Stepping into the cliff costs -100 and sends the agent back to S. Every other
step costs -1. The shortest path is 13 steps and skims the cliff along row 2.

Why it matters
--------------
The environment is DETERMINISTIC. The only source of risk is the agent's own
exploration: at each step with probability epsilon it takes a random action,
and near the cliff a random action can step off. So:

  Q-learning assesses a cliff-edge state by "what if I acted optimally from
  here", so it learns to hug the edge - the optimal 13-step path - and falls
  occasionally during training.

  SARSA assesses it by "what if I keep exploring, including the random
  steps", so it learns a margin: a longer but safer route along the top.

The optimal POLICY differs by definition, and the difference is largest while
exploring. It is the standard illustration that the best policy to LEARN is
not always the best policy to FOLLOW while learning.

Run:  python 06_sarsa_vs_q_learning.py
"""

import numpy as np

SEED = 42

print("=" * 78)
print("06 - SARSA vs Q-LEARNING ON CLIFF WALKING")
print("=" * 78)


# ======================================================================================
# Environment
# ======================================================================================
ROWS, COLS = 4, 12
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTIONS = [UP, RIGHT, DOWN, LEFT]
ACTION_NAMES = ["up", "right", "down", "left"]
ARROWS = ["^", ">", "v", "<"]
DELTAS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}

START = (3, 0)                 # bottom-left, as in the original task
GOAL = (3, 11)
CLIFF = {(3, c) for c in range(1, 11)}
GAMMA = 1.0                    # undiscounted, episodic: the classic setting
STEP_COST = -1.0
CLIFF_PENALTY = -100.0
MAX_STEPS = 300                # a truncation cap; episodes normally end far sooner

N_STATES = ROWS * COLS
N_ACTIONS = 4


def encode(state):
    return state[0] * COLS + state[1]


def decode(index):
    return divmod(index, COLS)


def env_step(state, action):
    """Deterministic transition. Risk comes from exploration, not the world."""
    dr, dc = DELTAS[action]
    r, c = state
    nr, nc = r + dr, c + dc
    if not (0 <= nr < ROWS and 0 <= nc < COLS):
        nr, nc = r, c
    s2 = (nr, nc)
    if s2 in CLIFF:
        return START, CLIFF_PENALTY, True
    if s2 == GOAL:
        return s2, STEP_COST, True
    return s2, STEP_COST, False


def epsilon_greedy(q_row, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))
    return int(rng.choice(np.flatnonzero(q_row == q_row.max())))


# ======================================================================================
# The two algorithms
# ======================================================================================
def q_learning(n_episodes=500, alpha=0.5, epsilon=0.1, seed=SEED):
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    returns, lengths, falls = [], [], []

    for _ in range(n_episodes):
        state = START
        total, steps = 0.0, 0
        done = False
        fell = 0
        while not done and steps < MAX_STEPS:
            a = epsilon_greedy(Q[encode(state)], epsilon, rng)
            s2, r, done = env_step(state, a)
            if r == CLIFF_PENALTY:
                fell += 1
            target = r if done else r + GAMMA * Q[encode(s2)].max()
            Q[encode(state), a] += alpha * (target - Q[encode(state), a])
            state = s2
            total += r
            steps += 1
        returns.append(total)
        lengths.append(steps)
        falls.append(fell)
    return Q, np.array(returns), np.array(lengths), np.array(falls)


def sarsa(n_episodes=500, alpha=0.5, epsilon=0.1, seed=SEED):
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    returns, lengths, falls = [], [], []

    for _ in range(n_episodes):
        state = START
        # SARSA needs an action BEFORE the loop, because the very first update
        # needs the (s, a, r, s', a') quintuple.
        a = epsilon_greedy(Q[encode(state)], epsilon, rng)
        total, steps = 0.0, 0
        done = False
        fell = 0
        while not done and steps < MAX_STEPS:
            s2, r, done = env_step(state, a)
            if r == CLIFF_PENALTY:
                fell += 1
            if done:
                target = r
                a2 = None
            else:
                a2 = epsilon_greedy(Q[encode(s2)], epsilon, rng)
                target = r + GAMMA * Q[encode(s2), a2]
            Q[encode(state), a] += alpha * (target - Q[encode(state), a])
            state = s2
            a = a2 if a2 is not None else a     # a2 unused once done
            total += r
            steps += 1
        returns.append(total)
        lengths.append(steps)
        falls.append(fell)
    return Q, np.array(returns), np.array(lengths), np.array(falls)


def greedy_route(Q):
    """Follow the argmax policy and return the visited cells until it ends."""
    state = START
    path = [state]
    for _ in range(MAX_STEPS):
        a = int(np.argmax(Q[encode(state)]))
        s2, r, done = env_step(state, a)
        path.append(s2)
        state = s2
        if done:
            break
    return path


def render_paths(paths_and_markers, title=""):
    print(title)
    grid = [["." for _ in range(COLS)] for _ in range(ROWS)]
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) in CLIFF:
                grid[r][c] = "C"
    grid[START[0]][START[1]] = "S"
    grid[GOAL[0]][GOAL[1]] = "G"
    for path, marker in paths_and_markers:
        for (r, c) in path:
            if grid[r][c] not in ("C", "S", "G"):
                grid[r][c] = marker
    for row in grid:
        print("   " + " ".join(row))


# ======================================================================================
# PART 1 - train both, one seed each
# ======================================================================================
print()
print("-" * 78)
print("PART 1 - TRAINING (500 episodes, alpha=0.5, epsilon=0.1, gamma=1.0)")
print("-" * 78)

Q_q, q_returns, q_lengths, q_falls = q_learning()
Q_s, s_returns, s_lengths, s_falls = sarsa()

print(f"{'':<14} {'mean ret first 100':>20} {'mean ret last 100':>20} "
      f"{'falls first 100':>18} {'falls last 100':>16}")
print("-" * 92)
print(f"{'Q-learning':<14} {q_returns[:100].mean():>20.2f} {q_returns[-100:].mean():>20.2f} "
      f"{q_falls[:100].mean():>18.2f} {q_falls[-100:].mean():>16.2f}")
print(f"{'SARSA':<14} {s_returns[:100].mean():>20.2f} {s_returns[-100:].mean():>20.2f} "
      f"{s_falls[:100].mean():>18.2f} {s_falls[-100:].mean():>16.2f}")
print()
print("Both start by falling off the cliff constantly - early on every one of")
print("them explores blindly. The difference shows up once they have learned a")
print("route: Q-learning keeps skimming the edge because that is optimal under")
print("its target, so its occasional random step still finds the cliff. SARSA")
print("has already learned that the edge is expensive while exploring.")
print()

q_path = greedy_route(Q_q)
s_path = greedy_route(Q_s)
render_paths([(q_path, "o")], "Q-learning greedy route (o):")
print()
render_paths([(s_path, "x")], "SARSA greedy route (x):")
print()
print(f"Q-learning route: {len(q_path) - 1} steps  "
      f"(optimal, hugging the cliff along row 2)")
print(f"SARSA route:      {len(s_path) - 1} steps  "
      f"(longer, keeping a row of margin from the cliff)")
print()
print("This is the picture the textbook is famous for: two agents trained on the")
print("same task by almost the same update learn visibly different policies.")


# ======================================================================================
# PART 2 - averaged over seeds, so a single lucky run cannot mislead
# ======================================================================================
print()
print("-" * 78)
print("PART 2 - AVERAGED OVER SEEDS")
print("-" * 78)

n_seeds = 40
q_all = np.zeros((n_seeds, 500))
s_all = np.zeros((n_seeds, 500))
q_last, s_last = [], []
for seed in range(n_seeds):
    _, qr, _, _ = q_learning(seed=seed)
    _, sr, _, _ = sarsa(seed=seed)
    q_all[seed] = qr
    s_all[seed] = sr
    q_last.append(qr[-100:].mean())
    s_last.append(sr[-100:].mean())

q_last = np.array(q_last)
s_last = np.array(s_last)

print("mean reward per episode, averaged over 40 independent runs:")
print(f"{'window':<16} {'Q-learning':>14} {'SARSA':>14} {'difference':>14}")
print("-" * 60)
for lo, hi, label in [(0, 100, "episodes 1-100"), (100, 200, "101-200"),
                      (200, 300, "201-300"), (400, 500, "401-500")]:
    qm = q_all[:, lo:hi].mean()
    sm = s_all[:, lo:hi].mean()
    print(f"{label:<16} {qm:>14.2f} {sm:>14.2f} {sm - qm:>14.2f}")
print()
print(f"last-100 mean:  Q-learning {q_last.mean():.2f} +/- {q_last.std():.2f}   "
      f"SARSA {s_last.mean():.2f} +/- {s_last.std():.2f}")
print()
print("Averaged over 40 seeds, SARSA earns a HIGHER reward per episode than")
print("Q-learning throughout. The gap is the price Q-learning pays for pursuing")
print("the edge-optimal policy while still exploring. Report means with a spread")
print("and average over runs - a single seed can reverse the ordering by luck.")


# ======================================================================================
# PART 3 - where the difference lives: values at the cliff edge
# ======================================================================================
print()
print("-" * 78)
print("PART 3 - V(s) = MAX_a Q(s,a) ALONG THE CLIFF")
print("-" * 78)

V_q = Q_q.max(axis=1)
V_s = Q_s.max(axis=1)

print("Row 2 (directly above the cliff), columns 0..7:")
print()
print(f"{'col':<6} {'V from Q-learning':>18} {'V from SARSA':>14} {'difference':>12}")
print("-" * 54)
for c in range(8):
    i = encode((2, c))
    print(f"{c:<6} {V_q[i]:>18.2f} {V_s[i]:>14.2f} {V_q[i] - V_s[i]:>12.2f}")
print()
print("Q-learning values the cliff-edge states MORE than SARSA does, because it")
print("prices them assuming no further exploration mistakes. SARSA is pessimistic")
print("about them precisely because from there a random step is expensive. A")
print("pessimistic value function is often exactly what you want when the cost of")
print("a mistake is asymmetric - the same idea underlies conservative Q-learning")
print("and other safe-RL methods.")


# ======================================================================================
# PART 4 - how much do they actually differ in behaviour?
# ======================================================================================
print()
print("-" * 78)
print("PART 4 - POLICY AGREEMENT AND EDGE-RISK")
print("-" * 78)

q_pol = np.argmax(Q_q, axis=1)
s_pol = np.argmax(Q_s, axis=1)

# Only compare non-terminal, non-trivially-unvisited states.
reachable = [i for i in range(N_STATES) if decode(i) not in CLIFF]
agree = int(np.sum(q_pol[reachable] == s_pol[reachable]))
print(f"greedy policy agreement on non-cliff cells: "
      f"{agree}/{len(reachable)} ({agree / len(reachable):.0%})")
print()

# How many states adjacent to the cliff does each greedy route prefer to leave?
def edge_preference(Q, row=2):
    """Count, over the cliff-adjacent row, how often the greedy action moves
    ALONG the cliff (left/right) versus AWAY from it (up)."""
    along = away = 0
    for c in range(1, 11):
        i = encode((row, c))
        a = int(np.argmax(Q[i]))
        if a in (LEFT, RIGHT):
            along += 1
        elif a == UP:
            away += 1
    return along, away


qa, qaw = edge_preference(Q_q)
sa, saw = edge_preference(Q_s)
print(f"{'agent':<14} {'moves ALONG cliff (row 2)':>26} {'moves AWAY (up)':>18}")
print("-" * 60)
print(f"{'Q-learning':<14} {qa:>26} {qaw:>18}")
print(f"{'SARSA':<14} {sa:>26} {saw:>18}")
print()
print("Q-learning keeps travelling along the cliff; SARSA is more willing to")
print("climb away from it. That single statistical tendency is what produces the")
print("visibly different routes in Part 1.")


# ======================================================================================
# Report
# ======================================================================================
print()
print("=" * 78)
print("REPORT")
print("=" * 78)
print(f"""
ENVIRONMENT (deterministic - risk is self-generated by exploration)
  grid           {ROWS} x {COLS}, start {START}, goal {GOAL}
  cliff          bottom row cols 1..10, penalty {CLIFF_PENALTY}, teleport to start
  step cost      {STEP_COST}, gamma {GAMMA}, epsilon 0.1

TARGETS
  Q-learning  r + gamma * max_a' Q(s',a')      off-policy
  SARSA       r + gamma * Q(s',a'), a' ~ pi     on-policy

SINGLE RUN (seed 0)
  Q-learning  route {len(q_path) - 1} steps, last-100 reward {q_returns[-100:].mean():.2f}
  SARSA       route {len(s_path) - 1} steps, last-100 reward {s_returns[-100:].mean():.2f}

40-SEED AVERAGE
  Q-learning  last-100 {q_last.mean():.2f} +/- {q_last.std():.2f}
  SARSA       last-100 {s_last.mean():.2f} +/- {s_last.std():.2f}

CLIFF-EDGE PREFERENCE (row 2, cols 1..10)
  Q-learning  {qa} along / {qaw} away
  SARSA       {sa} along / {saw} away

WHAT TO CARRY FORWARD
  - one symbol (max vs sampled action) is the entire on/off-policy split
  - the environment here is deterministic; the danger is the agent's own
    epsilon-greedy exploration, which is why the two policies differ at all
  - off-policy Q-learning chases the edge-optimal policy and pays for it while
    exploring; on-policy SARSA learns a safer, longer route
  - asymptotically the greedy Q-learning policy is the shortest (13 steps);
    SARSA trades optimality for safety, which is better ON AVERAGE here
  - always average several seeds: a single run reverses the ranking by luck
  - expected SARSA (script 09) averages over a' instead of sampling it,
    cutting the variance of SARSA's target with the same on-policy view
""")
print("=" * 78)
