"""
04 - Value Iteration and Policy Iteration
==========================================

Goal: compute the OPTIMAL value function V* and the optimal policy in a
GridWorld using both classic dynamic programming methods - value iteration and
policy iteration - then compare them against each other and against the random
policy baseline from script 03.

Why this is the pivot of the whole folder
-----------------------------------------
Scripts 01-03 evaluated a given policy. This script answers the question RL
actually cares about: which policy is best? Both algorithms here assume a
KNOWN model, which real problems never have - but every model-free algorithm
that follows (Q-learning, SARSA, DQN) is an approximation of exactly these two
updates with the model swapped out for samples.

The optimality backup
---------------------
    Q(s,a) = SUM_s' P(s'|s,a) [ R(s,a,s') + gamma * V(s') ]
    V(s)   = MAX_a Q(s,a)

The only change from policy evaluation is SUM_a -> MAX_a. That single operator
swap is the difference between "how good is this policy" and "how good can
things possibly get".

Value iteration: apply the optimality backup to every state until V stops
changing, then read off the greedy policy.

Policy iteration: evaluate the current policy exactly (one linear solve),
improve it greedily, repeat. It alternates two expensive-ish steps but usually
converges in a handful of policy swaps, far fewer than value iteration's
sweeps.

Run:  python 04_value_iteration.py
"""

import numpy as np

SEED = 42

print("=" * 78)
print("04 - VALUE ITERATION AND POLICY ITERATION")
print("=" * 78)


# ======================================================================================
# Environment (same 4x4 grid as script 03, model available as an explicit table)
# ======================================================================================
ROWS, COLS = 4, 4
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTIONS = [UP, RIGHT, DOWN, LEFT]
ACTION_NAMES = {UP: "^ up", RIGHT: "> right", DOWN: "v down", LEFT: "< left"}
ARROWS = {UP: "^", RIGHT: ">", DOWN: "v", LEFT: "<"}
DELTAS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}

START = (0, 0)
GOAL = (3, 3)
GAMMA = 0.9
STEP_COST = -1.0
GOAL_REWARD = 0.0

N_STATES = ROWS * COLS
N_ACTIONS = len(ACTIONS)


def encode(state):
    return state[0] * COLS + state[1]


def decode(index):
    return divmod(index, COLS)


def is_terminal(state):
    return state == GOAL


def step_model(state, action):
    """The FULL model in one function: returns a list of (p, s2, r)."""
    dr, dc = DELTAS[action]
    r, c = state
    nr, nc = r + dr, c + dc
    if not (0 <= nr < ROWS and 0 <= nc < COLS):
        nr, nc = r, c
    s2 = (nr, nc)
    if s2 == GOAL and not is_terminal(state):
        reward = GOAL_REWARD + STEP_COST
    else:
        reward = STEP_COST
    return [(1.0, s2, reward)]


# Precompute the model as arrays so the inner loops stay readable.
# P_model[i, a, j] = prob of j given (i, a);  R_model[i, a] = expected reward.
P_model = np.zeros((N_STATES, N_ACTIONS, N_STATES))
R_model = np.zeros((N_STATES, N_ACTIONS))
for i in range(N_STATES):
    state = decode(i)
    if is_terminal(state):
        continue
    for a in ACTIONS:
        for p, s2, r in step_model(state, a):
            P_model[i, a, encode(s2)] += p
            R_model[i, a] += p * r


# ======================================================================================
# Algorithm 1 - value iteration
# ======================================================================================
def value_iteration(gamma=GAMMA, tol=1e-10, max_sweeps=10_000):
    """
    Repeatedly apply V(s) = MAX_a SUM_s' P [R + gamma V(s')] everywhere.

    Note the update uses the NEW values of states already swept in this pass
    if you write in place, or the OLD ones if you double-buffer. Both converge
    to the same V*; double-buffering makes the contraction argument trivial,
    in-place sweeps converge about twice as fast.
    """
    V = np.zeros(N_STATES)
    history = [V.copy()]
    sweeps = 0

    for sweeps in range(1, max_sweeps + 1):
        V_new = np.zeros(N_STATES)
        for i in range(N_STATES):
            if is_terminal(decode(i)):
                V_new[i] = 0.0
                continue
            q_values = R_model[i] + gamma * P_model[i] @ V
            V_new[i] = q_values.max()
        delta = float(np.max(np.abs(V_new - V)))
        V = V_new
        history.append(V.copy())
        if delta < tol:
            break

    # Extract the greedy policy from the finished V.
    policy = np.zeros(N_STATES, dtype=int)
    q_table = np.zeros((N_STATES, N_ACTIONS))
    for i in range(N_STATES):
        q_table[i] = R_model[i] + gamma * P_model[i] @ V
        policy[i] = int(np.argmax(q_table[i]))
    return V, policy, q_table, sweeps, history


# ======================================================================================
# Algorithm 2 - policy iteration (evaluate exactly, improve greedily)
# ======================================================================================
def policy_evaluation_exact(policy, gamma=GAMMA):
    """The linear solve from script 03, reused as a subroutine."""
    A = np.eye(N_STATES)
    b = np.zeros(N_STATES)
    for i in range(N_STATES):
        if is_terminal(decode(i)):
            A[i, :] = 0.0
            A[i, i] = 1.0
            b[i] = 0.0
            continue
        a = policy[i]
        A[i] -= gamma * P_model[i, a]
        b[i] += R_model[i, a]
    return np.linalg.solve(A, b)


def policy_iteration(gamma=GAMMA, max_iterations=100):
    """
    Start from an arbitrary policy, then alternate:
      1. evaluate it EXACTLY (not approximately - exact evaluation is cheap
         here because the grid is tiny, and it removes evaluation noise from
         the comparison)
      2. act greedily with respect to that V
    Stop when the greedy policy equals the current policy: a policy that is
    greedy w.r.t. its own values is optimal (the policy improvement theorem).
    """
    policy = np.zeros(N_STATES, dtype=int)       # start: always "up"
    history = []
    for iteration in range(1, max_iterations + 1):
        V = policy_evaluation_exact(policy, gamma)
        q_table = R_model + gamma * np.einsum("iaj,j->ia", P_model, V)
        new_policy = np.argmax(q_table, axis=1)
        changed = int(np.sum(new_policy != policy))
        history.append((iteration, V.copy(), changed))
        if changed == 0:
            return V, policy, q_table, iteration, history
        policy = new_policy
    return V, policy, q_table, max_iterations, history


# ======================================================================================
# PART 1 - value iteration
# ======================================================================================
print()
print("-" * 78)
print("PART 1 - VALUE ITERATION")
print("-" * 78)

V_star, greedy_policy, q_star, vi_sweeps, vi_history = value_iteration()

print(f"converged in {vi_sweeps} sweeps (tol 1e-10)")
print()
print("V* (optimal value function):")
print("      " + " ".join(f"c{c:<5}" for c in range(COLS)))
for r in range(ROWS):
    print(f"  r{r}  " + " ".join(f"{V_star[encode((r, c))]:>6.3f}" for c in range(COLS)))
print()
print("Greedy policy arrows (S = start, G = goal):")
for r in range(ROWS):
    row = []
    for c in range(COLS):
        s = (r, c)
        if s == START:
            row.append(" S ")
        elif s == GOAL:
            row.append(" G ")
        else:
            row.append(f" {ARROWS[greedy_policy[encode(s)]]} ")
    print("   " + "".join(row))
print()
print("Reading the arrows: from the start the agent moves right, and from any")
print("cell it heads toward the goal along a shortest path. No cell says 'up'")
print("in the top row or 'left' in the first column, because those moves bump")
print("into the border and only burn a step.")


# ======================================================================================
# PART 2 - convergence behaviour of value iteration
# ======================================================================================
print()
print("-" * 78)
print("PART 2 - VALUE ITERATION CONVERGENCE")
print("-" * 78)

reference = V_star
print(f"{'sweep':>6} {'max |V_k - V*|':>18} {'ratio':>10} {'greedy policy changes':>24}")
print("-" * 62)

# How different is the greedy policy at sweep k from the final one?
def greedy_of(V):
    q = R_model + GAMMA * np.einsum("iaj,j->ia", P_model, V)
    return np.argmax(q, axis=1)

final_greedy = greedy_of(reference)
previous_err = None
for k in [0, 1, 2, 3, 5, 8, 12, 20, vi_sweeps]:
    if k >= len(vi_history):
        continue
    err = float(np.max(np.abs(vi_history[k] - reference)))
    ratio = "-" if previous_err in (None, 0.0) else f"{err / previous_err:.3f}"
    changed = int(np.sum(greedy_of(vi_history[k]) != final_greedy))
    print(f"{k:>6} {err:>18.6e} {ratio:>10} {changed:>24}")
    previous_err = err
print()
print("The value error shrinks geometrically by ~gamma per sweep, but the")
print("GREEDY POLICY usually stabilises long before the values do. That gap is")
print("practically important: you can stop value iteration early and still have")
print("the right policy, even though V is still visibly wrong.")


# ======================================================================================
# PART 3 - policy iteration
# ======================================================================================
print()
print("-" * 78)
print("PART 3 - POLICY ITERATION")
print("-" * 78)

pi_V, pi_policy, pi_q, pi_iters, pi_history = policy_iteration()

print(f"converged in {pi_iters} policy improvements (starting from 'always up')")
print()
print(f"{'iter':>5} {'policies changed':>18} {'V(start)':>10}  what happened")
print("-" * 70)
notes = {
    1: "always-up evaluated; almost every state prefers another action",
    2: "improved policy evaluated; only a few states still switch",
    3: "usually stable - greedy w.r.t. its own values",
}
for iteration, V, changed in pi_history:
    note = notes.get(iteration, "no change -> optimal (policy improvement theorem)")
    print(f"{iteration:>5} {changed:>18} {V[encode(START)]:>10.4f}  {note}")
print()
print(f"Policy iteration took {pi_iters} improvements; value iteration took")
print(f"{vi_sweeps} sweeps. Each policy-iteration step contains an exact linear")
print("solve, so per-step it is far more expensive - but the whole loop is short.")
print("On this 16-state grid neither cost is visible; on a 10^6-state problem")
print("the linear solve is impossible and value iteration (or Q-learning) wins.")


# ======================================================================================
# PART 4 - the two must agree
# ======================================================================================
print()
print("-" * 78)
print("PART 4 - DO THEY AGREE?")
print("-" * 78)

print(f"{'state':<10} {'V* value iteration':>20} {'V* policy iteration':>21} {'diff':>10}")
print("-" * 64)
for index in range(N_STATES):
    print(f"{str(decode(index)):<10} {V_star[index]:>20.6f} {pi_V[index]:>21.6f} "
          f"{abs(V_star[index] - pi_V[index]):>10.2e}")
print()
policy_agreement = float(np.mean(greedy_policy == pi_policy))
print(f"max |V difference|:        {np.max(np.abs(V_star - pi_V)):.3e}")
print(f"policy agreement:          {policy_agreement:.4f} "
      f"({int(policy_agreement * N_STATES)}/{N_STATES} states identical)")
print()
print("Identical up to floating-point noise, as theory predicts: V* is unique")
print("for a fixed gamma, so two correct algorithms cannot disagree about it.")
print()
print("Where they CAN disagree is the POLICY in states whose Q-values are tied.")
print("If two actions have exactly the same Q, argmax picks the lower index and")
print("the two methods might break the tie differently. Ties are common in grids")
print("with symmetric costs and are the main source of 'my Q-learning picked a")
print("different action than the paper' confusion.")


# ======================================================================================
# PART 5 - how the optimal policy responds to gamma
# ======================================================================================
print()
print("-" * 78)
print("PART 5 - GAMMA CHANGES THE POLICY, NOT JUST THE NUMBERS")
print("-" * 78)

print(f"{'gamma':>7} {'V*(start)':>11} {'path length implied':>22} {'sweeps':>8}")
print("-" * 52)
for gamma in [0.5, 0.7, 0.85, 0.9, 0.95, 0.99, 0.999]:
    V_g, policy_g, _, sweeps_g, _ = value_iteration(gamma=gamma)
    # A sanity number: value at the cell one step from the goal on the optimal path.
    near_goal = V_g[encode((3, 2))]
    print(f"{gamma:>7} {V_g[encode(START)]:>11.4f} {near_goal:>22.4f} {sweeps_g:>8}")
print()
print("At low gamma the future is nearly worthless, so the differences between")
print("policies flatten out and convergence is fast. At gamma=0.999 the values")
print("scale up roughly as 1/(1-gamma) and convergence slows dramatically.")
print()
print("The optimal POLICY here barely changes, because every action costs the")
print("same - there is no tradeoff for gamma to resolve. Introduce a shortcut")
print("that is longer (more steps) but safer, and gamma becomes the knob that")
print("decides between them. That is the exploration/exploitation and")
print("risk-sensitivity story in its simplest form.")


# ======================================================================================
# Report
# ======================================================================================
print()
print("=" * 78)
print("REPORT")
print("=" * 78)
print(f"""
METHODS
  value iteration     {vi_sweeps} sweeps to 1e-10, then argmax over Q
  policy iteration    {pi_iters} improvements, each with an exact linear solve

AGREEMENT
  max |V_VI - V_PI|   {np.max(np.abs(V_star - pi_V)):.3e}
  policy agreement    {policy_agreement:.4f}

OPTIMAL VALUES (gamma={GAMMA})
  V*({START})       {V_star[encode(START)]:.4f}
  V*({GOAL})       0.0000  (terminal by definition)

THE UPDATE THAT MATTERS
  Q(s,a) = SUM_s' P(s'|s,a) [ R + gamma V(s') ]
  V(s)   = MAX_a Q(s,a)          <- SUM_a swapped for MAX_a

WHAT TO CARRY FORWARD
  - V* is unique for fixed gamma; algorithms differ only in how they find it
  - the greedy policy usually stabilises before V does - early stopping is safe
  - policy improvement theorem: greedy w.r.t. V^pi is never worse than pi
  - model-free RL replaces SUM_s' P [...] with a sample: R + gamma V(s')
  - script 05 keeps the MAX and drops the model; that is Q-learning
""")
print("=" * 78)
