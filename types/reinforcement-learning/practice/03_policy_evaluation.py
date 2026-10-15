"""
03 - Policy Evaluation: Iterative vs Exact
===========================================

Goal: compute the state-value function V^pi for a fixed policy in a 4x4
GridWorld, twice - once by the iterative Bellman sweep that every RL algorithm
uses, and once by solving the linear system directly - then show the two agree
to machine precision, and show what breaks the iteration.

Why this matters
----------------
Policy evaluation is the simplest complete RL algorithm there is, and it is
the inner loop of policy iteration, the bootstrap target of every TD method,
and the "critic" half of every actor-critic. If you can predict V^pi for a
known policy and a known model, you already understand what DQN is doing with
far worse information.

Two ways to get V^pi
--------------------
Iterative (no model beyond one backup):

    V_{k+1}(s) = SUM_a pi(a|s) SUM_s' P(s'|s,a) [ R + gamma * V_k(s') ]

Exact (needs the whole model at once):

    (I - gamma * P_pi) V = r_pi      solve the linear system

The iterative version converges to the exact one because the backup is a
gamma-contraction: the maximum difference between two successive sweeps
shrinks by at least a factor of gamma each sweep. That single fact is why RL
is guaranteed to converge at all.

Run:  python 03_policy_evaluation.py
"""

import numpy as np

SEED = 42

print("=" * 78)
print("03 - POLICY EVALUATION: ITERATIVE vs EXACT")
print("=" * 78)


# ======================================================================================
# A small, fully known 4x4 GridWorld
# ======================================================================================
ROWS, COLS = 4, 4
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTIONS = [UP, RIGHT, DOWN, LEFT]
ACTION_NAMES = {UP: "up", RIGHT: "right", DOWN: "down", LEFT: "left"}
DELTAS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}

START = (0, 0)
GOAL = (3, 3)
GAMMA = 0.9
STEP_COST = -1.0
GOAL_REWARD = 0.0          # reaching the goal is free, but each step costs -1

N_STATES = ROWS * COLS


def encode(state):
    return state[0] * COLS + state[1]


def decode(index):
    return divmod(index, COLS)


def next_state(state, action):
    """Deterministic transition with a reflecting border: bumping into the edge
    leaves the agent where it was (and still pays the step cost)."""
    dr, dc = DELTAS[action]
    r, c = state
    nr, nc = r + dr, c + dc
    if not (0 <= nr < ROWS and 0 <= nc < COLS):
        nr, nc = r, c
    return (nr, nc)


def is_terminal(state):
    return state == GOAL


def reward_of(state, action, s2):
    if s2 == GOAL and not is_terminal(state):
        return GOAL_REWARD + STEP_COST
    return STEP_COST


def transitions(state, action):
    """Return [(probability, next_state, reward)]. Deterministic here, but the
    list-of-triples shape keeps every later algorithm model-agnostic."""
    s2 = next_state(state, action)
    return [(1.0, s2, reward_of(state, action, s2))]


# --- policies under test --------------------------------------------------------------
def make_random_policy():
    return {encode(s): {a: 0.25 for a in ACTIONS} for s in
            [(r, c) for r in range(ROWS) for c in range(COLS)]}


def make_fixed_policy(direction):
    """Deterministic: every state takes `direction`, expressed as a one-hot
    dict so the evaluators can treat it exactly like a stochastic policy. The
    goal has no action at all (a terminal state has no outgoing edges)."""
    policy = {}
    for r in range(ROWS):
        for c in range(COLS):
            s = (r, c)
            if is_terminal(s):
                policy[encode(s)] = None
            else:
                policy[encode(s)] = {direction: 1.0}
    return policy


# ======================================================================================
# Algorithm 1 - iterative policy evaluation
# ======================================================================================
def evaluate_iterative(policy, gamma=GAMMA, tol=1e-10, max_sweeps=10_000):
    """
    Synchronous Bellman sweeps: read the whole V_k, write the whole V_{k+1}.

    The alternative - updating V in place as you sweep (Gauss-Seidel / "in-place
    backup") - also converges and typically twice as fast, because information
    from the state you just updated propagates within the same sweep. Both are
    used in practice; the difference is bookkeeping, not correctness.
    """
    V = np.zeros(N_STATES)
    history = [V.copy()]

    for sweep in range(1, max_sweeps + 1):
        V_new = np.zeros(N_STATES)
        max_delta = 0.0
        for index in range(N_STATES):
            state = decode(index)
            if is_terminal(state):
                V_new[index] = 0.0            # terminal states have value 0 by definition
                continue
            total = 0.0
            for action, p_action in policy[index].items():
                for p_trans, s2, reward in transitions(state, action):
                    total += p_action * p_trans * (reward + gamma * V[encode(s2)])
            V_new[index] = total
        max_delta = float(np.max(np.abs(V_new - V)))
        V = V_new
        history.append(V.copy())
        if max_delta < tol:
            break

    return V, sweep, history, max_delta


# ======================================================================================
# Algorithm 2 - exact solution of the linear system
# ======================================================================================
def evaluate_exact(policy, gamma=GAMMA):
    """Build (I - gamma P_pi) V = r_pi and solve it in one shot."""
    A = np.eye(N_STATES)
    b = np.zeros(N_STATES)

    for index in range(N_STATES):
        state = decode(index)
        if is_terminal(state):
            # Value of a terminal state is 0; pin it with V = 0.
            A[index, :] = 0.0
            A[index, index] = 1.0
            b[index] = 0.0
            continue
        for action, p_action in policy[index].items():
            for p_trans, s2, reward in transitions(state, action):
                A[index, encode(s2)] -= gamma * p_action * p_trans
                b[index] += p_action * p_trans * reward

    return np.linalg.solve(A, b)


# ======================================================================================
# PART 1 - uniform random policy
# ======================================================================================
print()
print("-" * 78)
print("PART 1 - V^pi FOR THE UNIFORM RANDOM POLICY")
print("-" * 78)

random_policy = make_random_policy()
V_iter, n_sweeps, history, final_delta = evaluate_iterative(random_policy)
V_exact = evaluate_exact(random_policy)

print("policy: pick up/right/down/left uniformly at random")
print()
print("Iterative value function (rows = grid rows):")
for r in range(ROWS):
    print("   " + " ".join(f"{V_iter[encode((r, c))]:>7.3f}" for c in range(COLS)))
print()
print("Exact solve of (I - gamma P) V = r:")
for r in range(ROWS):
    print("   " + " ".join(f"{V_exact[encode((r, c))]:>7.3f}" for c in range(COLS)))
print()

abs_error = np.abs(V_iter - V_exact)
print(f"sweeps to converge (tol=1e-10): {n_sweeps}")
print(f"final max |V_{{k+1}} - V_k|:     {final_delta:.3e}")
print(f"max |iterative - exact|:         {abs_error.max():.3e}")
print()
print("The two agree to ~1e-10, which is the tolerance we asked for. They are")
print("not 'close because the algorithm is good' - they are the same fixed point")
print("found by two different routes.")


# ======================================================================================
# PART 2 - convergence, and the gamma-contraction made visible
# ======================================================================================
print()
print("-" * 78)
print("PART 2 - HOW FAST IT CONVERGES")
print("-" * 78)

print(f"{'sweep':>6} {'max |V_k - V_exact|':>22} {'ratio to previous':>20}")
print("-" * 52)
reference = V_exact
for k in [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30]:
    if k >= len(history):
        break
    err = float(np.max(np.abs(history[k] - reference)))
    if k == 0:
        ratio = float("nan")
        prev = err
        print(f"{k:>6} {err:>22.6e} {'-':>20}")
    else:
        ratio = err / prev if prev > 0 else float("nan")
        print(f"{k:>6} {err:>22.6e} {ratio:>20.3f}")
        prev = err
print()
print(f"The error ratio settles around {GAMMA} - exactly the discount factor.")
print("That is the contraction property: each sweep multiplies the remaining")
print("error by at most gamma. Contraction implies a unique fixed point, which")
print("is the theorem that guarantees V^pi exists and is well defined.")
print()
print(f"Rough sweep count to reach tolerance t:  log(t) / log(gamma) "
      f"= {np.log(1e-10) / np.log(GAMMA):.0f} sweeps.")
print("Smaller gamma converges faster and forgets the future harder - the same")
print("tradeoff shows up as slower DQN training when gamma is pushed near 1.")


# ======================================================================================
# PART 3 - policy shape changes the values completely
# ======================================================================================
print()
print("-" * 78)
print("PART 3 - THE SAME ENVIRONMENT, DIFFERENT POLICIES")
print("-" * 78)

policies = {
    "uniform random": make_random_policy(),
    "always right": make_fixed_policy(RIGHT),
    "always down": make_fixed_policy(DOWN),
    "right then down": make_fixed_policy(RIGHT),
}
# "right then down" - go right while possible, else down.
hybrid = {}
for r in range(ROWS):
    for c in range(COLS):
        s = (r, c)
        hybrid[encode(s)] = None if is_terminal(s) else {
            (RIGHT if c < COLS - 1 else DOWN): 1.0}
policies["right then down"] = hybrid

print(f"{'policy':<20} {'V(start)':>10} {'V(1,2)':>10} {'V(3,1)':>10} "
      f"{'mean V':>10} {'sweeps':>8}")
print("-" * 72)
results = {}
for name, policy in policies.items():
    value, sweeps, _, _ = evaluate_iterative(policy)
    results[name] = value
    print(f"{name:<20} {value[encode(START)]:>10.4f} {value[encode((1, 2))]:>10.4f} "
          f"{value[encode((3, 1))]:>10.4f} {value.mean():>10.4f} {sweeps:>8}")
print()

best_name = max(results, key=lambda n: results[n][encode(START)])
print(f"Best policy for the start state: '{best_name}' "
      f"with V(0,0) = {results[best_name][encode(START)]:.4f}.")
print()
print("Reading the values: V(0,0) is negative everywhere because every step")
print("costs -1 and random walking takes many steps to stumble into the goal.")
print("'always right' then 'always down' is the shortest path policy - 6 steps -")
print("and its V(0,0) is the least negative of the four.")
print()
print("Note that a policy is only 'optimal' relative to a gamma. With gamma=0")
print("every policy looks the same (all that counts is the next reward, which is")
print("-1 regardless). Ordering between policies only appears once the future")
print("is allowed to matter.")


# ======================================================================================
# PART 4 - what the value function predicts, concretely
# ======================================================================================
print()
print("-" * 78)
print("PART 4 - WHAT A VALUE MEANS: EXPECTED RETURN FROM EACH CELL")
print("-" * 78)

print("V is a prediction about the future, not a property of the cell. Starting")
print("from (0,0) under the uniform policy:")
print()
print(f"  V(0,0) = {V_iter[encode(START)]:.4f}  ->  expected discounted return before acting")
print(f"  gamma  = {GAMMA}            ->  how much later rewards are worth now")
print()
print("Monte Carlo check: actually run 4000 episodes and average the returns.")
print()

gamma = GAMMA
returns = []
for episode in range(4000):
    rng = np.random.default_rng(10_000 + episode)
    state = START
    g = 0.0
    for t in range(400):
        if is_terminal(state):
            break
        action = int(rng.choice(ACTIONS))
        s2 = next_state(state, action)
        reward = reward_of(state, action, s2)
        g = reward + gamma * g
        state = s2
    returns.append(g)

returns = np.array(returns)
mc_mean = returns.mean()
mc_sem = returns.std(ddof=1) / np.sqrt(len(returns))
print(f"  analytic V(0,0)          {V_iter[encode(START)]:>10.4f}")
print(f"  Monte Carlo mean of {len(returns)} returns {mc_mean:>10.4f}")
print(f"  standard error           {mc_sem:>10.4f}")
print(f"  difference               {abs(mc_mean - V_iter[encode(START)]):>10.4f}")
print()
print("Analytic evaluation and sampled evaluation estimate the same number.")
print("The difference is the sampling noise, and it shrinks like 1/sqrt(N) -")
print("which is why Monte Carlo methods (script 08) need many episodes while")
print("bootstrapping methods need few.")


# ======================================================================================
# Report
# ======================================================================================
print()
print("=" * 78)
print("REPORT")
print("=" * 78)
print(f"""
SETUP
  grid                {ROWS} x {COLS}, goal {GOAL}, step cost {STEP_COST}, gamma {GAMMA}
  uniform policy      V(start) = {V_iter[encode(START)]:.4f}

CONVERGENCE
  sweeps to 1e-10     {n_sweeps}
  final residual      {final_delta:.3e}
  error vs exact      {abs_error.max():.3e}
  contraction ratio   settles at ~{GAMMA} (the discount factor itself)

MONTE CARLO CROSS-CHECK
  analytic            {V_iter[encode(START)]:.4f}
  sampled (n=4000)    {mc_mean:.4f} +/- {mc_sem:.4f}

WHAT TO CARRY FORWARD
  - one backup is an expectation over (action, next state) weighted by pi and P
  - iterative evaluation = repeated application of a gamma-contraction, so a
    unique fixed point exists and is reached from any starting V
  - terminal states are pinned to 0; that boundary condition is what makes the
    finite-horizon problem well posed
  - evaluating a KNOWN policy under a KNOWN model is the easy direction; the
    hard direction - improving the policy - is scripts 04, 05 and 06
  - the same backup with MAX over actions instead of SUM gives V* (script 04)
""")
print("=" * 78)
