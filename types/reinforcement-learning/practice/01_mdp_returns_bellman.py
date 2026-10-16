"""
01 - MDP, Returns and the Bellman Equations
===========================================  i just addd one line to the top of this file to test git commit and push

Goal: take a 3-state Markov Decision Process, write out its transition and
reward tables by hand, and compute by brute force every quantity that the rest
of reinforcement learning is built on: the discounted return, the state-value
function, the action-value function, and the two Bellman equations.

Why start here
--------------
RL has a reputation for being "trial and error with neural networks", but the
core of it is a small amount of linear algebra on a finite table. If you can
follow a 3-state MDP end to end without a library, then Q-learning and DQN
later are just *approximation schemes* for the same numbers computed here
exactly.

The objects
-----------
An MDP is the tuple (S, A, P, R, gamma):

    S   states             here: {"home", "cafe", "work"}
    A   actions            here: {"stay", "walk"}
    P   P(s' | s, a)       transition probabilities, a dict of dicts
    R   R(s, a, s')        reward for the resulting transition
    gamma  discount         0 <= gamma < 1, how much future reward counts

The return from time t onward is

    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

gamma = 0 means "only care about the immediate reward". gamma -> 1 means
"care about the far future almost as much as now". Without gamma < 1 an
episodic task with no end has an infinite sum that may not converge.

The Bellman expectation equations
---------------------------------
    V(s)  = SUM_a pi(a|s) * SUM_s' P(s'|s,a) * [ R(s,a,s') + gamma * V(s') ]
    Q(s,a)=          SUM_s' P(s'|s,a) * [ R(s,a,s') + gamma * V(s') ]

Or, in one line for a fixed policy, V = r_pi + gamma * P_pi * V, which is a
linear system. Solving it directly with numpy gives the exact answer that
*iterative* policy evaluation (script 03) approximates step by step.

Run:  python 01_mdp_returns_bellman.py
"""

import numpy as np

SEED = 42

print("=" * 78)
print("01 - MDP, RETURNS AND THE BELLMAN EQUATIONS")
print("=" * 78)


# ======================================================================================
# The MDP itself
# ======================================================================================
STATES = ["home", "cafe", "work"]
ACTIONS = ["stay", "walk"]

# P[next_state][(state, action)] = probability
# Every distribution over next_state must sum to exactly 1.0 - that is the
# defining constraint of a valid transition model, and script 02 checks it.
P = {
    "home": {
        ("home", "stay"): {"home": 1.0},
        ("home", "walk"): {"cafe": 0.8, "work": 0.2},   # sometimes walks past the cafe
    },
    "cafe": {
        ("cafe", "stay"): {"cafe": 1.0},
        ("cafe", "walk"): {"work": 0.9, "home": 0.1},   # occasionally turns back
    },
    "work": {
        ("work", "stay"): {"work": 1.0},
        ("work", "walk"): {"home": 1.0},
    },
}

# R[(state, action, next_state)] - immediate reward, can be negative.
R = {
    ("home", "stay", "home"): 0.0,
    ("home", "walk", "cafe"): 1.0,
    ("home", "walk", "work"): 3.0,
    ("cafe", "stay", "cafe"): -0.2,   # doing nothing at the cafe costs money
    ("cafe", "walk", "work"): 2.0,
    ("cafe", "walk", "home"): -1.0,   # walking home by mistake
    ("work", "stay", "work"): 5.0,    # salary tick while at work
    ("work", "walk", "home"): 0.5,    # finally heading home
}

GAMMA = 0.9


def step(state, action, rng):
    """Sample (next_state, reward) from the model. This is the ONLY thing an
    agent ever gets to call - everything else is internal bookkeeping."""
    distribution = P[state][(state, action)]
    next_state = rng.choice(list(distribution.keys()),
                            p=list(distribution.values()))
    return next_state, R[(state, action, next_state)]


def reward_of(state, action, next_state):
    return R[(state, action, next_state)]


# ======================================================================================
# PART 1 - validity of the model
# ======================================================================================
print()
print("-" * 78)
print("PART 1 - THE MODEL IS WELL FORMED")
print("-" * 78)

print(f"{'state':<8} {'action':<8} {'distribution':<34} {'sum':>6}")
print("-" * 60)
for state in STATES:
    for action in ACTIONS:
        dist = P[state][(state, action)]
        total = sum(dist.values())
        rendered = ", ".join(f"{k}:{v:g}" for k, v in dist.items())
        flag = "" if abs(total - 1.0) < 1e-12 else "  <-- BROKEN"
        print(f"{state:<8} {action:<8} {rendered:<34} {total:>6.3f}{flag}")
print()
print("Every row sums to 1.0, so P is a proper conditional distribution. A model")
print("that leaks or invents probability makes every later number meaningless.")


# ======================================================================================
# PART 2 - a sampled trajectory and its discounted return
# ======================================================================================
print()
print("-" * 78)
print("PART 2 - ONE TRAJECTORY, AND ITS RETURN G_t")
print("-" * 78)

rng = np.random.default_rng(SEED)
gamma = GAMMA


def rollout(policy, max_steps=8, start="home"):
    """Follow `policy(state) -> action` until max_steps, recording the
    (s, a, r) triplets. This is what experience replay would store."""
    state = start
    trace = []
    for _ in range(max_steps):
        action = policy(state)
        next_state, reward = step(state, action, rng)
        trace.append((state, action, reward, next_state))
        state = next_state
    return trace


def discounted_return(rewards, gamma):
    """G_0 = r_0 + gamma*r_1 + gamma^2*r_2 + ... written as a Horner loop, so
    no large powers of gamma ever appear and precision is preserved."""
    total = 0.0
    for reward in reversed(rewards):
        total = reward + gamma * total
    return total


greedy_walk = lambda s: "walk"
trace = rollout(greedy_walk)

print(f"policy: always 'walk', gamma = {gamma}")
print()
print(f"{'t':>2} {'state':<8} {'action':<8} {'reward':>7} {'gamma^t':>9} {'disc.':>8} {'next':<8}")
print("-" * 60)
running_total = 0.0
for t, (s, a, r, s2) in enumerate(trace):
    weight = gamma ** t
    running_total += weight * r
    print(f"{t:>2} {s:<8} {a:<8} {r:>7.2f} {weight:>9.4f} {weight * r:>8.3f} {s2:<8}")

rewards = [r for _, _, r, _ in trace]
g0 = discounted_return(rewards, gamma)
print()
print(f"G_0 (full sum over {len(rewards)} steps) = {g0:.4f}")
print(f"undiscounted sum of the same rewards      = {sum(rewards):.4f}")
print()
print("Discounting pulls the number down. With gamma=0.9 a reward of 5 that")
print("arrives 6 steps later is worth only 5 * 0.9^6 = 2.66 today.")
print(f"check: 5 * 0.9^6 = {5 * 0.9 ** 6:.4f}")
print()
for g in [0.0, 0.5, 0.9, 0.99, 0.999]:
    value = discounted_return(rewards, g)
    print(f"  gamma={g:<6} -> G_0 = {value:>8.4f}")
print()
print("gamma=0 throws away everything after the first reward. As gamma -> 1 the")
print("return approaches the plain sum. gamma also sets the effective horizon:")
print(f"  effective horizon ~ 1 / (1 - gamma) = {1 / (1 - GAMMA):.0f} steps at gamma=0.9")


# ======================================================================================
# PART 3 - exact policy evaluation by solving the linear system
# ======================================================================================
print()
print("-" * 78)
print("PART 3 - V AND Q FOR A FIXED POLICY, SOLVED EXACTLY")
print("-" * 78)

# Uniform random policy: pi(a|s) = 1/|A| for every action.
PI = {s: {a: 1.0 / len(ACTIONS) for a in ACTIONS} for s in STATES}

n = len(STATES)
index = {s: i for i, s in enumerate(STATES)}

# Build V = r + gamma * P V  =>  (I - gamma * P) V = r
# P[i, j] = probability of moving from state i to state j under the policy.
A_mat = np.eye(n)
r_vec = np.zeros(n)

for s in STATES:
    i = index[s]
    for a, pa in PI[s].items():
        for s2, p_trans in P[s][(s, a)].items():
            j = index[s2]
            A_mat[i, j] -= gamma * pa * p_trans
            r_vec[i] += pa * p_trans * reward_of(s, a, s2)

V_exact = np.linalg.solve(A_mat, r_vec)
V = dict(zip(STATES, V_exact))

print("Policy: uniform random over {stay, walk}")
print()
print(f"{'state':<8} {'V(s) exact':>12}   computed as (I - gamma*P)^-1 * r")
print("-" * 60)
for s in STATES:
    print(f"{s:<8} {V[s]:>12.4f}")
print()

# Q(s,a) = expected immediate reward + gamma * expected V(next)
Q = {}
print(f"{'state':<8} {'action':<8} {'E[r]':>8} {'E[gamma V(s\')]':>16} {'Q(s,a)':>9}")
print("-" * 60)
for s in STATES:
    for a in ACTIONS:
        immediate = 0.0
        future = 0.0
        for s2, p_trans in P[s][(s, a)].items():
            immediate += p_trans * reward_of(s, a, s2)
            future += gamma * p_trans * V[s2]
        Q[(s, a)] = immediate + future
        print(f"{s:<8} {a:<8} {immediate:>8.3f} {future:>16.4f} {immediate + future:>9.4f}")
print()
print("Q(s,a) mixes two things: what the move pays right now, and what the")
print("state it lands you in is worth from then on. That second term is the")
print("whole reason RL works on delayed rewards.")


# ======================================================================================
# PART 4 - the Bellman backup, shown as an actual number-by-number sum
# ======================================================================================
print()
print("-" * 78)
print("PART 4 - ONE BELLMAN BACKUP, TERM BY TERM")
print("-" * 78)

state, action = "home", "walk"
print(f"Expanding Q({state}, {action}) with the uniform policy's V:")
print()
print(f"{'next':<8} {'prob':>6} {'r(s,a,s\')':>10} {'gamma*V(next)':>14} {'weighted':>10}")
print("-" * 52)
total = 0.0
for s2, p_trans in P[state][(state, action)].items():
    r = reward_of(state, action, s2)
    future = gamma * V[s2]
    contribution = p_trans * (r + future)
    total += contribution
    print(f"{s2:<8} {p_trans:>6.2f} {r:>10.2f} {future:>14.4f} {contribution:>10.4f}")
print(f"{'':<8} {'':>6} {'':>10} {'total Q':>14} {total:>10.4f}")
print()
print(f"Q(home, walk) = {total:.4f}  vs table value {Q[('home', 'walk')]:.4f} "
      f"(difference {abs(total - Q[('home', 'walk')]):.2e})")
print()
print("A backup is just a weighted average of (reward + discounted value of")
print("where you land). Every RL algorithm differs only in WHERE it gets V from")
print("and WHEN it applies the backup.")


# ======================================================================================
# PART 5 - greedy policy and optimal values via policy iteration on a tiny table
# ======================================================================================
print()
print("-" * 78)
print("PART 5 - IMPROVING THE POLICY: ONE SWEEP OF POLICY ITERATION")
print("-" * 78)

# For a fixed policy, the Bellman *optimality* backup is a max instead of a
# sum over actions. One round of: evaluate exactly, then act greedily, is the
# first step of policy iteration.
V_opt = np.zeros(n)
Q_opt = np.zeros((n, len(ACTIONS)))

for _ in range(200):
    # evaluate the greedy policy exactly (same linear solve as PART 3)
    A_mat = np.eye(n)
    r_vec = np.zeros(n)
    for s in STATES:
        i = index[s]
        best_a = max(ACTIONS, key=lambda a: Q_opt[i, ACTIONS.index(a)])
        for s2, p_trans in P[s][(s, best_a)].items():
            j = index[s2]
            A_mat[i, j] -= gamma * p_trans
            r_vec[i] += p_trans * reward_of(s, best_a, s2)
    V_new = np.linalg.solve(A_mat, r_vec)
    # recompute every Q from the new V
    for s in STATES:
        i = index[s]
        for a in ACTIONS:
            q = 0.0
            for s2, p_trans in P[s][(s, a)].items():
                q += p_trans * (reward_of(s, a, s2) + gamma * V_new[index[s2]])
            Q_opt[i, ACTIONS.index(a)] = q
    if np.max(np.abs(V_new - V_opt)) < 1e-10:
        V_opt = V_new
        break
    V_opt = V_new

print(f"{'state':<8} {'V random policy':>16} {'V* (optimal)':>14} {'gain':>9}")
print("-" * 52)
for s in STATES:
    i = index[s]
    print(f"{s:<8} {V[s]:>16.4f} {V_opt[i]:>14.4f} {V_opt[i] - V[s]:>9.4f}")
print()

print("Greedy policy under V*:")
for s in STATES:
    i = index[s]
    q_values = {a: Q_opt[i, ACTIONS.index(a)] for a in ACTIONS}
    best = max(q_values, key=q_values.get)
    ranked = ", ".join(f"{a}={v:.3f}" for a, v in sorted(q_values.items(),
                                                          key=lambda kv: -kv[1]))
    print(f"  {s:<8} -> {best:<6}   ({ranked})")
print()
print("The random policy dithers in 'stay' for a negative reward at the cafe.")
print("The optimal policy never wastes a step: it walks, and sits at work")
print("collecting +5 rather than staying anywhere else.")


# ======================================================================================
# Report
# ======================================================================================
print()
print("=" * 78)
print("REPORT")
print("=" * 78)
print(f"""
MDP
  states            {STATES}
  actions           {ACTIONS}
  gamma             {GAMMA}
  effective horizon ~{1 / (1 - GAMMA):.0f} steps

RETURNS (always-walk rollout, {len(rewards)} steps)
  G_0 at gamma=0.9   {g0:.4f}
  plain sum           {sum(rewards):.4f}

VALUES UNDER THE UNIFORM RANDOM POLICY
  """ + "\n  ".join(f"V({s}) = {V[s]:.4f}" for s in STATES) + f"""

OPTIMAL VALUES
  """ + "\n  ".join(f"V*({s}) = {V_opt[index[s]]:.4f}" for s in STATES) + f"""

WHAT TO CARRY FORWARD
  - return is a weighted sum; gamma sets both the horizon and the scale
  - V(s) is a sum over actions weighted by the policy; Q(s,a) is not
  - one Bellman backup = weighted average of r + gamma * V(next)
  - replacing SUM over actions with MAX over actions gives the optimality
    backup, which is what value iteration and Q-learning actually run
  - solving (I - gamma P) V = r is exact but needs the model; scripts 03-06
    learn the same numbers without ever building that matrix
""")
print("=" * 78)
