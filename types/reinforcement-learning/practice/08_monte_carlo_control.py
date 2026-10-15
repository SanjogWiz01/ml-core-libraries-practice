"""
08 - Monte Carlo Control
========================

Goal: learn the optimal policy WITHOUT a model and WITHOUT bootstrapping, by
averaging the returns that followed each (state, action) pair in complete
episodes. This is the other half of the temporal-difference family, and the
contrast with scripts 05-06 is the point.

What "Monte Carlo" means here
-----------------------------
An update waits until the episode ENDS, then pushes Q(s,a) toward the actual
discounted return G_t that followed it:

    Q(s,a) <- Q(s,a) + alpha [ G_t - Q(s,a) ],      G_t = r_t + gamma r_{t+1} + ...

Compare with one-step TD:

    Q(s,a) <- Q(s,a) + alpha [ r + gamma Q(s',a') - Q(s,a) ]
                                 ~~~~~~~~~~~~~~~
    MC uses the whole realised future; TD uses one step plus a guess.

The tradeoff, in one line: MC is unbiased but high variance; TD is biased (by
its own current estimates) but low variance and usable online, before the
episode ends.

Two flavours
------------
First-visit MC: each (s,a) is counted once per episode, the first time it is
visited. Its average is an unbiased estimate of Q^pi(s,a).

Every-visit MC: count every occurrence. Also converges, and is what you use
when the same state repeats (common in grids). The difference is usually small.

The exploration problem
-----------------------
MC only learns about (s,a) pairs it actually visits. A greedy policy visits
only a handful of states, so a naive MC agent can never learn the rest. Two
fixes:
  - exploring starts: begin each episode at a random (s,a) (assumes you can
    reset the world to any state)
  - epsilon-soft forever: the policy always keeps probability >= eps/|A| on
    every action, so every state is eventually revisited. 'Greedy in the limit
    with infinite exploration' (GLIE).

This script implements exploring-starts MC control (Monte Carlo ES), the
textbook algorithm, and then shows why a fixed start needs epsilon-soft
exploration - and why that alone still is not enough on this grid.

Run:  python 08_monte_carlo_control.py
"""

import numpy as np

SEED = 42

print("=" * 78)
print("08 - MONTE CARLO CONTROL")
print("=" * 78)


# ======================================================================================
# Environment: a 5x5 grid with a pit and a goal
# ======================================================================================
ROWS, COLS = 5, 5
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTIONS = [UP, RIGHT, DOWN, LEFT]
ACTION_NAMES = ["up", "right", "down", "left"]
ARROWS = ["^", ">", "v", "<"]
DELTAS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}

START = (0, 0)
GOAL = (4, 4)
PIT = {(1, 1), (1, 2), (2, 1), (3, 3)}     # stepping here ends the episode badly
STEP_COST = -1.0
GOAL_REWARD = +10.0
PIT_PENALTY = -10.0
GAMMA = 0.95

N_STATES = ROWS * COLS
N_ACTIONS = 4


def encode(state):
    return state[0] * COLS + state[1]


def decode(index):
    return divmod(index, COLS)


def env_step(state, action):
    dr, dc = DELTAS[action]
    r, c = state
    nr, nc = r + dr, c + dc
    if not (0 <= nr < ROWS and 0 <= nc < COLS):
        nr, nc = r, c
    s2 = (nr, nc)
    if s2 in PIT:
        return s2, PIT_PENALTY, True
    if s2 == GOAL:
        return s2, GOAL_REWARD + STEP_COST, True
    return s2, STEP_COST, False


def epsilon_soft_probabilities(q_row, epsilon):
    """Greedy in the limit with infinite exploration: the best action gets
    1 - eps + eps/|A|, every other action gets eps/|A|. This is the standard
    eps-soft policy - exploration is spread over ALL actions, including the
    ones already known to be bad - which is what guarantees coverage."""
    n = len(q_row)
    best = np.flatnonzero(q_row == q_row.max())
    p = np.full(n, epsilon / n)
    p[best] += (1.0 - epsilon) / len(best)
    return p


# ======================================================================================
# Monte Carlo control: exploring starts by default, epsilon-soft optionally
# ======================================================================================
NON_TERMINAL = [s for s in [(r, c) for r in range(ROWS) for c in range(COLS)]
                if s not in PIT and s != GOAL]


def mc_control(n_episodes=20000, epsilon=0.0, epsilon_decay=0.9999, epsilon_min=0.01,
               gamma=GAMMA, first_visit=True, alpha=None, exploring_starts=True,
               seed=SEED, max_steps=200):
    """
    Two coverage strategies, because MC only learns about (s,a) pairs it visits:

    exploring_starts=True   begin each episode at a uniformly random
                            non-terminal state with a uniformly random first
                            action, then follow the (greedy) policy. This is
                            Monte Carlo ES, the textbook MC control algorithm.
                            It guarantees every pair is tried, so even a fully
                            greedy policy converges.

    exploring_starts=False  always start at START; the policy must keep
                            epsilon > 0 forever (epsilon-soft). Coverage is
                            then only probabilistic, which is the price for not
                            needing the ability to teleport the world to an
                            arbitrary state.

    alpha=None   -> running sample average, i.e. alpha = 1 / N(s,a) (true MC)
    alpha=float  -> constant step size, i.e. exponential forgetting (better in
                    practice for non-stationary problems, and it matches the
                    TD scripts, making the comparison fair)
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    N = np.zeros((N_STATES, N_ACTIONS), dtype=int)

    returns_per_episode = []
    lengths = []
    eps = epsilon

    for _ in range(n_episodes):
        # --- choose a starting state (and, under ES, a starting action) --------
        if exploring_starts:
            state = NON_TERMINAL[int(rng.integers(len(NON_TERMINAL)))]
            forced_action = int(rng.integers(N_ACTIONS))
        else:
            state = START
            forced_action = None

        # --- generate one full episode ----------------------------------------
        episode = []                       # list of (state, action, reward)
        done = False
        steps = 0
        while not done and steps < max_steps:
            if forced_action is not None:
                a = forced_action
                forced_action = None
            else:
                probs = epsilon_soft_probabilities(Q[encode(state)], eps)
                a = int(rng.choice(N_ACTIONS, p=probs))
            s2, r, done = env_step(state, a)
            episode.append((state, a, r))
            state = s2
            steps += 1

        # --- compute the return that followed each time step (backward sweep) ---
        # G_t = r_t + gamma G_{t+1}. Computing it backward shares work; doing it
        # forward naively is O(T^2).
        T = len(episode)
        G = np.zeros(T)
        running = 0.0
        for t in reversed(range(T)):
            running = episode[t][2] + gamma * running
            G[t] = running

        # --- update Q ---
        visits = {}
        for t, (s, a, _) in enumerate(episode):
            key = (encode(s), a)
            if first_visit:
                if key in visits:
                    continue            # ignore repeats within this episode
                visits[key] = True

            i, j = key
            N[i, j] += 1
            step = alpha if alpha is not None else 1.0 / N[i, j]
            Q[i, j] += step * (G[t] - Q[i, j])

        returns_per_episode.append(sum(r for _, _, r in episode))
        lengths.append(T)
        eps = max(epsilon_min, eps * epsilon_decay)

    return Q, N, np.array(returns_per_episode), np.array(lengths)


def greedy_policy(Q):
    return np.argmax(Q, axis=1)


def evaluate_greedy(Q, max_steps=200):
    """Roll the argmax policy out from START and report (steps, discounted
    return). The environment is deterministic, so this is exact."""
    state = START
    rewards = []
    steps = 0
    while steps < max_steps:
        a = int(np.argmax(Q[encode(state)]))
        s2, r, done = env_step(state, a)
        rewards.append(r)
        state = s2
        steps += 1
        if done:
            break
    g = 0.0
    for r in reversed(rewards):
        g = r + GAMMA * g
    return steps, g


def coverage(N):
    """How many states and (state, action) pairs were ever visited."""
    states = int(np.sum(N.sum(axis=1) > 0))
    pairs = int(np.sum(N > 0))
    return states, pairs


def render(rows, cols, policy=None, title=""):
    print(title)
    for r in range(rows):
        row = []
        for c in range(cols):
            s = (r, c)
            if s == START:
                row.append(" S ")
            elif s == GOAL:
                row.append(" G ")
            elif s in PIT:
                row.append(" P ")
            elif policy is not None:
                row.append(f" {ARROWS[policy[encode(s)]]} ")
            else:
                row.append(" . ")
        print("   " + "".join(row))


# ======================================================================================
# PART 1 - Monte Carlo ES: exploring starts + first-visit sample averaging
# ======================================================================================
print()
print("-" * 78)
print("PART 1 - MONTE CARLO ES (EXPLORING STARTS), SAMPLE AVERAGING")
print("-" * 78)

Q, N, _, _ = mc_control(n_episodes=20000, alpha=None, first_visit=True,
                        exploring_starts=True)
policy = greedy_policy(Q)

render(ROWS, COLS, policy, "Learned greedy policy (S start, G goal, P pit):")
steps, g = evaluate_greedy(Q)
states_seen, pairs_seen = coverage(N)
print()
print(f"V(start) = Q(start, argmax a)          {Q[encode(START)].max():>8.3f}")
print(f"greedy rollout from START: {steps} steps, discounted return {g:>6.3f}")
print(f"coverage: {states_seen}/{N_STATES} states, "
      f"{pairs_seen}/{N_STATES * N_ACTIONS} (state, action) pairs")
print()
print("Exploring starts let the greedy policy itself be the behaviour policy:")
print("because every episode is kicked off from a uniformly random cell with a")
print("random first action, every (s, a) is tried infinitely often, so the")
print("greedy policy no longer starves itself of data. The agent routes around")
print("the pit and into the goal, and no transition model was ever consulted -")
print("only sampled episodes.")


# ======================================================================================
# PART 2 - first-visit vs every-visit
# ======================================================================================
print()
print("-" * 78)
print("PART 2 - FIRST-VISIT vs EVERY-VISIT")
print("-" * 78)

Q_first, N_first, _, _ = mc_control(n_episodes=20000, alpha=0.05,
                                    first_visit=True, exploring_starts=True)
Q_every, N_every, _, _ = mc_control(n_episodes=20000, alpha=0.05,
                                    first_visit=False, exploring_starts=True)

print(f"{'variant':<16} {'V(start)':>10} {'greedy ret':>12} {'steps':>7} "
      f"{'pairs covered':>14}")
print("-" * 62)
for name, Qv, Nv in [("first-visit", Q_first, N_first),
                     ("every-visit", Q_every, N_every)]:
    st, gg = evaluate_greedy(Qv)
    _, pairs = coverage(Nv)
    print(f"{name:<16} {Qv[encode(START)].max():>10.3f} {gg:>12.3f} {st:>7} "
          f"{pairs:>10}/{N_STATES * N_ACTIONS}")
print()
print("Both variants converge to near-identical policies. Every-visit gives")
print("repeated (s, a) pairs more weight within an episode; on a grid where the")
print("policy repeatedly revisits cells this is a subtle difference in mixing,")
print("not in the answer. First-visit has a cleaner theoretical justification")
print("(each update is an unbiased estimate), every-visit is simpler to code.")


# ======================================================================================
# PART 3 - coverage is the bottleneck: exploring starts vs staying at START
# ======================================================================================
print()
print("-" * 78)
print("PART 3 - WHY EXPLORING STARTS MATTER: START-ONLY GETS TRAPPED")
print("-" * 78)
print("True MC (sample averaging, no step size) from a FIXED start. The only")
print("lever is how much the behaviour policy explores.")
print()
print(f"{'scheme':<34} {'V(start)':>10} {'greedy ret':>12} {'steps':>7}")
print("-" * 66)
part3_results = []
for label, kwargs in [
    ("exploring starts, greedy", dict(exploring_starts=True, epsilon=0.0)),
    ("start-only, eps-soft 0.1", dict(exploring_starts=False, epsilon=0.1)),
    ("start-only, eps-soft 0.5", dict(exploring_starts=False, epsilon=0.5)),
]:
    Qc, Nc, _, _ = mc_control(n_episodes=20000, alpha=None, first_visit=True,
                              epsilon_decay=1.0, **kwargs)
    st, gg = evaluate_greedy(Qc)
    part3_results.append((label, Qc[encode(START)].max(), gg, st))
    print(f"{label:<34} {Qc[encode(START)].max():>10.3f} {gg:>12.3f} {st:>7}")
print()
print("From a fixed start, a near-greedy policy walks straight into a pit, the")
print("episode ends after two steps, and the goal basin is never explored - so")
print("its value stays unknown and the greedy route never changes. This is the")
print("deep-exploration problem: the chance of stumbling onto a distant reward")
print("by random wandering is tiny once greedy episodes terminate early. Only")
print("a very high exploration rate (0.5) stumbles onto the goal within the")
print("budget. Exploring starts sidesteps the problem entirely by construction.")
print()
print("The practical fix in real environments, where you cannot teleport to an")
print("arbitrary state, is to explore hard early and decay it (GLIE): epsilon")
print("must go to zero for the sample average to be unbiased, but only AFTER")
print("the goal has been found.")


# ======================================================================================
# PART 4 - the variance problem, quantified
# ======================================================================================
print()
print("-" * 78)
print("PART 4 - MC RETURNS HAVE HIGH VARIANCE")
print("-" * 78)

# Same start state, same policy, many episodes: how spread are the returns?
rng = np.random.default_rng(SEED)
Q_learned = Q  # from PART 1
start_returns = []
for _ in range(3000):
    state = START
    g = 0.0
    for _ in range(200):
        probs = epsilon_soft_probabilities(Q_learned[encode(state)], 0.1)
        a = int(rng.choice(N_ACTIONS, p=probs))
        s2, r, done = env_step(state, a)
        g = r + GAMMA * g
        state = s2
        if done:
            break
    start_returns.append(g)

start_returns = np.array(start_returns)
print(f"returned across 3000 episodes from the start state:")
print(f"  mean        {start_returns.mean():>9.3f}")
print(f"  std         {start_returns.std():>9.3f}")
print(f"  min / max   {start_returns.min():>9.3f} / {start_returns.max():.3f}")
print(f"  spread (max-min) {start_returns.max() - start_returns.min():.3f}")
print()
print("A single sampled return can be tens of units away from the mean because")
print("one unlucky episode falls into the pit. MC averages THESE, so its targets")
print("are noisy - the curse that one-step TD fixes by bootstrapping and that")
print("n-step returns and GAE (script 09) interpolate between.")


# ======================================================================================
# Report
# ======================================================================================
print()
print("=" * 78)
print("REPORT")
print("=" * 78)
print(f"""
ALGORITHM
  Monte Carlo control with exploring starts (Monte Carlo ES), greedy policy
  update    Q(s,a) <- Q(s,a) + alpha [ G_t - Q(s,a) ]

TRAINING (first-visit, sample averaging, 20000 episodes, exploring starts)
  V(start)                 {Q[encode(START)].max():.3f}
  greedy rollout           {steps} steps, discounted return {g:.3f}
  coverage                 {states_seen}/{N_STATES} states, {pairs_seen}/{N_STATES * N_ACTIONS} pairs

FIRST vs EVERY VISIT (alpha=0.05, exploring starts)
  first-visit V(start)     {Q_first[encode(START)].max():.3f}
  every-visit V(start)     {Q_every[encode(START)].max():.3f}

COVERAGE EXPERIMENT (true MC, 20000 episodes)
  scheme                    V(start)   greedy ret   steps
  {'-'*56}""")
for label, vstart, gg, st in part3_results:
    print(f"  {label:<24}{vstart:>9.3f}   {gg:>9.3f}   {st:>5}")
print(f"""
VARIANCE FROM THE START STATE (3000 eps-soft rollouts)
  mean {start_returns.mean():.2f}, std {start_returns.std():.2f}, spread {start_returns.max() - start_returns.min():.2f}

WHAT TO CARRY FORWARD
  - MC needs full episodes; it cannot update mid-trajectory or on a single
    transition, so it is unusable for continuing tasks without truncation
  - unbiased target (real returns), high variance (whole-episode noise)
  - coverage is the real bottleneck: exploring starts guarantees every (s, a)
    is visited, while a fixed start needs epsilon-soft exploration and still
    cannot find a distant reward once greedy episodes terminate early
  - the backward G_t sweep is O(T) and shared across the episode's updates
  - script 09 blends this with TD: n-step returns use n real rewards then a
    bootstrapped estimate, dialing variance against bias
""")
print("=" * 78)
