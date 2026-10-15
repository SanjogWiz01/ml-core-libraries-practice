"""
05 - Q-Learning From Scratch
============================

Goal: implement tabular Q-learning - the canonical model-free, off-policy,
temporal-difference algorithm - and train it on a GridWorld with a cliff, with
no library beyond NumPy. Then inspect the learned Q-table, the greedy policy,
the learning curve, and the effect of the learning rate.

The update rule
---------------
    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
                              \_________________  _________________/
                                                \/
                                       temporal-difference error (delta)

Three properties define Q-learning:

  **model-free** - it never uses P. The world supplies one (s, a, r, s') and
  that is the entire input.

  **bootstrapped** - the target uses the CURRENT Q of the next state, not the
  true value. It is a moving target, which is why convergence is a theorem
  about stochastic approximation rather than about plain fixed-point iteration.

  **off-policy** - the target uses max over actions regardless of what the
  behaviour policy actually did. So Q-learning learns the values of the
  GREEDY policy while being trained by an EXPLORING policy. SARSA (script 06)
  is the on-policy twin that replaces max with the next action actually taken.

Convergence: with a decaying alpha that satisfies SUM alpha = inf and
SUM alpha^2 < inf (e.g. 1/k), and every (s,a) visited infinitely often,
Q converges to Q* with probability 1. Constant alpha never fully converges -
it keeps chasing the newest samples, which is often what you want in a
non-stationary world.

Run:  python 05_q_learning.py
"""

import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)

print("=" * 78)
print("05 - Q-LEARNING FROM SCRATCH")
print("=" * 78)


# ======================================================================================
# Environment: the cliff grid from script 02, reduced and explicit
# ======================================================================================
ROWS, COLS = 4, 6
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTIONS = [UP, RIGHT, DOWN, LEFT]
ACTION_NAMES = ["up", "right", "down", "left"]
ARROWS = ["^", ">", "v", "<"]
DELTAS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}

START = (0, 0)
GOAL = (3, 5)
CLIFF = {(3, 1), (3, 2), (3, 3), (3, 4)}
STEP_COST = -1.0
GOAL_REWARD = +10.0
CLIFF_PENALTY = -100.0
GAMMA = 0.9

N_STATES = ROWS * COLS
N_ACTIONS = 4


def encode(state):
    return state[0] * COLS + state[1]


def decode(index):
    return divmod(index, COLS)


def step(state, action, slip=0.0, env_rng=None):
    """Returns (next_state, reward, terminated). Border bumps cost a step."""
    chosen = action
    if slip > 0 and env_rng is not None and env_rng.random() < slip:
        chosen = int(env_rng.choice([a for a in ACTIONS if a != action]))

    dr, dc = DELTAS[chosen]
    r, c = state
    nr, nc = r + dr, c + dc
    if not (0 <= nr < ROWS and 0 <= nc < COLS):
        nr, nc = r, c
    s2 = (nr, nc)

    if s2 in CLIFF:
        return START, CLIFF_PENALTY, True
    if s2 == GOAL:
        return s2, GOAL_REWARD + STEP_COST, True
    return s2, STEP_COST, False


def epsilon_greedy(q_row, epsilon, action_rng):
    """With prob epsilon pick uniformly at random, otherwise pick argmax.
    Ties are broken randomly among the best actions - breaking them
    deterministically can lock the agent into an arbitrary suboptimal action
    forever, because the tie never gets re-examined."""
    if action_rng.random() < epsilon:
        return int(action_rng.integers(N_ACTIONS))
    best = np.flatnonzero(q_row == q_row.max())
    return int(action_rng.choice(best))


# ======================================================================================
# The algorithm
# ======================================================================================
def q_learning(n_episodes=1500, alpha=0.2, gamma=GAMMA, epsilon=0.1,
               epsilon_decay=0.999, epsilon_min=0.01, slip=0.0,
               alpha_schedule=None, seed=SEED, max_steps=500,
               optimistic_init=0.0):
    """
    Tabular Q-learning.

    Parameters
    ----------
    alpha : float or callable
        Constant learning rate, or a function of the episode index.
    epsilon : float
        Starting exploration rate, decayed multiplicatively per episode.
    optimistic_init : float
        Start every Q at this value instead of 0. Because the target is
        max_a' Q(s',a'), over-estimating untried actions makes the agent try
        them - an exploration scheme that needs no epsilon at all.
    """
    env_rng = np.random.default_rng(seed + 1)
    action_rng = np.random.default_rng(seed + 2)

    Q = np.full((N_STATES, N_ACTIONS), float(optimistic_init))
    episode_returns = []
    episode_lengths = []
    eps_track = []
    alpha_track = []

    current_eps = epsilon

    for episode in range(n_episodes):
        state = START
        total_reward = 0.0
        done = False
        steps = 0
        ep_alpha = alpha(episode) if callable(alpha) else alpha

        while not done and steps < max_steps:
            a = epsilon_greedy(Q[encode(state)], current_eps, action_rng)
            s2, reward, done = step(state, a, slip=slip, env_rng=env_rng)
            i, j = encode(state), a

            if done:
                # Terminal target is just r: there is no next state to bootstrap
                # from. Forgetting this line is a classic bug that invents
                # value out of nowhere after the goal.
                target = reward
            else:
                target = reward + gamma * Q[encode(s2)].max()

            delta = target - Q[i, j]
            Q[i, j] += ep_alpha * delta

            state = s2
            total_reward += reward
            steps += 1

        episode_returns.append(total_reward)
        episode_lengths.append(steps)
        eps_track.append(current_eps)
        alpha_track.append(ep_alpha)
        current_eps = max(epsilon_min, current_eps * epsilon_decay)

    return {
        "Q": Q,
        "returns": np.array(episode_returns),
        "lengths": np.array(episode_lengths),
        "epsilon": np.array(eps_track),
        "alpha": np.array(alpha_track),
    }


def greedy_policy(Q):
    return np.argmax(Q, axis=1)


def render_policy(policy, title=""):
    print(title)
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            s = (r, c)
            if s == START:
                row.append(" S ")
            elif s == GOAL:
                row.append(" G ")
            elif s in CLIFF:
                row.append(" C ")
            else:
                row.append(f" {ARROWS[policy[encode(s)]]} ")
        print("   " + "".join(row))


# ======================================================================================
# PART 1 - the learning run
# ======================================================================================
print()
print("-" * 78)
print("PART 1 - TRAINING")
print("-" * 78)

result = q_learning(n_episodes=1500, alpha=0.2, epsilon=0.1)
Q = result["Q"]
returns = result["returns"]
lengths = result["lengths"]

window = 50
smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")

print(f"episodes: {len(returns)}   alpha: 0.2 (constant)   "
      f"epsilon: 0.1 -> 0.01 (x0.999 per episode)   gamma: {GAMMA}")
print()
print(f"{'episode block':<18} {'mean return':>13} {'mean length':>13} "
      f"{'mean last 50':>14}")
print("-" * 60)
for start in range(0, 1500, 250):
    block = returns[start:start + 250]
    last = smoothed[-1] if start + 250 >= 1500 else float("nan")
    print(f"{f'{start}-{start + 250}':<18} {block.mean():>13.2f} "
          f"{lengths[start:start + 250].mean():>13.2f} {last:>14.2f}")
print()
print(f"first 100 episodes mean return: {returns[:100].mean():.2f}")
print(f"last  100 episodes mean return: {returns[-100:].mean():.2f}")
print()
print("The curve is noisy episode to episode - that is exploration cost, not")
print("instability. With epsilon=0.1 roughly one step in ten is random, and one")
print("random step into the cliff costs -100, so occasional disasters continue")
print("even after the policy is perfect. Averaging over a window is the only")
print("honest way to read the curve.")


# ======================================================================================
# PART 2 - the learned Q-table and the policy it implies
# ======================================================================================
print()
print("-" * 78)
print("PART 2 - THE LEARNED Q-TABLE")
print("-" * 78)

policy = greedy_policy(Q)
render_policy(policy, "Greedy policy (S start, G goal, C cliff):")
print()

print("Q-values at the start state (r0, c0):")
i0 = encode(START)
for a in ACTIONS:
    marker = "  <-- greedy" if a == policy[i0] else ""
    print(f"  {ACTION_NAMES[a]:<7} Q = {Q[i0, a]:>9.3f}{marker}")
print()
print("Q-values one step to the right of the cliff's left end (3,0):")
for a in ACTIONS:
    state = (3, 0)
    i = encode(state)
    marker = "  <-- greedy" if a == policy[i] else ""
    print(f"  {ACTION_NAMES[a]:<7} Q = {Q[i, a]:>9.3f}{marker}")
print()
print("From (3,0) the cliff sits directly to the right. The agent still prefers")
print("'up' because the safe route along the top row has a much better return")
print("than a -100 detour - even though 'right' has not been tried much. That")
print("preference exists because Q-learning's target propagates values BACKWARD")
print("from the goal through the states that were actually visited.")


# ======================================================================================
# PART 3 - sweep through every state's best action
# ======================================================================================
print()
print("-" * 78)
print("PART 3 - ALL Q-VALUES, SORTED")
print("-" * 78)

flat = [(Q[i, a], i, a) for i in range(N_STATES) for a in ACTIONS]
flat.sort(reverse=True)

print("Highest 8 Q-values (what the agent most wants to do):")
print(f"{'state':<10} {'action':<8} {'Q(s,a)':>9}  note")
print("-" * 52)
for value, i, a in flat[:8]:
    state = decode(i)
    note = "terminal-adjacent" if state in {(3, 4), (2, 5)} else ""
    print(f"{str(state):<10} {ACTION_NAMES[a]:<8} {value:>9.3f}  {note}")
print()
print("Lowest 4 Q-values:")
for value, i, a in flat[-4:]:
    print(f"{str(decode(i)):<10} {ACTION_NAMES[a]:<8} {value:>9.3f}")
print()
print("The spread of Q is roughly [-100, +15]: the cliff dominates the bottom")
print("and the goal the top. An untrained table is all zeros, so the RANGE of")
print("the values is a direct measure of how much has been learned.")


# ======================================================================================
# PART 4 - hyperparameters
# ======================================================================================
print()
print("-" * 78)
print("PART 4 - WHAT THE HYPERPARAMETERS DO")
print("-" * 78)

configs = [
    ("baseline (alpha=0.2, eps=0.1)", dict(alpha=0.2, epsilon=0.1)),
    ("alpha=0.02 (too slow)", dict(alpha=0.02, epsilon=0.1)),
    ("alpha=0.9 (too jumpy)", dict(alpha=0.9, epsilon=0.1)),
    ("epsilon=0.0 (greedy, no eps)", dict(alpha=0.2, epsilon=0.0)),
    ("epsilon=0.5 (mostly random)", dict(alpha=0.2, epsilon=0.5)),
    ("optimistic init=5, eps=0", dict(alpha=0.2, epsilon=0.0, optimistic_init=5.0)),
    ("decaying alpha 0.5/sqrt(k+1)", dict(alpha=lambda k: 0.5 / np.sqrt(k + 1),
                                          epsilon=0.1)),
]

print(f"{'config':<32} {'mean ret first 100':>20} {'mean ret last 100':>20}")
print("-" * 74)
summary = {}
for name, kwargs in configs:
    run = q_learning(n_episodes=1500, seed=SEED, **kwargs)
    first = run["returns"][:100].mean()
    last = run["returns"][-100:].mean()
    summary[name] = (first, last, run)
    print(f"{name:<32} {first:>20.2f} {last:>20.2f}")
print()

print("Reading the table:")
print("  (single-seed numbers are noisy - read them as illustrations, not a")
print("   ranking; script 06 averages several seeds for exactly this reason)")
print("  - alpha too low means values propagate slowly from goal to start, so")
print("    early episodes are all noise; alpha too high means the newest sample")
print("    overwrites everything learned so far and the values never settle.")
print("  - epsilon=0 removes DELIBERATE exploration, yet still learns here. Do")
print("    not read that as 'exploration is unnecessary': ties in the initially")
print("    zero table are broken randomly, and every -1 step makes a rejected")
print("    move look worse, so the agent is nudged onto other actions by")
print("    accident. On a bandit (script 07) or a cliff where the first greedy")
print("    choice is catastrophic, the same agent locks in and never recovers.")
print("  - epsilon=0.5 explores so much it barely exploits - the best-known")
print("    route is abandoned half the time, so the return stays low.")
print("  - optimistic initialisation explores WITHOUT epsilon: untried actions")
print("    look good precisely because they are untried, so the agent tries them,")
print("    gets a worse-than-expected reward, and the value decays toward truth.")
print("  - decaying alpha gives the final numbers a chance to settle.")
print()

# Show how much of the space epsilon=0 actually covered.
zero_eps_run = summary["epsilon=0.0 (greedy, no eps)"][2]
zero_eps_q = zero_eps_run["Q"]
visited = int(np.sum(np.any(zero_eps_q != 0.0, axis=1)))
print(f"epsilon=0 coverage: {visited}/{N_STATES} states have a non-zero Q row "
      f"(={visited / N_STATES:.0%})")
print("  The grid is small and mostly reachable, so accidental exploration")
print("  covers it. On a larger or more deceptive environment the same policy")
print("  visits only a corner of the state space and the rest stays at 0 forever.")
print("  That is why epsilon, UCB or optimistic init exist - to guarantee the")
print("  coverage accidental exploration cannot promise.")


# ======================================================================================
# PART 5 - what a learned Q-table costs
# ======================================================================================
print()
print("-" * 78)
print("PART 5 - WHY TABLES STOP WORKING")
print("-" * 78)

print(f"{'grid size':<16} {'states':>10} {'Q entries (x4)':>16} {'memory (float64)':>18}")
print("-" * 64)
for size in [4, 6, 12, 50, 100, 200, 84 * 84]:
    states = size * size
    entries = states * 4
    print(f"{f'{size} x {size}':<16} {states:>10,} {entries:>16,} "
          f"{entries * 8 / 1024:>14.1f} KB")
print()
print("At 200x200 the table is still only ~1.3 MB - memory is not the problem.")
print("The problem is COVERAGE: each of those 160,000 states needs enough")
print("visits for its value to be accurate, and the agent simply cannot visit")
print("them all in a reasonable number of episodes. A neural network replacing")
print("the table shares statistical strength across states, which is exactly")
print("what DQN does and what script 10 sets up.")


# ======================================================================================
# Report
# ======================================================================================
print()
print("=" * 78)
print("REPORT")
print("=" * 78)
print(f"""
ALGORITHM   Q(s,a) <- Q(s,a) + alpha [ r + gamma max_a' Q(s',a') - Q(s,a) ]
            off-policy, bootstrapped, model-free, tabular

TRAINING
  episodes            1500
  alpha / epsilon     0.2 constant / 0.1 -> 0.01
  mean return first 100   {returns[:100].mean():.2f}
  mean return last  100   {returns[-100:].mean():.2f}

LEARNED POLICY
  shortest path length 6 steps
  """ + "\n  ".join(f"Q({decode(i)}) = " +
                    ", ".join(f"{ACTION_NAMES[a]}:{Q[i, a]:.2f}" for a in ACTIONS)
                    for i in [encode(START), encode((0, 1))]) + f"""

HYPERPARAMETER SWEEP (mean return, last 100 episodes)
""" + "\n  ".join(f"  {name:<34} {vals[1]:>8.2f}" for name, vals in summary.items()) + f"""

WHAT TO CARRY FORWARD
  - the target is r + gamma * max Q(s', .) - bootstrapped, never the true value
  - terminal targets use r alone; bootstrapping past a terminal state is a bug
  - off-policy means the behaviour policy (epsilon-greedy) and the learned
    policy (greedy) are different, and that is a FEATURE, not a defect
  - deliberate exploration is load-bearing: epsilon=0 only learns by accident
    through ties and step penalties, and that luck is environment-specific
  - constant alpha keeps adapting; decaying alpha converges to a fixed Q
  - SARSA (script 06) replaces max with the action actually taken
""")
print("=" * 78)
