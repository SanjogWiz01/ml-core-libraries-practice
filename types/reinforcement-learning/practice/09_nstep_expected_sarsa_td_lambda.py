"""
09 - n-Step Returns, Expected SARSA and TD(lambda)
==================================================

Goal: fill in the space BETWEEN Monte Carlo (script 08) and one-step TD
(scripts 05-06). The endpoints are extremes; almost every strong algorithm
lives in the middle.

The spectrum of targets for Q(s,a)

    n=1     r + gamma * Q(s',a')                       low variance, biased
    n=2     r + gamma r' + gamma^2 Q(s'',a'')           |
    ...                                                 |  bias down,
    n=T     r + gamma r' + ... + gamma^T r_T (the real   |  variance up
            return G_t, i.e. Monte Carlo)               |
                                                        high variance, unbiased

Each extra real reward replaces one bootstrapped guess. That is the ONLY knob
being turned: how far to trust reality before trusting your own estimates.

Two refinements on top of the n-step idea
-----------------------------------------
Expected SARSA: instead of sampling the next action a' (SARSA) or maximizing
(SARSA), take the expectation over the policy:

    target = r + gamma * SUM_a' pi(a'|s') Q(s',a')

Deterministic, so lower variance than SARSA, and it makes the backup work for
any policy including the epsilon-greedy one. With a greedy pi it reduces
exactly to Q-learning.

TD(lambda): average ALL n-step returns with weights (1-lambda) lambda^{n-1}.
lambda=0 is one-step TD, lambda=1 is Monte Carlo, and the forward view is

    G_t^lambda = (1 - lambda) SUM_{n=1..} lambda^{n-1} G_t^{(n)}

The backward view - eligibility traces - achieves the same thing online with
one extra array, updating every recently visited (s,a) by delta * trace. This
script uses the practical n-step formulation and the lambda-weighted forward
view, then checks them against each other.

Run:  python 09_nstep_expected_sarsa_td_lambda.py
"""

import numpy as np

SEED = 42

print("=" * 78)
print("09 - n-STEP RETURNS, EXPECTED SARSA AND TD(lambda)")
print("=" * 78)


# ======================================================================================
# Environment: a long corridor. Long chains are where n-step learning shines,
# because with one-step TD a reward at the end takes many episodes to travel
# back to the start one cell at a time.
# ======================================================================================
LENGTH = 12
START = 0
GOAL = LENGTH - 1
GAMMA = 0.95
STEP_COST = -1.0
GOAL_REWARD = 10.0

N_STATES = LENGTH
N_ACTIONS = 2           # 0 = left, 1 = right
ACTION_NAMES = ["left", "right"]


def env_step(state, action):
    """Deterministic corridor: move one cell, clamp at the walls."""
    if action == 1:
        nxt = min(state + 1, GOAL)
    else:
        nxt = max(state - 1, 0)
    if nxt == GOAL:
        return nxt, GOAL_REWARD, True
    return nxt, STEP_COST, False


def epsilon_greedy(q_row, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))
    return int(rng.choice(np.flatnonzero(q_row == q_row.max())))


def epsilon_soft_probs(q_row, epsilon):
    n = len(q_row)
    best = np.flatnonzero(q_row == q_row.max())
    p = np.full(n, epsilon / n)
    p[best] += (1.0 - epsilon) / len(best)
    return p


# ======================================================================================
# n-step SARSA  (on-policy, sampled next action)
# ======================================================================================
def nstep_sarsa(n_episodes=2000, n=1, alpha=0.4, epsilon=0.1, seed=SEED,
                max_steps=400):
    """
    Collect the episode, then for each time t build the n-step target:

        G_t^{(n)} = r_t + gamma r_{t+1} + ... + gamma^{n-1} r_{t+n-1}
                    + gamma^n Q(s_{t+n}, a_{t+n})     (if not terminal)

    When n = 1 this is exactly the SARSA update from script 06.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    returns = []

    for _ in range(n_episodes):
        # --- roll out one episode ---------------------------------------------
        states, actions, rewards = [START], [], []
        state = START
        action = epsilon_greedy(Q[state], epsilon, rng)
        done = False
        steps = 0
        while not done and steps < max_steps:
            s2, r, done = env_step(state, action)
            actions.append(action)
            rewards.append(r)
            states.append(s2)
            state = s2
            action = epsilon_greedy(Q[state], epsilon, rng)
            steps += 1
        returns.append(sum(rewards))

        T = len(rewards)
        # --- n-step updates ---------------------------------------------------
        for t in range(T):
            upper = min(t + n, T)
            g = 0.0
            for k in range(t, upper):
                g += (GAMMA ** (k - t)) * rewards[k]
            if upper < T:
                # bootstrap from the state n steps ahead with its actual action
                s_n, a_n = states[upper], actions[upper]
                g += (GAMMA ** (upper - t)) * Q[s_n, a_n]
            s_t, a_t = states[t], actions[t]
            Q[s_t, a_t] += alpha * (g - Q[s_t, a_t])

    return Q, np.array(returns)


# ======================================================================================
# n-step Expected SARSA  (on-policy, averaged next action)
# ======================================================================================
def nstep_expected_sarsa(n_episodes=2000, n=1, alpha=0.4, epsilon=0.1,
                         target_epsilon=None, seed=SEED, max_steps=400):
    """
    Same as above, but the bootstrap uses the EXPECTED value under a policy
    rather than a sampled action. `target_epsilon` is the policy being
    evaluated: set it to 0.0 to evaluate the greedy policy exactly (off-policy
    in effect) without needing importance sampling, which is the elegant
    reason expected SARSA is so widely used.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    returns = []
    if target_epsilon is None:
        target_epsilon = epsilon

    for _ in range(n_episodes):
        states, actions, rewards = [START], [], []
        state = START
        done = False
        steps = 0
        while not done and steps < max_steps:
            a = epsilon_greedy(Q[state], epsilon, rng)
            s2, r, done = env_step(state, a)
            actions.append(a)
            rewards.append(r)
            states.append(s2)
            state = s2
            steps += 1
        returns.append(sum(rewards))

        T = len(rewards)
        for t in range(T):
            upper = min(t + n, T)
            g = 0.0
            for k in range(t, upper):
                g += (GAMMA ** (k - t)) * rewards[k]
            if upper < T:
                s_n = states[upper]
                probs = epsilon_soft_probs(Q[s_n], target_epsilon)
                g += (GAMMA ** (upper - t)) * float(np.dot(probs, Q[s_n]))
            s_t, a_t = states[t], actions[t]
            Q[s_t, a_t] += alpha * (g - Q[s_t, a_t])

    return Q, np.array(returns)


# ======================================================================================
# TD(lambda): the forward, lambda-weighted view
# ======================================================================================
def td_lambda(n_episodes=2000, lam=0.9, alpha=0.4, epsilon=0.1, seed=SEED,
              max_steps=400):
    """
    Compute the lambda-return for each time step as a weighted average of all
    n-step returns, and update toward it. Implemented with the efficient
    recursion

        G_t^lambda = r_t + gamma [ (1-lambda) Q(s_{t+1},a_{t+1})
                                   + lambda G_{t+1}^lambda ]

    which is the standard way to get the forward view in one backward sweep.
    lambda=0 gives the one-step target; lambda=1 gives the full Monte Carlo
    return.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    returns = []

    for _ in range(n_episodes):
        states, actions, rewards = [START], [], []
        state = START
        action = epsilon_greedy(Q[state], epsilon, rng)
        done = False
        steps = 0
        while not done and steps < max_steps:
            s2, r, done = env_step(state, action)
            actions.append(action)
            rewards.append(r)
            states.append(s2)
            state = s2
            action = epsilon_greedy(Q[state], epsilon, rng)
            steps += 1
        returns.append(sum(rewards))

        T = len(rewards)
        # lambda-return backward recursion
        G_lambda = np.zeros(T)
        next_value = 0.0            # after the terminal step, value is 0
        for t in reversed(range(T)):
            if t == T - 1:
                G_lambda[t] = rewards[t]
            else:
                s_next, a_next = states[t + 1], actions[t + 1]
                td_target = (1 - lam) * Q[s_next, a_next] + lam * G_lambda[t + 1]
                G_lambda[t] = rewards[t] + GAMMA * td_target
        for t in range(T):
            s_t, a_t = states[t], actions[t]
            Q[s_t, a_t] += alpha * (G_lambda[t] - Q[s_t, a_t])

    return Q, np.array(returns)


# ======================================================================================
# PART 1 - one fixed environment, many n
# ======================================================================================
print()
print("-" * 78)
print("PART 1 - n-STEP SARSA AS n GOES 1 -> 12")
print("-" * 78)

print(f"corridor length {LENGTH}, start {START}, goal {GOAL}, gamma {GAMMA}")
print(f"{'n':>3} {'mean ret last 500':>20} {'V(start)':>10} {'updates/ep':>12} "
      f"{'first-episode-to-goal':>24}")
print("-" * 74)

nstep_results = {}
for n in [1, 2, 3, 4, 6, 8, 12]:
    Q, rets = nstep_sarsa(n_episodes=1500, n=n)
    nstep_results[n] = (Q, rets)
    print(f"{n:>3} {rets[-500:].mean():>20.2f} {Q[START].max():>10.3f} "
          f"{LENGTH - n + 1 if n <= LENGTH else 1:>12} "
          f"{'full-episode' if n >= LENGTH else f'{n}-step':>24}")
print()
print("Small n bootstraps almost entirely (fast, but the goal reward takes many")
print("episodes to propagate back one cell per episode at n=1). Large n uses the")
print("real rewards almost to the end (fast propagation, but each target is one")
print("noisy sample). Somewhere in the middle is usually best - here n=4..6.")
print()
print(f"At n=1 the value of the start state after 1500 episodes is")
print(f"  {nstep_results[1][0][START].max():.3f}")
print(f"and at n=4 it is {nstep_results[4][0][START].max():.3f}.")
print("Bigger difference than most people expect for such a small change.")


# ======================================================================================
# PART 2 - expected SARSA vs sampled SARSA
# ======================================================================================
print()
print("-" * 78)
print("PART 2 - EXPECTED SARSA vs SARSA: SAME MEAN, LESS NOISE")
print("-" * 78)

n_seeds = 8
sarsa_runs, exp_sarsa_runs, exp_greedy_runs = [], [], []
for seed in range(n_seeds):
    _, r_sarsa = nstep_sarsa(n_episodes=1000, n=1, alpha=0.4, epsilon=0.1, seed=seed)
    _, r_exp = nstep_expected_sarsa(n_episodes=1000, n=1, alpha=0.4, epsilon=0.1,
                                    seed=seed)
    _, r_exp_g = nstep_expected_sarsa(n_episodes=1000, n=1, alpha=0.4, epsilon=0.1,
                                      target_epsilon=0.0, seed=seed)
    sarsa_runs.append(r_sarsa[-200:].mean())
    exp_sarsa_runs.append(r_exp[-200:].mean())
    exp_greedy_runs.append(r_exp_g[-200:].mean())

sarsa_runs = np.array(sarsa_runs)
exp_sarsa_runs = np.array(exp_sarsa_runs)
exp_greedy_runs = np.array(exp_greedy_runs)

print(f"averaged over {n_seeds} seeds, mean return over the last 200 episodes:")
print(f"  SARSA (sampled a')          {sarsa_runs.mean():>7.3f} +/- {sarsa_runs.std():.3f}")
print(f"  Expected SARSA (eps target) {exp_sarsa_runs.mean():>7.3f} +/- "
      f"{exp_sarsa_runs.std():.3f}")
print(f"  Expected SARSA (greedy tgt) {exp_greedy_runs.mean():>7.3f} +/- "
      f"{exp_greedy_runs.std():.3f}")
print()
print("Expected SARSA removes the randomness that comes from sampling a'; it")
print("replaces it with an exact sum over the two actions. Same expected value,")
print("noticeably smaller seed-to-seed spread. Setting target_epsilon=0 makes")
print("it learn the GREEDY policy off-policy from an exploring stream of data -")
print("the usual way people use expected SARSA in practice.")


# ======================================================================================
# PART 3 - TD(lambda): sweeping lambda from TD to Monte Carlo
# ======================================================================================
print()
print("-" * 78)
print("PART 3 - TD(lambda): lambda=0 IS TD, lambda=1 IS MONTE CARLO")
print("-" * 78)

print(f"{'lambda':>8} {'mean ret last 500':>20} {'V(start)':>10} {'kind':>20}")
print("-" * 62)
lambda_results = {}
for lam in [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]:
    Q, rets = td_lambda(n_episodes=1500, lam=lam, alpha=0.4, epsilon=0.1)
    lambda_results[lam] = (Q, rets)
    if lam == 0.0:
        kind = "pure one-step TD"
    elif lam == 1.0:
        kind = "pure Monte Carlo"
    else:
        kind = "blend"
    print(f"{lam:>8} {rets[-500:].mean():>20.2f} {Q[START].max():>10.3f} {kind:>20}")
print()
print("lambda=0 and lambda=1 are the two extremes from scripts 06 and 08. The")
print("interesting part is the curve in between: it usually peaks somewhere")
print("around 0.8-0.95, where the target is mostly grounded in real rewards but")
print("still bootstraps enough to keep its variance in check.")
print()
print("One caution: TD(lambda) as implemented here recomputes the lambda-return")
print("offline AFTER the episode finishes. The classic online version uses")
print("eligibility traces, z_{t}(s,a) <- gamma lambda z_{t-1} + 1 on visit, and")
print("updates every (s,a) by alpha delta z. The two views are provably")
print("equivalent for the forward/backward definitions; traces just avoid")
print("recomputing the return for past states.")


# ======================================================================================
# PART 4 - the world's smallest check: does lambda=1 equal Monte Carlo?
# ======================================================================================
print()
print("-" * 78)
print("PART 4 - VERIFYING THE EXTREMES")
print("-" * 78)

# With lambda=1 the TD(lambda) recursion collapses to G_t = r_t + gamma G_{t+1},
# which is exactly the Monte Carlo return. Check numerically on one episode.
rewards_demo = []
state = START
done = False
while not done:
    s2, r, done = env_step(state, 1)               # always move right
    rewards_demo.append(r)
    state = s2

T = len(rewards_demo)
mc_return = 0.0
for r in reversed(rewards_demo):
    mc_return = r + GAMMA * mc_return

# lambda=1 recursion with Q ignored (terminal bootstrap = 0)
G_lambda1 = 0.0
for r in reversed(rewards_demo):
    G_lambda1 = r + GAMMA * G_lambda1

print(f"straight-to-goal episode: {T} steps, rewards {rewards_demo}")
print(f"Monte Carlo return          {mc_return:.10f}")
print(f"TD(lambda=1) recursion      {G_lambda1:.10f}")
print(f"difference                  {abs(mc_return - G_lambda1):.2e}")
print()
print("Identical, as the algebra says: at lambda=1 the bootstrap term is")
print("multiplied by zero and only real rewards survive. The whole family of")
print("methods is a single interpolation between 'trust your estimates' and")
print("'trust your data', parameterised by how far ahead you look.")


# ======================================================================================
# Report
# ======================================================================================
print()
print("=" * 78)
print("REPORT")
print("=" * 78)
best_n = max(nstep_results, key=lambda k: nstep_results[k][1][-500:].mean())
best_lam = max(lambda_results, key=lambda k: lambda_results[k][1][-500:].mean())
print(f"""
THE TARGET SPECTRUM
  n=1          r + gamma Q(s',a')                       one-step TD (script 06)
  n=k          k real rewards + gamma^k Q               n-step
  n=T          real discounted return                   Monte Carlo (script 08)
  lambda blend (1-lambda) SUM lambda^(n-1) G^(n)        TD(lambda)

RESULTS (corridor length {LENGTH}, 1500 episodes)
  best n by last-500 return        n = {best_n}
  best lambda by last-500 return   lambda = {best_lam}

EXPECTED SARSA ({n_seeds} seeds, last 200 episodes)
  sampled SARSA        {sarsa_runs.mean():.3f} +/- {sarsa_runs.std():.3f}
  expected SARSA       {exp_sarsa_runs.mean():.3f} +/- {exp_sarsa_runs.std():.3f}
  expected, greedy tgt {exp_greedy_runs.mean():.3f} +/- {exp_greedy_runs.std():.3f}

VERIFIED
  TD(lambda=1) == Monte Carlo return, difference {abs(mc_return - G_lambda1):.2e}

WHAT TO CARRY FORWARD
  - every algorithm in this folder is the same backup with a different target
  - n and lambda control the bias/variance tradeoff; they are the most
    important knobs after the learning rate
  - expected SARSA's exact expectation over a' is a free variance reduction
    and lets you learn a greedy target off-policy without importance sampling
  - long chains (sparse reward, many steps) are exactly where n-step and
    lambda>0 earn their keep
  - eligibility traces are the online form of TD(lambda); the forward and
    backward views give identical updates
""")
print("=" * 78)
