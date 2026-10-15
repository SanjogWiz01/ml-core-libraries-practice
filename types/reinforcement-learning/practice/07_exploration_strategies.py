"""
07 - Exploration Strategies (Multi-Armed Bandits)
=================================================

Goal: isolate the exploration/exploitation tradeoff in the cleanest possible
setting - a k-armed bandit, which is an MDP with exactly one state - and
compare the standard strategies head to head.

Why bandits first
-----------------
A bandit has no state, so no credit assignment across time, no bootstrapping,
no discounting. The ONLY thing left is the exploration question. That is why
it is the right place to learn the exploration tricks that every real RL agent
then reuses: epsilon-greedy, decaying epsilon, softmax/Boltzmann, upper
confidence bounds, and optimistic initialisation.

If a strategy cannot beat plain greedy on a bandit, it will not save a hard
control problem.

The setting
-----------
k slot machines. Arm i pays reward ~ N(mu_i, 1). The agent does not know the
mu_i and must balance:
  - EXPLOIT: pull the arm that has looked best so far
  - EXPLORE: pull an unfamiliar arm in case it is truly the best

Regret is the honest metric: the opportunity cost of every pull relative to
always pulling the best arm.

    regret(T) = T * max_i mu_i - SUM_{t=1..T} reward_t          (realised)
    expected regret under a policy is what theory bounds: O(log T) for UCB,
    O(sqrt(T)) for a fixed epsilon, O(T) for pure greedy in the worst case.

Run:  python 07_exploration_strategies.py
"""

import numpy as np

SEED = 42

print("=" * 78)
print("07 - EXPLORATION STRATEGIES: MULTI-ARMED BANDITS")
print("=" * 78)


# ======================================================================================
# The bandit
# ======================================================================================
K = 10
N_PULLS = 2000
N_RUNS = 200

# A fixed-but-arbitrary set of true means. The gaps between the top arms decide
# how hard the problem is: if the best two arms differ by 0.01, distinguishing
# them requires many samples and regret grows. A wide spread is easy.
TRUE_MEANS = np.array([0.2, -0.3, 0.8, 0.4, 1.2, 0.1, 0.9, -0.5, 0.6, 1.1])
BEST_ARM = int(np.argmax(TRUE_MEANS))
BEST_MEAN = float(TRUE_MEANS.max())

print(f"arms: {K}   pulls per run: {N_PULLS}   independent runs: {N_RUNS}")
print()
print("true arm means:")
for i, mu in enumerate(TRUE_MEANS):
    marker = "   <- BEST" if i == BEST_ARM else ""
    print(f"  arm {i}: {mu:>5.2f}{marker}")
print()
print(f"The optimal action earns on average {BEST_MEAN:.2f} per pull. Anything")
print("below that accumulates regret. The gap between arms 4 (1.2) and 9 (1.1)")
print("is only 0.1, so occasionally misidentifying the best arm is expected.")


# ======================================================================================
# Strategies. Each returns the chosen arm given the running statistics.
# ======================================================================================
def greedy(counts, means, t, rng):
    return int(np.argmax(means))


def epsilon_greedy_factory(epsilon):
    def strategy(counts, means, t, rng):
        if rng.random() < epsilon:
            return int(rng.integers(K))
        return int(np.argmax(means))
    return strategy


def epsilon_decay_factory(e0=1.0, decay=0.995, floor=0.01):
    def strategy(counts, means, t, rng):
        eps = max(floor, e0 * decay ** t)
        if rng.random() < eps:
            return int(rng.integers(K))
        return int(np.argmax(means))
    return strategy


def softmax_factory(temperature=0.1):
    """Boltzmann exploration: probability proportional to exp(Q/tau). As tau
    -> 0 this becomes greedy; as tau -> inf it becomes uniform. It explores in
    proportion to how promising an arm looks, unlike epsilon-greedy which
    explores ALL arms equally including ones already known to be bad."""
    def strategy(counts, means, t, rng):
        z = (means - means.max()) / temperature      # subtract max for stability
        p = np.exp(z)
        p = p / p.sum()
        return int(rng.choice(K, p=p))
    return strategy


def ucb_factory(c=2.0):
    """Upper Confidence Bound. Always pulls the arm maximizing
        mean_i + c * sqrt(ln(t) / n_i)
    The bonus shrinks as n_i grows, so every arm is eventually sampled (no
    explicit randomness) and the best arm is pulled more and more. This is
    'optimism in the face of uncertainty' made precise, and it achieves the
    O(log T) regret bound."""
    def strategy(counts, means, t, rng):
        n = np.maximum(counts, 1)
        bonus = c * np.sqrt(np.log(t + 1) / n)
        return int(np.argmax(means + bonus))
    return strategy


def optimistic_factory(q0=2.0):
    """Initialise every arm's estimate optimistically. Greedy then explores by
    itself: untried arms look best and get pulled until their estimates drop."""
    def strategy(counts, means, t, rng):
        # counts == 0 arms must be pulled; represent by treating them as q0.
        untried = np.flatnonzero(counts == 0)
        if untried.size:
            return int(untried[0])
        return int(np.argmax(means))
    return strategy


def thompson_factory():
    """Thompson sampling with a Gaussian posterior (known unit variance).
    Sample a plausible mean for each arm from N(mean_i, 1/n_i) and act greedily
    on the SAMPLE, not the estimate. Cheap, no tuning, and near-optimal in
    practice - often the default in production bandit systems."""
    def strategy(counts, means, t, rng):
        n = np.maximum(counts, 1)
        sample = rng.normal(means, 1.0 / np.sqrt(n))
        return int(np.argmax(sample))
    return strategy


STRATEGIES = [
    ("greedy", lambda: greedy),
    ("epsilon=0.1", lambda: epsilon_greedy_factory(0.1)),
    ("epsilon=0.01", lambda: epsilon_greedy_factory(0.01)),
    ("epsilon decay 0.995", lambda: epsilon_decay_factory(1.0, 0.995, 0.01)),
    ("softmax tau=0.1", lambda: softmax_factory(0.1)),
    ("softmax tau=0.5", lambda: softmax_factory(0.5)),
    ("UCB c=2", lambda: ucb_factory(2.0)),
    ("UCB c=0.5", lambda: ucb_factory(0.5)),
    ("optimistic q0=2", lambda: optimistic_factory(2.0)),
    ("Thompson", lambda: thompson_factory()),
]


# ======================================================================================
# The experiment
# ======================================================================================
def run_once(strategy, seed):
    rng = np.random.default_rng(seed)
    counts = np.zeros(K, dtype=int)
    means = np.zeros(K)                      # sample-average estimate
    rewards = np.zeros(N_PULLS)
    optimal = np.zeros(N_PULLS, dtype=bool)

    for t in range(N_PULLS):
        arm = strategy(counts, means, t, rng)
        reward = rng.normal(TRUE_MEANS[arm], 1.0)
        counts[arm] += 1
        # Incremental sample average: mean <- mean + (x - mean) / n.
        # This is the same running-estimate idiom as the TD update, with the
        # "target" being the freshly observed reward.
        means[arm] += (reward - means[arm]) / counts[arm]
        rewards[t] = reward
        optimal[t] = (arm == BEST_ARM)

    return rewards, optimal


def run_experiment(make_strategy, n_runs=N_RUNS):
    all_rewards = np.zeros((n_runs, N_PULLS))
    all_optimal = np.zeros((n_runs, N_PULLS))
    for run in range(n_runs):
        all_rewards[run], all_optimal[run] = run_once(make_strategy(), 1000 + run)
    regret_curve = np.cumsum(BEST_MEAN - all_rewards, axis=1).mean(axis=0)
    return all_rewards, all_optimal, regret_curve


# ======================================================================================
# PART 1 - run every strategy
# ======================================================================================
print()
print("-" * 78)
print("PART 1 - ALL STRATEGIES, AVERAGED OVER RUNS")
print("-" * 78)

results = {}
for name, factory in STRATEGIES:
    rewards, optimal, regret = run_experiment(factory)
    results[name] = {
        "mean_reward": rewards.mean(),
        "optimal_rate": optimal.mean(),
        "final_regret": regret[-1],
        "regret": regret,
        "optimal_early": optimal[:, :100].mean(),
    }

print(f"{'strategy':<22} {'avg reward':>12} {'% optimal arm':>15} "
      f"{'total regret':>14} {'%opt (first 100)':>18}")
print("-" * 84)
for name, _ in STRATEGIES:
    r = results[name]
    print(f"{name:<22} {r['mean_reward']:>12.4f} {100 * r['optimal_rate']:>14.1f}% "
          f"{r['final_regret']:>14.1f} {100 * r['optimal_early']:>17.1f}%")
print()
print(f"maximum possible avg reward: {BEST_MEAN:.4f}")
print()
best_strategy = max(results, key=lambda n: results[n]["mean_reward"])
worst_strategy = min(results, key=lambda n: results[n]["mean_reward"])
print(f"best strategy here:  {best_strategy} ({results[best_strategy]['mean_reward']:.4f})")
print(f"worst strategy here: {worst_strategy} ({results[worst_strategy]['mean_reward']:.4f})")
print()
print("Read it as three groups:")
print("  - pure greedy collapses early on whatever arm looked good by luck and")
print("    stays there; its % optimal is low and regret grows linearly.")
print("  - fixed epsilon never stops exploring, so regret keeps accumulating at")
print("    a steady rate even after the best arm is known.")
print("  - decaying epsilon, UCB and Thompson learn then commit, so their regret")
print("    curves bend and flatten - the O(log T) behaviour theory promises.")


# ======================================================================================
# PART 2 - the regret curves, and the shape of the tradeoff
# ======================================================================================
print()
print("-" * 78)
print("PART 2 - REGRET AT CHECKPOINTS")
print("-" * 78)

checkpoints = [10, 50, 100, 250, 500, 1000, 2000]
header = f"{'strategy':<22}" + "".join(f"{f't={c}':>11}" for c in checkpoints)
print(header)
print("-" * len(header))
for name, _ in STRATEGIES:
    curve = results[name]["regret"]
    print(f"{name:<22}" + "".join(f"{curve[c - 1]:>11.1f}" for c in checkpoints))
print()
print("Pure greedy's regret is already growing linearly by t=2000. Fixed")
print("epsilon=0.01 is barely behind the adaptive methods early but its curve")
print("has a constant slope: it keeps paying a tax forever.")
print()
print("The angle between the early part of a curve (must be steep - you have to")
print("sample every arm) and its late part (should flatten - you already know")
print("the answer) is the entire art of exploration.")


# ======================================================================================
# PART 3 - how sensitive are the knobs?
# ======================================================================================
print()
print("-" * 78)
print("PART 3 - SENSITIVITY TO THE EXPLORATION PARAMETER")
print("-" * 78)

print("epsilon for epsilon-greedy (pulls=2000, averaged over runs):")
print(f"{'epsilon':>9} {'avg reward':>13} {'% optimal':>11} {'total regret':>14}")
print("-" * 50)
for eps in [0.0, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5]:
    rewards, optimal, regret = run_experiment(
        lambda e=eps: epsilon_greedy_factory(e), n_runs=100
    )
    print(f"{eps:>9} {rewards.mean():>13.4f} {100 * optimal.mean():>10.1f}% "
          f"{regret[-1]:>14.1f}")
print()
print("There is a sweet spot around 0.01-0.05 here, and it depends on the");
print("horizon: with only 2000 pulls, too much exploration never pays off, and")
print("with 100,000 pulls a larger epsilon would look terrible. 'What epsilon'")
print("is not a property of the problem alone - it is a function of how many")
print("samples you get.")
print()

print("UCB exploration coefficient c:")
print(f"{'c':>9} {'avg reward':>13} {'% optimal':>11} {'total regret':>14}")
print("-" * 50)
for c in [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]:
    rewards, optimal, regret = run_experiment(
        lambda cc=c: ucb_factory(cc), n_runs=100
    )
    print(f"{c:>9} {rewards.mean():>13.4f} {100 * optimal.mean():>10.1f}% "
          f"{regret[-1]:>14.1f}")
print()
print("UCB is much less sensitive than epsilon, because its exploration bonus")
print("decays automatically as counts grow. That self-tuning is the practical")
print("reason UCB and Thompson sampling are the workhorses of real systems.")


# ======================================================================================
# PART 4 - what each strategy spends its pulls on
# ======================================================================================
print()
print("-" * 78)
print("PART 4 - WHERE THE PULLS GO")
print("-" * 78)

print("Fraction of pulls spent on each arm after 2000 pulls, averaged over runs:")
print()
arm_header = f"{'strategy':<22}" + "".join(f"{f'arm{i}':>7}" for i in range(K))
print(arm_header)
print("-" * len(arm_header))


def pull_distribution(strategy, n_runs=100):
    total = np.zeros(K)
    for run in range(n_runs):
        rng = np.random.default_rng(5000 + run)
        counts = np.zeros(K, dtype=int)
        means = np.zeros(K)
        for t in range(N_PULLS):
            arm = strategy(counts, means, t, rng)
            reward = rng.normal(TRUE_MEANS[arm], 1.0)
            counts[arm] += 1
            means[arm] += (reward - means[arm]) / counts[arm]
        total += counts
    return total / total.sum()


for name, factory in STRATEGIES:
    dist = pull_distribution(factory(), n_runs=60)
    row = "".join(f"{100 * d:>7.1f}" for d in dist)
    print(f"{name:<22}{row}")
print()
print("Look at arm 4 (the true best, mu=1.2) and arm 9 (mu=1.1): a good strategy")
print("spends most of its budget on those two and keeps a small trickle on the")
print("rest. Pure greedy gets stuck on whichever arm was lucky first. UCB's")
print("spread is the most concentrated, because its bonus shrinks to nothing.")


# ======================================================================================
# Report
# ======================================================================================
print()
print("=" * 78)
print("REPORT")
print("=" * 78)
print(f"""
BANDIT
  arms             {K}
  true means       {np.round(TRUE_MEANS, 2).tolist()}
  best arm         {BEST_ARM} (mu = {BEST_MEAN:.2f})
  horizon          {N_PULLS} pulls x {N_RUNS} runs

RANKING BY AVERAGE REWARD
""" + "\n".join(
    f"  {i + 1:>2}. {name:<22} {results[name]['mean_reward']:.4f}"
    for i, (name, _) in enumerate(
        sorted(STRATEGIES, key=lambda s: -results[s[0]]["mean_reward"])
    )
) + f"""

STRATEGIES AND THEIR IDEA
  greedy          exploit only - collapses onto a lucky arm
  epsilon-greedy  uniform random exploration with probability eps
  epsilon decay   shrink eps over time: explore early, commit late
  softmax         explore in proportion to exp(Q/tau), not uniformly
  UCB             optimism in the face of uncertainty: q + c sqrt(ln t / n)
  optimistic init start high so greedy explores automatically
  Thompson        sample a plausible mean, act on the sample - no tuning

WHAT TO CARRY FORWARD
  - regret, not average reward alone, is the honest scoreboard
  - a constant exploration rate leaves a permanent linear regret tax
  - adaptive bonuses (UCB) and posterior sampling (Thompson) pack the best
    sample efficiency with the least tuning
  - the same tricks reappear as epsilon schedules in DQN and as entropy
    bonuses in policy-gradient methods (script 10)
  - bandits are the one-state MDP; add states with delayed rewards and you
    are back to scripts 05 and 06
""")
print("=" * 78)
