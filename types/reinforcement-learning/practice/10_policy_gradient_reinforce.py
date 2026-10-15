"""
10 - Policy Gradient: REINFORCE From Scratch
============================================

Goal: implement the policy-gradient method - the second great family of RL,
alongside the value-based methods of scripts 04-06 - with nothing but NumPy,
and train it on the same kind of grid the earlier scripts used.

The change of mindset
---------------------
Value-based methods (Q-learning, SARSA) learn Q, then derive a policy by
argmax. Policy-gradient methods parameterise the policy DIRECTLY,

    pi_theta(a | s) = softmax_theta(s, .)

and climb the gradient of expected return with respect to theta. There is no
Q-table as an intermediate, no max, no epsilon-greedy bolted on top - the
exploration is built into the softmax itself, and the output can be a
distribution over actions (which is what you need for continuous control or
for stochastic optimal policies).

The policy gradient theorem
---------------------------
For a policy parametrised by theta,

    grad_theta J(theta) = E_theta [ grad_theta log pi_theta(a|t) * G_t ]

In English: run episodes with the current policy, and for every action taken,
push its log-probability UP in proportion to the return that followed. Actions
in good episodes become more likely; actions in bad episodes become less
likely. Subtracting a baseline b(s) does not change the expected gradient but
dramatically cuts its variance:

    grad_theta J(theta) = E_theta [ grad_theta log pi_theta(a|t) * (G_t - b(s_t)) ]

The step from here to PPO and to RLHF for language models is, roughly:
"replace the sampled G_t with an advantage estimate and clip the update." The
core estimator below is the ancestor of all of it.

Run:  python 10_policy_gradient_reinforce.py
"""

import numpy as np

SEED = 42

print("=" * 78)
print("10 - POLICY GRADIENT: REINFORCE FROM SCRATCH")
print("=" * 78)


# ======================================================================================
# Environment: a 6x6 grid with a pit field. Same step/pit/goal structure as 08.
# ======================================================================================
ROWS, COLS = 6, 6
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTIONS = [UP, RIGHT, DOWN, LEFT]
ACTION_NAMES = ["up", "right", "down", "left"]
ARROWS = ["^", ">", "v", "<"]
DELTAS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}

START = (0, 0)
GOAL = (5, 5)
PIT = {(2, 1), (2, 2), (2, 3), (3, 3), (1, 4)}
STEP_COST = -0.02
GOAL_REWARD = 1.0
PIT_PENALTY = -1.0
GAMMA = 0.99

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
        return s2, GOAL_REWARD, True
    return s2, STEP_COST, False


# ======================================================================================
# The parametrised policy: a softmax over a theta table, one row per state
# ======================================================================================
def softmax(logits):
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def policy_probs(theta, state):
    return softmax(theta[encode(state)])


def sample_action(theta, state, rng):
    probs = policy_probs(theta, state)
    return int(rng.choice(N_ACTIONS, p=probs)), probs


def log_policy_grad(theta, state, action):
    """
    grad_theta log pi_theta(a|s) for a softmax.

    d/dtheta_j log pi(a) = 1[a == j] - pi(j)

    So the gradient is (one-hot(a) - pi) placed at the row of `s`. This is the
    single expression the whole algorithm multiplies by the return.
    """
    probs = policy_probs(theta, state)
    onehot = np.zeros(N_ACTIONS)
    onehot[action] = 1.0
    grad = np.zeros_like(theta)
    grad[encode(state)] = onehot - probs
    return grad


# ======================================================================================
# REINFORCE with a baseline and an entropy bonus
# ======================================================================================
def reinforce(n_episodes=6000, alpha=0.02, gamma=GAMMA, use_baseline=True,
              entropy_coef=0.01, seed=SEED, max_steps=200, theta0=None):
    """
    Parameters
    ----------
    alpha : float
        Gradient-ascent learning rate.
    use_baseline : bool
        Subtract a running mean of returns. On by default because without it
        the gradient is dominated by the constant offset of the returns and
        learning sits still for thousands of episodes.
    entropy_coef : float
        Bonus proportional to the entropy of the action distribution
        H(pi) = -SUM_a pi(a) log pi(a). It keeps the policy from collapsing to
        a single deterministic action before it has explored enough - the
        policy-gradient equivalent of epsilon, but smooth and self-annealing.
    """
    rng = np.random.default_rng(seed)
    theta = np.zeros((N_STATES, N_ACTIONS)) if theta0 is None else theta0.copy()
    returns, lengths = [], []
    baseline = 0.0
    baseline_alpha = 0.01

    for episode in range(n_episodes):
        # --- roll out an episode, recording the (state, action, reward) triples --
        state = START
        states, actions, rewards = [], [], []
        done = False
        steps = 0
        total = 0.0
        while not done and steps < max_steps:
            action, _ = sample_action(theta, state, rng)
            s2, r, done = env_step(state, action)
            states.append(state)
            actions.append(action)
            rewards.append(r)
            total += r
            state = s2
            steps += 1
        returns.append(total)
        lengths.append(steps)

        if use_baseline:
            baseline += baseline_alpha * (total - baseline)

        # --- compute the discounted returns-to-go G_t for each time step ------
        T = len(states)
        g = 0.0
        returns_to_go = np.zeros(T)
        for t in reversed(range(T)):
            g = rewards[t] + gamma * g
            returns_to_go[t] = g

        # --- accumulate the gradient ------------------------------------------
        grad = np.zeros_like(theta)
        for t in range(T):
            state, action = states[t], actions[t]
            advantage = returns_to_go[t] - baseline
            grad += advantage * log_policy_grad(theta, state, action)

            # entropy bonus: grad_theta H(pi) for a softmax, vectorised.
            # H = -SUM pi log pi, and its softmax gradient reduces to the
            # same (onehot - pi) form, which is why it is cheap to add.
            probs = policy_probs(theta, state)
            log_probs = np.log(probs + 1e-12)
            mean_logp = float(np.sum(probs * log_probs))
            ent_grad = -probs * (log_probs + 1.0 + mean_logp)
            grad[encode(state)] += entropy_coef * ent_grad

        # --- one gradient-ascent step -----------------------------------------
        # Normalise by the episode length so the step size does not grow with
        # episode duration (a simple, effective variance control).
        theta += alpha * grad / max(T, 1)

    return theta, np.array(returns), np.array(lengths), baseline


def greedy_policy(theta):
    return np.argmax(theta, axis=1)


def render(policy, title=""):
    print(title)
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            s = (r, c)
            if s == START:
                row.append(" S ")
            elif s == GOAL:
                row.append(" G ")
            elif s in PIT:
                row.append(" P ")
            else:
                row.append(f" {ARROWS[policy[encode(s)]]} ")
        print("   " + "".join(row))


def evaluate_greedy(theta, n_episodes=300, seed=999):
    """Rollout the argmax policy (no sampling) to get a clean score."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_episodes):
        state = START
        total, steps, done = 0.0, 0, False
        while not done and steps < 200:
            a = int(np.argmax(theta[encode(state)]))
            s2, r, done = env_step(state, a)
            total += r
            state = s2
            steps += 1
        scores.append(total)
    return np.array(scores)


# ======================================================================================
# PART 1 - train
# ======================================================================================
print()
print("-" * 78)
print("PART 1 - REINFORCE WITH BASELINE AND ENTROPY BONUS")
print("-" * 78)

theta, returns, lengths, baseline = reinforce(n_episodes=6000)
policy = greedy_policy(theta)

render(policy, "Learned greedy policy (S start, G goal, P pit):")
print()
print(f"episodes: {len(returns)}")
print(f"mean return, first 500: {returns[:500].mean():>9.4f}")
print(f"mean return, last  500: {returns[-500:].mean():>9.4f}")
print(f"mean length, last  500: {lengths[-500:].mean():>9.2f}")
print()
print("Note the reward scale: +1 for the goal and -0.02 per step, so the maximum")
print("undiscounted return is roughly 1 - 0.02 * path_length. The policy learns")
print("to reach the goal and to do it in few steps, purely from the log-prob")
print("gradient - argmax is only used to READ the policy afterwards.")


# ======================================================================================
# PART 2 - what the policy distribution looks like
# ======================================================================================
print()
print("-" * 78)
print("PART 2 - THE POLICY IS A DISTRIBUTION, NOT A CHOICE")
print("-" * 78)

print("Action probabilities at the start state:")
probs = policy_probs(theta, START)
for a in ACTIONS:
    bar = "#" * int(round(probs[a] * 40))
    print(f"  {ACTION_NAMES[a]:<7} {probs[a]:>6.3f}  {bar}")
print()
print("Action probabilities on a pit-adjacent state, (2, 0):")
probs2 = policy_probs(theta, (2, 0))
for a in ACTIONS:
    note = "  <-- pit to the right" if a == RIGHT else ""
    print(f"  {ACTION_NAMES[a]:<7} {probs2[a]:>6.3f}{note}")
print()
print("The softmax keeps a little probability on every action. That residual is")
print("the policy's own exploration, and it is why REINFORCE needs no separate")
print("epsilon-greedy step: an action never becomes exactly impossible, so it can")
print("always recover. The entropy bonus directly controls how much of that")
print("residual survives to the end of training.")

# Entropy of the current policy at each state, averaged.
entropies = []
for i in range(N_STATES):
    p = softmax(theta[i])
    entropies.append(-float(np.sum(p * np.log(p + 1e-12))))
print()
print(f"mean policy entropy over states: {np.mean(entropies):.4f} nats "
      f"(max possible ln(4) = {np.log(4):.4f})")
print("Low entropy at reachable states = confident; the entropy bonus in the")
print("objective is what keeps it from hitting exactly 0.")


# ======================================================================================
# PART 3 - the baseline is not optional
# ======================================================================================
print()
print("-" * 78)
print("PART 3 - WITH AND WITHOUT A BASELINE")
print("-" * 78)

n_seeds = 5
with_base, without_base = [], []
for seed in range(n_seeds):
    _, r_with, _, _ = reinforce(n_episodes=3000, use_baseline=True, seed=seed)
    _, r_without, _, _ = reinforce(n_episodes=3000, use_baseline=False, seed=seed)
    with_base.append(r_with[-300:].mean())
    without_base.append(r_without[-300:].mean())

with_base = np.array(with_base)
without_base = np.array(without_base)

print(f"averaged over {n_seeds} seeds, mean return over the last 300 episodes:")
print(f"  with baseline     {with_base.mean():>8.4f} +/- {with_base.std():.4f}")
print(f"  without baseline  {without_base.mean():>8.4f} +/- {without_base.std():.4f}")
print()
print("The baseline does not change the EXPECTED gradient (it is subtracted from")
print("an estimate whose expectation is zero), but it removes the large common")
print("offset that every action in an episode shares. Without it the updates are")
print("dominated by 'this episode was good or bad overall' rather than 'this")
print("action was better or worse than usual', and learning is much slower and")
print("noisier across seeds.")
print()
print("This '_ minus b(s)_' trick is the seed of the actor-critic: replace the")
print("sampled return G_t with a learned value V(s) and you have A(s,a) = G - V,")
print("the advantage, and an agent that updates every step instead of every")
print("episode.")


# ======================================================================================
# PART 4 - the policy gradient, checked by hand
# ======================================================================================
print()
print("-" * 78)
print("PART 4 - VERIFYING grad log pi FOR THE SOFTMAX")
print("-" * 78)

demo_theta = np.zeros((N_STATES, N_ACTIONS))
demo_theta[encode(START)] = np.array([0.3, 1.0, -0.2, 0.5])
state = START
action = RIGHT

analytic = log_policy_grad(demo_theta, state, action)[encode(state)]

# Numerical gradient by central differences, so we are not trusting the algebra.
eps = 1e-6
numeric = np.zeros(N_ACTIONS)
for j in range(N_ACTIONS):
    plus = demo_theta.copy()
    minus = demo_theta.copy()
    plus[encode(state), j] += eps
    minus[encode(state), j] -= eps
    lp_plus = np.log(policy_probs(plus, state)[action])
    lp_minus = np.log(policy_probs(minus, state)[action])
    numeric[j] = (lp_plus - lp_minus) / (2 * eps)

print(f"state {state}, action '{ACTION_NAMES[action]}', params {demo_theta[encode(state)]}")
print()
print(f"{'j':>3} {'analytic':>12} {'numeric':>12} {'diff':>12}")
print("-" * 44)
for j in range(N_ACTIONS):
    print(f"{j:>3} {analytic[j]:>12.6f} {numeric[j]:>12.6f} "
          f"{abs(analytic[j] - numeric[j]):>12.2e}")
print()
print("The 'one-hot minus softmax' formula matches central differences to ~1e-9.")
print("Every policy-gradient method you will meet - A2C, PPO, TRPO, GRPO - uses")
print("exactly this gradient; they differ only in how the scalar in front of it")
print("(the advantage) is estimated and how far the step is allowed to go.")


# ======================================================================================
# PART 5 - where this goes next
# ======================================================================================
print()
print("-" * 78)
print("PART 5 - FROM THIS SCRIPT TO DEEP RL")
print("-" * 78)

print("Everything here used a table to store theta and Q. Swapping the table for")
print("a neural network is the single change that turns these toy agents into")
print("the systems that play Atari and fine-tune language models:")
print()
print("  tabular theta[s, a]        ->  a network  f_theta(s) -> logits")
print("  tabular Q[s, a]            ->  a network  Q_theta(s) -> values")
print()
print("The library you would reach for next is stable-baselines3, which wraps")
print("PPO, DQN, SAC and friends over gymnasium environments:")
print()
print("    pip install gymnasium stable-baselines3 torch")
print()
print("  import gymnasium as gym")
print("  from stable_baselines3 import PPO")
print("  env = gym.make('LunarLander-v3')")
print("  model = PPO('MlpPolicy', env).learn(total_timesteps=1_000_000)")
print()
print("The API is two lines. The two lines only make sense because you now know")
print("what PPO's objective, advantage and entropy bonus are, which is what")
print("scripts 05-09 built up.")


# ======================================================================================
# Report
# ======================================================================================
print()
print("=" * 78)
print("REPORT")
print("=" * 78)
greedy_scores = evaluate_greedy(theta)
print(f"""
ALGORITHM
  REINFORCE with baseline and entropy bonus, tabular softmax policy
  grad J = E[ grad log pi_theta(a|s) * (G_t - b(s)) ]
  grad log pi for softmax = one-hot(a) - pi

TRAINING (6000 episodes)
  mean return first 500   {returns[:500].mean():.4f}
  mean return last  500   {returns[-500].mean():.4f}
  greedy eval return      {greedy_scores.mean():.4f} (+/- {greedy_scores.std():.4f})
  final baseline b        {baseline:.4f}

BASELINE ABLATION ({n_seeds} seeds, last 300 episodes)
  with baseline    {with_base.mean():.4f} +/- {with_base.std():.4f}
  without baseline {without_base.mean():.4f} +/- {without_base.std():.4f}

GRADIENT CHECK
  analytic softmax grad vs central differences: max diff
  {max(abs(log_policy_grad(demo_theta, state, action)[encode(state)][j] - numeric[j]) for j in range(N_ACTIONS)):.2e}

POLICY ENTROPY
  mean over states {np.mean(entropies):.4f} nats (max {np.log(4):.4f})

WHAT TO CARRY FORWARD
  - policy gradient optimises the policy directly; no argmax, no Q-table
  - the score-function estimator grad log pi * return is unbiased but noisy;
    a baseline b(s) cuts the variance without changing the expectation
  - softmax policies explore natively; entropy bonuses keep them from
    collapsing too early
  - REINFORCE is for EPISODIC, high-variance, small problems; the fixes are
    actor-critic (learn V for the baseline) and clipping (PPO)
  - the two families are fully compatible: value-based Q-learning gives the
    critic, policy gradients give the actor, and modern methods use both

FURTHER READING IN THIS FOLDER
  ../ (README) - the loop, vocabulary, and the model-based/model-free map
  gymnasium + stable-baselines3 - the production versions of scripts 05-10
""")
print("=" * 78)
