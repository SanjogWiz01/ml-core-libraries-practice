"""
02 - A GridWorld Environment From Scratch
==========================================

Goal: write a small, fully working GridWorld environment with nothing but
Python and NumPy - reset, step, terminal states, walls, slippery actions,
cliff penalties - and then inspect it: the state space, the action space, and
a few hundred sampled transitions.

Why this comes before any algorithm
-----------------------------------
Every RL algorithm in this folder is written against the same two-method
contract:

    state = env.reset()
    state, reward, done, info = env.step(action)

That contract is what `gymnasium` standardised. If you can build it yourself
once, the library API stops being a black box and every algorithm below can be
tested against an environment you fully understand.

The grid
--------
    4 rows x 12 columns, the classic "Cliff Walking" layout:

        S . . . . . . . . . . .
        . . . . . . . . . . . .
        . . . . . . . . . . . .
        . . C C C C C C C C C G

    S = start (0,0)      G = goal (3,11)
    C = cliff: stepping in gives -100 and teleports back to S
    . = safe ground: -1 per step (a time cost, so the agent cannot idle)
    Actions: 0 up, 1 right, 2 down, 3 left

Slippery mode: with probability `slip` the action is ignored and a uniformly
random other action is taken. Slippery transitions are what make the problem
*stochastic*, and therefore what separates SARSA from Q-learning (script 06).

Run:  python 02_gridworld_env.py
"""

import numpy as np

SEED = 42

print("=" * 78)
print("02 - A GRIDWORLD ENVIRONMENT FROM SCRATCH")
print("=" * 78)


# ======================================================================================
# The environment
# ======================================================================================
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_NAMES = {UP: "up", RIGHT: "right", DOWN: "down", LEFT: "left"}
# (delta_row, delta_col) - row grows DOWNWARD, which is the usual confusion
# when reading a grid printed as text.
DELTAS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}


class GridWorld:
    """
    Deterministic-or-slippery grid with a cliff, a step cost and a goal bonus.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions.
    cliff : iterable of (r, c)
        Cells that cost `cliff_penalty` and teleport the agent to the start.
    slip : float
        Probability that an intended action is replaced by a uniformly random
        DIFFERENT action. 0.0 means fully deterministic.
    step_cost : float
        Reward applied on every non-terminal transition. Negative, so that
        reaching the goal quickly matters.
    goal_reward : float
        Reward for entering the goal cell; the episode then ends.
    max_steps : int
        Safety cap. A Gym-style `truncated` flag is returned when it fires -
        distinct from `terminated`, which means the goal/cliff was reached.
    """

    metadata = {"render_mode": "ansi"}

    def __init__(self, rows=4, cols=12, cliff=None, slip=0.0,
                 step_cost=-1.0, goal_reward=+10.0, cliff_penalty=-100.0,
                 max_steps=200):
        self.rows = rows
        self.cols = cols
        self.cliff = set(cliff if cliff is not None else
                         [(rows - 1, c) for c in range(1, cols - 1)])
        self.slip = slip
        self.step_cost = step_cost
        self.goal_reward = goal_reward
        self.cliff_penalty = cliff_penalty
        self.max_steps = max_steps

        self.start = (0, 0)
        self.goal = (rows - 1, cols - 1)
        self.n_actions = 4
        self.n_states = rows * cols
        self._rng = np.random.default_rng(SEED)

        # A wall layout is kept even though this grid has none, so that adding
        # walls later does not change the algorithm-facing API.
        self.walls = set()

    # --- helpers ---------------------------------------------------------------------
    def encode(self, state):
        """(r, c) -> flat index. Flat indices are what a Q-table is indexed by."""
        r, c = state
        return r * self.cols + c

    def decode(self, index):
        return divmod(index, self.cols)

    @property
    def observation(self):
        """The agent's full observation is the true state - this is a fully
        observable MDP. Partial observability would return something else."""
        return self.encode(self.state)

    # --- gym-style API ----------------------------------------------------------------
    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.state = self.start
        self.steps = 0
        return self.observation, {}

    def step(self, action):
        """
        Apply `action` (with slip), charge the step cost, and report the
        outcome. Returns (observation, reward, terminated, truncated, info).
        """
        assert self.steps < self.max_steps, "call reset() first"
        intended = int(action)

        # --- slip ---------------------------------------------------------------------
        chosen = intended
        if self.slip > 0 and self._rng.random() < self.slip:
            alternatives = [a for a in range(self.n_actions) if a != intended]
            chosen = int(self._rng.choice(alternatives))

        dr, dc = DELTAS[chosen]
        r, c = self.state
        nr, nc = r + dr, c + dc

        # --- the border pushes back instead of letting the agent walk off the map ---
        if not (0 <= nr < self.rows and 0 <= nc < self.cols) or (nr, nc) in self.walls:
            nr, nc = r, c

        self.state = (nr, nc)
        self.steps += 1

        # --- rewards and termination -------------------------------------------------
        terminated = False
        truncated = False
        info = {"action": ACTION_NAMES[intended], "executed": ACTION_NAMES[chosen]}

        if self.state in self.cliff:
            reward = self.cliff_penalty
            self.state = self.start          # teleport, as in the classic grid
            terminated = True
            info["cliff"] = True
        elif self.state == self.goal:
            reward = self.goal_reward
            terminated = True
            info["goal"] = True
        else:
            reward = self.step_cost
            if self.steps >= self.max_steps:
                truncated = True

        return self.observation, float(reward), terminated, truncated, info

    # --- inspection --------------------------------------------------------------------
    def render(self, policy=None, q=None):
        """ASCII map. With `policy` or a Q-table it also draws the greedy arrows."""
        lines = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                cell = (r, c)
                if cell == self.start:
                    ch = "S"
                elif cell == self.goal:
                    ch = "G"
                elif cell in self.cliff:
                    ch = "C"
                elif cell in self.walls:
                    ch = "#"
                elif policy is not None or q is not None:
                    if q is not None:
                        q_values = q[self.encode(cell)]
                        best = int(np.argmax(q_values))
                    else:
                        best = policy[self.encode(cell)]
                    ch = {UP: "^", RIGHT: ">", DOWN: "v", LEFT: "<"}[best]
                else:
                    ch = "."
                row.append(ch)
            lines.append(" ".join(row))
        return "\n".join(lines)


# ======================================================================================
# PART 1 - the state and action spaces
# ======================================================================================
print()
print("-" * 78)
print("PART 1 - STATE AND ACTION SPACES")
print("-" * 78)

env = GridWorld()
print(f"grid: {env.rows} x {env.cols} = {env.n_states} states, {env.n_actions} actions")
print(f"start {env.start}  goal {env.goal}  cliff cells: {len(env.cliff)}")
print()
print("The empty map:")
print()
print(env.render())
print()
print("Encoding is a simple row-major flattening:")
for cell in [env.start, (0, 1), (1, 0), env.goal]:
    code = env.encode(cell)
    print(f"  {str(cell):<10} -> index {code:>3}  -> back to {env.decode(code)}")
print()
print("Size matters: 52 states x 4 actions = 208 Q-values, trivially a table.")
print("A 100x100 grid is 40,000 values - still fine. Atari at 84x84 grayscale")
print("is ~10^8 states, and that is exactly where a table fails and a neural")
print("network (script 10's problem, scaled up) becomes necessary.")


# ======================================================================================
# PART 2 - deterministic single steps
# ======================================================================================
print()
print("-" * 78)
print("PART 2 - ONE STEP AT A TIME (DETERMINISTIC)")
print("-" * 78)

env = GridWorld(slip=0.0)
env.reset(seed=SEED)

print(f"{'action':<8} {'executed':<9} {'from':<12} {'to':<12} {'reward':>8} {'done':>6}")
print("-" * 62)
demo_actions = [RIGHT, RIGHT, DOWN, LEFT, UP]
for a in demo_actions:
    before = env.state
    _, reward, terminated, truncated, info = env.step(a)
    print(f"{info['action']:<8} {info['executed']:<9} {str(before):<12} "
          f"{str(env.state):<12} {reward:>8.1f} {str(terminated or truncated):>6}")
print()
print("Two things to notice: walking into the border does NOT move the agent")
print("(the state is unchanged but the step cost is still paid), and every")
print("transition costs -1, so a lazy agent bleeds reward.")


# ======================================================================================
# PART 3 - stochastic transitions from slipping
# ======================================================================================
print()
print("-" * 78)
print("PART 3 - SLIPPING: THE SAME ACTION, DIFFERENT OUTCOMES")
print("-" * 78)

slip_env = GridWorld(slip=0.3)
print("slip probability = 0.3, starting at (0,1), intended action = 'right'")
print()
print(f"{'trial':>5} {'executed':<10} {'resulting state':<18} {'reward':>8}")
print("-" * 46)
outcomes = {}
for trial in range(10):
    slip_env.reset(seed=1000 + trial)
    slip_env.state = (0, 1)
    _, reward, terminated, _, info = slip_env.step(RIGHT)
    outcomes[info["executed"]] = outcomes.get(info["executed"], 0) + 1
    print(f"{trial:>5} {info['executed']:<10} {str(slip_env.state):<18} {reward:>8.1f}")
print()
print("Execution counts over those 10 trials:", outcomes)
print()
print("The agent ASKED for right and sometimes got up/down/left. This is why")
print("the transition model is P(s'|s,a) and not a function: the same (s, a)")
print("pair can lead anywhere. It is also the entire reason SARSA and Q-learning")
print("diverge near the cliff - script 06 measures exactly how much.")


# ======================================================================================
# PART 4 - Monte Carlo estimate of the transition function
# ======================================================================================
print()
print("-" * 78)
print("PART 4 - LEARNING P(s'|s,a) FROM SAMPLES, NOT FROM THE TABLE")
print("-" * 78)

print("No algorithm is given the map. Here is what the agent can actually")
print("observe: sample transitions from (0,0) taking 'right', 2000 times.")
print()

mc_env = GridWorld(slip=0.2)
counts = {}
for trial in range(2000):
    mc_env.reset(seed=2000 + trial)
    mc_env.state = (0, 0)
    before = mc_env.state
    _, reward, _, _, info = mc_env.step(RIGHT)
    key = (before, "right", mc_env.state, reward)
    counts[key] = counts.get(key, 0) + 1

print(f"{'from':<12} {'action':<7} {'to':<12} {'reward':>7} {'count':>6} {'freq':>7}")
print("-" * 56)
for (before, action, after, reward), count in sorted(counts.items(),
                                                      key=lambda kv: -kv[1]):
    print(f"{str(before):<12} {action:<7} {str(after):<12} {reward:>7.1f} "
          f"{count:>6} {count / 2000:>7.3f}")
print()
print("With slip=0.2 the empirical frequencies are close to 0.8/0.1/0.1 - the")
print("true model - but never exact. Model-free algorithms do not care: they")
print("never estimate P at all, they fold the samples straight into Q.")


# ======================================================================================
# PART 5 - an episode from the start to the goal, unrolled
# ======================================================================================
print()
print("-" * 78)
print("PART 5 - A FULL EPISODE")
print("-" * 78)

env = GridWorld(slip=0.0, max_steps=60)
env.reset(seed=SEED)

# A hand-written policy: go right until the last column, then go down.
def hand_policy(state):
    r, c = state
    return RIGHT if c < env.cols - 1 else DOWN


trajectory = []
done = False
while not done:
    state = env.observation
    action = hand_policy(env.state)
    _, reward, terminated, truncated, info = env.step(action)
    trajectory.append((state, ACTION_NAMES[action], reward, env.state))
    done = terminated or truncated

print("policy: move right to column 11, then move down to the goal")
print()
print(f"{'t':>2} {'state':<8} {'action':<7} {'reward':>7} {'next':<8}")
print("-" * 42)
total = 0.0
for t, (s, a, r, s2) in enumerate(trajectory):
    total += r
    print(f"{t:>2} {str(env.decode(s)):<8} {a:<7} {r:>7.1f} {str(s2):<8}")
print()
print(f"steps: {len(trajectory)}   undiscounted return: {total:.1f}   "
      f"gamma=0.9 return: ", end="")

gamma = 0.9
g = 0.0
for _, _, r, _ in reversed(trajectory):
    g = r + gamma * g
print(f"{g:.4f}")
print()
print("The shortest path is 14 steps (11 right + 3 down) and the theoretical")
print("maximum undiscounted return is 14*(-1) + 10 = -4. Every extra step costs")
print("exactly one more unit, so return and path length are interchangeable here.")


# ======================================================================================
# Report
# ======================================================================================
print()
print("=" * 78)
print("REPORT")
print("=" * 78)
print(f"""
ENVIRONMENT
  grid                {env.rows} x {env.cols} = {env.n_states} states
  actions             {env.n_actions} (up, right, down, left)
  transitions         stochastic when slip > 0
  terminals           goal (+{env.goal_reward}), cliff ({env.cliff_penalty} + teleport)
  step cost           {env.step_cost} per move

SAMPLE EPISODE
  steps               {len(trajectory)}
  undiscounted return {total:.1f}
  gamma=0.9 return    {g:.4f}

CONTRACT THIS ENV PROVIDES
  reset(seed) -> (observation, info)
  step(action) -> (observation, reward, terminated, truncated, info)

WHAT TO CARRY FORWARD
  - terminated (goal/cliff) and truncated (time limit) are different signals;
    bootstrapping past a truncation is allowed, past a termination it is not
  - P(s'|s,a) is a distribution; a deterministic env is the special case where
    one outcome has probability 1
  - the flat index encoding is what every Q-table below is keyed on
  - slippery transitions make the OPTIMAL policy risk-aware, which is the
    entire point of script 06
""")
print("=" * 78)
