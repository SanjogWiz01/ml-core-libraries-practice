# 11. Deep Q-networks

Deep Q-networks (DQN) combine Q-learning with a neural network to estimate action values for discrete actions. The network maps state `s` to a vector of Q-values, one per action.

## Basic target and loss

For a sampled transition `(s,a,r,s',done)`, a common target is:

`y = r + γ (1-done) max_{a'} Q_target(s',a')`

The online network minimizes a squared or robust loss between `Q_online(s,a)` and `y`. The target network is a delayed copy that changes more slowly, reducing the feedback loop between prediction and target.

## Experience replay

Store transitions and sample minibatches to reduce correlation between neighboring updates and reuse experience. Replay can become stale as the behavior policy changes; prioritizing transitions changes the sampling distribution and requires appropriate correction if unbiased objectives matter.

## Double DQN and variants

The max operator tends to overestimate values when estimates are noisy. Double DQN selects the next action with the online network and evaluates it with the target network. Other extensions address action selection, exploration, or value distributions; extensions add complexity and should be justified by evidence.

## DQN assumptions and limits

- The standard formulation assumes a finite discrete action set.
- It is primarily designed for off-policy learning from transitions.
- It can be unstable and sensitive to replay size, learning rate, target update rate, reward scale, and exploration schedule.
- It may perform poorly when logged data have limited coverage or when rare catastrophic outcomes matter.

## Debugging sequence

First validate the environment and reward. Check that terminal transitions are masked correctly. Confirm Q-values and targets have plausible ranges. Overfit a tiny fixed dataset or tiny environment as a plumbing check. Compare against random and heuristic baselines. Then run repeated seeds and held-out evaluations.

## Practice

Implement DQN on a small discrete-action simulator. Log episode return, loss, action frequencies, and Q-value magnitudes. Explain why decreasing training loss alone does not prove better control.
