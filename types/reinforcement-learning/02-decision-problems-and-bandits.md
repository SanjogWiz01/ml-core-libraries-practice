# 2. Decision problems and bandits

Start by asking what kind of decision problem you have. The distinction determines whether RL is appropriate and how much experimentation is needed.

## A decision problem has five practical pieces

1. **Decision maker:** the system choosing an action.
2. **Information:** what it observes before acting.
3. **Available actions:** what it is permitted and able to do.
4. **Outcome and feedback:** what happens and what is measured afterward.
5. **Objective and constraints:** what should improve, over what time horizon, and what must not be violated.

An action is consequential when it changes future outcomes or future information. If each choice has an immediate reward and does not influence future decisions, a bandit may suffice. If actions alter future state, use a sequential control formulation such as an MDP.

## Multi-armed bandits

Each action (arm) has an unknown reward distribution. At each round, choose an arm and observe its reward. The tension is **exploration** (learn uncertain arms) versus **exploitation** (use the best currently estimated arm).

For action `a`, estimate its mean reward with the sample average:

`Q_n(a) = (r_1 + ... + r_n) / n`

An epsilon-greedy policy selects a random action with probability `ε` and otherwise selects the action with the largest estimate. Upper-confidence-bound methods add an uncertainty bonus to estimated reward. Thompson sampling samples plausible reward parameters and chooses the action that would be best under that sample.

## Contextual bandits

At each round, observe context `x`, choose action `a`, then observe reward `r`. The action does not influence the next context in the classic contextual-bandit setup. This models some recommendation, ranking, and offer-selection problems, subject to feedback and consent constraints.

Logged data require care: if the logging policy chose actions non-uniformly, outcomes for unchosen actions are missing. A logged row should include context, chosen action, reward, and the probability with which the behavior policy selected that action (the propensity).

## Bandit versus RL: a diagnostic

Ask: “Would this action change the future state, opportunities, or observations?” If yes, myopic bandit optimization may miss delayed effects. A treatment decision may change a patient's later health; a maintenance action may change future failure risk; a pricing decision may alter customer retention. These are sequential effects, though the model and safety burden can make online RL inappropriate.

## Common mistakes

- Calling any repeated recommendation problem RL when actions have no sequential effect.
- Optimizing clicks while ignoring long-term value or user harm.
- Treating missing rewards as zero; unobserved outcomes are not necessarily zero outcomes.
- Reporting reward without specifying the exploration cost paid to obtain it.

## Practice

Implement epsilon-greedy on a simulated three-arm bandit. Track cumulative reward and regret against the best arm in hindsight. Then explain which assumptions would fail if an action affected the next round's context.
