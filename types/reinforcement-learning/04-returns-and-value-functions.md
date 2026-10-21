# 4. Returns and value functions

The **return** is the objective accumulated after a decision. In a discounted task:

`G_t = R_{t+1} + γ R_{t+2} + γ² R_{t+3} + ...`

For a finite episode ending at `T`, stop the sum at `T`. The discount `γ` controls the weight of delayed rewards. With `γ=0`, only immediate reward matters. Values near one place more weight on distant outcomes, but can increase variance and make estimates sensitive to long-term model errors.

## Value functions

For policy `π`, the state value is:

`Vπ(s) = Eπ[G_t | S_t=s]`

The action value is:

`Qπ(s,a) = Eπ[G_t | S_t=s, A_t=a]`

These are expectations over uncertain transitions and future policy actions, not guarantees for any one trajectory. An optimal value, `V*(s)`, is the largest achievable expected return from state `s`; `Q*(s,a)` is defined similarly.

## Why values matter

They convert delayed consequences into a score for states or actions. A policy can choose the action with the highest estimated action value, or directly assign probabilities to actions. Values also provide baselines that reduce gradient-estimation variance.

## Discount factor is a business assumption

Do not set `γ` by habit. Connect it to task duration, reward timing, and the actual objective. Discounting is not the same as uncertainty about the future, and it does not fix a poorly specified horizon. If rewards are monetary, selecting a discount can imply a time preference that should be reviewed explicitly.

## Reward scaling and units

Multiplying every reward by a positive constant does not change the exact optimal policy in a basic unconstrained discounted MDP, but it changes value magnitudes and can affect optimization stability. Combining dollars, clicks, and safety penalties requires explicit scaling and a defensible preference structure.

## Variance and reporting

Two policies can have the same mean return and very different tail outcomes. For consequential decisions, report distributional summaries: mean, median, quantiles, worst-case or conditional-tail metrics where appropriate, and constraint-violation rates. A high average reward does not establish acceptable individual outcomes.

## Practice

For a three-step reward sequence, calculate `G_0` at `γ=0.5` and `γ=0.95`. Explain why the preferred policy might change when the horizon or discount changes. State one limitation of using mean return alone.
