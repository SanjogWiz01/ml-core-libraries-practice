# 10. Function approximation

Tabular values require a separate entry for each state or state-action pair. Real problems have large, continuous, or structured inputs, so we represent value or policy with a function such as `V(s; w)`, `Q(s,a; w)`, or `π(a|s; θ)`.

## Linear approximation

With features `φ(s)`, approximate `V(s;w) = wᵀφ(s)`. A semi-gradient TD update is:

`w ← w + α [r + γ V(s';w) - V(s;w)] φ(s)`

Features encode generalization. Poor features can omit important differences; overly rich features can overfit. Normalize numeric inputs, encode categorical variables thoughtfully, and include action information for action-value models.

## Neural approximation

Neural networks learn representations from data and can handle high-dimensional observations. They also introduce nonconvex optimization, sensitivity to scale and initialization, and greater data demands. A good supervised prediction score does not automatically yield a good policy because decisions depend on comparative errors across actions.

## The deadly triad

The interaction of function approximation, bootstrapping, and off-policy learning can cause instability or divergence. It is a warning about interacting sources of error, not a claim that every such combination fails. Replay buffers, target networks, and gradient constraints can help in particular algorithms, but they do not remove the need for careful evaluation.

## Generalization and distribution shift

The policy visits states according to its own behavior. Learning data collected by another policy may cover a different region. Evaluate on held-out initial conditions, environment parameters, and scenarios that reflect deployment, while retaining stress tests for rare but important cases.

## Practical checklist

- Establish a tabular or linear baseline on a reduced problem.
- Verify feature ranges and reward scale.
- Inspect action values, not only aggregate losses.
- Track visitation and identify unobserved or weakly supported states.
- Change one training choice at a time and use multiple random seeds.

## Practice

Build linear value approximation for a small continuous-state task. Compare raw coordinates with normalized features. Plot prediction error and policy return separately; explain why they need not move together.
