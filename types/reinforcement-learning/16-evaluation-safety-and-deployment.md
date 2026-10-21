# 16. Evaluation, safety, and deployment

An RL result is only as useful as its evidence. A high training score can come from overfitting a simulator, lucky initialization, reward exploitation, or evaluation leakage.

## Evaluation protocol

1. Define the primary outcome and hard constraints before comparing policies.
2. Compare against random, heuristic, existing, and domain-specific baselines.
3. Use separate training, validation, and final test environments or scenarios.
4. Run multiple independent seeds and report uncertainty, not only the best run.
5. Evaluate on held-out initial conditions and relevant dynamics shifts.
6. Report learning cost, interaction budget, compute, and failure rate.

Use episode or environment-level resampling for uncertainty when observations within episodes are dependent. A confidence interval over individual transitions can be misleading.

## Offline policy evaluation

When a policy cannot be deployed for a direct experiment, off-policy evaluation estimates its outcome from logged data. Methods include importance sampling, weighted importance sampling, doubly robust estimators, and fitted value approaches. Each relies on assumptions such as adequate action coverage, correct propensities or models, and consistency between logging and target environments. Present sensitivity analyses and uncertainty; do not treat a single estimate as proof.

## Safety and governance

- Keep immutable logs of state/context, action, policy version, propensity where applicable, reward/outcome, and constraint events.
- Use access controls and minimize sensitive data. Define retention and consent rules.
- Start with simulation, replay, shadow evaluation, and limited pilots where appropriate.
- Add action bounds, fallback policies, rate limits, and a human override for consequential deployments.
- Monitor outcome drift, state coverage, action distribution, and subgroup-level harm.
- Define rollback conditions before launch and rehearse the rollback path.

## Reward exploitation and simulator gap

Inspect trajectories, not only aggregate scores. Search for implausibly high reward, repeated boundary actions, ignored constraints, or behavior that exploits simulator shortcuts. Compare simulation behavior with domain experts and real system limits. A simulator is a tool for evidence, not reality itself.

## Deployment decision memo

Summarize the intended decision, evidence, assumptions, supported population, known failure modes, constraints, monitoring, fallback, and conditions under which the policy must be disabled. If the evidence does not support safe deployment, the correct result is to recommend more data, a restricted pilot, or a simpler baseline.

## Capstone rubric

The project is complete when another data scientist can reproduce the formulation, baseline, training setup, evaluation, uncertainty estimates, and recommendation. A good result includes cases where the method failed and a clear explanation of whether RL added value over simpler approaches.
