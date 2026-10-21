# 14. Model-based RL and planning

Model-based methods use a transition/reward model to predict consequences and plan. The model may be specified by experts, provided by a simulator, or learned from data. Model-free methods learn values or policies without explicitly planning through such a model.

## Planning with a model

If a model is available, use dynamic programming, search, or model-predictive control (MPC). MPC repeatedly plans over a finite lookahead horizon, executes the first action, observes the new state, and replans. Replanning can correct some prediction errors, but a short horizon can miss delayed consequences.

## Learning a model

Estimate `P(s'|s,a)` or predict next state and reward. Evaluate model quality on held-out transitions and, crucially, on multi-step rollouts. Small one-step errors can compound over long imagined trajectories. Track uncertainty and avoid trusting predictions far outside training coverage.

## Dyna-style learning

Dyna methods interleave real experience with simulated updates from a learned model. This can improve sample efficiency when the model is accurate. Model errors can also reinforce themselves because the agent plans into states where its model is wrong.

## Sample efficiency versus model bias

Model-based methods can reuse data efficiently, but performance depends on model fidelity in decision-relevant regions. A model's average prediction metric may hide errors that reverse action rankings or miss rare high-cost events. Evaluate decision quality and tail risks, not only one-step accuracy.

## When this is a good fit

- A trusted simulator or operational model exists.
- Real interactions are costly, slow, or unsafe.
- The state/action space permits useful planning.
- There is enough data to validate the model in relevant regions.

## Practice

Compare a known-model planner with a learned-model planner on a small task. Introduce a controlled model error in one transition and observe how planning responds. Write down a validation test that would detect the error before deployment.
