# 15. Special settings and modern practice

Many real decision problems depart from the fully observed, single-agent, online-learning MDP. Recognize the setting before selecting an algorithm.

## Partial observability and POMDPs

When the true state is hidden, the agent receives observations. A partially observable MDP (POMDP) models hidden state and observation likelihoods. A belief state is a probability distribution over possible hidden states and can serve as a Markov information state in theory. In practice, recurrent networks, filters, and history summaries are common approximations. Test whether the policy handles ambiguous observations and missing sensors.

## Hierarchical RL and options

Hierarchical methods use temporally extended actions, often called options, that include an initiation condition, an internal policy, and a termination rule. They can make long-horizon tasks easier to explore and interpret, but designing useful abstractions is difficult. Ensure the high-level action duration and lower-level policy match the task's safety constraints.

## Multi-agent RL

Multiple learning agents create nonstationarity: each agent's environment changes as others update. Distinguish cooperative, competitive, and mixed objectives. Evaluate against diverse partners or opponents; performance against one fixed policy may not generalize. Coordination, communication, and equilibrium concepts may matter more than single-agent reward maximization.

## Imitation learning and offline RL

Imitation learning learns behavior from demonstrations. Behavior cloning treats actions as supervised labels and can fail when the learned policy reaches states absent from demonstrations. Offline RL learns from a fixed dataset without new interaction. It must avoid assigning high value to unsupported actions; conservative methods and uncertainty checks address this challenge, but cannot create evidence absent from the data.

## Safe and constrained RL

Some tasks have constraints such as a maximum failure probability, budget, or exposure. A penalty in reward does not guarantee a hard constraint. Constrained formulations, action shields, control barriers, conservative policies, and human oversight can help, but guarantees depend on assumptions and validated models. State the constraint metric and how violations are measured.

## Transfer and generalization

Transfer learning reuses representations, models, or policies across tasks. A transferred policy can fail when dynamics, user populations, or action meanings change. Validate on the new context and identify what was assumed invariant.

## Practice

Take your capstone and modify one assumption: hide part of the state, restrict learning to a fixed log, or add a hard constraint. Explain which baseline and evaluation changes are required.
