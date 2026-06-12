# Conference Research Brief

## Working Title

Memory-Augmented Routing for Adaptive Mixture-of-Experts Models

## Research Question

Can expert-specific performance memory improve MoE routing decisions under non-stationary task drift?

## Core Claim Under Test

Memory should not be treated as an extra hidden layer. The intended contribution is an explicit routing signal: each expert carries a compact record of recent success, failure, and stability, and the router uses that record when choosing experts.

## Current Experimental Setup

- Dataset: synthetic binary classification stream with abrupt linear-rule drift.
- Drift protocol: 1,500 online training steps, task swap every 300 steps.
- Seeds: 42, 43, 44.
- Batch size: 64.
- Experts: 4 MLP experts.
- Routing: soft routing for the first controlled comparison.
- Metrics: evaluation accuracy on the final regime, mean train loss/accuracy, normalized routing entropy, selected expert share, and adaptation speed after drift.

## Implemented Variants

| Family | Variant | Question it answers |
|--------|---------|---------------------|
| Non-MoE baseline | Dense MLP | Is the task already solved without routing? |
| Weak MoE baseline | Random-routing MoE | Does expert capacity alone help without learned routing? |
| Main baseline | Standard MoE | What does a learned stateless router achieve? |
| Alternative explanation | Load-balanced MoE | Are gains just from less expert collapse? |
| Memory v1 | Scoreboard Memory MoE | Does expert-level EMA reward memory accelerate recovery? |
| Memory v2 | Queue Memory MoE | Does short-term loss/stability memory help? |
| Memory v3 | Contextual Memory MoE | Does input-conditioned expert memory improve drift response? |

## First-Pass Results

| Model | Eval accuracy | Adaptation speed |
|-------|---------------|------------------|
| Queue Memory MoE | 0.979 ± 0.009 | 54.0 steps |
| Standard MoE | 0.979 ± 0.008 | 54.2 steps |
| Dense MLP | 0.978 ± 0.003 | 71.2 steps |
| Scoreboard Memory MoE | 0.978 ± 0.008 | 51.4 steps |
| Load-balanced MoE | 0.978 ± 0.007 | 52.5 steps |
| Contextual Memory MoE | 0.973 ± 0.012 | 51.8 steps |
| Random MoE | 0.968 ± 0.005 | 99.7 steps |

## Interpretation

The current task is easy enough that final accuracy compresses near 0.97-0.98 for most learned models. The more informative signal is recovery after drift: memory-aware and load-balanced routing recover faster than the dense baseline, while random routing is clearly slower. This supports the idea that explicit routing state may improve adaptation, but it does not yet prove that memory is uniquely responsible.

The strongest present claim is:

> On a controlled synthetic drift stream, memory-augmented MoE variants preserve competitive final accuracy while reducing post-drift recovery time relative to dense and random-routing baselines.

The claim that still needs more evidence is:

> Memory is superior to standard learned routing and load balancing across harder drift settings.

## Reviewer-Ready Caveats

| Concern | Current answer | Next evidence needed |
|---------|----------------|----------------------|
| "Is the benchmark too easy?" | Yes, final accuracy is saturated. Adaptation speed is the useful metric. | Harder drift: rotated rules, noisy labels, gradual drift, recurring regimes. |
| "Is memory just load balancing?" | Load-balanced MoE is already included and is competitive. | Add memory + load balance combinations and expert-collapse plots. |
| "Does memory use privileged feedback?" | Current memory uses per-expert local losses in the controlled prototype. | Add observed-only memory where only selected expert reward is updated. |
| "Does this work with sparse top-1 routing?" | The first suite uses soft routing for stable comparison. | Run top-1 straight-through and sparse observed-only routing. |
| "Is this statistically significant?" | Three seeds are enough for smoke evidence, not a paper claim. | Run 10-20 seeds and bootstrap confidence intervals. |
| "Does it generalize beyond synthetic data?" | Not tested yet. | MNIST/FashionMNIST drift and a tiny sequence task. |
| "What is the overhead?" | Parameter counts are logged; runtime overhead is not yet reported. | Add wall-clock timing and memory-update complexity. |

## Next Experimental Matrix

| Axis | Values |
|------|--------|
| Routing | soft, top1 straight-through, observed-only sparse |
| Memory alpha | 0.0, 0.1, 0.25, 0.5, 1.0 |
| Memory decay | 0.8, 0.9, 0.95, 0.99 |
| Queue size | 10, 25, 50, 100 |
| Number of experts | 2, 4, 8 |
| Context count | 2, 4, 8, 16 |
| Drift pattern | abrupt, gradual, recurring, noisy-label, rule-rotation |
| Baselines | dense, random, standard MoE, load-balanced MoE, oracle synthetic router |

## Conference Narrative

1. MoE routers are usually trained as parametric functions of the current input.
2. Non-stationary environments make current-input-only routing brittle because the best expert can change over time.
3. We add compact expert-specific memory as a routing prior.
4. The first controlled benchmark shows that final accuracy alone hides the effect; adaptation speed exposes it.
5. The remaining work is to separate memory from load balancing, remove privileged memory updates, and test harder non-stationarity.
