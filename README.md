# Memory-Augmented Routing for Adaptive Mixture-of-Experts Models

This repository contains the research project investigating **Contextual Expert Memory for Adaptive Mixture-of-Experts (MoE) Routing**.

The core objective is to design and evaluate an MoE router that utilizes expert-specific historical performance memory (success, failure, stability) to make more adaptive and reliable routing decisions, especially under non-stationary task drifts.

---

## Research Question

> **Can expert-specific performance memory improve MoE routing decisions, especially under changing or non-stationary tasks?**

## Main Hypothesis

> **A memory-aware MoE router should adapt faster than a standard MoE router because it remembers which experts recently succeeded or failed on similar inputs.**

---

## Directory Structure

```
├── notebooks/
│   └── 01_synthetic_baseline.ipynb   # Baseline training on synthetic task
├── src/
│   ├── data.py                       # Data loaders and synthetic dataset generation
│   ├── models.py                     # MLP, Standard MoE, and Memory-Augmented MoE models
│   ├── train.py                      # Training loop and experiment execution
│   └── metrics.py                    # Adaptation speed, entropy, and performance metrics
└── results/
    ├── figures/                      # Plots and figures
    └── tables/                       # Experiment log tables
```

---

## Getting Started

1. **Environment Setup**:
   Ensure you have Conda installed. Activate or install the `maj` environment:
   ```bash
   conda activate maj
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Week 1 baseline**:
   ```bash
   python -m src.train
   ```
   This writes metrics to `results/tables/` and the drift plot to `results/figures/`.

4. **Run the current MoE research suite**:
   ```bash
   python -m src.train --suite --steps 1500 --swap-interval 300 --batch-size 64 --num-experts 4 --seeds 42,43,44 --experiment-name synthetic_moe_comparison
   ```
   This compares dense, random-routing MoE, standard MoE, load-balanced MoE, scoreboard-memory MoE, queue-memory MoE, and contextual-memory MoE under the same synthetic drift protocol.

---

## Current Research Status

The repository now supports the first conference-style comparison suite:

| Variant | Purpose |
|---------|---------|
| Dense MLP | Non-MoE baseline. |
| Random MoE | Tests whether expert specialization alone helps without learned routing. |
| Standard MoE | Learned router without explicit memory. |
| Load-balanced MoE | Tests whether balanced expert usage explains gains. |
| Scoreboard Memory MoE | Adds expert-level EMA reward memory to router logits. |
| Queue Memory MoE | Adds rolling loss/stability memory per expert. |
| Contextual Memory MoE | Adds expert memory conditioned on deterministic input contexts. |

Latest synthetic-drift suite: 3 seeds, 1,500 steps, 4 experts, swap interval 300.

| Model | Eval accuracy | Adaptation speed after drift |
|-------|---------------|------------------------------|
| Queue Memory MoE | 0.979 ± 0.009 | 54.0 steps |
| Standard MoE | 0.979 ± 0.008 | 54.2 steps |
| Dense MLP | 0.978 ± 0.003 | 71.2 steps |
| Scoreboard Memory MoE | 0.978 ± 0.008 | 51.4 steps |
| Load-balanced MoE | 0.978 ± 0.007 | 52.5 steps |
| Contextual Memory MoE | 0.973 ± 0.012 | 51.8 steps |
| Random MoE | 0.968 ± 0.005 | 99.7 steps |

Interpretation: final accuracy is tightly clustered on the current toy task, but memory-aware routing improves drift recovery speed relative to dense and random baselines. The next research step is to stress-test this claim with top-1 routing, memory-alpha/decay ablations, harder drift types, observed-only memory updates, and image/task benchmarks.
