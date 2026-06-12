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
