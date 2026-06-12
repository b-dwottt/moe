# Entry #2 — 12 June 2026 — Week 1 Synthetic Baseline

#### 1. Session overview

| Field | Your answer |
|-------|-------------|
| Date & time | 12 June 2026, 15:29-16:20 |
| Duration (hrs) | 0.8 |
| Week in roadmap | Week 1 |
| Phase | Baseline |
| Goal for this session | Build the drifting synthetic task and train a dense MLP baseline with saved outputs. |
| Did you achieve it? | Yes |

---

#### 2. What you built or changed

- **Files changed/created:**
  - `src/data.py` — synthetic drifting stream generator and evaluation-set helper.
  - `src/models.py` — dense MLP baseline.
  - `src/metrics.py` — accuracy and moving-average helpers.
  - `src/train.py` — runnable training/evaluation script that saves metrics and a plot.
  - `src/__init__.py` — package marker for module execution.
  - `results/figures/synthetic_baseline_loss_accuracy.png` — loss/accuracy figure.
  - `results/tables/synthetic_baseline_metrics.csv` — per-step training metrics.
  - `results/tables/synthetic_baseline_summary.csv` — summary metrics.
- **Key code decisions made:**
  - Used deterministic linear rules that swap every fixed interval so the drift is reproducible.
  - Kept the first model intentionally simple: a dense MLP baseline, so later MoE variants have a clean comparison point.
  - Wrote outputs to `results/` so the experiment leaves a persistent paper trail.
- **Git commit hash (if applicable):**
  - N/A

---

#### 3. Experiment details

| Hyperparameter | Value used | Reason / source |
|----------------|------------|-----------------|
| Learning rate | 0.001 | Stable default for the first baseline run. |
| Batch size | 64 | Small enough for quick iteration. |
| Number of experts | N/A | Dense baseline, no experts yet. |
| Router hidden size | N/A | No router in the dense baseline. |
| Memory alpha (weight) | N/A | No memory module yet. |
| Memory decay | N/A | No memory module yet. |
| Queue size | N/A | No memory module yet. |
| Number of clusters/contexts | N/A | No contextual memory yet. |
| Routing type | N/A | No routing; plain dense classifier. |
| Optimiser | Adam | Standard baseline optimizer. |
| Number of training steps | 1000 | Enough to see drift across four regimes. |
| Task swap interval | 250 | Frequent enough to show adaptation behavior. |
| Random seed(s) | 42 | Single deterministic seed for the first run. |
| Hardware | CPU | Local machine, no GPU used. |
| PyTorch version | 2.9.0 | Active environment version. |

---

#### 4. Results

**Quantitative results:**

| Model | Accuracy (mean ± std) | Loss | Adaptation speed (steps) | Expert entropy | Notes |
|-------|-----------------------|------|--------------------------|----------------|-------|
| Dense baseline | 0.958 ± 0.000 | 0.119 | N/A | N/A | Evaluation on the final regime after drift. |
| Standard MoE | N/A | N/A | N/A | N/A | Not implemented yet. |
| This session's model | 0.958 ± 0.000 | 0.119 | N/A | N/A | Same as dense baseline. |

- Number of seeds run: 1
- Are results averaged? No
- Confidence intervals / std reported? No

**Figures saved:**
- [x] `results/figures/synthetic_baseline_loss_accuracy.png` — loss and accuracy over training steps with drift markers.

**Tables saved:**
- [x] `results/tables/synthetic_baseline_metrics.csv`
- [x] `results/tables/synthetic_baseline_summary.csv`

---

#### 5. Comparison to baseline

- Which baseline(s) did you compare against today? None; this session created the baseline.
- Was the comparison fair (same data, same seeds, same compute)? N/A.
- Did your method win / lose / tie? N/A.
- If it lost: what is your hypothesis for why? N/A.

---

#### 6. What worked

The synthetic stream, dense MLP, and training loop all ran end to end, and the script saved both a CSV metrics table and a drift plot. The final-regime evaluation accuracy reached 0.958 on CPU with a single seed.

---

#### 7. What failed or surprised you

Nothing failed in the final run. The only minor issue was that the first draft of the noise sampling needed a PyTorch-compatible tensor call, which was corrected before execution.

---

#### 8. Threats to validity

This is still a toy linear drift benchmark, so it does not say anything about real-world routing or memory usefulness yet. The single-seed result is also not statistically robust.

---

#### 9. Limitations identified today

- This result does not test any MoE or memory mechanism.
- It uses one synthetic task family with one seed.
- It does not prove robustness to harder drift patterns or noisy labels.

---

#### 10. Paper writing notes

- **Method section note:** Describe the drifting linear rule generator and the dense MLP baseline used for the first comparison point.
- **Results section note:** Add the synthetic baseline loss/accuracy figure and the summary metrics row.
- **Limitations section note:** Note that the current result is only a dense baseline on a toy drift benchmark.
- **New claim supported today:** The experiment pipeline is reproducible and can capture drift-sensitive performance curves.
- **Claim that was weakened today:** None yet; no memory method has been tested.

---

#### 11. Reviewer pre-emption

| Likely reviewer concern | Your planned response |
|-------------------------|-----------------------|
| "Memory adds parameters — is it fair?" | This session is only the baseline setup; later comparisons will keep compute and data fixed. |
| "Why not just use a larger router?" | The current result does not address this yet; later ablations will. |
| "Does this work on real data beyond MNIST?" | Not yet; the first target is controlled synthetic drift. |
| "What is the compute overhead of memory updates?" | Not measured yet; baseline instrumentation is in place first. |
| "Could load balancing alone achieve the same effect?" | Not tested yet. |
| "Is the non-stationary benchmark too artificial?" | Yes, for now; it is meant as a controlled starting point. |
| "Is 2 experts enough to show the effect?" | Not applicable yet; the dense baseline comes first. |
| "Are results statistically significant?" | Not yet; only one seed has been run. |

---

#### 12. Next session plan

- **Next immediate task:** Implement the standard 2-expert MoE baseline on the same synthetic stream.
- **Blocking issue (if any):** None.
- **Reading needed before next session:** Standard MoE routing and top-k gating notes.
- **Estimated time needed:** 2-3 hours.

---

#### 13. Quick health check

| Check | Status |
|-------|--------|
| Code committed to Git? | No |
| At least one graph saved? | Yes |
| At least one number recorded? | Yes |
| Research log updated? | Yes |
| Paper note written? | Yes |
