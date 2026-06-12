# Entry #3 — 12 June 2026 — Standard MoE and Memory Variant Suite

#### 1. Session overview

| Field | Your answer |
|-------|-------------|
| Date & time | 12 June 2026, 15:39-15:50 |
| Duration (hrs) | 0.2 |
| Week in roadmap | Week 2-5 bridge |
| Phase | Standard MoE / Memory-v1 / Memory-v2 / Contextual |
| Goal for this session | Move beyond the dense baseline by implementing and comparing learned MoE routing, load balancing, and multiple memory-routing variants. |
| Did you achieve it? | Yes |

---

#### 2. What you built or changed

- **Files changed/created:**
  - `src/models.py` — added shared MoE implementation, random-routing baseline, load-balancing support, scoreboard memory, queue memory, and contextual memory.
  - `src/metrics.py` — added normalized routing entropy, adaptation speed, final-window summaries, and regime summaries.
  - `src/train.py` — added `--suite` comparison mode, multi-seed runs, model selection, routing diagnostics, memory traces, and aggregate outputs.
  - `README.md` — documented the current research suite and first comparative table.
  - `paper/conference_brief.md` — drafted the conference-facing framing, caveats, and next experimental matrix.
  - `results/tables/synthetic_moe_comparison_*.csv` — metrics, summary, aggregate summary, and regime summary.
  - `results/figures/synthetic_moe_comparison_*.png` — accuracy/loss, routing diagnostics, and memory traces.
- **Key code decisions made:**
  - Kept all MoE variants in a shared interface so diagnostics and training conditions are consistent.
  - Used soft routing for the first controlled comparison because it gives stable gradients and avoids conflating memory with sparse-router training instability.
  - Logged adaptation speed after drift because final accuracy is already near saturation on the current synthetic task.
  - Included random routing and load-balanced MoE to separate expert capacity and usage balance from explicit memory effects.
- **Git commit hash (if applicable):**
  - N/A

---

#### 3. Experiment details

| Hyperparameter | Value used | Reason / source |
|----------------|------------|-----------------|
| Learning rate | 0.001 | Same stable optimizer setting as the dense baseline. |
| Batch size | 64 | Matches Week 1 baseline scale. |
| Number of experts | 4 | Tests nontrivial routing beyond a 2-expert toy. |
| Router hidden size | 32 | Lightweight router relative to expert capacity. |
| Memory alpha (weight) | 0.5 | First-pass value large enough to expose a routing effect. |
| Memory decay | 0.95 | Smooth EMA-style memory. |
| Queue size | 50 | Short recent-performance window. |
| Number of clusters/contexts | 4 | Matches number of experts for first contextual run. |
| Routing type | soft | Stable first comparison before sparse top-1 experiments. |
| Optimiser | Adam | Standard across all variants. |
| Number of training steps | 1500 | Five regimes with four drift boundaries. |
| Task swap interval | 300 | Frequent enough to measure recovery repeatedly. |
| Random seed(s) | 42, 43, 44 | First multi-seed comparison. |
| Hardware | CPU | Local run. |
| PyTorch version | 2.9.0 | Active environment version from earlier baseline. |

Command:

```bash
python -m src.train --suite --steps 1500 --swap-interval 300 --batch-size 64 --num-experts 4 --seeds 42,43,44 --experiment-name synthetic_moe_comparison
```

---

#### 4. Results

**Quantitative results:**

| Model | Accuracy (mean ± std) | Loss | Adaptation speed (steps) | Expert entropy | Notes |
|-------|-----------------------|------|--------------------------|----------------|-------|
| Dense baseline | 0.978 ± 0.003 | 0.078 | 71.2 | N/A | Strong non-MoE baseline; final accuracy nearly saturated. |
| Random MoE | 0.968 ± 0.005 | 0.161 | 99.7 | 0.000 | Expert capacity without learned routing is weaker. |
| Standard MoE | 0.979 ± 0.008 | 0.096 | 54.2 | 0.629 | Learned routing improves adaptation over dense baseline. |
| Load-balanced MoE | 0.978 ± 0.007 | 0.101 | 52.5 | 0.679 | Load balance is a strong competing explanation. |
| Scoreboard Memory MoE | 0.978 ± 0.008 | 0.081 | 51.4 | 0.644 | Fastest recovery in this run while matching final accuracy. |
| Queue Memory MoE | 0.979 ± 0.009 | 0.092 | 54.0 | 0.630 | Best/equal final accuracy; adaptation similar to standard MoE. |
| Contextual Memory MoE | 0.973 ± 0.012 | 0.085 | 51.8 | 0.645 | Fast recovery but lower final accuracy on this easy setup. |

- Number of seeds run: 3
- Are results averaged? Yes
- Confidence intervals / std reported? Standard deviation across seeds

**Figures saved:**
- [x] `results/figures/synthetic_moe_comparison_loss_accuracy.png` — multi-model loss and accuracy over drift.
- [x] `results/figures/synthetic_moe_comparison_routing_diagnostics.png` — routing entropy and selected expert usage.
- [x] `results/figures/synthetic_moe_comparison_memory_traces.png` — memory value traces for memory variants.

**Tables saved:**
- [x] `results/tables/synthetic_moe_comparison_metrics.csv`
- [x] `results/tables/synthetic_moe_comparison_summary.csv`
- [x] `results/tables/synthetic_moe_comparison_aggregate.csv`
- [x] `results/tables/synthetic_moe_comparison_regime_summary.csv`

---

#### 5. Comparison to baseline

- Which baseline(s) did you compare against today? Dense MLP, random-routing MoE, standard MoE, and load-balanced MoE.
- Was the comparison fair (same data, same seeds, same compute)? Mostly: same stream settings, seeds, optimizer, batch size, and expert count where applicable. Dense has a different parameterization by design.
- Did your method win / lose / tie? Memory variants tied on final accuracy but improved adaptation speed relative to dense and random baselines. They did not clearly dominate standard/load-balanced MoE.
- If it lost: contextual memory had lower final accuracy, likely because the simple random-projection contexts are too crude for the current task.

---

#### 6. What worked

The research harness now runs a full first-pass comparison across seven model families and produces paper-ready artifacts. Scoreboard memory recovered in 51.4 steps on average after drift, compared with 71.2 for the dense baseline and 99.7 for random routing.

---

#### 7. What failed or surprised you

Final accuracy saturated across dense, standard MoE, and memory variants. This means the current synthetic task is too easy for final accuracy to be the main evidence. Adaptation speed and routing behavior are more informative.

---

#### 8. Threats to validity

The current memory update uses per-expert local losses, which is useful for controlled diagnosis but may be too informative compared with a deployed sparse MoE that only observes the selected expert's reward. The current suite also uses soft routing, only three seeds, and one synthetic drift family.

---

#### 9. Limitations identified today

- This result does not prove memory beats standard MoE under all drift settings.
- Load-balanced MoE is competitive, so future work must separate memory from usage smoothing.
- Contextual memory needs better context construction than fixed random projections.
- Adaptation speed thresholding at 0.85 is useful but should be complemented with area-under-recovery and regret-style metrics.

---

#### 10. Paper writing notes

- **Method section note:** Define memory bonus as an additive expert-specific routing prior applied to router logits.
- **Results section note:** Report both final accuracy and post-drift adaptation speed; final accuracy alone hides the difference.
- **Limitations section note:** Current prototype uses soft routing and local per-expert losses, so observed-only sparse memory is required before strong deployment claims.
- **New claim supported today:** Memory-aware routing can match standard MoE final accuracy while improving drift recovery over dense and random baselines on controlled synthetic drift.
- **Claim that was weakened today:** Memory is not yet clearly better than load-balanced MoE or standard MoE on final accuracy.

---

#### 11. Reviewer pre-emption

| Likely reviewer concern | Your planned response |
|-------------------------|-----------------------|
| "Memory adds parameters — is it fair?" | Current memory stores small buffers, not learned dense layers; report parameter counts and overhead separately. |
| "Why not just use a larger router?" | Add router-size ablation and compare against standard MoE with matched or larger router. |
| "Does this work on real data beyond MNIST?" | Not yet; next phase should add MNIST/FashionMNIST drift before any broad claim. |
| "What is the compute overhead of memory updates?" | Add wall-clock timing and memory update complexity to the suite. |
| "Could load balancing alone achieve the same effect?" | Load-balanced MoE is already competitive; future runs should include memory+load balance and harder drift. |
| "Is the non-stationary benchmark too artificial?" | Yes; this is a diagnostic benchmark. Add recurring, gradual, noisy, and image drift tasks. |
| "Is 2 experts enough to show the effect?" | This run used 4 experts; next ablation should include 2, 4, and 8. |
| "Are results statistically significant?" | Not yet; three seeds are a first-pass screen. Run 10-20 seeds and bootstrap intervals. |

---

#### 12. Next session plan

- **Next immediate task:** Add top-1 routing, memory-alpha/decay ablations, and observed-only memory updates.
- **Blocking issue (if any):** None.
- **Reading needed before next session:** Sparse MoE credit assignment and contextual bandit exploration notes.
- **Estimated time needed:** 2-4 hours.

---

#### 13. Quick health check

| Check | Status |
|-------|--------|
| Code committed to Git? | No |
| At least one graph saved? | Yes |
| At least one number recorded? | Yes |
| Research log updated? | Yes |
| Paper note written? | Yes |
