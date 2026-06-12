# Research Log — Memory-Augmented MoE Project
**Project:** Contextual Expert Memory for Adaptive Mixture-of-Experts Routing
**Started:** 12 June 2026
**Log maintained by:** Antigravity (AI Assistant)

---

> **How to use this log**
> Copy the entry template below for every work session. Fill every field — even if the answer is "N/A" or "unclear yet". Bad results and failures are as important as good ones. Reviewers care that you thought carefully, not just that it worked.

---

## Log Entry Template

Copy this block for each work session into a new file under `logs/entry_XXX_[title].md`.

```markdown
### Entry #[N] — [Date] — [Session title]

#### 1. Session overview

| Field | Your answer |
|-------|-------------|
| Date & time | |
| Duration (hrs) | |
| Week in roadmap | Week [0–12] |
| Phase | Setup / Baseline / Memory-v1 / Memory-v2 / Contextual / Ablation / Analysis / Writing |
| Goal for this session | One sentence: what were you trying to do or answer today? |
| Did you achieve it? | Yes / Partial / No |

---

#### 2. What you built or changed

- **Files changed/created:**
  - `src/models.py` — added ...
- **Key code decisions made:**
  - Why did you write it this way and not another way?
- **Git commit hash (if applicable):**

---

#### 3. Experiment details

| Hyperparameter | Value used | Reason / source |
|----------------|------------|-----------------|
| Learning rate | | |
| Batch size | | |
| Number of experts | | |
| Router hidden size | | |
| Memory alpha (weight) | | |
| Memory decay | | |
| Queue size | | |
| Number of clusters/contexts | | |
| Routing type | top-1 / soft / epsilon-greedy |  |
| Optimiser | | |
| Number of training steps | | |
| Task swap interval | | |
| Random seed(s) | | |
| Hardware | CPU / GPU (model) | |
| PyTorch version | | |

---

#### 4. Results

**Quantitative results:**

| Model | Accuracy (mean ± std) | Loss | Adaptation speed (steps) | Expert entropy | Notes |
|-------|-----------------------|------|--------------------------|----------------|-------|
| Dense baseline | | | | | |
| Standard MoE | | | | | |
| This session's model | | | | | |

- Number of seeds run: ___
- Are results averaged? Yes / No
- Confidence intervals / std reported? Yes / No

**Figures saved:**
- [ ] `results/figures/[filename].png` — description of what it shows

**Tables saved:**
- [ ] `results/tables/[filename].csv`

---

#### 5. Comparison to baseline

- Which baseline(s) did you compare against today?
- Was the comparison fair (same data, same seeds, same compute)?
- Did your method win / lose / tie?
- If it lost: what is your hypothesis for why?

---

#### 6. What worked

Write specifically — not just "it worked". E.g., "Memory-MoE recovered 12% faster after the task swap at step 500, averaged over 3 seeds."

---

#### 7. What failed or surprised you

- What did not work as expected?
- Were there bugs? How did you find and fix them?
- Any unexpected behaviour?

---

#### 8. Threats to validity

Write your honest answers here.

---

#### 9. Limitations identified today

- What does this result NOT prove?
- What assumptions were made today that might not hold in general?
- What would break this in a harder setting?

---

#### 10. Paper writing notes

- **Method section note:** One sentence to add/update in the method section.
- **Results section note:** What table row or figure caption should be added.
- **Limitations section note:** Any limitation identified today to acknowledge.
- **New claim supported today:** e.g., "Memory-MoE adapts faster than standard MoE on synthetic drift."
- **Claim that was weakened today:** e.g., "Scoreboard memory alone is not sufficient — queue memory needed."

---

#### 11. Reviewer pre-emption

| Likely reviewer concern | Your planned response |
|-------------------------|-----------------------|
| "Memory adds parameters — is it fair?" | |
| "Why not just use a larger router?" | |
| "Does this work on real data beyond MNIST?" | |
| "What is the compute overhead of memory updates?" | |
| "Could load balancing alone achieve the same effect?" | |
| "Is the non-stationary benchmark too artificial?" | |
| "Is 2 experts enough to show the effect?" | |
| "Are results statistically significant?" | |

---

#### 12. Next session plan

- **Next immediate task:**
- **Blocking issue (if any):**
- **Reading needed before next session:**
- **Estimated time needed:**

---

#### 13. Quick health check

| Check | Status |
|-------|--------|
| Code committed to Git? | Yes / No |
| At least one graph saved? | Yes / No |
| At least one number recorded? | Yes / No |
| Research log updated? | Yes (you're reading it) |
| Paper note written? | Yes / No |
```

---

## Active Log Entries

- [**Entry #1 (12 June 2026)**: Repository Setup and Environment Verification](file:///Users/bidwot/Desktop/moe/logs/entry_001_setup.md)

---

## Cumulative experiment tracker

Update this table after every experiment session. This becomes your results section.

| # | Date | Model | Dataset | Key metric | Value (mean ± std) | Seeds | Better than baseline? | Notes |
|---|------|-------|---------|------------|-------------------|-------|-----------------------|-------|
| 1 | | Dense baseline | Synthetic | Accuracy | | | — | |
| 2 | | Standard MoE | Synthetic | Accuracy | | | — | |
| 3 | | Scoreboard Memory MoE | Synthetic | Accuracy | | | | |
| 4 | | Queue Memory MoE | Synthetic | Accuracy | | | | |
| 5 | | Contextual Memory MoE | Synthetic | Accuracy | | | | |
| 6 | | Standard MoE | MNIST drift | Accuracy | | | — | |
| 7 | | Memory MoE (best) | MNIST drift | Accuracy | | | | |

---

## Ablation summary table

Fill as ablations are run (Week 8).

| Ablation | Values tested | Best value | Effect on accuracy | Effect on adaptation speed |
|----------|---------------|------------|-------------------|---------------------------|
| alpha | 0, 0.01, 0.05, 0.1, 0.2 | | | |
| Memory decay | 0.8, 0.9, 0.95, 0.99 | | | |
| Queue size | 10, 50, 100, 200 | | | |
| Num experts | 2, 4, 8 | | | |
| Num contexts | 2, 4, 8, 16 | | | |
| Routing type | top-1, soft, epsilon-greedy | | | |

---

## Bug log

| # | Date | Bug description | How found | Fix applied | Experiments affected |
|---|------|-----------------|-----------|-------------|----------------------|
| 1 | | | | | |

---

## Decision log

Record every significant design decision so you can justify it in the paper.

| # | Date | Decision | Alternatives considered | Reason chosen |
|---|------|----------|------------------------|---------------|
| 1 | | e.g., used EMA not raw reward | Raw reward, windowed average | EMA is smoother and avoids outlier spikes | |
| 2 | | | | |

---

## Weekly summary (fill at end of each week)

### Week 0 summary

| Field | Answer |
|-------|--------|
| Week goal | Initialize repository, folder structure, verify environment and dependencies. |
| Achieved? | Yes |
| Key result this week | Git repo initialized, requirements.txt created, folder layout prepared, and conda environment maj checked. |
| Key failure this week | None |
| Biggest thing learned | Project structure separation of logs and code structure layout. |
| Paper progress | Drafted research question and hypothesis in README.md. |
| Commits this week | Initial commits for structure, requirements, README, and logs reorganization. |
| Hours spent | 0.5 |
| Next week's goal | Implement synthetic data generation and MLP baseline. |

---

*Log started: 12 June 2026. Last updated: 12 June 2026.*
