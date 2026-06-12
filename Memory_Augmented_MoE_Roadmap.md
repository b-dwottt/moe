# Memory-Augmented MoE Roadmap
**Complete Learn-by-Building Roadmap**
Contextual Expert Memory for Adaptive Mixture-of-Experts Routing
*A practical research pathway from zero setup to paper draft*

Prepared: 11 June 2026

> **Core idea in one sentence**
> Build a Mixture-of-Experts model whose router does not only look at the current input, but also consults expert-specific memory of recent success, failure, stability, and context.

> **Important principle for this roadmap**
> Do not learn everything first and then build later. Every week combines learning, coding, testing, logging results, and writing a small part of the final paper.

**Project working title:** Memory-Augmented Routing for Adaptive Mixture-of-Experts Models

---

## Table of Contents

1. Project overview
2. What you are trying to prove
3. What you need to know while building
4. Tools, environment and repository setup
5. The learn-by-building method
6. Full 12-week roadmap
7. Model designs to implement
8. Experiment plan and metrics
9. Baselines and ablations
10. Common problems and fixes
11. Paper writing plan
12. Final deliverables checklist
13. Reading plan and references

---

## 1. Project Overview

This roadmap is for building a research prototype called Contextual Expert Memory for Adaptive MoE Routing. The roadmap assumes you start from scratch, but it is designed so that you learn each concept only when you need it for the next coding step.

### 1.1 Simple explanation

A standard Mixture-of-Experts (MoE) model has several expert networks and a router. The router decides which expert should process each input. Most routers are mostly stateless: they use the current input and learned parameters, but they do not keep an explicit memory of which expert recently worked well or badly.

Your research idea is to give each expert a small memory. The router then combines the current input with this memory before choosing the expert. This should help the model adapt faster when tasks change over time.

| Component | Plain meaning | In this project |
|-----------|---------------|-----------------|
| Expert | A small neural network | MLP, CNN, or Transformer FFN expert |
| Router | Chooses which expert to use | MLP router that produces expert logits |
| Memory | Stores recent wins/losses | Moving average, queue, or context-specific expert score |
| Reward | How well expert performed | Negative loss or success score |
| Non-stationary task | Task changes over time | Rules swap every few hundred steps |

**Final research question**
Can expert-specific performance memory improve MoE routing decisions, especially under changing or non-stationary tasks?

---

## 2. What You Are Trying to Prove

### 2.1 Main hypothesis

A memory-aware MoE router should adapt faster than a standard MoE router because it remembers which experts recently succeeded or failed on similar inputs.

### 2.2 Expected contributions

1. A lightweight expert-specific memory mechanism for MoE routing.
2. A contextual memory extension where experts remember performance by input type or cluster.
3. A learn-by-building experimental benchmark for non-stationary routing.
4. Evidence that memory-aware routing improves adaptation speed and reduces poor expert assignments.
5. A clear comparison against standard MoE, load-balanced MoE, random routing, and oracle routing.

**Novelty statement**
The novelty is not simply adding memory. The novelty is using expert-specific success/failure history as an active routing signal.

---

## 3. What You Need to Know While Building

You do not need to master all topics before starting. Each topic below is learned through a coding task. Use this section as a map of what to understand as you build.

| Topic | What to learn | When you learn it |
|-------|---------------|-------------------|
| PyTorch basics | nn.Module, tensors, training loops, optimizers | When building the first MLP and standard MoE |
| MoE basics | Experts, router, softmax, top-k, load balance | When implementing StandardMoE |
| Moving averages | Exponential moving average, decay, stability | When implementing Scoreboard Memory |
| Queues | Recent loss history and variance | When implementing Queue Memory |
| Clustering | Input clusters or simple context groups | When implementing Contextual Memory |
| Bandits | State, action, reward, exploration | When implementing UCB or epsilon-greedy routing |
| Credit assignment | Which expert caused the loss? | When comparing local expert loss vs global loss |
| Experiment design | Baselines, metrics, ablation studies | Throughout all phases |
| Research writing | Method, results, limitations | Start writing from Week 1 |

---

## 4. Tools, Environment and Repository Setup

### 4.1 Minimum tools

- Python 3.10 or newer.
- PyTorch.
- Jupyter Notebook or Google Colab for quick experiments.
- VS Code or any Python editor.
- Git and GitHub for version control.
- Matplotlib and pandas for graphs/tables.
- Optional: CUDA GPU or Google Colab GPU. The first prototype can run on CPU.

### 4.2 Recommended project folder

```
memory_moe_project/
  README.md
  requirements.txt
  notebooks/
    01_synthetic_standard_moe.ipynb
    02_scoreboard_memory_moe.ipynb
    03_queue_memory_moe.ipynb
    04_contextual_memory_moe.ipynb
  src/
    data.py
    models.py
    memory.py
    train.py
    metrics.py
    plots.py
  experiments/
    synthetic_swap/
    mnist_drift/
    tiny_language_model/
  results/
    figures/
    tables/
  paper/
    outline.md
    draft.md
```

**Setup success criterion**
By the end of setup, you should be able to run a simple PyTorch MLP, save a loss curve, and push code to GitHub.

---

## 5. The Learn-by-Building Method

Every unit of work should follow the same cycle. This prevents passive learning and forces research progress from the first week.

| Step | Do this | Output |
|------|---------|--------|
| 1. Read just enough | Read only the concept needed for the next coding step. | One paragraph note in your research log. |
| 2. Implement small | Build the smallest working version first. | A runnable script or notebook cell. |
| 3. Test visibly | Plot loss, accuracy, or expert usage. | One graph saved to results/figures. |
| 4. Compare | Run one baseline against your method. | A small table. |
| 5. Write immediately | Write what happened, even if results are bad. | A few lines for the future paper. |

**Daily rule**
Every study session must produce at least one of these: code commit, graph, table, bug note, experiment result, or paper paragraph.

---

## 6. Full 12-Week Roadmap

This roadmap assumes 8 to 12 hours of work per week. If you have less time, keep the same order but extend the timeline.

| Week | Focus | Build | Learn while doing | Deliverable |
|------|-------|-------|-------------------|-------------|
| 0 | Setup and problem framing | Install tools, create repository, write research question | PyTorch environment, Git workflow, MoE idea | Repo + one-page project note |
| 1 | Standard MLP and synthetic data | Create changing synthetic task and train simple MLP | Tensors, losses, training loop | Synthetic data notebook + baseline graph |
| 2 | Standard MoE | Build 2-expert MoE without memory | Router, softmax, top-k, expert usage | Standard MoE working on synthetic task |
| 3 | Scoreboard memory | Add one memory score per expert | Moving average, reward = negative loss | Memory-MoE v1 + adaptation graph |
| 4 | Queue memory | Store last N losses per expert | Mean, variance, stability, decay | Memory-MoE v2 + queue ablation |
| 5 | Contextual memory | Add memory by expert and input cluster | Clustering and context-conditioned routing | Memory[e, cluster] prototype |
| 6 | Clean experiment system | Turn notebooks into reusable scripts | Experiment reproducibility | src/ training pipeline |
| 7 | MNIST/FashionMNIST drift | Test on changing image tasks | CNN/MLP experts and task shifts | Image benchmark results |
| 8 | Baselines and ablations | Compare against random, standard, load-balanced, oracle | Fair evaluation and ablation logic | Main result table |
| 9 | Bandit routing | Try epsilon-greedy or UCB memory router | Contextual bandits, exploration bonus | Bandit-MoE comparison |
| 10 | Tiny language model | Try 2-layer Transformer or character model with experts | Perplexity, language modelling basics | Small language experiment |
| 11 | Analysis and failure cases | Study expert collapse, memory bias, adaptation speed | Interpretation and limitations | Analysis figures |
| 12 | Paper draft | Write full research paper and clean code | Research writing and packaging | Paper draft + GitHub-ready code |

### 6.1 Detailed weekly guidance

#### Week 0: Setup and framing

| Build now | Learn while doing | Deliverable |
|-----------|-------------------|-------------|
| Create GitHub repository and local project folder. Install Python, PyTorch, pandas, matplotlib, scikit-learn, tqdm. Write the research question and hypothesis in README.md. Create an experiment log file called research_log.md. | How to run Python scripts/notebooks. How to save graphs and tables. How to use Git commit messages. | Repo created, environment working, first commit done. |

#### Week 1: Synthetic data and first baseline

| Build now | Learn while doing | Deliverable |
|-----------|-------------------|-------------|
| Create a function that generates input vectors x and labels y. Create a rule that changes every 500 steps. Train a simple dense MLP on the task. Plot loss and accuracy over time. | Tensors, DataLoader, training loop, loss curve. Concept drift: the data rule changes over time. | Notebook 01 with synthetic dataset and dense baseline. |

#### Week 2: Standard MoE without memory

| Build now | Learn while doing | Deliverable |
|-----------|-------------------|-------------|
| Build two expert MLPs. Build router MLP that outputs logits for each expert. Route each sample to top-1 expert. Track how often each expert is used. | Softmax logits, top-k routing, expert usage, expert collapse. | Standard MoE baseline + expert usage graph. |

#### Week 3: Scoreboard memory

| Build now | Learn while doing | Deliverable |
|-----------|-------------------|-------------|
| Add memory vector M[e] for each expert. Update M[e] using exponential moving average of reward. Add alpha * M[e] to router logits. Compare adaptation after task swap. | Exponential moving average, reward shaping, routing bias. | First Memory-MoE result showing faster/slower adaptation. |

#### Week 4: Queue memory

| Build now | Learn while doing | Deliverable |
|-----------|-------------------|-------------|
| Replace single score with queue of recent losses per expert. Calculate mean loss, standard deviation, and trend. Use these as memory bonus or penalty. Run ablation: no memory vs scoreboard vs queue. | Stability, variance, moving windows, memory decay. | Queue-Memory MoE and ablation table. |

#### Week 5: Contextual memory

| Build now | Learn while doing | Deliverable |
|-----------|-------------------|-------------|
| Cluster inputs into contexts using simple k-means or rule-based bins. Store Memory[expert, context]. Use context-specific memory bonus for routing. Compare with global memory. | Why one global score is weak; context-specific success/failure. | Contextual Expert Memory model. |

#### Week 6: Reusable experiment pipeline

| Build now | Learn while doing | Deliverable |
|-----------|-------------------|-------------|
| Move code from notebooks into src/. Add command-line experiment configs. Save metrics to CSV. Create plot scripts. | Reproducibility, experiment logging, random seeds. | A clean training pipeline that can rerun experiments. |

#### Week 7: MNIST/Fashion-MNIST task drift

| Build now | Learn while doing | Deliverable |
|-----------|-------------------|-------------|
| Train MoE on MNIST variants. Create task shifts: normal digits, noisy digits, rotated digits, even/odd task. Run standard MoE and Memory-MoE. | Image classification basics, simple CNN/MLP experts. | First non-synthetic benchmark results. |

#### Week 8: Baselines and ablations

| Build now | Learn while doing | Deliverable |
|-----------|-------------------|-------------|
| Implement random routing baseline. Implement load-balanced MoE baseline. Implement oracle routing for synthetic data. Run ablation for alpha, memory decay, queue size, cluster count. | Fair comparison, ablation study design, hyperparameter sensitivity. | Main results table. |

#### Week 9: Bandit routing

| Build now | Learn while doing | Deliverable |
|-----------|-------------------|-------------|
| Implement epsilon-greedy routing. Implement UCB-style expert bonus. Compare bandit memory to learned router memory. Measure exploration vs exploitation. | Contextual bandits, uncertainty, reward estimates. | Bandit-MoE comparison section. |

#### Week 10: Tiny language experiment

| Build now | Learn while doing | Deliverable |
|-----------|-------------------|-------------|
| Use character-level Shakespeare or TinyStories subset. Build small Transformer or RNN with expert FFN layer. Track perplexity and expert usage. Keep model small. | Language modelling basics, perplexity, sequence batches. | Optional but strong extra experiment. |

#### Week 11: Deep analysis

| Build now | Learn while doing | Deliverable |
|-----------|-------------------|-------------|
| Plot adaptation speed after each task change. Check expert collapse and memory bias. Analyse cases where memory hurts. Write limitations honestly. | Research interpretation, error analysis, limitations. | Analysis figures and limitations section. |

#### Week 12: Paper and final packaging

| Build now | Learn while doing | Deliverable |
|-----------|-------------------|-------------|
| Write abstract, introduction, method, experiments, results, limitations, conclusion. Clean code and README. Prepare figures and final tables. Create final research poster or slides later if needed. | Academic writing, reproducibility, contribution framing. | Complete paper draft and project repository. |

---

## 7. Model Designs to Implement

### 7.1 Standard MoE baseline

The standard MoE is the reference model. It has experts and a router, but no memory signal.

```python
base_logits = router(x)         # shape: [batch, num_experts]
expert_id = top1(base_logits)   # choose expert
output = expert[expert_id](x)   # expert prediction
loss = criterion(output, y)
```

### 7.2 Scoreboard Memory MoE

Each expert has one memory score. The router adds this score to its normal logits.

```python
base_logits = router(x)
final_logits = base_logits + alpha * memory_score
expert_id = top1(final_logits)
loss = criterion(output, y)
reward = -loss.detach()
memory_score[e] = decay * memory_score[e] + (1 - decay) * reward
```

> **Use continuous reward first**
> Do not use only success/fail at the beginning. Use reward = negative loss because it gives smoother memory updates.

### 7.3 Queue Memory MoE

Each expert stores a queue of recent losses. The router uses mean loss and stability as a signal.

```python
expert_loss_queue[e].append(local_loss)
mean_loss = average(expert_loss_queue[e])
stability = standard_deviation(expert_loss_queue[e])
memory_bonus[e] = -mean_loss - beta * stability
final_logits = router(x) + alpha * memory_bonus
```

### 7.4 Contextual Expert Memory MoE

This is the strongest version. Each expert has memory for each input type or context cluster.

```python
context_id = cluster_or_bin(x)
bonus_for_input = memory_table[:, context_id]
final_logits = router(x) + alpha * bonus_for_input
chosen_expert = top1(final_logits)
memory_table[chosen_expert, context_id] = update_with_recent_reward(...)
```

> **Main final model**
> Your final research model should be Contextual Expert Memory MoE, not only simple Scoreboard Memory MoE.

### 7.5 Bandit Memory Router

Once simple memory works, treat routing as a contextual bandit: the router chooses an expert, observes a reward, and updates expert memory. Start with epsilon-greedy or UCB before trying REINFORCE.

---

## 8. Experiment Plan and Metrics

### 8.1 Synthetic task

- Input: random vector x.
- Rule 1: if sum(x) > 0, one expert should be better; otherwise another expert should be better.
- Rule swap: every 500 steps, change which pattern is best.
- Purpose: test whether memory helps the router adapt after task changes.

### 8.2 MNIST/Fashion-MNIST drift task

- Task A: normal classification.
- Task B: noisy images.
- Task C: rotated images.
- Task D: even/odd classification.
- Purpose: test whether memory works beyond synthetic data.

### 8.3 Tiny language task

- Use a tiny character-level model or small Transformer.
- Replace the feed-forward layer with small experts.
- Track validation perplexity and expert usage.
- Keep this optional if time or compute is limited.

| Metric | Meaning | Desired direction |
|--------|---------|-------------------|
| Accuracy | Classification correctness | Higher is better |
| Loss | Training/validation error | Lower is better |
| Adaptation speed | Steps needed to recover after task switch | Lower is better |
| Expert usage entropy | How evenly experts are used | Medium/high is usually better |
| Expert collapse rate | How often one expert dominates | Lower is better |
| Recovery area | Performance area after drift point | Higher is better |
| Perplexity | Language model quality | Lower is better |

---

## 9. Baselines and Ablations

| Model | Why needed |
|-------|------------|
| Dense model | Shows whether MoE is actually useful. |
| Standard MoE | Main baseline: router has no memory. |
| Standard MoE + load balancing loss | Checks whether memory is better than common balancing tricks. |
| Random routing | Sanity check. Your model must beat this. |
| Oracle routing | Upper bound for synthetic task only. |
| Scoreboard Memory MoE | Simple memory baseline. |
| Queue Memory MoE | Tests whether recent loss history helps. |
| Contextual Memory MoE | Main proposed model. |

### 9.1 Ablation studies

- Memory weight alpha: 0, 0.01, 0.05, 0.1, 0.2.
- Memory decay: 0.8, 0.9, 0.95, 0.99.
- Queue size: 10, 50, 100, 200.
- Number of experts: 2, 4, 8.
- Number of contexts/clusters: 2, 4, 8, 16.
- Routing type: top-1, soft routing, epsilon-greedy.

---

## 10. Common Problems and Fixes

| Problem | Symptom | Fix |
|---------|---------|-----|
| Expert collapse | One expert receives almost all inputs. | Add exploration bonus, load balancing, entropy regularisation, or minimum routing probability. |
| Memory bias | An expert fails early and is avoided forever. | Use decay, queue reset, exploration, or uncertainty bonus. |
| No improvement over standard MoE | Memory curve is similar to baseline. | Increase task drift, tune alpha, use contextual memory instead of global memory. |
| Unstable training | Loss jumps or becomes NaN. | Lower learning rate, lower alpha, clip gradients, normalise memory bonus. |
| Cheating with task labels | Model uses explicit task ID unfairly. | Use input clusters or embeddings for context in harder experiments. |
| Bad credit assignment | You cannot tell which expert caused loss. | Use local expert loss first; later test bandit/REINFORCE methods. |

---

## 11. Paper Writing Plan

Start writing from Week 1. Do not wait until all experiments are complete.

| Paper section | Write when | What to include |
|---------------|------------|-----------------|
| Abstract | Week 12 | Problem, method, datasets, main results, contribution. |
| Introduction | Weeks 1-3 | MoE routers are mostly stateless; expert memory may improve adaptation. |
| Related work | Weeks 2-8 | MoE routing, load balancing, expert choice, MPI router, memory systems, bandits. |
| Method | Weeks 3-6 | Scoreboard memory, queue memory, contextual memory, formulas. |
| Experiments | Weeks 6-10 | Datasets, baselines, metrics, hyperparameters. |
| Results | Weeks 8-11 | Tables and figures. Explain where memory helps or fails. |
| Discussion | Week 11 | Expert collapse, memory bias, credit assignment, limitations. |
| Conclusion | Week 12 | Summary and future work. |

### 11.1 Suggested abstract draft

Current Mixture-of-Experts models route tokens using input-dependent router scores, but these routers typically do not maintain explicit expert-level memory of recent success or failure. This project proposes a memory-augmented routing mechanism in which each expert maintains lightweight performance memory that is used as an additional signal during routing. We evaluate scoreboard memory, queue-based memory, and contextual expert memory under non-stationary task shifts. The aim is to test whether expert-specific memory improves adaptation speed, routing stability, and expert utilisation compared with standard MoE baselines.

---

## 12. Final Deliverables Checklist

| Deliverable | Minimum requirement | Done? |
|-------------|---------------------|-------|
| Code repository | Clean README, requirements, src folder, notebooks, results folder. | ☐ |
| Synthetic benchmark | Standard MoE and Memory-MoE tested under task swaps. | ☐ |
| Contextual memory model | Memory table by expert and context implemented. | ☐ |
| Baseline comparison | Dense, random, standard MoE, load-balanced MoE, oracle where possible. | ☐ |
| Ablation study | alpha, decay, queue size, contexts, experts. | ☐ |
| Graphs | Loss, accuracy, adaptation speed, expert usage. | ☐ |
| Paper draft | Abstract, intro, related work, method, experiments, results, limitations. | ☐ |
| Reproducibility | Random seeds, saved config, experiment logs. | ☐ |

---

## 13. Reading Plan and References

Read papers only when they directly help the next build step. The first three sources are the uploaded papers that inspired this roadmap.

| When | Read/study | Why |
|------|------------|-----|
| Week 0-2 | Redesign Mixture-of-Experts Routers with Manifold Power Iteration | Understand why router design matters and how expert-aware routing can improve MoE. |
| Week 1-3 | Sparsely-Gated Mixture-of-Experts / Switch Transformer | Understand classical MoE routing and top-k gating. |
| Week 3-5 | PROJECTMEM | Understand event-sourced memory and why explicit memory can guide future actions. |
| Week 4-8 | BASE Layers | Understand routing as a balanced assignment problem. |
| Week 4-8 | Expert Choice Routing | Understand expert-side token selection and load balance. |
| Week 5-9 | Contextual bandit tutorials/papers | Understand state, action, reward, exploration and credit assignment. |
| Week 8-11 | Toward Trustworthy AI: Multi-Target Adversarial Attacks and Robust Defenses for Continuous Data Summarization | Understand robustness, task shift, evaluation under perturbation, and downstream reliability. |

### 13.1 References

- Lian, Y., Guo, L., Zhao, Z., Lu, Z., Cai, Y., Pang, S., Xu, D., & Xue, J. (2026). Toward Trustworthy AI: Multi-Target Adversarial Attacks and Robust Defenses for Continuous Data Summarization. arXiv:2606.11804v1.
- Malo, R. C., & Qiu, T. (2026). PROJECTMEM: A Local-First, Event-Sourced Memory and Judgment Layer for AI Coding Agents. arXiv:2606.12329v1.
- Wu, S., Lv, A., Xie, R., & Lin, Y. (2026). Redesign Mixture-of-Experts Routers with Manifold Power Iteration. arXiv:2606.12397v1.
- Fedus, W., Zoph, B., & Shazeer, N. (2021). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. arXiv:2101.03961.
- Lewis, M., Bhosale, S., Dettmers, T., Goyal, N., & Zettlemoyer, L. (2021). BASE Layers: Simplifying Training of Large, Sparse Models. arXiv:2103.16716.
- Zhou, Y., Lei, T., Liu, H., Du, N., Huang, Y., Zhao, V., Dai, A., Chen, Z., Le, Q., & Laudon, J. (2022). Mixture-of-Experts with Expert Choice Routing. arXiv:2202.09368.

---

## Appendix A: First Prototype Mini-Spec

Use this mini-spec for the first working prototype. Do not add complexity until this works.

| Item | Decision |
|------|----------|
| Dataset | Synthetic 10-dimensional input vector. |
| Experts | 2 MLP experts, each with one hidden layer. |
| Router | Small MLP that outputs 2 logits. |
| Routing | Top-1 routing. |
| Memory | Scoreboard memory: one scalar per expert. |
| Memory update | Exponential moving average of reward = negative local loss. |
| Task drift | Swap decision rule every 500 steps. |
| Main metric | Adaptation speed after each swap. |
| Baseline | Standard MoE with no memory. |
| Success condition | Memory-MoE recovers accuracy faster after swaps in at least 3 random seeds. |

---

## Appendix B: Research Log Template

```
Date:
Experiment name:
Question:
Model version:
Dataset/task:
Settings changed:
Result summary:
Graph/table saved:
What worked:
What failed:
Next action:
Paper note:
```
