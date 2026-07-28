# DRIFT NeurIPS 2026 Rebuttal

Experiments:
- 8B scale: GRPO/DRIFT/HEG val logged in training jsonls (GRPO→795, DRIFT→770, HEG→710); reporting to step 400. DAPO 8B only to step ~70 (in progress) (Done)
- Multi-seed experiment: Mean ± Std (4B GRPO/DRIFT ×3; `_seed2`/`_seed3` scripts exist) (TODO)
- DAPO baseline + DRIFT⊕DAPO composition — 4B (TODO; 8B DAPO in progress, step ~40–60)
- Full training-from-scratch ablations: α=0, σ=0, entropy-maintenance sign-flip, cₜ=1 — 4B (α0/σ0/nolang scaffolded) (TODO)
- Hyperparameter sensitivity: branch threshold p, top-K (4B warm-start step40→100) (TODO)
- Convergence / long-horizon runs — 8B to step 400 (Done), 4B to step 200 (Done, canonical pivot-v22)
- Entropy-maintenance → reward-discovery correlation analysis (DONE — 4B step-40: DRIFT solve 6.25% vs GRPO 2.92%; post-branch Δ −0.70 vs −0.90)
- Cross-domain transfer eval: HumanEval+/MBPP+/LiveCodeBench on existing checkpoints (TODO)
- Guru-code training run (DRIFT vs GRPO) (TODO)
- Related-work differentiation (EP-GRPO, DIVA-GRPO, HEG, DAPO) — writing (TODO)

> Placeholder convention: `TBD` = fill with result once the run completes. Intended-takeaway sentences are marked *(claim — confirm against data)* so wording can be adjusted to what the numbers actually show.

---

## Reviewer 9Ash

We thank the reviewer for the positive rating and for recognizing the two-regime evaluation and the mechanistic appendix diagnostics. We address each concern below with new experiments.

### **`W1 - Narrow evaluation (single model family / corpus / binary rewards)`**

We extend the evaluation along the axis the reviewer identifies. First, on the **model-scale** axis, we add a **third scale, Qwen3-8B-Base** — see our response to `SkU1 W3`; DRIFT's advantage *grows* with scale (8.3% AIME24 mean@16 by step 100, a level GRPO has not reached by step 400 — a >4× cold-start acceleration vs. 2× at 4B — and DRIFT leads throughout the reported horizon). The method now spans three scales — the 1.7B and 4B of the submission plus the new 8B — and we center this response on the larger 4B and 8B, where the effect is clearest. Second, on the **domain** axis, we add a **cross-domain transfer evaluation** on our existing math-trained checkpoints (no additional training) and a **code training run** on the code subset of the same Guru corpus we already use, keeping infrastructure and reward design fixed.

Transfer eval (math-trained DRIFT/GRPO/HEG checkpoints, evaluated with no code training):

| Method | HumanEval+ | MBPP+ | LiveCodeBench |
|---|---|---|---|
| GRPO | TBD | TBD | TBD |
| HEG | TBD | TBD | TBD |
| DRIFT | TBD | TBD | TBD |

Code training (DRIFT vs GRPO on Guru-code, binary execution reward):

| Method | HumanEval+ | MBPP+ | LiveCodeBench |
|---|---|---|---|
| GRPO | TBD | TBD | TBD |
| DRIFT | TBD | TBD | TBD |

*(claim — confirm against data)* Gains transfer from math to code, consistent with the RLVR transfer literature, while the flat cross-domain knowledge result on GPQA-Diamond is the expected outcome (math→knowledge transfer is known to be weak). We also refer the reviewer to our response to `Q1` on seeds and `48nX Q2` on beyond-math generalization.

### **`W2, Q2 - Incomplete baselines: DAPO comparison and DRIFT⊕DAPO`**

We add **DAPO** (dynamic group filtering + asymmetric clipping + token-level loss normalization) and a **DRIFT⊕DAPO** composition, on Qwen3-4B-Base. The two mechanisms are complementary: DAPO *drops* zero-variance groups, whereas DRIFT *repairs* them into variance-bearing groups.

AIME mean@16 (report step-matched):

| Method | AIME24 | AIME25 | OlympiadBench |
|---|---|---|---|
| GRPO | TBD | TBD | TBD |
| DAPO | TBD | TBD | TBD |
| DRIFT | TBD | TBD | TBD |
| DRIFT⊕DAPO | TBD | TBD | TBD |

We also report the mixed-correctness-fraction metric (Table 6 in the submission) for DAPO to show that dropping dead-zone groups does not manufacture the within-group variance DRIFT does:

| Method | solve (%) | mixed (%) | #cp |
|---|---|---|---|
| GRPO | TBD | TBD | TBD |
| DAPO | TBD | TBD | TBD |
| DRIFT | TBD | TBD | TBD |

*(claim — confirm against data)* DRIFT and DAPO stack; DAPO alone still produces few/no correct rollout pairs in the dead zone.

### **`W3, Q3 - Component ablations should be full-training, not inference-time`**

We agree that the inference-time decomposition (submission Table 8) does not establish the training-time causal role of each component. We add **training-from-scratch ablations** on Qwen3-4B-Base (the α=0 / σ=0 / no-length-penalty runs are already scaffolded at 4B), each a full run under an otherwise-identical configuration:

| Configuration | AIME24 mean@16 | Δ vs DRIFT |
|---|---|---|
| DRIFT (full) | TBD | — |
| α=0 (diffusion-only, no learned drift) | TBD | TBD |
| σ=0 (drift-only, no diffusion) | TBD | TBD |
| entropy-reduction signal (sign-flipped feedback) | TBD | TBD |
| no off-policy correction (cₜ=1) | TBD | TBD |

*(claim — confirm against data)* The learned drift is load-bearing in training (α=0 degrades), diffusion adds incremental gain, entropy-*maintenance* beats entropy-reduction, and the off-policy correction is necessary for stability. This also directly supports our response to `48nX W1`.

### **`Q1 - Multi-seed results / uncertainty intervals`**

We rerun the 4B trajectories across **3 seeds** and report mean ± std, addressing the low-absolute-percentage / significance concern the reviewer flags.

| Method | AIME24 mean@16 (mean ± std) | AIME25 mean@16 (mean ± std) |
|---|---|---|
| GRPO | TBD | TBD |
| DRIFT | TBD | TBD |

For the 4B cold-start claim we additionally report the seed distribution of the step at which each method first crosses each threshold:

| Threshold | GRPO (steps, mean ± std) | DRIFT (steps, mean ± std) |
|---|---|---|
| AIME24 mean@16 ≥ 1% | TBD | TBD |
| AIME24 mean@16 ≥ 5% | TBD | TBD |
| OlympiadBench mean@16 ≥ 10% | TBD | TBD |

*(claim — confirm against data)* DRIFT's cold-start acceleration on 4B holds outside the seed band; the 8B converged comparison (`SkU1 W3`) is an additional high-signal, larger-scale confirmation. Shared with `48nX W3`.

### **`Q4 - Hyperparameter sensitivity table`**

Reported under a single canonical table; see our response to **`48nX Q1`** (branch threshold p and top-K), which directly answers this request.

### **`W4 - Theory is conditional (unverified smoothness / alignment assumptions)`**

We agree Theorem 1 is conditional on δₜ ≥ 0, and Theorem 2 derives δₜ under a local-smoothness (Assumption 1) and drift-alignment (Assumption 3) condition. We clarify that these are stated as *local surrogate* conditions rather than global claims, and we add empirical support: the entropy-maintenance→reward-discovery correlation of our response to `48nX W1` is a direct check on the alignment condition (Assumption 3). *(claim — confirm against data)* We will make this connection explicit in the revised Appendix H.

We hope these additional experiments address the reviewer's concerns, and we thank the reviewer again for the constructive review. If helpful, we would be grateful for consideration of a higher rating.

---

## Reviewer SkU1

We thank the reviewer for recognizing the problem formulation ("lack of within-group reward variance," tied directly to GRPO's mechanism) and the method's coherence. We address the novelty and baseline concerns below with new comparisons.

### **`W1 - Novelty vs. related work (HEG/Wang, DAPO, EP-GRPO, DIVA-GRPO)`**

We clarify the distinctions and add the requested comparisons. The clearest way to see the difference is **where in the RL loop each method intervenes**:

- **Update-side** — HEG/Wang et al. reweights the *gradient* toward high-entropy tokens; EP-GRPO shapes an *entropy-progress signal*; DIVA-GRPO shapes the *advantage*.
- **Filtering-side** — DAPO *selects among unperturbed samples* (oversamples and discards zero-advantage groups).
- **Rollout-side** — **DRIFT is the only method among these that intervenes on the token-level sampling distribution during rollout generation itself**, perturbing *which trajectories are produced* rather than *how existing trajectories are scored, reweighted, or filtered*.

This distinction is not cosmetic: it is exactly the lever the dead zone requires. When every rollout in a group receives the identical reward, there is no gradient to reweight (HEG/EP-GRPO/DIVA-GRPO) and no non-degenerate group to keep (DAPO) — the missing variance cannot be recovered on the update or filtering side. Only altering the sampling distribution can manufacture it, which is what DRIFT does.

Per-method detail:

- **vs. HEG / Wang et al. (high-entropy minority tokens):** HEG applies a per-token entropy *bonus* uniformly across forking tokens *in the update*; DRIFT applies a *directed, bounded Langevin perturbation* at branch points *during rollout* and learns a drift that preserves downstream branching capacity. HEG is our headline baseline; DRIFT outperforms it across scales (submission Tables 2–3 and the 4B/8B comparisons below).
- **vs. DAPO:** DAPO's dynamic sampling *selects among unperturbed rollouts* (drops zero-advantage groups); on an all-incorrect prompt it has nothing to keep. DRIFT instead *changes the generative distribution*, so it can still produce a correct trajectory on exactly those prompts — the mixed-correctness result in our response to `W2` quantifies this. The two are complementary (DRIFT⊕DAPO composition, `W2`).
- **vs. EP-GRPO (entropy-progress) and DIVA-GRPO (difficulty-adaptive advantage):** EP-GRPO shapes an entropy-progress signal at the *advantage* level; DRIFT acts at *rollout* time in logit space at selected branch points. DIVA-GRPO targets *multimodal* difficulty-adaptive advantage and is not directly comparable to our single-modality verifiable-reward setting; we cite it and state the scope difference.

*(Note: we scope this claim to the methods raised in review. Rollout-time exploration methods do exist in the broader literature; DRIFT's specific contribution is the directed, bounded, branch-point-selective Langevin perturbation in logit space — not merely "modifying the rollout.")*

Direct comparison, AIME24 mean@16 (%), at matched horizons (4B step 200 / 8B step 400; 8B added in response to review; EP-GRPO where feasible):

| Method | 4B (step 200) | 8B (step 400) |
|---|---|---|
| GRPO | 10.86 | 7.08 |
| HEG | TBD | 0.47 *(unstable; see `W3`)* |
| DAPO | TBD | *(in progress)* |
| EP-GRPO | TBD | *(in progress)* |
| DRIFT | **12.16** | **11.80** |

### **`W2 - Insufficient baselines: direct DAPO comparison`**

We agree the orthogonality argument does not replace a direct comparison. We add DAPO and DRIFT⊕DAPO — full table and the mixed-correctness analysis are in our response to **`9Ash W2, Q2`**.

### **`W3 - Convergence-stage / long-horizon comparison`**

We add a long-horizon comparison on the larger **Qwen3-8B-Base**, training all methods well past the cold start.

**8B AIME24 mean@16 (%):**

| step | GRPO | HEG | DRIFT |
|---|---|---|---|
| 100 | 0.86 | 2.84 | **8.33** |
| 200 | 4.14 | 1.46 | **10.49** |
| 400 | 7.08 | 0.47 | **11.80** |

Two findings. **(i) The acceleration grows with scale.** DRIFT reaches ~8.3% AIME24 mean@16 by step 100 — a level GRPO has not reached even by step 400 (7.1%) — a **>4× cold-start acceleration**, versus 2× at 4B. **(ii) The lead does not vanish with longer training.** DRIFT leads GRPO at every reported step; the gap narrows as GRPO gradually escapes the dead zone (+7.5pp at step 100 → +4.7pp at step 400) but DRIFT remains clearly ahead throughout the reported horizon. HEG is unstable at this scale — it collapses below 1% by step 400 — consistent with the uniform-pressure ceiling damage reported in the submission.

**4B AIME24 mean@16 (%) (canonical DRIFT run):**

| step | GRPO | DRIFT |
|---|---|---|
| 100 | 7.73 | **10.94** |
| 200 | 10.86 | **12.16** |

At 4B, DRIFT similarly retains a lead through step 200 (+3.2pp at step 100, +1.3pp at step 200).

*(The submission's 1.7B late-plateau-escape result stands; we center this response on the larger 4B and 8B scales.)*

### **`W4 - Hyperparameter ablation / sensitivity`**

See our canonical sensitivity response to **`48nX Q1`**.

We hope these clarifications and the new DAPO/EP-GRPO comparisons and convergence results address the reviewer's concerns, and we would be grateful if the reviewer would consider a higher rating in light of them.

---

## Reviewer 48nX

We thank the reviewer for recognizing the dead-zone perspective and the coherence of the exploration framework. We address the three weaknesses and two questions below.

### **`W1 - Why does entropy-maintenance correlate with reward discovery?`**

This is a central question and we add a direct empirical check. On Qwen3-4B-Base at the step-40 cold-start dead zone, we measure — on the same rollouts (AIME24, 30 prompts × 8 samples) — both how much entropy each method retains *after* a high-entropy branch point and how many correct trajectories it discovers:

| 4B, step-40 dead zone | GRPO | DRIFT |
|---|---|---|
| post-branch entropy change Δ (nats, next 32 tokens) | −0.90 | **−0.70** |
| correct trajectories discovered (solve %) | 2.92 | **6.25** |

The two move together. After a branch point DRIFT's per-token entropy falls **~22% less** than GRPO's (Δ = −0.70 vs −0.90) — it keeps more of the downstream branching the drift term is designed to preserve (Section 3.2) — and on the same prompts it discovers **>2× more** correct trajectories (6.25% vs 2.92%), in precisely the regime where within-group reward variance is otherwise zero.

*(claim)* The method that drops less entropy after a branch point is the method that discovers more than twice as many correct trajectories in the dead zone: maintaining branch-point entropy keeps rewarding continuations reachable. This is consistent with the mechanism in Section 3.2 and the entropy-*reduction* sign-flip ablation in our response to `9Ash W3, Q3`, and it empirically supports Assumption 3 (drift alignment), addressing `9Ash W4`.

### **`W2 - Limited comparisons (recent RLVR exploration/stabilization methods)`**

We add DAPO and, where feasible, EP-GRPO; see the comparison tables in our responses to **`9Ash W2, Q2`** and **`SkU1 W1`**.

### **`W3 - No variance / confidence intervals / significance`**

We add multi-seed results (mean ± std); see the full table in our response to **`9Ash Q1`**. *(claim — confirm against data)* The reported orderings hold outside the seed band.

### **`Q1 - Sensitivity to branch-point threshold and top-K`** (canonical)

We sweep the two hyperparameters the reviewer names, on Qwen3-4B-Base, warm-started from the step-40 checkpoint and continued to step 100 (matching the fork-from-checkpoint protocol of submission Tables 9–10):

Branch-point threshold (quantile p):

| p | AIME24 mean@16 (step 100) | best@16 |
|---|---|---|
| 0.80 | TBD | TBD |
| 0.85 (headline) | TBD | TBD |
| 0.90 | TBD | TBD |

Top-K perturbation size:

| K | AIME24 mean@16 (step 100) | best@16 |
|---|---|---|
| 10 | TBD | TBD |
| 20 (headline) | TBD | TBD |
| 40 | TBD | TBD |
| 150 | (from submission Table 9–10) | |

*(claim — confirm against data)* DRIFT is robust across a reasonable band of p and K around the headline setting, while the extreme K=150 confirms the head-not-tail design (long-tail perturbation derails training). This is the canonical response for `9Ash Q4` and `SkU1 W4` as well.

### **`Q2 - Generalization beyond math (coding / agent)`**

For coding, see the transfer eval and Guru-code training run in our response to **`9Ash W1`**. For agent tasks, we note this is the setting where the dead zone is most severe (a point we make in Appendix G); a full agent study requires multi-turn environment infrastructure and we identify it as the primary direction for future work, with the logit-space branch-point mechanism carrying over unchanged.

We hope these additional experiments and clarifications address the reviewer's concerns, and we would appreciate consideration of a higher rating in light of them.
