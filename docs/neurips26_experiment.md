# NeurIPS 2026 — DRIFT Rebuttal Experiment Plan

Submission 19493 · "Learned DRIFT on the Logit Manifold for LLM Reasoning Exploration"

Internal naming: **DRIFT = `pivot_v22`**. Baselines: `grpo` (vanilla), `*_high_ent` (HEG), `*_dapo` (DAPO).

## Current standing

| Reviewer | Rating | Conf | Quality | Orig | Signif |
|---|---|---|---|---|---|
| 9Ash | **4** borderline accept | 3 | 3 | 3 | 3 |
| SkU1 | **3** borderline reject | 4 | 2 | 2 | 3 |
| 48nX | **3** borderline reject | 3 | 3 | 2 | 2 |

Overall borderline reject. The most negative reviewer (SkU1, confidence 4) is driven almost entirely by **missing baselines + missing ablations/sensitivity** — that is where the leverage is.

## Measured compute costs (this cluster, 8×H200 = 1 node)
From checkpoint mtimes on completed runs (includes eval overhead):

| Model | per-step | 60 steps | 100 steps | 150 steps |
|---|---|---|---|---|
| 1.7B | ~6.7 min | ~7 h | ~11 h | ~17 h |
| 4B | ~20 min | ~20 h | ~33 h | — |

Implications: **full runs are the expensive line-items** (seeds, DAPO — each needs a full run). **Sensitivity is cheapest** because short warm-started continuations suffice. Anything on 4B is ~3× the per-step cost of 1.7B. Sensitivity/ablation runs are embarrassingly parallel across nodes if available.

## Consensus asks (the score-movers)

| Ask | 9Ash | SkU1 | 48nX | Status |
|---|---|---|---|---|
| Seeds + CIs / significance | Q1 | (implied) | W3 | seed scripts exist for 4B |
| DAPO baseline (+ DRIFT⊕DAPO) | Q2 | W1,W2 | W2 | scripts exist |
| Full **training-time** ablations | W3,Q3 | W4 | — | α0/σ0/nolang scripts exist |
| Hyperparameter sensitivity table | Q4 | W4 | Q1 | TODO |

---

## P0 — Score-critical (do these first)

### P0-1. Multi-seed + uncertainty (all three reviewers)
The headline "exceeds the GRPO ceiling" is a ~0.5–1 pp gap on a **30-problem** AIME-25 set with no error bars. This is the single biggest liability.

- **1.7B: GRPO ×3, DRIFT ×3** — non-negotiable. Report mean ± std or bootstrap CI on every AIME trajectory.
- **4B: GRPO ×2–3, DRIFT ×2–3** — report cold-start escape as a *distribution* of "step to cross 1% / 5% / Olymp 10%" (Table 1), not a single number. The "2× faster" claim must survive seed noise.
- HEG can remain 1 seed if compute-bound (state it).
- Scripts: `run_qwen3_4b_base_pivot_v22_seed2.sh`, `run_qwen3_4b_base_pivot_v22_seed3.sh` (exist). Need 1.7B seed variants + GRPO seed variants.

### P0-2. DAPO baseline + composition (SkU1's central objection)
SkU1 explicitly rejects the "orthogonal, so we skip it" argument. Required:
- **DAPO** on 1.7B and 4B — `run_qwen3_4b_base_dapo.sh` (exists), need 1.7B.
- **DRIFT ⊕ DAPO** on ≥1.7B — `run_qwen3_4b_base_pivot_v22_dapo.sh` (exists), need 1.7B. Directly tests the "could be stacked" claim (paper line 296).
- Framing: DAPO *drops* dead-zone groups; DRIFT *repairs* them. Use the mixed-correctness-fraction metric (Table 6) as the weapon — DAPO still yields 0 correct pairs at 4B step 60 where DRIFT yields 105.

### P0-3. Full training-from-scratch ablations (9Ash W3/Q3)
Table 8 is **inference-time on a fixed checkpoint** → the causal claim that learned drift drives *training dynamics* is unproven. Train from scratch on 1.7B:
- **diffusion-only** (α=0) — `_alpha0.sh` (exists for 4B; add 1.7B). Isolates whether SPSA drift matters in training at all.
- **drift-only** (σ=0) — `_sigma0.sh` (exists for 4B; add 1.7B).
- **entropy-maintenance vs. entropy-reduction** — flip the signal sign/target. Also answers 48nX W1.
- **no off-policy correction** (cₜ=1).
- (optional) random-drift control.

### P0-4. Hyperparameter sensitivity (all three; 48nX Q1 asks branch-threshold + top-K verbatim)

**Chosen design — cheap warm-start on 4B (defends the headline model directly).**
Warm-start each config from the existing **4B DRIFT checkpoint at step 40** (`.../pivot-v22/qwen3_4b_base/global_step_40`), run **+60 steps → step 100** (paper's standard comparison point), `test_freq=20`. This matches the paper's own fork-from-checkpoint ablation methodology (Table 9/10), so it's defensible; frame it honestly as *continued-training (local) sensitivity from a shared checkpoint*, not from-scratch.

3 values per axis; **the center of each axis is the existing headline run → reuse it, no new run:**

| Axis (config key) | grid | new runs |
|---|---|---|
| branch threshold `trig_percentile` p | {80, **85**, 90} | 80, 90 |
| top-K `langevin_top_k` | {10, **20**, 40} | 10, 40 |

- **Center (p=85, K=20)** = existing headline curve to step 100 (reuse).
- **K=150** already done → free 4th point on the K axis (collapse at extreme K).
- **Total: 4 new runs ≈ 80 node-hours** (4B ≈ 20 min/step × 60 steps ≈ 20 h/run).
- Optional: re-run center (p=85, K=20) once under the same warm-start command (+1 run, ~20 h) for strict protocol symmetry. Likely immaterial (see below) — skip unless a reviewer is pedantic.

**Verified: no top-K resume blocker (checked `verl/utils/vllm/pivot_patch.py`).** The drift vector `_G` is init `None`, built via the γ-EMA, lives in **full-vocab `(N,V)`** coordinates with top-K applied as a **mask** (not a K-dim vector), and is **never checkpointed**. So (a) resuming at a different K cannot mismatch dimensions — K runs are viable with zero code changes; (b) `_G` re-inits on resume and re-warms within a single rollout, so reusing the existing (continuously-trained) headline curve as the center is fine.

**Secondary axes (lower priority; do on 1.7B short-horizon only if time):** positional guard t_min ∈ {0, 500, 1000}; η/σ ratio (2–3 pts); λ_len ∈ {0, 1, 2}. **λ_len doubles as the length-reward confound fix** (see below); `_nolang.sh` = λ_len 0 (exists for 4B).

---

## P1 — Strong single/double-reviewer asks

### P1-1. Convergence / long-horizon (SkU1 W3 — his most concrete objection)
SkU1 suspects the gap shrinks with longer training, "especially on 4B where the difference is almost negligible." Extend GRPO and DRIFT to convergence (1.7B → ~300–400 steps; 4B well past step 100). Either show DRIFT wins on *final* performance, or honestly reframe the 4B contribution as bootstrap acceleration and lean the "final ceiling" claim on 1.7B.

### P1-2. Entropy-maintenance → reward correlation (48nX W1)
48nX's unique technical objection: no evidence that maintaining entropy *correlates* with finding reward. Analysis experiment: correlate the maintenance signal sₜ (or H@trig) at branch points against downstream discovery probability / eventual rollout reward. A positive relationship closes W1.

---

## P2 — Breadth (expensive; addresses "narrow evaluation")

### P2-1. Second model family (9Ash, 48nX Q2)
One run on a non-Qwen base (Llama-3.2-1B/3B-Base or OLMo) to show it is not a Qwen artifact.

### P2-2. Non-math domain (48nX Q2)
Code (verifiable-reward subset) or logic. One clean transfer beats the current GPQA table, which is within-noise and arguably hurts.

*If time-boxed, defer P2 to camera-ready and promise it in the rebuttal.*

---

## P3 — Writing only, no compute (do all regardless)

### P3-1. Related-work differentiation (SkU1 W1)
Add a paragraph + small table separating DRIFT from:
- **HEG / Wang et al. 2025** (branch-point *selection* vs. their forking-token *bonus*)
- **DAPO** (Yu et al.)
- **EP-GRPO** (arXiv 2605.04960) — entropy-progress aligned GRPO
- **DIVA-GRPO** (arXiv 2603.01106) — multimodal / difficulty-adaptive advantage; argue partial non-comparability but cite and address.

### P3-2. Updated limitations (9Ash)
Add: seed variance, that component ablations were inference-time (fixed by P0-3), code-release status.

### P3-3. Release code
Both 9Ash and the checklist flag the deferred release. Anonymized repo removes an easy ding if employer approval lands.

---

## Not raised by reviewers — fix before an alert AC does
The significance runs will *expose* these, so patch proactively:

1. **Length-reward confound.** Paper claims all three methods "share the reward function" (line 287–291) but DRIFT optimizes length-adjusted r̃ (Eq. 9, λ_len=2.0). Either apply r̃ to baselines or ablate it (P0-4 covers λ_len=0) and drop the identical-reward claim.
2. **Seeded-checkpoint confound.** 4B DRIFT is "seeded from a step-20 checkpoint" (Table 5 ‡). Make cold-start same-initialization or justify.
3. **Perturbed vs. unperturbed in-training eval.** Confirm the in-training AIME-24 logger measures the *unperturbed* policy for DRIFT (Table 8 shows the perturbation alone adds +0.83 pp). Prefer post-hoc unperturbed sweep for all head-to-heads.

---

## Minimal run matrix if compute-bound
Concentrate on 1.7B (cheap); buy 4B only where the claim requires it.

1. 1.7B: GRPO ×3, DRIFT ×3, DAPO ×1, DRIFT⊕DAPO ×1  → seeds + DAPO (P0-1, P0-2)
2. 1.7B: 4 training ablations ×1 (α=0, σ=0, entropy-sign flip, cₜ=1)  → (P0-3)
3. 4B: sensitivity — 4 warm-start-from-step-40 runs (p∈{80,90}, K∈{10,40}), +60 steps → step 100, ~80 node-hrs  → (P0-4)
4. 1.7B + 4B: extend GRPO/DRIFT to convergence  → (P1-1)
5. 4B: GRPO ×2 + DRIFT ×2 for cold-start-timing distribution  → (P0-1)

This ordering closes SkU1 and 48nX (baselines + ablations + sensitivity + significance) and shores up 9Ash's remaining asks.

## Existing scripts to reuse
- Seeds: `run_qwen3_4b_base_pivot_v22_seed2.sh`, `_seed3.sh`
- Ablations: `run_qwen3_4b_base_pivot_v22_alpha0.sh`, `_sigma0.sh`, `_nolang.sh`
- DAPO: `run_qwen3_4b_base_dapo.sh`, `run_qwen3_4b_base_pivot_v22_dapo.sh`, `_dapo_nofg.sh`
- HEG: `run_qwen3_1p7b_base_grpo_high_ent.sh`, `run_qwen3_4b_base_high_ent_grpo.sh`
- Missing: 1.7B counterparts of the ablations/DAPO/seeds; entropy-sign-flip and cₜ=1 ablations; sensitivity sweep.
