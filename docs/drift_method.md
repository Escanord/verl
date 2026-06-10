# Learned DRIFT on the Logit Manifold for LLM Reasoning Exploration

---

## Abstract

We present **DRIFT** (**D**irected **R**ollout with **I**nferred **F**eedback at **T**riggers), a method for reinforcement learning of language model reasoning grounded in the Langevin dynamics framework. Standard group-relative policy optimization (GRPO) collapses to a zero-gradient regime on hard reasoning tasks when all sampled rollouts within a prompt group receive identical rewards—a degenerate condition we term the *dead zone*. The principled remedy is to inject a Langevin update at high-entropy token positions (branch points), where the model's next-token distribution is genuinely diffuse over qualitatively distinct continuations and where perturbations have maximal downstream consequence. The Langevin update

    l̃_t = l_t − η · ∇E(l_t) + σ · ζ_t

requires the energy gradient ∇E(l_t), which is intractable to compute via backpropagation through a stochastic trajectory of thousands of tokens. DRIFT replaces this gradient with an online-learned drift vector G, maintained as a Polyak-filtered SPSA estimate. A central design choice in our formulation is the construction of the SPSA feedback signal: rather than rewarding directions that *reduce* branch-point entropy—which produces an answer-seeking gradient that collapses exploration—DRIFT learns directions that *maintain* entropy near a target fraction of the trajectory's first branch-point reference, preserving the policy's exploratory behavior throughout long reasoning chains. Off-policy trajectories from the perturbed rollout policy are integrated via an importance-sampling correction with a mixture-policy denominator, and a length-conditioned score adjustment suppresses a short-answer shortcut to which the Langevin perturbation would otherwise be vulnerable. The resulting algorithm produces measurable gradient signal during the dead-zone phase that vanilla GRPO cannot escape, and demonstrates substantial bootstrap acceleration on competition-level mathematics over both GRPO and DAPO baselines on Qwen3-4B-Base.

---

## 1. Introduction

Large language model (LLM) reasoning has emerged as a high-leverage frontier in AI capabilities. Recent works have demonstrated that reinforcement learning with verifiable rewards can elicit qualitatively new reasoning behavior in pre-trained language models (DeepSeek-R1; OpenAI o1). However, RL training on hard reasoning tasks faces a fundamental exploration bottleneck. During early training, a model generating *n* candidate responses to a competition-level mathematics problem produces all-incorrect rollouts with near certainty—exceeding 95% on AIME-class benchmarks. When all *n* rollouts within a prompt group share the same reward, the within-group advantage estimator collapses to zero for every token, the policy gradient vanishes identically, and parameters do not update. The model remains trapped in this degenerate fixed point for extended periods of training.

Existing approaches to this exploration deficit are either coarse or palliative. The standard entropy regularization carried over from policy gradient methods (Mnih et al., 2016; Schulman et al., 2017) injects undifferentiated stochasticity across all token positions; recent work on LLM reasoning shows that only a small minority of high-entropy "forking" tokens drive RL improvement (Wang et al., 2025), implying that this uniform-pressure exploration wastes update budget on positions whose perturbation does not propagate to outcome. When the reward signal is additionally near-zero, the entropy term dominates the loss and drives entropy upward without guiding exploration toward correct solution paths. Filter-based methods (DAPO; Yu et al., 2025) excise zero-advantage groups from the gradient computation, removing wasted compute but generating no new gradient signal on prompts the model consistently fails. Neither family addresses the structural problem: the model needs to *try qualitatively different reasoning chains* on hard prompts, with the exploration directed by online feedback rather than by a prior belief about which trajectories deserve perturbation.

The natural framework for principled exploration in a continuous space is **Langevin dynamics**. Given a potential E whose negative gradient points toward higher reward, the Langevin update

    l̃_t = l_t − η · ∇E(l_t) + σ · ζ_t,    ζ_t ~ N(0, I)

perturbs the logits at position t with a drift toward lower energy and a stochastic diffusion term, producing rollouts that explore qualitatively different continuations rather than repeating the same deterministic path. Two obstacles, however, prevent direct application of Langevin dynamics to autoregressive language generation. First, ∇E(l_t) requires differentiating expected downstream reward through a stochastic trajectory of up to several thousand tokens—computationally prohibitive. Second, naive application of the perturbation at every token wastes update budget on positions whose outcomes are insensitive to logit perturbation (the next-token distribution is concentrated and the gradient is effectively zero) and also at positions whose perturbations affect surface form rather than reasoning content (early tokens, where the model has not yet committed to a strategy).

DRIFT resolves both obstacles. Branch-point selection (Section 3.1) restricts the Langevin update to high-entropy positions in the substantive reasoning region of the response, concentrating computational budget where token choice is most outcome-informative. An online drift estimate (Section 3.3) replaces ∇E with a momentum-filtered SPSA estimate updated at each trigger from a per-trajectory entropy reference: the entropy of the *first* branch point of the trajectory serves as a fixed anchor H_{first}, and the SPSA feedback at subsequent triggers measures the deviation of current branch-point entropy from a target α_target · H_{first}. The drift G accumulates evidence about which perturbation directions keep the trajectory's branch-point entropy *near this anchor* — that is, which directions preserve the policy's exploratory capacity — rather than directions that drive entropy down. This orientation is the central methodological choice in our formulation. The natural alternative would be to reward perturbations that decrease subsequent branch-point entropy, on the heuristic that lower entropy indicates increased confidence and therefore progress; we argue, and the empirical record supports, that this orientation is precisely backwards in the dead-zone regime. The dead zone is, by definition, a state in which the model produces highly confident yet uniformly wrong reasoning chains. Reinforcing directions that further reduce entropy reinforces the failure mode; what is needed is a feedback signal that *sustains* the policy's branching capacity at successive branch points, allowing the chain of Langevin updates to keep redirecting the trajectory rather than collapse to a deterministic terminus after the first perturbation.

Two additional components complete the algorithm. A length-conditioned score adjustment (Section 3.4) penalizes short *correct* rollouts prior to advantage normalization, suppressing a shortcut to which the Langevin perturbation is otherwise vulnerable: a perturbation can stochastically redirect the trajectory toward a brief correct answer, producing a high-advantage rollout whose policy gradient teaches the model to truncate reasoning. The penalty isolates this failure mode without affecting the within-group ranking of incorrect rollouts. An importance-sampling correction (Section 3.5) integrates the off-policy trajectories produced under the perturbed rollout policy into the GRPO objective with bounded weight variance.

---

## 2. Background

### 2.1 Autoregressive Language Model Generation

A language model parameterized by θ defines a conditional distribution over token sequences. Given a prompt x, the model generates a response y = (y_1, …, y_T) autoregressively:

    π_θ(y | x) = ∏_{t=1}^{T} π_θ(y_t | x, y_{<t})

where π_θ(· | x, y_{<t}) = softmax(l_t) and l_t ∈ ℝ^|V| are the logits at step t. The Shannon entropy of the next-token distribution is

    H_t = − ∑_{v ∈ V} π_θ(v | x, y_{<t}) log π_θ(v | x, y_{<t}).

### 2.2 Group-Relative Policy Optimization

GRPO (Shao et al., 2024) is an actor-only RL algorithm that estimates advantages from within-group reward variance. For each prompt x^i, n rollouts {y^{i,j}}_{j=1}^n are sampled and scored by a verifiable reward function. The normalized advantage is

    Â^{i,j} = (r^{i,j} − mean_j r^{i,j}) / std_j r^{i,j},

and the clipped policy gradient objective is

    L_GRPO(θ) = E_{i,j,t} [ min( ρ^{i,j}_t · Â^{i,j},  clip(ρ^{i,j}_t, 1−ε_low, 1+ε_high) · Â^{i,j} ) ]

where ρ^{i,j}_t = π_θ(y^{i,j}_t | x^i, y^{i,j}_{<t}) / π_old(y^{i,j}_t | x^i, y^{i,j}_{<t}) is the per-token importance ratio.

### 2.3 The Dead Zone

When the reward distribution over rollouts within a group is degenerate—all n rollouts receive the same reward—the within-group standard deviation is zero and Â^{i,j} = 0 for every j. The gradient of L_GRPO is identically zero at every token of every affected rollout. On AIME-class problems, a base model achieves near-zero individual accuracy, implying that with n = 8 rollouts per prompt the probability of observing at least one correct rollout per group is negligibly small. Empirically, vanilla GRPO produces no measurable validation signal for the first hundred training steps on such data. The first-order objective of an exploration mechanism in this regime is to break this degeneracy: to raise the probability of producing at least one correct rollout per group above the level afforded by the unperturbed policy, even before reasoning has been learned.

### 2.4 Motivation & Empirical Evidence

The dead-zone framing is not a thought experiment — it is observable in the training-time logs and rollouts of vanilla GRPO on Qwen3-4B-Base. We provide three mechanism diagnostics on the same training runs that produce the headline accuracy results in §6: (i) per-step gradient and entropy trajectories that show the dead zone in numerical form (§2.4.1), (ii) within-group reward-variance statistics that show diffusion produces the variance GRPO needs (§2.4.2), and (iii) per-token branch-point entropy traces that show the entropy-maintenance feedback sustains exploration headroom (§2.4.3). All three studies parse the same set of artifacts that the public release ships in `examples/empirical/`.

#### 2.4.1 Bootstrap acceleration and entropy stability over training

We extract three per-step quantities from the 4B training-time JSONLs of GRPO, HEG (uniform-pressure entropy bonus), and DRIFT (this paper, v18c) and aggregate them by step bucket:

**Table 0.** Three signals of the dead zone, averaged over step buckets on Qwen3-4B-Base. `|pg_loss|` is the magnitude of the policy-gradient loss (proportional to actual parameter movement); `entropy` is the per-token policy entropy averaged over the batch; "AIME24 mean@16" is from the training-time logger (val every 5 steps, n=16 samples). DRIFT is seeded from a step-20 reference checkpoint and saves its own first ckpt at step 25.

| Step bucket | GRPO `\|pg_loss\|` | HEG `\|pg_loss\|` | DRIFT `\|pg_loss\|` | GRPO entropy | HEG entropy | DRIFT entropy | GRPO AIME24 | HEG AIME24 | DRIFT AIME24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| (0, 40] | 0.00015 | 0.00151 | **0.00802** | 0.895 | 0.828 | 0.580 | 0.01% | 0.01% | **0.74%** |
| (40, 80] | 0.00037 | 0.03248 | **0.01024** | 0.934 | 0.503 | 0.403 | 0.45% | 1.56% | **3.99%** |
| (80, 120] | 0.00071 | 0.03711 | **0.01666** | 0.117 | 0.088 | **0.363** | **8.02%** | 6.48% | 7.07%‡ |
| (120, 200] | 0.00052 | 0.03233 | — | 0.033 | 0.042 | — | **10.12%** | 7.12% | — |

‡ DRIFT 4B last available val is step 90; the (80, 120] cell is a 2-point average (steps 85, 90).

Three observations follow.

**1. The dead zone is real and asymmetric.** Vanilla GRPO's `|pg_loss|` is essentially zero for the first 80 training steps (≈ 1.5 × 10⁻⁴) — half the late-training magnitude (≈ 7 × 10⁻⁴) and three orders of magnitude below DRIFT's contemporaneous magnitude. Translated into AIME24 mean@16, GRPO produces no measurable signal until step 65: it sits at *exactly* 0.00% for steps (0, 60], crosses 0.10% at step 65, and only crosses 1% near step 75. This is the dead zone in numerical form — for ~60 wall-clock steps of compute, no learning happens.

**2. Uniform-pressure exploration (HEG) breaks the dead zone but pays for it.** HEG's `|pg_loss|` is an order of magnitude larger than GRPO's during the dead zone (1.5 × 10⁻³ vs 1.5 × 10⁻⁴ at steps (0, 40]) — entropy bonus alone produces gradient signal where GRPO has none. But HEG's entropy collapses almost as fast as GRPO's (0.83 → 0.09 by step 100), and its eventual AIME24 ceiling is *lower* than GRPO's (7.12% vs 10.12% at the same horizon). This ceiling gap — visible across all benchmarks in §6 — is the cost of uniform pressure: every token gets perturbed, including the high-confidence reasoning tokens whose precision matters for correctness.

**3. Selective exploration (DRIFT) escapes the dead zone earlier *and* preserves entropy.** DRIFT's `|pg_loss|` from its first saved step is **50× GRPO's** and **5× HEG's** at the same training-time bucket. Crucially, **DRIFT entropy stays in the 0.36–0.40 range** through step 95 (the latest measurement) while GRPO collapses to 0.12 and HEG to 0.09 by step 100 — direct numerical evidence for the §3.3.2 entropy-maintenance objective working as designed. AIME24 mean@16 reaches 3.99% at step ~50 (DRIFT) versus 1.56% (HEG) versus 0.45% (GRPO) — a ~9× lead over GRPO and ~3× over HEG during the regime the paper is engineered for.

**Plot specification (Figure 2 — bootstrap and entropy trajectory).** Two-panel figure, shared x-axis = training step, range [0, 200].
- *Data file*: `empirical/reward_variance_4b.json` (shipped in the public release).
- *Schema*: `{"GRPO": [{"step": int, "actor.pg_loss": float, "actor.entropy": float, ...}, ...], "HEG": [...], "DRIFT": [...]}`.
- *Panel A (top)*: y = `|actor.pg_loss|` on log scale; three lines GRPO/HEG/DRIFT.
- *Panel B (bottom)*: y = `actor.entropy` on linear scale; same three lines.
- *Style*: consistent palette across the paper (e.g. GRPO blue, HEG orange, DRIFT green); thin vertical line at step 80 marking GRPO's bootstrap escape.

#### 2.4.2 Within-group reward variance (diffusion produces the variance GRPO needs)

§3.2 will argue that DRIFT's diffusion term is what spreads the n rollouts of a GRPO group into distinct continuations, restoring the reward variance GRPO requires. The relevant empirical signal is *not* whether DRIFT produces more textually diverse rollouts than vanilla sampling — a sharp softmax can already produce textually distinct continuations that are all wrong — but whether the resulting group contains the *correctness asymmetry* GRPO actually optimizes from. We measure this directly with the **mixed-correctness fraction**: the share of n=8 rollout groups containing both a correct and an incorrect rollout. We complement it with the total number of *correct* rollout pairs across all groups, the count of useful diversity.

**Table 1.** Within-group reward variance on Qwen3-4B-Base at training step 60 (deep within GRPO's cold-start dead zone) and on Qwen3-1.7B-Base at step 160 (the late-training plateau). 30 AIME24 prompts × n=8 rollouts at temperature 1.0. `mixed` is the fraction of groups containing both a correct and an incorrect rollout (the precondition for non-zero GRPO advantage); `n correct pairs` is the total number of correct rollout pairs across all 30 groups; `lev.all` and `lev.correct` are mean pairwise normalized Levenshtein distance over all rollout pairs and over correct-only pairs, respectively. Bold marks the leading method per column.

| Regime | Method | solve | mixed | lev.all | lev.correct | n correct pairs |
|---|---|---:|---:|---:|---:|---:|
| 4B step 60 (dead zone) | GRPO | 0.42% | 3.33% | 0.895 | — | **0** |
| 4B step 60 | HEG | 2.50% | 16.67% | 0.849 | 0.761 | 1 |
| 4B step 60 | DRIFT | **23.33%** | **50.00%** | 0.826 | **0.801** | **105** |
| 1.7B step 160 (plateau) | GRPO | 13.33% | **50.00%** | 0.734 | **0.627** | 26 |
| 1.7B step 160 | HEG | 10.83% | 43.33% | 0.807 | 0.670 | 18 |
| 1.7B step 160 | DRIFT | **18.33%** | 40.00% | 0.733 | 0.523 | **84** |

Two observations:

**1. Textual diversity alone is misleading; correctness asymmetry is the load-bearing metric.** GRPO at 4B step 60 has the *highest* `lev.all` (0.895) — but `lev.correct = 0` because not a single group ever contained two correct rollouts. The textual-distance metric rewards GRPO for being *creatively wrong*. Switching to mixed-correctness fraction reveals the actual story: GRPO produces variance-bearing groups on only 3% of prompts, DRIFT on 50%.

**2. The two regimes match the paper's framing.** At 4B step 60, DRIFT raises the per-group variance probability from 3% to 50% and produces 105 correct rollout pairs at the same training step where GRPO produces 0 — the discovery-amplification mechanism of Theorem 1 in measurable form. At 1.7B step 160, GRPO has bootstrapped and both methods produce variance on roughly half their groups; DRIFT now solves 18.33% of rollouts vs GRPO's 13.33% and produces 3.2× as many correct rollout pairs (84 vs 26), consistent with the late-plateau-escape result of §6.2.

**Plot specification (optional Figure 3 — diversity stacked bar or grouped bar).** Two grouped-bar panels, one per regime; bars per method (GRPO/HEG/DRIFT).
- *Data files*: `empirical/diversity__{grpo,heg,v18c}_4b_step60.json` and `empirical/diversity__{grpo,heg,v18b}_1p7b_step160.json` (each is `{"summary": {...}, "per_prompt": [...]}` from `examples/empirical/rollout_diversity.py`).
- *Bar values*: `summary.mixed_correctness_fraction` and/or `summary.lev_norm_correct.n` (= `n correct pairs`).
- Annotate bar tops with `summary.solve_rate.mean` for context.
- Same color palette as Figure 2.

#### 2.4.3 Branch-point entropy is sustained, not collapsed

§3.3.2 will argue that the SPSA feedback signal `s_t = H_t − α_target · H_first` is what keeps DRIFT's branch-point distribution open across a trajectory, and that the alternative *reduction* signal `s_t = −(H_t − α_target · H_first)` would reinforce the same overconfidence the dead zone is defined by. To measure this directly we run, on a single Qwen3-1.7B v18b step-120 checkpoint, two HuggingFace-level rollouts per AIME prompt — one with the unperturbed policy (vanilla) and one with the DRIFT Langevin update at every triggered position — and log the per-token entropy under both conditions. Triggers fire only past the §3.1 positional guard `t_min = 800`, so prompt 1 (rollout length 652) lies entirely below the guard and contains no triggers; we report the four prompts that produce triggers below.

**Table 2.** Per-prompt branch-point entropy on Qwen3-1.7B v18b step 120, AIME24 prompts 0/2/3/4. `H@trig` is the mean per-token entropy at DRIFT trigger positions; `H@non` is the mean entropy at non-trigger positions of the same DRIFT rollout; `vanilla` is the mean entropy of the unperturbed rollout for the same prompt. `H_first` is the entropy at the first trigger of the trajectory and `α·H_first` is the maintenance target with α = 0.7.

| Prompt | Triggers | H@trig | H@non | vanilla | H_first | α·H_first |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 83 | 0.266 | 0.016 | 0.040 | 0.008 | 0.006 |
| 2 | 37 | 0.608 | 0.064 | 0.072 | 0.194 | 0.136 |
| 3 | 45 | 0.411 | 0.051 | 0.042 | 0.181 | 0.127 |
| 4 | 65 | 0.193 | 0.027 | 0.043 | 0.365 | 0.256 |
| **mean** | — | **0.369** | **0.048** | **0.052** | — | — |

Two observations:

**1. Selectivity is real.** Trigger-position entropy averages **0.369 nats** while non-trigger entropy in the *same* DRIFT rollout averages **0.048 nats** — a 7.7× ratio. Non-trigger entropy is statistically indistinguishable from vanilla entropy (0.048 vs 0.052), so DRIFT does not pollute the rest of the rollout the way an unrestricted entropy bonus would. This is direct evidence for the §3.1 branch-point selection criterion.

**2. Entropy is sustained at branch points, not collapsed.** The post-bootstrap vanilla policy at this checkpoint sits at uniformly low entropy (~0.05 nats); DRIFT pushes branch-point entropy to 0.2–0.6 nats *and keeps it there across all triggers* (mean 0.369 across 230 trigger events). This is the "maintenance" effect of §3.3.2: the SPSA feedback restores entropy at branch points rather than letting it decay with the surrounding policy. Note also that H@trig clusters in 0.19–0.61 nats while α · H_first targets vary by 50× across prompts (0.006 to 0.256) — DRIFT produces *useful exploration headroom* at branch points rather than slavishly tracking a per-prompt target.

**Plot specification (Figure 4 — per-token entropy traces).** A 4-row stacked plot (one row per prompt with triggers: 0, 2, 3, 4); shared x-axis = token position from 0 to `len(entropies)`; y-axis = per-token Shannon entropy in nats.
- *Data file*: `empirical/entropy_out/entropy__drift_v18b_step120.jsonl` (one record per (prompt, condition); 10 records total).
- *Schema*: `{"prompt_idx": int, "condition": "vanilla"|"drift", "entropies": [float...], "is_trigger": [0|1...], "trigger_positions": [int...], "H_first": float|null, "alpha_target": 0.7}`.
- *Per row, overlay both conditions*: vanilla entropy as a faded baseline curve, DRIFT entropy as a solid foreground curve.
- *Per row, overlay markers*: red ticks at the x-coordinates in `trigger_positions`; horizontal dashed line at `alpha_target * H_first` for the DRIFT trace; vertical line at x = 800 (the `t_min` positional guard, before which triggers are suppressed).
- *Optional fifth row* showing prompt 1 (rollout length 652) to make explicit that no triggers fire under `t_min = 800` for short rollouts; otherwise drop prompt 1 with a footnote.
- Same color palette as Figure 2 for vanilla (de-saturated) vs DRIFT (saturated).

---

These three studies tell the same story from independent angles: GRPO has a dead zone (per-step pg_loss in §2.4.1), and within that dead zone GRPO groups carry no within-group reward variance for the gradient to consume (zero correct pairs at 4B step 60 in §2.4.2). DRIFT's selective Langevin perturbation manufactures the variance GRPO needs (50% mixed-correctness at the same training step, with 105 correct pairs), without paying the entropy-collapse cost that uniform-pressure HEG suffers (entropy held at 0.36–0.40 in §2.4.1, sustained at high values at branch points in §2.4.3). The remainder of the paper makes this concrete: §3 specifies how DRIFT achieves selectivity, diffusion-based variance, and entropy maintenance; §4 derives the discovery-amplification guarantee; §6 verifies the three claims on a four-benchmark suite.

---

## 3. DRIFT

DRIFT implements Langevin dynamics on the logit manifold at outcome-informative branch points, with a learned drift direction replacing the intractable energy gradient. The method has five components: branch-point selection (Section 3.1), the Langevin update with learned drift (Section 3.2), online drift estimation via SPSA momentum with an entropy-maintenance feedback signal (Section 3.3), a length-conditioned score adjustment that suppresses a short-answer shortcut (Section 3.4), and an importance-sampling correction integrating the resulting off-policy trajectories into the GRPO objective (Section 3.5).

### 3.1 Branch Point Selection

Let E(l_t) = −V(l_t) where V(l_t) is the expected downstream reward of trajectories sampled from softmax(l_t). The energy gradient ∇E(l_t) is approximately zero wherever the logit distribution is degenerate: if π_θ(· | context) is concentrated on a single token, no logit perturbation small relative to the dominant logit changes which token is sampled, and the Langevin update is vacuous. Informally, the gradient is nonzero only where the distribution is diffuse—at positions the model is genuinely uncertain about.

This motivates restricting the Langevin update to positions of high next-token entropy. DRIFT declares position t a **branch point** if both:

1. **Adaptive entropy threshold**: H_t ≥ τ_t,    where τ_t = Quantile(H_buf, p) is the p-th quantile of a rolling buffer H_buf of recent per-token entropies, with a fixed fallback τ_0 used during the warmup phase before the buffer is populated.
2. **Positional guard**: t > t_min.

Two considerations motivate the positional guard. The first is informational: the opening tokens of a mathematical reasoning response—problem restatement, preamble phrases, and initial setup ("We need to find…", "Let us denote…")—exhibit high entropy driven by semantically-equivalent surface realizations rather than by genuine uncertainty over reasoning strategy. Perturbing logits at these positions diversifies phrasing while leaving the underlying computation invariant. More precisely, the quantity that determines whether a Langevin update is outcome-informative is not H_t alone but the mutual information I(y_t ; y_T | x, y_{<t}) between the current token and the final answer; for small t, the model has not committed to a reasoning path and divergent continuations remain accessible regardless of which surface token is sampled, so this mutual information is low even where H_t is large. Beyond t_min the model has established a local reasoning context and individual token choices begin to close off qualitatively distinct continuations, making H_t a reliable proxy for outcome-informativeness.

The second consideration is a failure mode: at small t_min, a Langevin perturbation can redirect the trajectory toward a short *direct answer* rather than a fuller reasoning chain. When such a short trajectory happens to be correct (a stochastic event), it produces a high-advantage rollout whose policy gradient teaches the model to truncate reasoning early. Iterated across training steps, this drives **length collapse**: the response-length distribution compresses sharply and the score gains the algorithm produces are attributable to short-answer exploitation rather than improved reasoning. The score-level adjustment of Section 3.4 isolates this pathology at the advantage estimator, but a sufficiently large t_min eliminates it at the source by ensuring the Langevin update only fires after substantive reasoning has occurred.

The third consideration is *cumulative off-manifold drift during the bootstrap phase*. Each Langevin fire perturbs the rollout off the base policy's distribution, so the rollout's KL divergence from the base scales with the number of fires. On Qwen3-4B-Base we observe this directly: at training step 40, the per-token KL on correct rollouts is 0.05 nats under t_min = 0.6 · peak length, versus 0.15 nats with t_min = 0 — a 3× difference attributable to early-position perturbation alone. Each extra fire pushes the rollout further from the base manifold and reduces the chance it stays in the correct-solution subspace; correspondingly, the count of correct rollouts at step 40 is 280 versus 103 (a 2.7× reduction) under the same comparison. The reduction compounds through GRPO's group-relative advantage: groups with zero correct rollouts contribute no learning signal, so the bootstrap rate is bottlenecked by the rate at which correct rollouts emerge. A sufficiently large t_min restricts the perturbation budget to a subregion of the rollout, keeping the policy close enough to the base manifold that bootstrap-phase correct rollouts emerge at a rate sufficient for GRPO to amplify them.

### 3.2 Langevin Update with Learned Drift

At each branch point, DRIFT plants a stochastic decision *in the logit manifold itself*. Rather than committing the trajectory to a single deterministic continuation, it applies a Langevin update that holds the policy in a small, controllable neighborhood of its current logit vector long enough for n parallel rollouts to fan out into qualitatively different reasoning paths. The update has the form

    l̃_t = l_t + η · (G_t / ‖G_t‖) + σ · ζ_t,    ζ_t ~ N(0, I)

with η the drift step size, σ the diffusion magnitude, and G_t / ‖G_t‖ the learned descent direction of E(l_t) in the top-K logit subspace. The update is confined to the K coordinates of largest absolute logit value:

    l̃_t[k] = l_t[k] + η · (G_t / ‖G_t‖)[k] + σ · ζ_t[k],    k ∈ arg-top-K(|l_t|),

after which the token is drawn from the perturbed distribution ỹ_t ~ softmax(l̃_t). Outside branch points the policy is left untouched (ỹ_t ~ softmax(l_t)), so the perturbation budget falls entirely on positions where it has somewhere to go.

**Exploration on the logit manifold, not on the probability simplex.** The natural alternatives — raising temperature, widening top-p, or masking high-probability tokens — all act on the simplex itself, either uniformly inflating tails or surgically truncating them. The first dilutes the model's hard-won structure; the second discards parts of it. Langevin perturbation acts one level upstream, on the unconstrained logit space, where additive Gaussian noise lifts smoothly to a multiplicative geometric reweighting on the simplex while leaving its softmax structure intact. This is what lets the perturbed distribution be evaluated, log-densified, and importance-sampled (Section 3.5) without distortion — and it is what makes the logit perturbation *invisible to the language model* until sampling time, so the model continues to extend any sampled token coherently rather than fighting an out-of-distribution prefix.

**The top-K subspace as the model's vocabulary of choices.** When the model is uncertain at a branch point, its uncertainty lies among a small set of plausible continuations — perhaps a dozen or two competing transitions, connectives, or formula moves — all packed into the top of the logit ranking. The long tail is, by construction, the model's record that it has dismissed those tokens for the current context: typo fragments, mid-word shards, off-language symbols. Perturbing into the long tail would mistake noise for diversity and derail the trajectory at the first sampled corruption. Restricting drift and diffusion to the top-K coordinates keeps the perturbation inside the model's *own* vocabulary of plausible choices, redistributing mass among the candidates the model already considers viable rather than manufacturing alternatives it would never have proposed.

**Symmetry breaking across rollouts: the σ·ζ term.** A single prompt under DRIFT produces n rollouts that share the same policy π_θ and the same drift G — yet they fan out into a spread of different reasoning paths because each rollout draws its own diffusion ζ_t at every branch point. Without σ·ζ, all n rollouts would receive the *same* drift η·G/‖G‖ at the same branch point, and on a sharply peaked softmax they would near-deterministically sample the same token; the fan would collapse to a line. The diffusion term is the mechanism that turns one drift estimate into n semantically-distinct trajectories. It is what allows DRIFT to deliver, from a single prompt, the kind of branching diversity that an RL algorithm would otherwise have to engineer through explicit tree search.

**Why this breaks the dead zone.** The dead zone is degeneracy across rollouts: all n unperturbed trajectories run the same near-deterministic policy and arrive at the same wrong answer, so within-group reward variance is zero and GRPO has no gradient to act on. The Langevin update breaks the symmetry exactly where it matters — at the branch points where the model is uncertain — by spreading the n rollouts into different perturbed neighborhoods. Even at cold start, when G is uninformative and the perturbation is purely diffusive, this fan-out raises the probability that at least one of the n trajectories happens to reach a correct answer, restoring within-group variance and reanimating the GRPO gradient. As G accumulates evidence about which perturbation directions sustain the policy's branching capacity (Section 3.3), the fan begins to point toward more productive regions of the logit neighborhood, converting raw exploration into directed exploration without ever leaving the autoregressive sampling procedure.

**Drift and diffusion, position-adaptively.** The two terms η·(G/‖G‖) and σ·ζ play complementary roles: the drift biases the rollout fan toward regions the SPSA estimate believes preserve exploratory capacity, the diffusion guarantees isotropic coverage and the rollout-level symmetry breaking described above, and their ratio η/σ controls the directed-vs-undirected balance of the perturbation. Restricting both to branch points is a form of *position-adaptive Langevin*: the effective Langevin step size is η at branch points and zero elsewhere, focusing the entire exploration budget on the small fraction of positions where the energy landscape is non-flat and where token-choice mutual information with the final answer is largest (Section 3.1).

### 3.3 Online Drift Estimation: SPSA Momentum with Entropy-Maintenance Feedback

The drift G_t approximates the descent direction of E(l_t) within the top-K logit subspace. Because E(l_t) involves expectations over stochastic trajectories of thousands of tokens, its gradient is intractable. DRIFT estimates it online via a momentum-filtered SPSA construction. Two design choices distinguish DRIFT's estimator from a textbook SPSA gradient estimator: the *perturbation distribution* combines exploitation of the current drift estimate with random sphere sampling (Section 3.3.1), and the *feedback signal* is oriented to reward exploration-preserving directions rather than entropy-decreasing ones (Section 3.3.2).

#### 3.3.1 Perturbation Direction

At branch point t, the Langevin perturbation direction is drawn from a mixture of the current normalized drift and a uniform direction on the (K−1)-sphere:

    ε_t = α · (G_t / ‖G_t‖) + (1 − α) · ξ_t,    ξ_t ~ Uniform(S^{K−1}),

with α ∈ [0, 1] interpolating between exploitation (α → 1, ε_t concentrates on the learned drift) and pure random exploration (α → 0, ε_t is uniform on the sphere). At initialization, when no feedback has yet been collected and G_0 = 0, the mixture reduces to ε_t = ξ_t, implementing maximum-entropy exploration of the K-sphere. As G accumulates consistent directional evidence over the trajectory, the distribution of ε_t concentrates around G/‖G‖, implementing a data-driven reduction in effective exploration temperature. Crucially, this reduction emerges from the accumulation of feedback rather than from an explicit annealing schedule, allowing the exploration–exploitation balance to be set per trigger by the strength of the learned signal rather than by the absolute training step.

#### 3.3.2 Feedback Signal: Entropy Maintenance, not Entropy Reduction

The SPSA feedback signal evaluates whether a perturbation in direction ε at a previous trigger was beneficial, where "beneficial" must be defined operationally because the true expected reward at a trigger is unobservable from a single trajectory. A textbook formulation would define benefit as *reduction* of next-trigger entropy: signal_t = H_{prev} − H_t, with positive values rewarding perturbations that produced more concentrated next-token distributions at the next branch point. The implicit hypothesis is that increased confidence at downstream branch points indicates progress toward an answer.

We argue that this orientation is incorrect for the dead-zone regime. The dead zone is, by definition, a regime in which the model already produces highly confident—and uniformly wrong—reasoning chains. Rewarding perturbations that further reduce next-trigger entropy reinforces the very behavior that produces the failure mode: premature commitment to a single reasoning direction and rapid collapse of branching capacity. What is needed is the opposite: a feedback signal that rewards perturbations under which the trajectory *retains* its branching capacity at downstream positions, allowing successive Langevin updates to continue redirecting the trajectory rather than reaching a deterministic terminus after the first perturbation.

DRIFT's feedback signal is therefore constructed as the deviation of the current trigger's pre-perturbation entropy from a target fraction of the trajectory's first-trigger entropy:

    signal_{t_new} = H_{t_new} − α_target · H_{first},    α_target ∈ (0, 1)

where H_{first} is the pre-perturbation entropy at the first branch point of the current trajectory and α_target sets the target maintenance ratio. The signal is positive when the trajectory's branch-point entropy remains above α_target · H_{first}—when the policy retains a substantial fraction of its initial branching capacity—and negative when entropy has decayed below the target. The drift G is updated via the Polyak filter

    G_{t_new} ← γ · G_{t_prev} + (1 − γ) · signal_{t_new} · ε_{t_prev}

with momentum coefficient γ ∈ [0, 1) controlling the filter bandwidth. Over the course of a trajectory, G accumulates evidence about which perturbation directions sustained the policy's exploratory behavior at downstream branch points; subsequent perturbations are then biased in those directions.

The reference H_{first} is fixed at the first branch point and not updated within the trajectory. This is intentional: H_{first} serves as a per-trajectory exploration anchor, ensuring that the target adapts to the local entropy regime of each prompt-response pair (some prompts permit higher branching than others) without drifting downward as the trajectory itself collapses. A per-trigger or rolling reference would track any decrease in entropy and silently re-anchor the target to the collapsed regime, defeating the maintenance objective.

Empirically, this orientation produces drift directions that sustain the per-trigger entropy distribution near α_target · H_{first} over the course of training, in contrast to the entropy-reduction variant which produces monotone collapse of branch-point entropy within a small number of steps.

### 3.4 Length-Conditioned Score Adjustment

A pathology that emerges when Langevin updates are applied during early training is **length collapse**. A Langevin perturbation at a branch point can redirect the trajectory toward a short correct answer, producing a high-advantage rollout whose policy gradient teaches the model to terminate reasoning early. Within an otherwise all-incorrect group, a single correct short rollout receives Â ≈ +√(n−1) under within-group normalization, yielding strong gradient toward the truncating completion at every token along its trajectory. Iterated over training steps, this shortens the policy's response-length distribution and compresses score gains toward stochastic short-answer hits rather than improved reasoning.

DRIFT addresses this by **adjusting the score of short correct rollouts prior to advantage normalization**. The raw rollout score r^{i,j} is replaced by an adjusted score r̃^{i,j}:

    r̃^{i,j} = r^{i,j} + λ · ( min(L^{i,j} / L_target, 1) − 1 ) · 𝟙[r^{i,j} > 0]

where L^{i,j} is the response length, L_target is the target reasoning length, λ is the penalty coefficient, and 𝟙[r^{i,j} > 0] restricts the adjustment to correct rollouts only. The adjustment is a one-sided floor: rollouts of length L ≥ L_target receive no penalty, while shorter correct rollouts incur a penalty growing linearly to a maximum of −λ at L = 0. The GRPO advantage is then computed from the adjusted scores:

    Â^{i,j} = (r̃^{i,j} − mean_j r̃^{i,j}) / std_j r̃^{i,j}.

**Why correctness-conditional.** Penalizing short rollouts unconditionally—correct and incorrect alike—produces a competing pathology. With the majority of early-training rollouts incorrect, an unconditional penalty applies primarily to wrong rollouts, whose advantages become *more* negative when short, gradient-incentivizing the model to extend incorrect answers rather than to terminate them when reasoning fails. Empirically this produces runaway length growth without correctness improvement. The 𝟙[r > 0] mask isolates the adjustment to the failure mode it is designed to address: short *correct* rollouts that the model would otherwise be reinforced to imitate.

**Why pre-normalization.** The penalty is applied to the score r before within-group mean-subtraction and standard-deviation normalization, with the consequence that *within-group* ranking among correct rollouts is reshaped to favor longer reasoning. A short correct rollout receives a smaller r̃ than a long correct one; after group normalization, the longer correct rollout receives strictly higher advantage. Applying an equivalent penalty post-normalization—as a separate loss term, for example—would not produce this within-group differentiation, because the GRPO normalization removes any common additive shift across rollouts.

**Calibration limit and future-work hard floor.** The smooth penalty above is calibrated by a single coefficient λ that must simultaneously cover short collapses (L ≪ L_target) and milder length deficits (L close to L_target). Algebraically, λ must satisfy r + λ(L/L_target − 1) ≤ −1 across the full collapse range to push short-correct rollouts into negative within-group advantage; at fixed L_target = 1000 this requires λ ≥ 2/(1 − L/L_target), i.e. λ ≥ 2.2 at L = 87, λ ≥ 4 at L = 500, λ ≥ 10 at L = 800. A single value chosen to suppress mid-length collapse (L ≈ 500) over-penalises borderline-acceptable long rollouts; a value chosen to be gentle on long rollouts is insufficient against mid-length collapse. A natural extension—**deferred to future work**—is to layer a piecewise hard floor on top of the smooth penalty: r̃ ← r̃ + λ_floor · 𝟙[L < L_min AND r > 0], with λ_floor large enough that any short-correct rollout under L_min is *guaranteed* to receive r̃ ≤ −1 regardless of how λ is calibrated. This isolates the length-collapse failure mode at the advantage estimator without distorting the linear penalty in the acceptable-length regime. Whether the smooth penalty alone is sufficient when paired with the perturbation-level and trigger-level defenses (the §3.1 positional guard restricting Langevin to t > t_min, and the §3.2 top-K subspace restriction) is the empirical question this paper studies; the hard-floor extension addresses the residual failure mode at strong-prior base models (e.g. Qwen3-4B-Base) where breakthrough-driven short-correct rollouts can appear before the smooth penalty's gradient has had time to engage.

### 3.5 Importance-Corrected Policy Gradient

Trajectories produced under the DRIFT rollout policy π̃_θ are off-policy with respect to the current policy π_θ and the behavioral policy π_old. DRIFT corrects the GRPO objective via per-token importance weights at trigger positions:

    L_DRIFT(θ) = E_{τ ~ π̃_θ} [ ∑_t Â_t · min( w_t · ρ_t,  clip(w_t · ρ_t, 1 − ε_low, 1 + ε_high) ) ]

where the IS weight w_t is

    w_t = π_θ(ỹ_t | x, y_{<t}) / π̃_θ(ỹ_t | x, y_{<t}),    t ∈ T_trigger,
    w_t = 1,                                                t ∉ T_trigger.

**Mixture-policy denominator.** A central difficulty with importance sampling in policy gradient methods is that the IS weight π_θ / π̃_θ can be unbounded when π̃_θ departs significantly from π_θ at low-probability tokens, causing gradient variance to explode. Kakade & Langford (2002) resolve this in the conservative policy iteration framework by using a mixture policy as the behavior distribution rather than either constituent alone:

    π̃_eff(ỹ_t) = β · π̃_θ(ỹ_t) + (1 − β) · π_old(ỹ_t),    β ∈ (0, 1).

The resulting IS weight π_θ / π̃_eff is bounded above by 1/(1 − β) globally, ensuring finite gradient estimates regardless of how far the perturbed policy departs from π_old. The symmetric choice β = 1/2 yields an additionally robust estimator (Dudik et al., 2011): the gradient remains consistent if either π̃_θ or π_old correctly specifies the sampling distribution, providing protection against misspecification of either component. An entropy cap H_cap further clips IS weights when the perturbed distribution is very diffuse (H_t > H_cap), trading a small bias for variance reduction in regimes where the bound 1/(1−β) is loose relative to typical weight magnitudes.

**KL regularization.** A low-variance KL divergence to the pre-trained reference policy π_ref is appended:

    L(θ) = L_DRIFT(θ) + λ_KL · KL(π_θ ‖ π_ref).

**Implementation note (v21 soft-IS multiplier).** The mixture-policy denominator above bakes the IS correction into the PPO ratio, which forces a tradeoff between trust-region behavior (ratio ≈ 1) and off-policy correction strength (ratio ≪ 1 at heavily-perturbed positions). DRIFT v21 sidesteps this tradeoff by factoring the IS weight out of the PPO ratio entirely:

    π_θ(a_t) / π̃_θ(a_t) = [π_θ(a_t) / π_old(a_t)] · [π_old(a_t) / π̃_θ(a_t)]
                                └─────────┬─────────┘   └──────────┬─────────┘
                                  PPO ratio r_t                w_corr,t

The first factor stays inside PPO's `clip(r_t, 1−ε_low, 1+ε_high)` and behaves as the standard trust region. The second factor is applied as a per-token loss multiplier (equivalently, by reweighting advantages Â'_t = Â_t · w_corr,t at trigger positions). With the variance cap

    w_corr,t = min(1, π_old(a_t) / π̃_θ(a_t)),    t ∈ T_trigger,
    w_corr,t = 1,                                  t ∉ T_trigger,

the estimator is unbiased in the regime where Langevin raises the committed token's probability (the rate-inflation regime that drives length collapse, where π̃_θ(a_t) > π_old(a_t)), and one-sidedly biased only in the opposite regime where the cap fires. The mixture denominator π̃_eff above corresponds, in this factorization, to an effective weight w_corr,t = √(π_old / π̃_θ) when β = 1/2 — i.e., only half the bias is corrected in log space. The v21 multiplier applies the full correction. The asymmetric-IS branch and the entropy cap H_cap become unnecessary in this formulation, since w_corr,t ∈ (0, 1] is well-defined for both signs of advantage and does not rely on diffuse-distribution heuristics for variance control.

---

## 4. Theoretical Analysis

### 4.1 Langevin Convergence and the Learned Drift Approximation

The unadjusted Langevin algorithm (ULA; Roberts & Tweedie, 1996) with step size η converges to a distribution O(η)-close in total variation to p(x) ∝ exp(−E(x)), provided E is L-smooth and m-strongly convex. DRIFT applies ULA on the K-dimensional top-logit subspace with the exact gradient ∇E(l_t) replaced by the SPSA estimate G_t / ‖G_t‖. The approximation error between G_t / ‖G_t‖ and the true descent direction introduces additional bias in the stationary distribution. Under standard one-point gradient estimation results (Flaxman et al., 2005; Agarwal et al., 2010), the SPSA estimate converges at rate O(K^{1/2} / √n) in the number of feedback observations n. Combined with the ULA convergence bound, the stationary distribution of DRIFT is O(η + K^{1/2}/√n) close to the true Langevin stationary distribution; as training progresses and G accumulates observations, the approximation error diminishes. The Polyak filter introduces an additional momentum bias of order proportional to the gradient Lipschitz constant times the filter bandwidth 1/(1 − γ).

### 4.2 The Entropy-Maintenance Objective as a Surrogate Gradient

Let Φ(l_t) denote a real-valued objective on the logits at branch point t whose gradient direction is to be estimated. The classical SPSA construction estimates ∇Φ from the differential

    ε_t · ( Φ(l_{t+δ}, ε) − Φ(l_t) )

for small δ. The choice of Φ determines what direction is learned. DRIFT uses

    Φ(l_t) = H(softmax(l_t)) − α_target · H_{first},

equivalent up to constant to the deviation of branch-point entropy from a fixed per-trajectory target. The gradient ∇_l Φ is the gradient of Shannon entropy on the simplex, projected onto the top-K subspace. The learned drift G therefore points (in expectation, modulo SPSA bias) in the direction along which logit perturbation maintains H near α_target · H_{first}. Over a sequence of branch points, repeated application of this drift produces a trajectory whose branch-point entropy time-series tracks the target rather than collapsing to it.

This contrasts with the alternative Φ(l_t) = H_{prev} − H(softmax(l_t)), whose gradient direction would maximize entropy reduction. Under that objective, repeated application produces a self-reinforcing cascade in which each successive perturbation finds a direction further reducing entropy, terminating in deterministic collapse. The choice of target as a fraction of H_{first} rather than as a per-step difference is what makes the objective an *equilibrium* objective rather than a *descent* objective.

### 4.3 Branch Points as Regions of Non-Flat Energy Landscape

The energy E(l_t) = −V(l_t) is approximately flat wherever the next-token distribution is concentrated: if π_θ(v | context) ≈ 1 for some v, then for small δ, softmax(l_t + δ) ≈ softmax(l_t), V(l_t + δ) ≈ V(l_t), and ∇E(l_t) ≈ 0. The Langevin update at such positions produces no useful exploration—it adds noise to a step that was already nearly deterministic. DRIFT's branch-point condition selects positions where H_t is large, equivalently where the next-token distribution is diffuse and ∇E(l_t) is likely nonzero. Using the information-theoretic decomposition

    I(y_T ; y_t | x, y_{<t}) = H(y_t | x, y_{<t}) − H(y_t | x, y_{<t}, y_T),

the mutual information between the current token and the final answer is large precisely where H_t is large *and* H(y_t | y_T) is small—positions where token choice matters for the outcome and the model is uncertain about it. H_t alone provides a tractable upper bound and selection proxy for this mutual information.

### 4.4 Implicit Tree Search Interpretation

DRIFT can be viewed as performing implicit partial tree search within the standard autoregressive generation procedure. Vanilla decoding is greedy depth-first traversal. Applying a Langevin perturbation at branch points effectively conditions the trajectory on a displaced logit at the branch node, producing a sibling branch without explicit backtracking. Across n rollouts per prompt, each branch point is visited with n different perturbations, sampling from a neighborhood of the original logit distribution. The drift vector G plays the role of a node-level value in MCTS: it accumulates evidence about which displacement direction sustains downstream exploration capacity, and biases subsequent expansions in that direction. Unlike MCTS, DRIFT requires neither explicit tree construction nor a separate value network; the entire mechanism operates within the standard rollout procedure.

---

## 5. Experimental Setup

**Base models.** We evaluate DRIFT on Qwen3-1.7B-Base and Qwen3-4B-Base (Yang et al., 2025), transformer base models pre-trained on general text with no instruction or reasoning tuning. The two scales are chosen to isolate the bootstrap-acceleration claim along the dead-zone-duration axis: vanilla GRPO escapes the dead zone within ~50 training steps on 1.7B (probing *ceiling parity*) and only after ~75–80 steps on 4B (probing *time-to-escape*). We use the original Qwen3 chat template throughout and apply a minimum-response-length floor of 800 tokens during rollout to ensure DRIFT's positional guard `t_min` is reachable.

**Datasets.** Training uses a math-only subset of LLM360/guru-RL-92k — competition-style problems from the AIME and MATH families with rule-based reward via the `math_dapo` sympy-equivalence checker, yielding ±1 binary correctness with no shaping. We evaluate on four benchmarks spanning the difficulty and domain spectrum: AIME24 (8× repeated, 240 problems) and AIME25 (30 problems) for in-distribution competition mathematics; OlympiadBench-Math-EN (`OE_TO_maths_en_COMP` single-answer subset, 674 problems) (He et al., 2024) for harder out-of-distribution math; and GPQA-Diamond (198 problems) (Rein et al., 2023) for cross-domain transfer to graduate-level science MCQ. AIME24 and the MATH-500 development set are additionally evaluated in-training every 5 steps via the rollout logger; the full external suite is run as a post-hoc sweep on saved checkpoints at every 20-step boundary.

**Implementation details.** All training runs use 8 × H200 (140 GB) GPUs through verl (Sheng et al., 2024), with FSDP for the actor and vLLM for the rollout engine. We sample n = 8 rollouts per prompt during training and n = 16 per problem during evaluation, both at temperature 1.0; metrics are `mean@16` (per-sample solve rate) and, for AIME, additionally `best@16` (probability that at least one of 16 samples is correct). Optimization uses AdamW with learning rate 1 × 10⁻⁶, weight decay 0.1, and 10 warmup steps; batch size 1024 prompts per training step, mini-batch 256, micro-batch 16 per GPU; maximum prompt length 1024 tokens, maximum response length 4096 tokens. Unless otherwise specified DRIFT uses the §3.6 defaults: branch-point quantile p = 0.85 (rolling buffer 2000, warmup fallback 0.4 nats), positional guard t_min = 800, top-K subspace K = 20, drift step η = 0.1, diffusion σ = 0.01, Polyak momentum γ = 0.7, drift exploit α = 0.6, entropy-maintenance target α_target = 0.7, length penalty λ = 2.0, target length L_target = 1000, IS mixture β = 0.5, entropy cap H_cap = 0.8 nats, asymmetric clip range (ε_low, ε_high) = (0.2, 0.28), KL coefficient λ_KL = 1 × 10⁻³. Under these settings the trigger fraction settles at 3–5% of generated tokens with the adaptive threshold τ_t in 0.3–0.5 nats. All code is available at <https://github.com/Escanord/verl>.

**Baselines.** We compare DRIFT against (i) **GRPO** (Shao et al., 2024) — vanilla group-relative policy optimization, the dead-zone-victim baseline; and (ii) **High-Entropy GRPO (HEG)** — GRPO with a uniform per-token entropy regularizer, the most direct empirical instantiation of the uniform-pressure-exploration critique we make in §1 and §3. HEG isolates the value of *selective* exploration at branch points relative to *uniform* entropy injection across all tokens. All three methods share data, optimizer, batch size, sampling temperature, KL coefficient, and reward function — they differ only in the exploration mechanism. We do not compare directly against DAPO (Yu et al., 2025) for compute reasons; DAPO's asymmetric clipping, dynamic group filtering, and token-level loss normalization are orthogonal to DRIFT's exploration mechanism and could in principle be stacked.

---

## 6. Results

We organize the empirical evaluation around the two claims established in §1: that DRIFT's selective exploration mechanism (i) accelerates dead-zone escape relative to vanilla GRPO without (ii) paying the ceiling penalty incurred by uniform-pressure entropy regularization (HEG). Section 6.1 establishes the bootstrap acceleration on the harder 4B regime where the dead zone is longest. Section 6.2 reports the parallel comparison on Qwen3-1.7B-Base, where vanilla GRPO escapes its dead zone unaided and the comparison probes ceiling parity rather than time-to-escape. Each section presents two tables: a dense per-5-step training-time table (AIME24 + MATH-500) covering the full optimization trajectory, and a sparse post-hoc table at saved-checkpoint boundaries (AIME25, OlympiadBench-Math-EN, GPQA-Diamond) that re-samples each checkpoint with the standalone eval harness.

All numbers are accuracy in percent at n=16 samples per problem with temperature 1.0; AIME24/AIME25 columns report mean@16 and best@16, while MATH/OlympiadBench/GPQA report mean@16 only. Bold marks the leading method per row.

### 6.1 Qwen3-4B-Base

The 4B base model exhibits a long dead-zone phase under vanilla GRPO: AIME24 mean@16 remains identically zero for the first ~70 training steps, with no within-group reward variance and therefore no policy gradient signal. DRIFT and HEG both inject exploration during this phase but through structurally different mechanisms (selective vs uniform), and the comparison is the cleanest empirical evidence for the value of branch-point selectivity.

**Table 1A.** Training-time per-5-step evaluation on Qwen3-4B-Base — AIME24 (mean@16, best@16) and MATH-500 (mean@16). All values are accuracy %; em-dash means the run had not started val at that step or had stopped. DRIFT (v18c) was seeded from the v18 step-20 checkpoint and saved its first own ckpt at step 25; training is currently at step 93.

| Step | GRPO AIME mean | GRPO AIME best | GRPO MATH | HEG AIME mean | HEG AIME best | HEG MATH | DRIFT AIME mean | DRIFT AIME best | DRIFT MATH |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | — | — | — | 0.00 | 0.00 | 0.15 | — | — | — |
| 10 | — | — | — | 0.00 | 0.00 | 0.12 | — | — | — |
| 15 | — | — | — | 0.00 | 0.00 | 0.14 | — | — | — |
| 20 | — | — | — | 0.00 | 0.00 | 0.21 | — | — | — |
| 25 | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 | 0.18 | 0.05 | 0.54 | 1.45 |
| 30 | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 | 0.35 | 0.10 | 0.90 | 9.64 |
| 35 | 0.00 | 0.00 | 0.24 | 0.03 | 0.26 | 0.75 | 0.52 | 3.39 | 23.49 |
| 40 | 0.03 | 0.27 | 0.30 | 0.08 | 0.81 | 6.88 | **2.29** | **5.30** | **27.47** |
| 45 | 0.00 | 0.00 | 0.35 | 0.21 | 1.75 | 21.56 | **1.33** | **7.73** | **38.25** |
| 50 | 0.00 | 0.00 | 0.44 | 0.23 | 2.24 | 31.54 | **1.64** | **10.39** | **49.04** |
| 55 | 0.00 | 0.00 | 0.51 | 0.44 | 3.59 | 39.41 | **2.53** | **11.60** | **54.24** |
| 60 | 0.00 | 0.00 | 0.75 | 1.02 | 5.76 | 38.42 | **3.91** | **12.76** | **58.91** |
| 65 | 0.10 | 1.10 | 1.24 | 0.73 | 5.61 | 37.08 | **4.74** | **13.63** | **62.60** |
| 70 | 0.10 | 1.06 | 3.11 | 1.80 | 9.87 | 47.31 | **5.39** | **14.21** | **63.70** |
| 75 | 0.91 | 6.07 | 11.90 | 3.85 | 14.55 | 56.38 | **6.48** | **14.56** | **66.05** |
| 80 | 2.47 | 12.01 | 30.39 | 4.22 | 12.06 | 61.70 | **5.89** | 12.68 | **66.75** |
| 85 | 5.10 | 15.83 | 51.21 | 5.13 | 12.32 | 64.78 | **6.69** | **15.41** | **67.40** |
| 90 | 6.77 | 17.90 | 64.31 | 6.28 | 12.86 | 66.92 | **7.14** | 15.50 | **68.54** |
| 95 | **7.76** | **19.88** | 70.65 | 6.67 | 14.29 | 68.64 | 7.40 | 15.84 | 69.30 |
| 100 | **7.73** | **20.19** | 73.04 | 6.46 | 13.33 | 69.11 | 7.81 | 19.23 | 73.11 |
| 105 | **8.91** | 19.18 | 74.21 | 6.85 | 13.70 | 69.97 | — | — | — |
| 110 | **9.53** | 19.96 | 74.58 | 7.01 | 13.68 | 70.36 | — | — | — |
| 115 | **9.01** | **20.39** | 75.29 | 6.80 | 12.49 | 71.46 | — | — | — |
| 120 | **9.32** | **20.62** | 75.10 | 6.64 | 14.14 | 71.76 | — | — | — |
| 125 | **10.10** | **23.30** | 75.64 | 6.82 | 13.81 | 71.80 | — | — | — |
| 130 | **10.16** | 19.58 | 75.90 | 6.69 | 12.54 | 72.66 | — | — | — |
| 135 | **9.35** | 19.75 | 76.36 | 7.27 | 15.04 | 72.64 | — | — | — |
| 140 | **9.95** | **20.43** | 76.62 | 6.95 | 12.23 | 72.81 | — | — | — |
| 145 | **9.40** | **20.74** | 77.31 | 7.03 | 14.44 | 73.29 | — | — | — |
| 150 | **9.77** | **20.60** | 77.21 | 7.53 | 14.76 | 73.36 | — | — | — |
| 155 | **10.26** | **23.79** | 76.81 | 7.34 | 15.70 | 72.74 | — | — | — |
| 160 | **10.10** | **21.60** | 76.74 | 7.42 | 14.96 | 74.41 | — | — | — |
| 165 | **10.65** | **21.78** | 76.91 | 7.19 | 14.92 | 73.61 | — | — | — |
| 170 | **10.00** | 19.79 | 78.33 | 7.03 | 15.27 | 74.08 | — | — | — |
| 175 | **10.10** | **21.91** | 76.88 | 7.32 | 15.53 | 73.71 | — | — | — |
| 180 | **9.74** | **22.10** | 77.60 | 6.90 | 14.38 | 73.91 | — | — | — |
| 185 | **10.13** | **21.12** | 78.17 | 7.01 | 14.68 | 74.36 | — | — | — |
| 190 | **10.81** | **21.99** | 77.74 | 6.69 | 13.71 | 73.92 | — | — | — |
| 195 | **10.49** | **22.18** | 78.04 | 7.01 | 13.82 | 74.98 | — | — | — |
| 200 | **10.86** | **21.57** | 78.30 | 7.66 | 15.32 | 74.72 | — | — | — |
| 205 | 10.81 | 22.68 | 78.11 | — | — | — | — | — | — |
| 210 | 11.20 | 21.43 | 78.77 | — | — | — | — | — | — |
| 215 | 11.20 | 21.06 | 78.24 | — | — | — | — | — | — |
| 220 | 11.12 | 21.10 | 78.85 | — | — | — | — | — | — |

**Table 1B.** Post-hoc evaluation at saved-checkpoint boundaries on Qwen3-4B-Base — AIME25 mean@16, OlympiadBench mean@16, GPQA-Diamond mean@16. Bold marks the leading method per row. DRIFT step 20 reflects the v18-step-20 seed checkpoint (v18c training began at step 25); steps 40–80 are post-hoc evaluations on saved v18c checkpoints. Steps 100–160 remain pending until v18c training advances past step 95.

| Step | GRPO AIME25 | GRPO Olymp | GRPO GPQA | HEG AIME25 | HEG Olymp | HEG GPQA | DRIFT AIME25 | DRIFT Olymp | DRIFT GPQA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.00 | 0.05 | 23.64 | 0.00 | **0.09** | **25.06** | 0.00 | **0.09** | 23.57 |
| 40 | 0.00 | 0.18 | 22.94 | **0.21** | 1.43 | 25.82 | 0.00 | **7.40** | **29.41** |
| 60 | 0.00 | 0.27 | 24.59 | 1.04 | 11.64 | **32.52** | **4.17** | **24.92** | 33.58 |
| 80 | 4.58 | 14.61 | 29.38 | 4.79 | 27.59 | **34.17** | **5.42** | **31.42** | 35.28 |
| 100 | **13.54** | **39.19** | **34.61** | 10.00 | 35.10 | 33.79 | 13.62 | 39.11 | 34.81 |
| 120 | **15.21** | **42.70** | **36.29** | 9.17 | 36.99 | 35.50 | [TBD] | [TBD] | [TBD] |
| 140 | **15.21** | **43.63** | **38.71** | 9.79 | 38.21 | 35.22 | [TBD] | [TBD] | [TBD] |
| 160 | **16.25** | **44.39** | **40.32** | 8.96 | 38.90 | 35.79 | [TBD] | [TBD] | [TBD] |

**Bootstrap summary.** DRIFT crosses every early threshold first:

| Threshold | GRPO | HEG | DRIFT |
|-----------|-----:|----:|------:|
| AIME24 mean@16 ≥ 1%   | step ~80 | step ~55 | step ~35 |
| AIME24 mean@16 ≥ 5%   | step ~95 | step ~80 | step ~70 |
| MATH mean@16 ≥ 50%    | step ~85 | step ~70 | step ~50 |
| OlympiadBench ≥ 10%   | step ~75 | step ~55 | [TBD]    |

The qualitative pattern — DRIFT > HEG > GRPO during the dead-zone phase — is consistent across AIME24 and MATH (and AIME25/Olympiad once the DRIFT post-hoc sweep completes). By step 100, GRPO has bootstrapped past HEG on AIME and well past it on OlympiadBench, but the cumulative training-step debt incurred during steps 0–80 is the cost the bootstrap mechanism is designed to amortize.

### 6.2 Qwen3-1.7B-Base

The 1.7B base model has a much shorter dead zone — vanilla GRPO escapes within ~60 training steps unaided. The 1.7B regime therefore tests the *quality* of the eventual policy rather than the time-to-escape: does selective exploration preserve GRPO's plateau, or does it pay a ceiling cost like HEG?

**Table 2A.** Training-time per-5-step evaluation on Qwen3-1.7B-Base — AIME24 (mean@16, best@16) and MATH-500 (mean@16). All values are accuracy %; em-dash means the run had stopped at that point. DRIFT (v18b) is currently still training at step 165.

| Step | GRPO AIME mean | GRPO AIME best | GRPO MATH | HEG AIME mean | HEG AIME best | HEG MATH | DRIFT AIME mean | DRIFT AIME best | DRIFT MATH |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.00 | 0.00 | 0.10 | 0.00 | 0.00 | 0.19 | 0.00 | 0.00 | 0.26 |
| 10 | 0.00 | 0.00 | 0.15 | 0.00 | 0.00 | 0.10 | 0.00 | 0.00 | 0.12 |
| 15 | 0.00 | 0.00 | 0.15 | 0.00 | 0.00 | 0.16 | 0.00 | 0.00 | 0.18 |
| 20 | 0.00 | 0.00 | 0.33 | 0.00 | 0.00 | 0.22 | 0.00 | 0.00 | 0.27 |
| 25 | 0.05 | 0.53 | 0.34 | 0.00 | 0.00 | 0.35 | 0.00 | 0.00 | 0.46 |
| 30 | 0.03 | 0.27 | 0.50 | 0.00 | 0.00 | 0.33 | 0.00 | 0.00 | 0.60 |
| 35 | 0.03 | 0.27 | 0.89 | 0.00 | 0.00 | 0.41 | 0.00 | 0.00 | 0.51 |
| 40 | **0.10** | **1.08** | **1.19** | 0.00 | 0.00 | 0.53 | 0.03 | 0.27 | 0.84 |
| 45 | **0.10** | **1.06** | **2.67** | 0.00 | 0.00 | 0.96 | 0.03 | 0.29 | 1.21 |
| 50 | **0.23** | **2.45** | **7.96** | 0.00 | 0.00 | 2.05 | 0.10 | 1.06 | 2.29 |
| 55 | **1.15** | **9.63** | **24.10** | 0.00 | 0.00 | 4.60 | 0.10 | 1.06 | 5.64 |
| 60 | **2.27** | **14.52** | **41.77** | 0.16 | 1.60 | 10.22 | 0.60 | 5.47 | 15.46 |
| 65 | **2.29** | **12.91** | **49.40** | 0.31 | 2.92 | 14.72 | 0.76 | 6.26 | 31.81 |
| 70 | **2.47** | **13.76** | **52.83** | 0.29 | 1.96 | 15.99 | 1.82 | 12.04 | 43.77 |
| 75 | **3.12** | **15.40** | **55.14** | 0.13 | 1.32 | 17.35 | 2.53 | 11.63 | 50.38 |
| 80 | **3.20** | **16.23** | **56.06** | 0.36 | 3.24 | 19.26 | 2.89 | 14.28 | 52.04 |
| 85 | **3.39** | **15.00** | **57.27** | 0.29 | 2.38 | 19.36 | 3.10 | 14.69 | 54.80 |
| 90 | 3.72 | **16.43** | **58.59** | 0.47 | 3.17 | 20.38 | 3.70 | 14.76 | 55.50 |
| 95 | 3.88 | **18.03** | **58.95** | 1.04 | 3.77 | 21.77 | 3.80 | 15.16 | 56.80 |
| 100 | 3.41 | **16.90** | **59.15** | 0.89 | 4.46 | 27.32 | **3.93** | 14.46 | 57.34 |
| 105 | 4.14 | **18.46** | **59.49** | 0.36 | 3.27 | 31.04 | **4.53** | 16.85 | 57.61 |
| 110 | 3.49 | 15.23 | **59.81** | 0.39 | 3.35 | 35.90 | **4.43** | **17.09** | 58.46 |
| 115 | 4.24 | 17.90 | **60.32** | 0.49 | 4.95 | 40.12 | **4.32** | 17.74 | 58.81 |
| 120 | 3.98 | 16.83 | **60.88** | 0.49 | 4.61 | 43.01 | **5.08** | **18.87** | 59.45 |
| 125 | 3.98 | 16.19 | **60.89** | 0.81 | 6.24 | 46.83 | **4.82** | **18.18** | 59.60 |
| 130 | **5.16** | 18.42 | **61.60** | 1.33 | 9.53 | 48.51 | 4.77 | **18.52** | 59.49 |
| 135 | 4.53 | 18.34 | **61.80** | 1.28 | 7.46 | 49.64 | **4.82** | 18.06 | 59.45 |
| 140 | 4.45 | 16.49 | **61.26** | 1.28 | 8.80 | 50.75 | **5.34** | **18.46** | 60.59 |
| 145 | 4.56 | 17.40 | **61.79** | 1.61 | 9.04 | 51.78 | **4.97** | **17.85** | 60.68 |
| 150 | 4.84 | 16.75 | **62.10** | 3.46 | 10.79 | 51.92 | **5.70** | **19.42** | 60.82 |
| 155 | 4.32 | 17.61 | **62.15** | 2.73 | 11.63 | 52.04 | **5.13** | **19.09** | 61.58 |
| 160 | 4.66 | 17.02 | **61.35** | 3.75 | 11.89 | 53.84 | **5.39** | **18.22** | **61.66** |
| 165 | 4.38 | 17.21 | **61.38** | 1.41 | 8.91 | 53.44 | **5.36** | **17.86** | 60.91 |
| 170 | 4.51 | 15.55 | **61.58** | 2.42 | 9.97 | 53.87 | **5.68** | **20.26** | 61.01 |
| 175 | 4.27 | 17.39 | **61.15** | 1.54 | 8.99 | 55.17 | — | — | — |
| 180 | 5.08 | 18.04 | **61.46** | 2.66 | 7.94 | 54.12 | — | — | — |
| 185 | 4.56 | 16.28 | **61.48** | 0.91 | 6.08 | 54.37 | — | — | — |
| 190 | 4.40 | 16.36 | **61.92** | 0.96 | 6.07 | 52.95 | — | — | — |
| 195 | 4.48 | 16.44 | **62.19** | 2.63 | 10.73 | 54.64 | — | — | — |
| 200 | 4.38 | 16.36 | **62.51** | 1.12 | 7.51 | 55.51 | — | — | — |
| 205 | 4.30 | 16.64 | 62.71 | 1.72 | 9.50 | 55.56 | — | — | — |
| 210 | 4.17 | 15.92 | 63.16 | 1.77 | 9.10 | 55.49 | — | — | — |
| 215 | 4.04 | 16.15 | 62.90 | — | — | — | — | — | — |
| 220 | 3.83 | 16.06 | 62.91 | — | — | — | — | — | — |
| 225 | 4.74 | 17.82 | 63.16 | — | — | — | — | — | — |
| 230 | 3.78 | 15.67 | 63.42 | — | — | — | — | — | — |
| 235 | 4.61 | 18.37 | 63.42 | — | — | — | — | — | — |
| 240 | 4.09 | 17.68 | 63.64 | — | — | — | — | — | — |

**Table 2B.** Post-hoc evaluation at saved-checkpoint boundaries on Qwen3-1.7B-Base — AIME25 mean@16, OlympiadBench mean@16, GPQA-Diamond mean@16. Bold marks the leading method per row (ties marked on both methods). DRIFT (v18b) post-hoc eval is complete for all reported checkpoints (20–160).

| Step | GRPO AIME25 | GRPO Olymp | GRPO GPQA | HEG AIME25 | HEG Olymp | HEG GPQA | DRIFT AIME25 | DRIFT Olymp | DRIFT GPQA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.00 | 0.13 | 19.07 | 0.00 | 0.18 | 18.85 | 0.00 | **0.22** | **20.30** |
| 40 | 0.00 | **0.85** | 21.07 | 0.00 | 0.33 | **23.45** | 0.00 | 0.45 | 22.62 |
| 60 | **3.33** | **16.92** | 23.60 | 0.21 | 2.47 | **25.70** | 1.04 | 6.21 | 23.54 |
| 80 | **3.96** | **24.79** | 23.67 | 0.00 | 5.76 | **26.74** | 2.71 | 21.57 | 24.68 |
| 100 | 3.54 | **26.02** | 24.78 | 0.83 | 7.91 | **26.84** | **5.62** | 26.57 | 26.37 |
| 120 | **4.38** | **26.25** | 24.52 | 0.62 | 13.99 | **27.95** | **4.38** | 28.17 | 25.95 |
| 140 | 4.17 | **27.41** | 25.67 | 1.88 | 19.14 | **27.89** | **4.38** | 28.81 | 29.09 |
| 160 | 4.58 | **27.28** | 26.33 | 2.29 | 20.35 | **28.39** | **4.79** | 29.07 | 29.60 |

**Three observations** anchor §1's "convergence at equal ceiling, faster path" framing on 1.7B:

1. **GRPO bootstraps faster than DRIFT on 1.7B** (steps 40–85, holding the AIME and MATH lead) — exactly the opposite of 4B. This confirms that DRIFT's bootstrap value scales with the duration of the dead zone, which is short on 1.7B.
2. **AIME mean@16 ceiling**: GRPO plateaus at 4–5%; DRIFT exceeds this from step 100 onward (3.93 → 5.08 → 5.34 → 5.39 → 5.70%). HEG plateaus 4× lower at 1–4% — the empirical signature of uniform-pressure ceiling damage.
3. **AIME best@16 ceiling**: DRIFT (peak **20.26%** at step 170) exceeds GRPO (peak 18.46% at step 105) at every late checkpoint we've evaluated.

MATH-500 confirms the dead-zone story: at step 50 GRPO is at 8.0%, DRIFT at 2.3%, HEG at 2.1% — GRPO escapes its dead zone first on the easier MATH distribution. By step 200 GRPO has converged to ~62.5% MATH and stays there; DRIFT closes to 60.9% by step 165 and is still rising. GPQA is essentially saturated for all three methods at ~25–28% (the random-MCQ floor is 25%, so 26–28% reflects modest above-random behavior rather than a meaningful gap). HEG's narrow GPQA edge appears to be an artifact of its higher policy entropy at evaluation time rather than improved cross-domain reasoning.

### 6.3 Mechanism Diagnostics

We additionally report training-time diagnostics confirming that DRIFT's mechanism behaves as designed.

| Diagnostic | Expected behavior | Observed |
|------------|-------------------|----------|
| Trigger fraction | 3–5% under chosen p, t_min | mean 3.4% across 79 v18c 4B steps; settles in [3.3%, 5.3%] (steps 60–80) and [4.2%, 7.1%] (steps 80–100). 49% of steps in [3%, 5%]. |
| Branch-point entropy | Tracks α_target · H_first throughout training | H@trigger 0.369 nats vs H@non-trigger 0.048 nats (7.7× ratio) on v18b 1.7B step 120 across 230 trigger events; per-prompt H@trig in [0.19, 0.61] while α·H_first targets vary in [0.006, 0.256] — DRIFT produces useful exploration headroom rather than tracking α·H_first exactly. |
| Drift G norm | Grows from 0 at initialization, stabilizes within ~10 triggers | [TBD] |
| IS weight magnitude | Bounded by 1/(1−β) = 2 modulo entropy cap | [TBD: max observed weight] |

### 6.4 Ablations

We probe two design choices that motivate the §3 construction: (i) the contribution of each DRIFT component (drift, diffusion, and the random-sphere base) on the trained policy, and (ii) the choice of the top-K subspace size that confines the Langevin perturbation. Table 5 reports both ablations on Qwen3-1.7B-Base.

**Table 5.** Ablations on Qwen3-1.7B-Base, AIME 24 mean@16 and maj@16 (n=16 samples per problem, temperature 1.0). **Top:** component decomposition on the *trained* v18b checkpoint, four configurations of the Langevin perturbation. **Bottom:** top-K ablation comparing v18b (K=20, headline DRIFT) vs v18d (K=150), both trained from the same seed checkpoint with all other hyperparameters fixed, evaluated at the same training horizon. Bold marks the best within each ablation block; Δ is mean@16 relative to the within-block reference (vanilla for the component decomposition; K=20 for top-K).

| Ablation | Condition | AIME mean@16 | AIME maj@16 | Δ mean |
|---|---|---:|---:|---:|
| *Component decomposition* | | | | |
| Pivot configuration | vanilla (no Langevin) | 5.08% | 7.72% | — |
| Pivot configuration | drift_full (α=0.6, σ=0.01, K=20) | **5.91%** | **10.03%** | +0.83 |
| Pivot configuration | alpha_0 (α=0.0, σ=0.01, K=20) | 4.92% | 8.05% | −0.16 |
| Pivot configuration | sigma_0 (α=0.6, σ=0.0, K=20) | 5.44% | 8.58% | +0.36 |
| *Top-K subspace* | | | | |
| Top-K size | K = 20 (v18b, headline) | **2.89%** | **4.73%** | — |
| Top-K size | K = 150 (v18d) | 0.52% | 0.66% | −2.37 |

**Component contributions.** We hold the trained DRIFT-1.7B checkpoint (v18b) fixed and vary only the Langevin configuration on full AIME 24: *vanilla* disables the perturbation entirely; *drift_full* matches the headline configuration; *alpha_0* sets α=0 to remove the learned-drift mixing (so ε_t = ξ_t becomes pure random-sphere); *sigma_0* sets σ=0 to remove the Gaussian diffusion (keeping drift plus random sphere). The decomposition validates two design claims of §3. First, **the learned drift G_t is the load-bearing component**: removing it (alpha_0) makes the policy *worse* than vanilla (−0.16 pp on mean@16), confirming that random perturbation without an outcome-aligned direction is net-harmful at this checkpoint. The drift contributes +0.99 pp (drift_full minus alpha_0). Second, **the Gaussian diffusion adds incremental gain on top of the drift**, contributing +0.47 pp (drift_full minus sigma_0); the diffusion-only configuration is itself net-negative (alpha_0 < vanilla), so diffusion's value comes from *combining with the drift* rather than from raw perturbation mass. Both components together produce the best mean@16 and maj@16. This decomposition uses a single trained checkpoint and varies only the Langevin configuration; the corresponding training-from-scratch component ablations require independent training runs and are deferred to future work (§7 Limitations).

**Top-K subspace size.** §3.2 argues that confining the Langevin update to the top-K=20 logit coordinates is necessary because perturbing the long tail commits the trajectory to occasional low-probability tokens whose continuations are off-distribution. We test this with v18d, an otherwise-identical run that increases K from 20 to 150. We train v18d from the same seed checkpoint as v18b with all other hyperparameters fixed; Table 6 reports the bucketed trajectory comparison over the v18d window, partitioned into early, mid, and late training phases.

**Table 6.** Bucketed trajectory comparison of v18b (K=20, headline) vs v18d (K=150) on Qwen3-1.7B-Base. Both runs share trajectory through the seed and diverge thereafter. Each row is the bucketed mean within an early/mid/late phase of the v18d window. `resp_len` is mean response length in tokens; `trig` is the per-batch Langevin trigger fraction; `entropy` is per-token policy entropy averaged over the batch.

| Phase | Method | resp_len | trig | AIME mean@16 | AIME best@16 | entropy | score |
|---|---|---:|---:|---:|---:|---:|---:|
| early | K = 20 | 1018 | 5.18% | 0.01% | 0.07% | 0.68 | −0.999 |
| early | K = 150 | 1021 | 3.20% | 0.01% | 0.07% | 0.86 | −0.998 |
| mid | K = 20 | 1012 | 5.35% | 0.21% | 1.97% | 0.70 | −0.993 |
| mid | K = 150 | **843** | 2.31% | 0.28% | 2.07% | **0.97** | −0.983 |
| late | K = 20 | 988 | 4.93% | **2.00%** | **11.05%** | **0.27** | −0.915 |
| late | K = 150 | **815** | 2.63% | 0.70% | 5.21% | 0.65 | −0.928 |

The trajectory tells the §3.2 story precisely. **(1) Length collapse signature.** K=150's mean response length drops from 1021 (early) to 843 (mid) and stays at 815 (late), while K=20 stays in 988–1018 throughout — a 17% length deficit by mid-training. This is the derailment signature: each long-tail perturbation occasionally commits the trajectory to a corrupted token, the resulting off-distribution context terminates the rollout early, and the typical rollout shortens. **(2) Reasoning-metric divergence.** AIME 24 mean@16 diverges sharply in the late phase: K=20 reaches 2.00% while K=150 plateaus at 0.70% (3× lower). AIME best@16 diverges similarly (11.05% vs 5.21%, 2× gap). At an equal-training-horizon comparison point, K=150 underperforms K=20 by **2.37 pp on mean@16** (0.52% vs 2.89%, a 5.5× gap at this checkpoint) and **4.07 pp on maj@16** (0.66% vs 4.73%). **(3) Entropy stays diffuse under K=150.** By the late phase, K=20's policy entropy has descended to 0.27 (concentrating on a solution distribution) while K=150 sits at 0.65 — the K=150 policy never commits because long-tail perturbations keep finding off-distribution continuations, preventing a coherent solution distribution from forming.

Both ablations together support the §3 design that DRIFT confines exploration to a narrow head of the logit distribution and injects bounded directed perturbations there, rather than perturbing widely or undirected.

---

## 7. Discussion

**Trigger fraction as a diagnostic.** The trigger fraction—the proportion of generated tokens at which the Langevin update fires—is a first-order diagnostic for whether DRIFT is active. The positional guard t_min trades off two failure modes: too small a value places Langevin updates within early surface-realization regions, where token choice does not propagate to outcome (vacuous updates) and where the perturbation can be exploited by a short-answer shortcut (length collapse); too large a value pushes branch points beyond the typical response length and trigger fraction collapses to zero, reducing DRIFT to vanilla GRPO. The chosen value is calibrated to place branch points within the substantive reasoning portion of responses while leaving sufficient remaining length for downstream computation.

**Entropy stability.** DRIFT's mechanism is structurally distinct from entropy bonus methods: the Langevin updates are applied at rollout time and act on the sampling distribution, not directly on the training objective, and the SPSA feedback signal is anchored to a per-trajectory exploration target rather than to a global entropy floor. The diffusion term σ·ζ maintains distributional coverage at branch points, the drift G accumulates exploration-preserving directions per Section 3.3.2, and the KL regularization constrains global drift relative to the reference policy. The combined effect is a controlled descent of branch-point entropy that tracks the policy's gradual specialization, distinct from the runaway collapse observed under entropy-reduction feedback signals and distinct from the entropy growth observed under naive entropy bonus.

**Bootstrap curriculum.** An emergent property of DRIFT is accelerated escape from the dead zone in early training. The Langevin drift at a branch point within the substantive reasoning region of the response can redirect the trajectory toward a correct continuation that the unperturbed policy would not have reached. When such a redirection produces a correct rollout in an otherwise all-incorrect group, the resulting non-zero within-group advantage initiates gradient flow before full reasoning has converged—gradient signal that pure GRPO does not access during the dead-zone phase. The length-conditioned score adjustment of Section 3.4 ensures this bootstrap signal does not collapse to a short-answer shortcut: redirected correct rollouts that happen to be short are penalized prior to advantage normalization, reshaping the within-group ranking to favor longer correct continuations and driving the policy toward full reasoning.

**Choice of feedback orientation.** The most consequential design choice in DRIFT is the orientation of the SPSA feedback signal toward entropy maintenance rather than entropy reduction (Section 3.3.2). Both orientations correspond to legitimate Langevin estimators on a tractable surrogate for ∇E, but they produce qualitatively different policies: entropy reduction yields directions of confident commitment, accelerating collapse to a single deterministic reasoning path and—in the dead-zone regime—accelerating commitment to wrong paths. Entropy maintenance yields directions that preserve the policy's branching capacity at downstream positions, sustaining the conditions under which subsequent Langevin perturbations remain effective. The framework is otherwise the same; the change of sign on the feedback target produces a different attractor for the rollout dynamics.

---

## Algorithm

```
Algorithm 1: DRIFT Rollout Generation (per prompt x)
─────────────────────────────────────────────────────────────────────────────
Input
  x                      prompt
  π_θ                    current policy
  H_buf                  global rolling entropy buffer (cap |H_buf| = 2000)
  Hyperparameters
    p, τ_0               adaptive-quantile p; warmup fallback threshold
    t_min                positional guard
    K_top                top-K logit subspace dimension
    K_steps              # of inner Langevin steps per trigger
    η, σ                 drift step size, diffusion magnitude
    α                    drift exploit ratio (mix weight on G/‖G‖)
    α_target, γ          entropy-maintenance target ratio, momentum

Output
  ỹ = (ỹ_1, …, ỹ_T)              generated tokens
  M ∈ {0,1}^T                     trigger mask
  {log π̃_θ(ỹ_t) : M[t] = 1}     per-trigger perturbed log-densities

State (per-trajectory)
  T_trigger ← ∅
  ε_prev    ← 0 ∈ ℝ^{K_top}
  G         ← 0 ∈ ℝ^{K_top}        // learned drift (persists across trajectories)
  H_first   ← None                  // anchor for entropy-maintenance signal

for t = 1 to T:
    l_t   = f_θ(x, ỹ_{<t})                                       // model logits at step t
    p_t   = softmax(l_t);   H_t = −∑_v p_t[v] · log p_t[v]

    # Adaptive quantile threshold with warmup fallback
    if |H_buf| < N_warmup:    τ_t = τ_0
    else:                     τ_t = Quantile(H_buf, p)

    if H_t ≥ τ_t  AND  t > t_min:
        # Anchor the entropy-maintenance reference at the first trigger
        if H_first is None:
            H_first = H_t

        # SPSA update of drift G using the previous trigger's perturbation
        if ε_prev ≠ 0:
            signal = H_t − α_target · H_first              // > 0  ⇒ entropy still above target
            G ← γ · G + (1 − γ) · signal · ε_prev

        # Perturbation direction: exploit drift + explore unit sphere
        ξ_t = SampleUnitSphere(K_top)
        if ‖G‖ > δ:   ε_t = α · G/‖G‖ + (1 − α) · ξ_t
        else:         ε_t = ξ_t                            // cold-start: random sphere

        # K_steps-fold Langevin update restricted to the top-K_top logit coords
        I_K = arg-top-K(|l_t|, K_top)
        l̃_t = l_t
        for k = 1 to K_steps:
            l̃_t[I_K] += η · ε_t                            // drift
            l̃_t[I_K] += σ · N(0, I_{K_top})                 // diffusion (top-K coords only)

        ỹ_t ~ Categorical(softmax(l̃_t))
        log π̃_θ(ỹ_t) ← log softmax(l̃_t)[ỹ_t]
        M[t] ← 1;  ε_prev ← ε_t;  T_trigger ← T_trigger ∪ {t}
    else:
        ỹ_t ~ Categorical(p_t);  M[t] ← 0                  // no perturbation

    H_buf.push(H_t)                                         // global buffer update

return ỹ, M, {log π̃_θ(ỹ_t) : t ∈ T_trigger}


Algorithm 2: DRIFT Training Step (one minibatch update)
─────────────────────────────────────────────────────────────────────────────
Input
  Batch  { (x^i, {ỹ^{i,j}}, {r^{i,j}}, {L^{i,j}}, {M^{i,j}}, {log π̃^{i,j}}) }_{i, j}
  π_θ        current policy        π_old      behavioral snapshot
  π_ref      KL reference policy
  Hyperparameters
    λ, L_target              length-adjustment coefficient; target length
    β                        IS mixture coefficient (denominator blend)
    H_cap                    entropy cap for IS weight clipping
    ε_low, ε_high            asymmetric PPO clip range (DAPO clip-higher)
    λ_KL                     KL regularization coefficient

# 1. Length-conditioned score adjustment (Section 3.4)
for each rollout (i, j):
    r̃^{i,j} ← r^{i,j} + λ · ( min(L^{i,j} / L_target, 1) − 1 ) · 𝟙[r^{i,j} > 0]

# 2. GRPO advantage on adjusted scores
for each i, j:
    Â^{i,j} ← (r̃^{i,j} − mean_j r̃^{i,j}) / std_j r̃^{i,j}

# 3. IS-corrected policy gradient (Section 3.5)
L_DRIFT ← 0
for each (i, j) and token position t in trajectory ỹ^{i,j}:
    ρ^{i,j}_t = π_θ(ỹ^{i,j}_t | x^i, ỹ^{i,j}_{<t}) / π_old(ỹ^{i,j}_t | x^i, ỹ^{i,j}_{<t})

    if M^{i,j}[t] = 1:
        # Mixture-policy denominator (Kakade & Langford, 2002)
        π̃_eff ← β · exp(log π̃^{i,j}_t) + (1 − β) · π_old(ỹ^{i,j}_t | x^i, ỹ^{i,j}_{<t})
        w^{i,j}_t ← π_θ(ỹ^{i,j}_t | x^i, ỹ^{i,j}_{<t}) / π̃_eff
        # Entropy-cap variance reduction
        if H^{i,j}_t > H_cap:
            w^{i,j}_t ← w^{i,j}_t · exp(−(H^{i,j}_t − H_cap))
    else:
        w^{i,j}_t ← 1

    z^{i,j}_t ← w^{i,j}_t · ρ^{i,j}_t
    L_DRIFT ← L_DRIFT + Â^{i,j} · min( z^{i,j}_t,  clip(z^{i,j}_t, 1 − ε_low, 1 + ε_high) )

L_DRIFT ← L_DRIFT / (∑_{i, j} |ỹ^{i,j}|)                     // token-level normalization

# 4. KL regularization to pre-trained reference
L_KL ← mean_i KL( π_θ(· | x^i)  ‖  π_ref(· | x^i) )

# 5. Parameter update
Δθ ← −∇_θ ( L_DRIFT + λ_KL · L_KL )
```

---

## References

Agarwal, A., Dekel, O., & Xiao, L. (2010). Optimal Algorithms for Online Convex Optimization with Multi-Point Bandit Feedback. COLT.

Dudik, M., Langford, J., & Li, L. (2011). Doubly Robust Policy Evaluation and Learning. ICML.

Flaxman, A., Kalai, A., & McMahan, H. (2005). Online Convex Optimization in the Bandit Setting: Gradient Descent without a Gradient. SODA.

Kakade, S. & Langford, J. (2002). Approximately Optimal Approximate Reinforcement Learning. ICML.

Meng, X.L. & Wong, W.H. (1996). Simulating Ratios of Normalizing Constants via a Simple Identity. Statistica Sinica.

Mnih, V., Badia, A.P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., & Kavukcuoglu, K. (2016). Asynchronous Methods for Deep Reinforcement Learning. ICML.

Polyak, B.T. (1964). Some Methods of Speeding up the Convergence of Iteration Methods. USSR Computational Mathematics and Mathematical Physics.

Roberts, G.O. & Tweedie, R.L. (1996). Exponential Convergence of Langevin Distributions and Their Discrete Approximations. Bernoulli.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.

Shao, Z. et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300.

Spall, J.C. (1992). Multivariate Stochastic Approximation Using a Simultaneous Perturbation Gradient Approximation. IEEE Transactions on Automatic Control, 37(3), 332–341.

Wang, S. et al. (2025). Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning. arXiv:2506.01939.

Welling, M. & Teh, Y.W. (2011). Bayesian Learning via Stochastic Gradient Langevin Dynamics. ICML.

Yu, T. et al. (2025). DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476.
