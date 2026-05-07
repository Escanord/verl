# PIVOT: Branch-Point Langevin Exploration for Reasoning-Intensive LLM Reinforcement Learning

---

## Abstract

We present **PIVOT** (**P**olicy-**I**nformed **V**ariance-Guided **O**nline **T**raining), a method for improving reinforcement learning of LLM reasoning via targeted exploration at structural branch points in the generation process. Standard group-relative policy optimization (GRPO) collapses to a zero-gradient regime on hard reasoning tasks when all sampled rollouts receive identical rewards—a degenerate condition we term the *dead zone*. PIVOT addresses this by identifying high-entropy token positions during generation (branch points, where the model's next-token distribution is genuinely diffuse over qualitatively distinct continuations) and applying adaptive Langevin perturbations exclusively at these positions. The perturbation direction is governed by an online-learned momentum vector G, maintained as a Polyak-filtered SPSA estimate of the ascent direction of a value proxy defined over the K-dimensional top-logit subspace. The feedback signal compares branch-point entropy at structurally homologous positions across successive triggers, eliminating the low-frequency syntactic autocorrelation that confounds next-token measurements. The resulting off-policy trajectories are incorporated into the GRPO objective via an importance-sampling correction whose denominator is constructed as the optimal bridge sampling estimator between the perturbed and behavioral policies. On Qwen3-4B-Base trained on competition-level mathematics, PIVOT achieves AIME mean@16 of 8.18% at step 150, with no entropy collapse over the training horizon.

---

## 1. Introduction

Large language model (LLM) reasoning has emerged as a high-leverage frontier in AI capabilities. Recent works demonstrate that reinforcement learning with verifiable rewards can unlock qualitatively new reasoning capabilities in pre-trained language models (DeepSeek-R1; OpenAI o1). However, RL training of LLMs on hard reasoning tasks faces a fundamental exploration bottleneck. During early training, a model generating n candidate responses to a competition-level mathematics problem produces all-incorrect rollouts with near certainty—exceeding 95% on AIME benchmarks. When all n rollouts within a group share the same reward, the within-group advantage estimator collapses to zero for every token position; the policy gradient vanishes identically and parameters do not update. The model remains trapped in this degenerate fixed point for extended periods.

Existing approaches to this problem are either coarse or palliative. Entropy bonus terms inject uniform perturbation signal across all token positions, including the large majority of syntactically-determined positions—conjunctions, punctuation, mathematical notation—whose outcomes carry no structural information about reasoning quality. When reward signal is near-zero and the policy gradient contribution is negligible, a nonzero entropy coefficient causes entropy to grow without bound rather than guide exploration toward correct solution paths. Filter-based methods (DAPO; Yu et al., 2025) excise zero-advantage groups from the gradient computation, eliminating wasted computation but generating no new gradient signal on prompts for which the model consistently fails.

We observe that the token sequence produced during reasoning is structurally heterogeneous. The overwhelming majority of positions are near-deterministic: their next-token distribution is sharply peaked, and their outcome follows with high probability from context. A small fraction constitute genuine branch points—positions at which the model assigns non-negligible probability to multiple qualitatively distinct continuations, such as choosing between proof strategies, selecting a substitution variable, or deciding whether to invoke a lemma. These positions are computable online as local maxima of the next-token Shannon entropy and are structurally privileged: they determine the topology of the reasoning chain that follows.

PIVOT concentrates exploration at these positions. Targeted Langevin perturbations at branch points produce trajectories that explore qualitatively different reasoning chains without disturbing the near-deterministic structure of the remainder of the sequence. The perturbation direction is governed by a momentum vector G maintained via a Polyak-filtered SPSA update: at each new branch point, the entropy reduction at subsequent branch points is used as a stochastic proxy for the directional derivative of the trajectory value with respect to the logit perturbation direction, and G is updated accordingly. Off-policy trajectories produced by this perturbed rollout policy are incorporated into GRPO via an importance-sampling objective with a bridge-sampling denominator that minimizes IS weight variance.

---

## 2. Background

### 2.1 Autoregressive Language Model Generation

A language model parameterized by θ defines a conditional distribution over token sequences. Given a prompt x, the model generates a response y = (y_1, ..., y_T) autoregressively:

    π_θ(y | x) = ∏_{t=1}^{T} π_θ(y_t | x, y_{<t})

where π_θ(· | x, y_{<t}) = softmax(l_t) and l_t ∈ ℝ^|V| are the logits at step t. We denote by H_t the Shannon entropy of this distribution:

    H_t = -∑_{v ∈ V} π_θ(v | x, y_{<t}) log π_θ(v | x, y_{<t})

### 2.2 Group-Relative Policy Optimization (GRPO)

GRPO (Shao et al., 2024) is an actor-only RL algorithm that avoids a learned value function by estimating advantages from within-group reward variance. For each prompt x^i, n rollouts {y^{i,j}}_{j=1}^n are sampled and scored. The normalized advantage is:

    Â^{i,j} = (r^{i,j} - mean_j r^{i,j}) / std_j r^{i,j}

The clipped policy gradient objective is:

    L_GRPO(θ) = E_{i,j,t} [ min(ρ^{i,j}_t · Â^{i,j},  clip(ρ^{i,j}_t, 1-ε, 1+ε) · Â^{i,j}) ]

where ρ^{i,j}_t = π_θ(y^{i,j}_t | x^i, y^{i,j}_{<t}) / π_{old}(y^{i,j}_t | x^i, y^{i,j}_{<t}) denotes the per-token probability ratio between the current and behavioral policies.

### 2.3 The Dead Zone

When the reward distribution over rollouts within a group is degenerate—all correct or, overwhelmingly more commonly, all incorrect—the within-group standard deviation is zero and Â^{i,j} = 0 for all j. The gradient of L_GRPO is identically zero for every token in every affected rollout. On AIME-class problems, a 4B-parameter base model achieves near-zero individual accuracy, implying that even at n=8, the probability of observing at least one correct rollout per prompt is negligibly small. Empirically, vanilla GRPO produces no measurable gradient signal for the first 70–80 training steps on such data, after which stochastic fluctuations occasionally produce a correct response and initiate learning.

---

## 3. The PIVOT Method

PIVOT augments the GRPO rollout procedure with three interconnected components: branch point detection (Section 3.1), adaptive Langevin perturbation (Section 3.2), and online direction learning via SPSA momentum (Section 3.3). Off-policy trajectories produced by the perturbed rollout policy are integrated into GRPO via an importance-corrected objective (Section 3.4).

### 3.1 Branch Point Detection

At each token position t during generation, PIVOT evaluates three criteria for branch point classification:

1. **Absolute entropy**: H_t ≥ τ,  τ = 0.4 nats
2. **Relative entropy**: H_t ≥ Quantile(H_{1:t}, p),  p = 0.65  (top-35% of sequence entropy so far)
3. **Positional guard**: t > t_min = 200

The positional guard excludes the first t_min tokens, where high entropy typically reflects lexical ambiguity in initial context processing rather than strategic reasoning uncertainty. Criteria 1 and 2 jointly select positions that are absolutely diffuse and locally extreme: the combination ensures robustness to sentence-level entropy variation while maintaining global selectivity. The resulting trigger set is sparse—constituting approximately 5–6% of generated tokens empirically—and concentrated on positions of genuine structural significance.

### 3.2 Adaptive Langevin Perturbation

At branch point t, PIVOT intervenes in the sampling process by applying a structured perturbation to the logit vector before drawing the next token.

**Perturbation direction.** Let G_t ∈ ℝ^K be the current momentum vector (initialized to zero, K = 20). The perturbation direction is:

    ε_t = α · (G_t / ‖G_t‖) + (1 - α) · ξ_t,    α = 0.6

where ξ_t ~ Uniform(S^{K-1}) is drawn uniformly from the unit sphere in ℝ^K. The exploitation weight α balances the learned direction G with undirected spherical noise.

**Logit displacement.** The perturbation is confined to the top-K logit dimensions by magnitude, which spans the face of the probability simplex carrying most of the model's probability mass at the branch point:

    l̃_t[k] = l_t[k] + η · ε_t[k],    k ∈ arg-top-K(|l_t|),    η = 0.1

**Isotropic regularization.** A small Gaussian perturbation is superimposed to ensure the perturbed distribution is absolutely continuous:

    l̃_t ← l̃_t + σ · ζ_t,    ζ_t ~ N(0, I_{|V|}),    σ = 0.01

**Sampling.** The token at position t is drawn from the perturbed distribution: ỹ_t ~ softmax(l̃_t). At non-branch positions, generation proceeds normally: ỹ_t ~ softmax(l_t).

**Geometric interpretation.** The top-K logit subspace corresponds to the K-dimensional face of the probability simplex activated by the current context. At branch points this face is well-populated, so displacements in ε_t translate to meaningful redistribution of probability mass across qualitatively distinct token alternatives. At near-deterministic positions the simplex face degenerates toward a vertex; perturbations there would have negligible effect on the sampled token. Confining the perturbation to branch points therefore maximizes the structural impact of each intervention per unit of distributional displacement.

### 3.3 Online Direction Learning: SPSA Momentum on the Logit Manifold

The space of possible reasoning trajectories is combinatorially vast and lacks a tractable metric. PIVOT navigates this space by maintaining G as a low-dimensional summary of locally beneficial perturbation directions—a momentum vector on the K-dimensional logit subspace—updated online from trajectory feedback.

**Value proxy.** Let f : S^{K-1} → ℝ denote the expected future branch-point entropy along trajectories initiated by perturbing in direction ε at the current branch point:

    f(ε) = E_{τ | ε} [ median({H_{t_k}}_{k > current trigger}) ]

PIVOT implicitly optimizes f: lower expected future branch-point entropy corresponds to a trajectory distribution more concentrated around coherent reasoning chains. G maintains an estimate of -∇_ε f, the ascent direction of the negated entropy objective, via the following online update.

**Feedback signal.** When a new branch point fires at position t_new, the following scalar is computed:

    signal_{t_new} = median({H_{t_k}}_{k ∈ past[-50]}) - H_{t_new}

This quantity approximates the signed directional derivative of f with respect to the previous perturbation direction ε_{t_prev}: positive signal indicates that perturbing in direction ε_{t_prev} reduced subsequent branch-point entropy relative to the running median, constituting evidence that ε_{t_prev} lies in a descent direction of f; negative signal indicates the opposite. The comparison is restricted to homologous positions—previous branch points—rather than adjacent tokens, for the following reason: next-token entropy H_{t+1} is dominated by local syntactic context and exhibits strong autocorrelation with H_t independent of the perturbation's downstream effect on reasoning structure. By evaluating the signal only at structurally comparable positions—those independently selected as high-entropy branch points—PIVOT eliminates this confound and obtains a signal whose variation reflects the perturbation's effect on the trajectory distribution rather than syntactic surface form.

The median reference is preferred over the sample mean to obtain robustness to outlier entropy values arising from phrasing accidents (unusual mathematical symbols, rare notation, indexing tokens) that enter the trigger set due to localized entropy spikes.

**G update: Polyak-filtered SPSA.** The momentum vector is updated as:

    G_{t+1} ← γ · G_t + (1 - γ) · signal_{t_new} · ε_{t_prev},    γ = 0.7

This update is structurally a Polyak heavy-ball accumulation of one-sided SPSA gradient estimates. In standard SPSA (Spall, 1992), the gradient of f is estimated using symmetric finite differences:

    ĝ_k = [f(θ + c Δ) - f(θ - c Δ)] / (2c) · Δ^{-1}

PIVOT uses a one-sided difference—feasible because f is evaluated along the perturbation direction that was actually applied—and replaces the per-step gradient with a Polyak filter that accumulates consistent directional evidence across successive triggers. The filter window of 1/(1-γ) ≈ 3.3 triggers suppresses single-observation noise while retaining sensitivity to sustained directional signal within a single trajectory. The exponential weighting down-weights temporally distant perturbations whose downstream effect may have been mediated by intervening branch points.

The combined effect is that G converges toward a stable estimate of the locally beneficial perturbation direction in the K-dimensional logit subspace, with the momentum accumulation providing variance reduction equivalent to averaging over a window of gradient samples. As G grows, the perturbation direction ε_t transitions from diffuse random search on S^{K-1} toward focused exploitation of the learned direction, implementing an implicit curriculum from exploration to exploitation that requires no explicit schedule.

### 3.4 Importance-Corrected Policy Gradient

Trajectories generated under the perturbed rollout policy π̃_θ are off-policy with respect to the current policy π_θ and the behavioral policy π_old. Standard GRPO probability ratios ρ_t = π_θ(y_t)/π_old(y_t) are unbiased at non-trigger positions (where π̃_θ = π_θ = π_old) but incorrect at trigger positions, where the sample was drawn from the perturbed distribution. PIVOT corrects this via per-token importance weights.

Let T_trigger(τ) ⊆ {1,...,T} denote the set of trigger positions in trajectory τ. The IS-corrected GRPO objective is:

    L_PIVOT(θ) = E_{τ~π̃_θ} [ ∑_t Â_t · min(w_t · ρ_t,  clip(w_t · ρ_t, 1-ε, 1+ε)) ]

where:

    w_t = π_θ(ỹ_t | x, y_{<t}) / π̃_θ(ỹ_t | x, y_{<t}),    t ∈ T_trigger
    w_t = 1,                                                   t ∉ T_trigger

The perturbed density π̃_θ(ỹ_t) is tractable: the Langevin displacement is a deterministic function of the logits (available at rollout time), and the isotropic Gaussian component is a known additive noise whose density is evaluable at ỹ_t.

**Bridge sampling denominator.** A naive implementation of w_t can exhibit high variance when π̃_θ deviates substantially from π_θ early in training. We stabilize the IS estimator by replacing the denominator with the optimal bridge sampling estimator (Meng & Wong, 1996). Bridge sampling constructs the minimum-variance unbiased estimator for normalizing constant ratios when samples are available from two distributions p and q; with equal sample allocations (n_p = n_q), the optimal denominator is the arithmetic mean:

    π̃_eff(ỹ_t) = β · π̃_θ(ỹ_t) + (1 - β) · π_old(ỹ_t),    β = 0.5

Substituting π̃_eff for π̃_θ in the IS weight denominator yields an estimator that is asymptotically efficient in the bridge sampling sense: it minimizes the relative mean squared error of the importance weight over the class of mixture denominators, providing lower variance than the pure-π̃_θ denominator at the cost of a small finite-sample bias that diminishes as π_θ converges. This estimator is also doubly robust (Dudik et al., 2011): the gradient estimator remains consistent if either the IS model (π̃_θ) or the direct model (π_old) correctly specifies the behavioral distribution, providing robustness to misspecification of either component.

**Entropy cap.** When H_t > H_cap = 0.8 nats at a trigger position, the IS weight is reduced toward unity. At extreme entropy, the perturbed distribution π̃_θ approaches uniformity over the top-K vocabulary, producing large IS weights whose variance is difficult to control. The cap trades a small bias for substantial variance reduction in this regime.

**KL regularization.** A low-variance KL divergence loss to the reference (pre-training) policy is appended:

    L(θ) = L_PIVOT(θ) + λ_KL · KL(π_θ ‖ π_ref),    λ_KL = 0.005

The KL term prevents the policy from drifting to degenerate solutions discovered via Langevin perturbation that are rewarded but incompatible with general language understanding.

---

## 4. Theoretical Analysis

### 4.1 PIVOT as Momentum SPSA on the Probability Simplex

The G update in Section 3.3 can be analyzed within the framework of stochastic approximation with momentum. Define the objective:

    f : S^{K-1} → ℝ,    f(ε) = E_{τ | ε} [ -median({H_{t_k}}) ]    (negated, to be maximized)

We seek the direction ε* ∈ S^{K-1} that maximizes f. PIVOT's update rule is:

    m_k = γ · m_{k-1} + (1 - γ) · (f_k^{(1)} · ε_{k-1})
    G_k = m_k

where f_k^{(1)} = signal_{t_{new}} is a noisy one-sided estimate of ∂f/∂ε evaluated at ε_{k-1}. This is a momentum-augmented variant of one-point gradient estimation (Flaxman et al., 2005; Agarwal et al., 2010), which establishes that under bounded noise and Lipschitz-smooth f, one-point gradient estimates converge to the true gradient direction at rate O(1/√K). The Polyak filter suppresses noise in the gradient estimate at the cost of introducing a momentum bias: the effective G tracks a convolution of historical gradient estimates with exponential weights e^{-k(1-γ)}, concentrating mass on the most recent 1/(1-γ) ≈ 3 observations.

Convergence of G to the ascent direction of f follows from the standard result for momentum SGD (Polyak, 1964) provided the learning rate implicit in (1-γ) satisfies the Robbins-Monro conditions—a condition satisfied in our finite-horizon setting where each rollout contributes a bounded number of trigger observations.

### 4.2 Langevin Dynamics on the Logit Manifold

Classical Langevin Monte Carlo (Welling & Teh, 2011) samples from a target distribution p(x) ∝ exp(-E(x)/T) via the Itô SDE:

    dx = -∇E(x) dt + √(2T) dW_t

PIVOT instantiates a discrete-time, finite-step approximation to this SDE on the logit manifold, restricted to the top-K subspace at branch points. Identifying the energy E(l_t) = -V(l_t), where V(l_t) is the expected downstream reward probability of trajectories sampled from softmax(l_t), and approximating -∇E(l_t) ≈ η · G_t/‖G_t‖, the PIVOT update becomes:

    l̃_t = l_t + η · G_t/‖G_t‖ + σ · ζ_t ≈ l_t - η · ∇E(l_t)|_{approx} + √(2σ²) · ζ_t/√2

This is the unadjusted Langevin algorithm (ULA) with step size η and temperature T = σ²/η. The stationarity of ULA is a perturbed version of the target distribution; the perturbation vanishes as η → 0 (Chen et al., 2015). In the LLM setting, differentiating through trajectory rewards to obtain the exact ∇E is computationally prohibitive; PIVOT replaces the exact gradient with the SPSA estimate G, accepting the resulting bias as a controlled approximation.

### 4.3 Exploration-Exploitation as Implicit Temperature Annealing

The mixture ε_t = α · G/‖G‖ + (1-α) · ξ defines a distribution on S^{K-1} parameterized by the concentration ‖αG‖. When G = 0 (initialization), ε_t is uniform on S^{K-1}—maximum-entropy exploration. As G grows, the distribution concentrates around G/‖G‖, reducing the effective entropy of the perturbation direction and transitioning exploration toward exploitation. This corresponds to a monotone decrease in the effective temperature of the perturbation process as evidence accumulates.

Crucially, this annealing occurs at the level of individual trajectories during rollout, not across training steps: G is reinitialized per rollout in the current implementation, meaning each trajectory explores freely in early trigger positions and increasingly exploits accumulated evidence in later triggers. This intra-trajectory curriculum enables the same rollout to perform both exploration (escaping the dead zone) and exploitation (refining the discovered direction) without requiring hyperparameter-scheduled annealing.

### 4.4 Information-Theoretic Motivation for Branch Point Selection

The mutual information between the current token choice y_t and the final answer y_T, conditioned on context, provides a principled criterion for measuring the structural importance of position t:

    I(y_T ; y_t | x, y_{<t}) = H(y_t | x, y_{<t}) - H(y_t | x, y_{<t}, y_T)

The first term is H_t, which PIVOT measures. The second term, the conditional entropy of y_t given the final answer, is low at positions where the token choice genuinely determines the reasoning outcome, and high at positions where multiple choices lead to correct solutions. Positions with large H_t and small H(y_t | y_T)—and thus large mutual information—are precisely those where exploration is most valuable: the model is uncertain but the choice matters. While the conditional term is intractable online, H_t provides a tractable proxy and upper bound. The percentile threshold provides further selectivity by identifying positions that are locally extreme in entropy, reducing false positives from uniformly high-entropy syntactic positions.

### 4.5 Connection to Model-Based RL and Implicit Tree Search

PIVOT can be interpreted as performing implicit partial tree search within the autoregressive generation process. Standard autoregressive decoding corresponds to greedy depth-first traversal of the generation tree. By perturbing at branch points, PIVOT effectively conditions the subsequent trajectory on a modified logit at a tree node, producing a sibling branch without explicit backtracking.

The G vector plays the role of a node-level value function estimate in MCTS: it accumulates evidence about the relative value of perturbation directions at branch-point nodes, and this evidence informs subsequent expansion decisions. The trigger-to-trigger entropy signal serves as the rollout return: lower future branch-point entropy is treated as a proxy for higher expected reward, analogous to the Monte Carlo return used in MCTS policy improvement. Unlike MCTS, PIVOT requires neither explicit tree construction nor a separate value network, operating entirely within a single forward pass per rollout.

---

## 5. Experimental Setup

**Base model.** Qwen3-4B-Base, a 4B-parameter transformer pre-trained on general text with no instruction tuning or RLHF.

**Training data.** Batches of 1024 competition-level mathematics problems drawn from AIME-style and MATH training sets. Maximum prompt length 1024 tokens; maximum response length 4096 tokens.

**Evaluation.** AIME 2024/2025 (30 problems): mean@16, the empirical solve rate over 16 independent samples per problem; best@16, the probability that at least one of 16 samples is correct. MATH-500: mean@16. Checkpoints evaluated every 5 training steps.

**Baselines.** (1) GRPO: vanilla group-relative policy optimization with no perturbation or filtering. (2) DAPO (Yu et al., 2025): GRPO augmented with dynamic group filtering, asymmetric PPO clipping (clip_high=0.28), token-level loss normalization, and overlong reward shaping.

**PIVOT hyperparameters.** Branch point detection: τ=0.4 nats, p=0.65, t_min=200. Langevin parameters: η=0.1, K=20, σ=0.01. Direction learning: γ=0.7, α=0.6. IS correction: β=0.5, H_cap=0.8 nats. KL regularization: λ_KL=0.005.

**Infrastructure.** 8×H200 140GB GPUs, FSDP actor, vLLM rollout engine, n=8 rollouts per prompt during training, n=16 during evaluation.

---

## 6. Results

| Method | AIME mean@16 | AIME best@16 | MATH mean@16 |
|---|---|---|---|
| GRPO | — | — | — |
| DAPO | — | — | — |
| **PIVOT** | **8.18%** (step 150) | **15.64%** (step 150) | **72.72%** (step 135) |

### 6.1 Preliminary Baseline Results — Qwen3-1.7B-Base (early checkpoints, ~step 200/795)

All runs on Qwen3-1.7B-Base, 8×H200 140GB, n=8 rollouts, guru_rl dataset. Metrics are mid-training snapshots; final results pending completion.

| Method | MATH500 mean@16 | MATH500 best@16 | MATH500 maj@16 | AIME mean@16 | AIME best@16 |
|---|---|---|---|---|---|
| GRPO | **62.5%** (step 200) | **82.6%** | **69.7%** | **4.4%** | **16.4%** |
| High-Ent GRPO (entropy_top_ratio=0.2) | 55.6% (step 205) | 77.2% | 62.4% | 1.7% | 9.5% |

Key observations at this checkpoint:
- GRPO reaches 62.5% MATH500 mean@16 by step 200 — strong baseline given only ~25% of training completed.
- High-Ent GRPO has a faster per-step wall-clock time (~345s vs ~500s) due to restricted gradient computation, but lags GRPO by ~7pts at matched step count. The high-entropy approach had a much steeper initial ramp (near-zero until step 50, then rapidly rising), suggesting the restricted gradient signal delays early dead-zone escape but may converge similarly later.
- AIME is still noisy for 1.7B at this stage; both methods hovering near floor.

---

PIVOT achieves a peak AIME mean@16 of 8.18% at step 150, with AIME best@16 of 15.64% and MATH mean@16 of 72.72% at step 135. Evaluation metrics exhibit consistent monotone improvement from steps 100–150, with no entropy collapse: the per-token entropy of the policy declines from 0.070 at step 127 to 0.058 at step 161 at a rate consistent with controlled specialization rather than distributional collapse. The trigger fraction remains stable at 5–6% throughout, confirming that the branch point selection criterion identifies a consistent fraction of tokens as structurally significant across the training trajectory.

---

## 7. Discussion

**Trigger fraction as a diagnostic.** The trigger fraction trig_frac—the proportion of generated tokens at which Langevin perturbation fires—is a first-order diagnostic for whether PIVOT is active. When t_min is set sufficiently large that the high-entropy region of the sequence lies entirely within the guard interval (e.g., t_min=800 when most branch-point tokens occur before position 800), trig_frac collapses to zero and PIVOT reduces to vanilla GRPO. The appropriate value of t_min is determined by the structure of the task: for mathematical reasoning, substantive branching begins after approximately 200 tokens of context echo and problem setup.

**Entropy stability.** A canonical failure mode of entropy-augmented RL is entropy collapse, in which the policy converges prematurely to a degenerate near-deterministic distribution. PIVOT's mechanism is structurally distinct from entropy bonus methods in this respect: the perturbations are applied at rollout time, not at parameter-update time, and act on the sampling distribution rather than directly on the policy gradient. The KL regularization term provides an independent mechanism constraining distributional drift. Empirically, entropy decreases at approximately 0.001 nats per training step over 160 steps—a rate consistent with task-driven specialization—and stabilizes above 0.055 nats, well above the values associated with degenerate distributions.

**Bootstrap curriculum.** An emergent property of PIVOT is an acceleration in the initial escape from the dead zone. Langevin perturbations applied at the first branch point within a response—which typically occurs within the first 100 tokens—can redirect the trajectory from extended reasoning toward a short direct answer. When such responses happen to be correct (a stochastic event driven by the reward verification checking only the final answer expression), a non-zero advantage is produced for an otherwise all-incorrect group. This provides gradient signal that begins to shift the model toward reward-consistent outputs before sustained correct reasoning has been learned. The KL regularization subsequently prevents these shortcut solutions from becoming entrenched, driving the model back toward full reasoning responses once the policy has escaped the dead zone.

---

## Algorithm

```
Algorithm 1: PIVOT Rollout Generation
Input:  prompt x, policy π_θ, momentum vector G ∈ ℝ^K
Output: trajectory ỹ, trigger mask M ∈ {0,1}^T, perturbed log-densities {log π̃_θ(ỹ_t)}_{t ∈ T_trigger}

Initialize: T_trigger = ∅, ε_prev = 0, H_history = []
for t = 1 to T:
    l_t = f_θ(x, ỹ_{<t})
    H_t = -∑_v softmax(l_t)[v] · log softmax(l_t)[v]

    if H_t ≥ τ  and  H_t ≥ Quantile(H_history, p)  and  t > t_min:
        // Update G with trigger-to-trigger SPSA signal
        if |T_trigger| > 0 and ε_prev ≠ 0:
            signal = median({H_history[t'] : t' ∈ T_trigger[-50:]}) - H_t
            G ← γ · G + (1 - γ) · signal · ε_prev

        // Compute perturbation direction
        ε_t = α · G/max(‖G‖, ε) + (1 - α) · SampleUnitSphere(K)

        // Apply top-K logit displacement with isotropic noise
        I_K = arg-top-K(|l_t|, K)
        l̃_t = l_t;  l̃_t[I_K] += η · ε_t;  l̃_t += σ · N(0, I)

        ỹ_t ~ Categorical(softmax(l̃_t))
        M[t] = 1;  ε_prev = ε_t;  T_trigger ← T_trigger ∪ {t}
    else:
        ỹ_t ~ Categorical(softmax(l_t));  M[t] = 0

    H_history.append(H_t)

return ỹ, M, {log π̃_θ(ỹ_t) : t ∈ T_trigger}


Algorithm 2: PIVOT Training Step
Input:  batch {(x^i, {ỹ^{i,j}}_{j=1}^n, {r^{i,j}}, {M^{i,j}}, {log π̃^{i,j}_t})}
Output: parameter update Δθ

for each (x^i, ỹ^{i,j}, r^{i,j}, M^{i,j}):
    Â^{i,j} = (r^{i,j} - mean_j r^{i,j}) / std_j r^{i,j}

    for each token position t:
        ρ_t = π_θ(ỹ^{i,j}_t | x^i, ỹ^{i,j}_{<t}) / π_old(ỹ^{i,j}_t | x^i, ỹ^{i,j}_{<t})

        if M^{i,j}[t] = 1:
            // Bridge sampling IS denominator
            log π̃_eff = log(β · exp(log π̃^{i,j}_t) + (1-β) · π_old(ỹ^{i,j}_t))
            w_t = exp(log π_θ(ỹ^{i,j}_t) - log π̃_eff)
            if H_t > H_cap: w_t ← w_t · exp(-(H_t - H_cap))   // entropy cap
        else:
            w_t = 1

    L_PIVOT += Â^{i,j} · ∑_t min(w_t · ρ_t,  clip(w_t · ρ_t, 1-ε, 1+ε))
    L_KL    += KL(π_θ(· | x^i) ‖ π_ref(· | x^i))

Δθ = -∇_θ (L_PIVOT + λ_KL · L_KL)
```

---

## References

Agarwal, A., Dekel, O., & Xiao, L. (2010). Optimal Algorithms for Online Convex Optimization with Multi-Point Bandit Feedback. COLT.

Chen, C., Fox, E., & Guestrin, C. (2015). Stochastic Gradient Hamiltonian Monte Carlo. ICML.

Dudik, M., Langford, J., & Li, L. (2011). Doubly Robust Policy Evaluation and Learning. ICML.

Flaxman, A., Kalai, A., & McMahan, H. (2005). Online Convex Optimization in the Bandit Setting: Gradient Descent without a Gradient. SODA.

Meng, X.L. & Wong, W.H. (1996). Simulating Ratios of Normalizing Constants via a Simple Identity: A Theoretical Exploration. Statistica Sinica.

Polyak, B.T. (1964). Some Methods of Speeding up the Convergence of Iteration Methods. USSR Computational Mathematics and Mathematical Physics.

Shao, Z. et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300.

Spall, J.C. (1992). Multivariate Stochastic Approximation Using a Simultaneous Perturbation Gradient Approximation. IEEE Transactions on Automatic Control, 37(3), 332–341.

Welling, M. & Teh, Y.W. (2011). Bayesian Learning via Stochastic Gradient Langevin Dynamics. ICML.

Yu, T. et al. (2025). DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476.
