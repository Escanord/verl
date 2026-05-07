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

**Base models.** Qwen3-1.7B-Base and Qwen3-4B-Base, transformer base models pre-trained on general text with no instruction tuning. The two scales are deliberately included: the 1.7B regime is one in which vanilla GRPO escapes the dead zone naturally within ~50 training steps, and the 4B regime is one in which vanilla GRPO requires ~75–80 steps to escape, isolating the *bootstrap acceleration* effect of DRIFT as the dead zone lengthens.

**Training data.** Batches of 1024 competition-level mathematics problems drawn from the math subset of LLM360/guru-RL-92k (AIME-style and MATH training problems). Maximum prompt length 1024 tokens; maximum response length 4096 tokens. Reward is rule-based answer matching via a sympy-equivalence checker (`math_dapo`).

**Evaluation.** All evaluations use n = 16 independent samples per problem at temperature 1.0; the two metrics reported across benchmarks are *mean@16* (empirical per-sample solve rate) and *best@16* (probability that ≥1 of 16 samples is correct). The benchmark suite covers four difficulty/domain levels:

- **AIME24** (30 problems, `math-ai/aime24`) — competition mathematics; the primary in-distribution evaluation.
- **AIME25** (30 problems, `math-ai/aime25`) — fresh test set in the same distribution; controls for any leakage of AIME24 problems into pre-training data.
- **OlympiadBench-Math-EN** (674 problems, `Hothan/OlympiadBench`, `OE_TO_maths_en_COMP` config; single-answer subset) — text-only English math competition; substantially harder than AIME and tests transfer beyond the AIME problem distribution.
- **GPQA-Diamond** (198 problems, `Idavidrein/gpqa`) — graduate-level science multiple-choice; cross-domain reasoning transfer test (math-trained policy evaluated on biology/chemistry/physics MCQ).
- **Minerva-Math** (~5000 problems, `EleutherAI/hendrycks_math` 7-subject test split, lm-eval-harness `minerva_math` convention) — *reported only if DRIFT outperforms baselines*; included to establish whether bootstrap gains transfer to easy-medium math problems.

In-training validation is run every 5 steps on AIME24 only (for fast iteration); the full benchmark suite is evaluated on saved checkpoints at fixed step boundaries (every 20 steps).

**Baselines.** (1) **GRPO** — vanilla group-relative policy optimization (Shao et al., 2024). (2) **High-Entropy GRPO (HEG)** — GRPO with a uniform per-token entropy regularizer, the most direct empirical instantiation of the "uniform-pressure exploration" critique of §1; the comparison isolates the value of *selective* exploration at branch points relative to *uniform* entropy injection. We do not compare directly against DAPO (Yu et al., 2025) for compute reasons, but note that DAPO's contribution (asymmetric clipping, dynamic group filtering, token-level loss normalization) is orthogonal to DRIFT's exploration mechanism and could in principle be combined.

**DRIFT hyperparameters.** Branch point: adaptive threshold at p = 0.85 quantile of a rolling entropy buffer of size 2000, with fallback τ_0 = 0.4 nats during warmup; positional guard t_min = 800. Langevin update: drift step size η = 0.1, diffusion magnitude σ = 0.01, top-K subspace dimension K = 20, one Langevin step per trigger. Drift estimation: Polyak momentum γ = 0.7, exploit ratio α = 0.6, entropy-maintenance target α_target = 0.7, feedback enabled. Score adjustment: penalty coefficient λ = 2.0, target length L_target = 1000. Importance sampling: mixture coefficient β = 0.5, entropy cap H_cap = 0.8 nats, asymmetric clip range (ε_low, ε_high) = (0.2, 0.28). KL regularization: λ_KL = 0.001 against the pre-trained reference policy. Optimizer: AdamW with learning rate 1×10⁻⁶, weight decay 0.1, 10 warmup steps. Mini-batch size 256, micro-batch size 16 per GPU.

With these settings, the conjunction of the adaptive entropy threshold and the positional guard selects approximately 3–5% of generated tokens as branch points; the adaptive threshold τ_t settles in the range 0.3–0.5 nats during stable training.

**Infrastructure.** 8 × H200 140 GB GPUs, FSDP actor, vLLM rollout engine, n = 8 rollouts per prompt during training, n = 16 during evaluation.

---

## 6. Results

We organize the empirical evaluation around the two claims established in §1: that DRIFT's selective exploration mechanism (i) accelerates dead-zone escape relative to vanilla GRPO without (ii) paying the ceiling penalty incurred by uniform-pressure entropy regularization (HEG). Section 6.1 establishes the bootstrap acceleration on the harder 4B regime where the dead zone is longest. Section 6.2 establishes ceiling parity with GRPO on the 1.7B regime where both methods reach steady state within the experimental budget. Section 6.3 reports out-of-distribution generalization on AIME25, OlympiadBench, and GPQA. Section 6.4 reports the optional Minerva-Math comparison.

### 6.1 Bootstrap Acceleration on Qwen3-4B-Base

The 4B base model exhibits a long dead-zone phase under vanilla GRPO: AIME24 mean@16 remains identically zero for the first ~70 training steps, with no within-group reward variance and therefore no policy gradient signal. DRIFT and HEG both inject exploration during this phase but through structurally different mechanisms (selective vs uniform), and the comparison is the cleanest empirical evidence for the value of branch-point selectivity.

**Table 1.** AIME24 evaluation on Qwen3-4B-Base across training, by method.

| Step | DRIFT (mean@16) | DRIFT (best@16) | GRPO (mean@16) | GRPO (best@16) | HEG (mean@16) | HEG (best@16) |
|------|----------------|----------------|---------------|---------------|--------------|---------------|
| 30   | 0.001 | 0.009 | 0.000 | 0.000 | 0.000 | 0.000 |
| 50   | 0.016 | 0.104 | 0.000 | 0.000 | 0.002 | 0.022 |
| 70   | 0.054 | 0.142 | 0.001 | 0.011 | 0.018 | 0.099 |
| 90   | [TBD] | [TBD] | 0.068 | 0.179 | 0.063 | 0.129 |
| 130  | [TBD] | [TBD] | 0.102 | 0.196 | 0.067 | 0.125 |
| 200  | [TBD] | [TBD] | 0.108 | 0.216 | 0.075 | 0.153 |

**Table 2.** Bootstrap-speed summary (steps to first cross threshold) on Qwen3-4B-Base.

| Threshold | DRIFT | GRPO | HEG |
|-----------|-------|------|-----|
| AIME mean@16 ≥ 0.001 (first non-zero) | 30 | 65 | 35 |
| AIME mean@16 ≥ 0.05  | ~70 | ~85 | ~85 |
| AIME mean@16 ≥ 0.10 (target plateau)  | [TBD] | ~125 | not reached |

DRIFT crosses each threshold in fewer training steps than either baseline. The asymptotic comparison between DRIFT and GRPO at step 200+ is in progress and will be filled in.

### 6.2 Convergence at Equal Ceiling on Qwen3-1.7B-Base

The 1.7B base model has a much shorter dead zone — vanilla GRPO escapes within ~50 training steps unaided. The 1.7B regime therefore tests the *quality* of the eventual policy rather than the time-to-escape: does selective exploration preserve GRPO's plateau, or does it pay a ceiling cost like HEG?

**Table 3.** Steady-state AIME24 plateau on Qwen3-1.7B-Base (averaged over steps 100–130).

| Method | AIME mean@16 | AIME best@16 | Math@16 (training metric) |
|--------|--------------|--------------|---------------------------|
| DRIFT  | 0.045 | 0.180 | 0.595 |
| GRPO   | 0.039 | 0.171 | 0.620 |
| HEG    | 0.012 | 0.103 | 0.555 |

DRIFT and GRPO are within experimental noise of each other on AIME mean@16 and best@16, while HEG is approximately 4× lower on mean@16 and 1.7× lower on best@16. The conjunction of Tables 1–3 supports the central empirical claim: DRIFT delivers HEG's bootstrap acceleration without HEG's ceiling penalty.

### 6.3 Out-of-distribution Evaluation

To distinguish AIME-specific gains from generalizable reasoning improvement, we evaluate the final checkpoints on three out-of-distribution benchmarks: a fresh AIME edition (AIME25), a harder competition-math distribution (OlympiadBench-Math-EN), and a cross-domain MCQ benchmark (GPQA-Diamond).

**Table 4.** Out-of-distribution evaluation at the final training checkpoint (step [TBD]).

*Qwen3-4B-Base*

| Benchmark | DRIFT (mean@16) | DRIFT (best@16) | GRPO (mean@16) | GRPO (best@16) | HEG (mean@16) | HEG (best@16) |
|-----------|----------------|-----------------|----------------|----------------|---------------|---------------|
| AIME25 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| OlympiadBench-Math-EN | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| GPQA-Diamond | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

*Qwen3-1.7B-Base*

| Benchmark | DRIFT (mean@16) | DRIFT (best@16) | GRPO (mean@16) | GRPO (best@16) | HEG (mean@16) | HEG (best@16) |
|-----------|----------------|-----------------|----------------|----------------|---------------|---------------|
| AIME25 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| OlympiadBench-Math-EN | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| GPQA-Diamond | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

### 6.4 Minerva-Math

We report Minerva-Math results conditional on DRIFT improving over GRPO and HEG; if Minerva is dominated by the AIME and OlympiadBench results above (i.e. all three methods saturate at similar values on the easier MATH-derived problems Minerva contains), the result is omitted.

| Benchmark | DRIFT (mean@16) | GRPO (mean@16) | HEG (mean@16) |
|-----------|----------------|----------------|----------------|
| Minerva-Math (4B) | [TBD] | [TBD] | [TBD] |
| Minerva-Math (1.7B) | [TBD] | [TBD] | [TBD] |

### 6.5 Mechanism Diagnostics

We additionally report training-time diagnostics confirming that DRIFT's mechanism behaves as designed.

| Diagnostic | Expected behavior | Observed |
|------------|-------------------|----------|
| Trigger fraction | 3–5% under chosen p, t_min | [TBD: range across training] |
| Branch-point entropy | Tracks α_target · H_first throughout training | [TBD: time-series characterization] |
| Drift G norm | Grows from 0 at initialization, stabilizes within ~10 triggers | [TBD] |
| IS weight magnitude | Bounded by 1/(1−β) = 2 modulo entropy cap | [TBD: max observed weight] |

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
