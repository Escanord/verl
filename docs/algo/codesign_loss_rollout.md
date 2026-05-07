# PIVOT: Pivot-point Identification Via On-the-fly walk Tracking

*Co-designing loss and rollout for RL training of language models around a single structural signal. No external reward model. Works on any transformer.*

---

## The Problem

RL training for LLMs has two components: **rollout** (how you generate training data) and **loss** (how you learn from it). These are almost always designed independently.

The rollout is standard autoregressive sampling — every token position gets the same process. The loss assigns credit uniformly via normalized advantages — every token in a sequence carries the same advantage scalar. Neither component knows what the other is doing. Neither is designed around the structure of the problem.

This is wasteful. RL reward is sparse: a binary outcome signal is spread uniformly across 800+ tokens, but only a small fraction of those tokens causally determine whether the answer is correct. The rest are connective tissue, formatting, elaboration. Gradient at those positions is noise. Exploration at those positions wastes compute.

Both problems have the same root: neither the loss nor the rollout knows *which* tokens matter.

---

## The Signal: Walk-Based Structural Change

### What "Decisive" Actually Means

A position t is decisive if the **choice of y_t has high variance in downstream outcome** — different tokens at position t lead to significantly different final rewards. This is the outcome-sensitivity definition.

Simple attention entropy H(α_t) is a known proxy (Beyond-80/20): when the model focuses sharply, it is making a specific choice. But entropy is blind to *what* the model is focusing on — a sharp focus on a formatting token and a sharp focus on a structural constraint look identical under entropy alone.

We want a signal that captures: **does the choice here change the causal structure of the reasoning?**

### Temporal Walk

Define a running block-level matrix R_t ∈ ℝ^{B×B} (B = num_blocks) that accumulates multi-hop attention importance across generation steps:

```
R_0 = 0
R_t = A_t_block + A_t_block · R_{t-1}
```

where A_t_block is the blocked attention from the current token's last transformer layer.

Unrolling:
```
R_t = A_t + A_t·A_{t-1} + A_t·A_{t-1}·A_{t-2} + ...
```

R_t[q, b] = "block q depends on block b, directly AND through any chain of prior generation steps." This is the temporal analog of Sketch&Walk's layer-wise walk — but accumulated across **time steps** rather than transformer depth.

**Distinction from Sketch&Walk**: Sketch&Walk's walk R^k = R^{k-1} · A^k accumulates across transformer layers (vertical/depth) for a single next-token prediction. The temporal walk accumulates across generation steps (horizontal/temporal), capturing how the causal structure of the reasoning chain evolves as tokens are produced.

### Why φ_t (Direct Walk Score) Is Insufficient

A naive use of the temporal walk as a Langevin trigger:

```
φ_t = Σ_b α_t[b] · spread_{t-1}[b]    (current attention · prior column sums of R)
```

This asks: "is the model attending to blocks that have been important in the past?" But this has no guarantee of predicting decisive points:

1. **Prompt-token inflation**: prompt tokens are included in every R update for all T steps. Their spread scores grow monotonically. φ_t is high whenever the model glances at the question prefix — regardless of whether this is a decision point.
2. **Persistence ≠ causality**: consistent reference to a prior position means it has been useful, not that the *current* choice is pivotal.
3. **Measures history, not change**: a high φ_t token could be routine grounding from an important prior block.

The fundamental gap: φ_t measures where the walk *has accumulated*, not whether the current choice *changes* the causal structure.

---

## Walk Change as the Core Signal

### What to Actually Measure

A decisive token introduces a **new causal pathway** — it changes the multi-hop structure of the reasoning. This is captured by the walk change:

```
||ΔR_t||_F = ||R_t - R_{t-1}||_F = ||A_t_block · (I + R_{t-1}) - R_{t-1}||_F
```

**Structural motivation**: tokens that do not change the walk (connective tissue, formatting, pronoun resolution) produce small `||ΔR||`. Tokens that open new reasoning chains — introducing a key equation, applying a constraint, branching a deduction — produce large `||ΔR||`. The metric directly measures whether the current step is structurally significant.

This is not a guarantee (we cannot compute actual outcome sensitivity on-the-fly) but has a clearer theoretical basis than entropy: "tokens that significantly change the multi-hop causal structure are more likely to fork downstream outcomes."

### Walk Variance Over Top-k Candidates (Strongest Version)

The forward pass at time t already computes attention patterns. For the top-k candidate next tokens {y¹, ..., y^k}, their attention patterns A^j_t_block differ. Compute:

```python
for j in range(k):    # k = 3 is sufficient
    delta_R_j = A_t_block_j @ (I + R_prev)
var_walk = Var({delta_R_1, ..., delta_R_k})    # element-wise, then Frobenius
```

**If var_walk is high**: different token choices produce structurally different walks → the current choice forks the causal structure → decisive point.
**If var_walk is low**: all plausible next tokens produce similar walk structure → choice here does not fork reasoning.

This directly operationalizes the definition of decisive: "does my choice here fork the causal structure?" The cost is k extra attention computations (not full forward passes) at generation time.

### Walk Curvature (Cheap Alternative)

Second derivative of the walk trajectory:

```
κ_t = ||R_t - 2·R_{t-1} + R_{t-2}||_F
```

High curvature = the walk is changing direction structurally — an inflection point in the reasoning chain. Requires no extra computation beyond the running R matrices.

### Signal Summary

| Metric | Computation | What it detects | Theoretical basis |
|---|---|---|---|
| φ_t = α_t · spread | O(B) | Attention on prior hubs | Persistence (weak) |
| `\|\|ΔR_t\|\|_F` | O(B²) | New causal pathway introduced | Structural change |
| Walk variance over top-k | k × O(B²) | Choice actually forks walk | Direct fork detection |
| Walk curvature κ_t | O(B²) | Inflection in reasoning structure | Dynamical change |

---

## The Two Co-Designed Components

### Rollout: Langevin Sampling at Structural Change Points

At each generation step, use the **expected** walk change as the Langevin trigger:

```python
R_prev = zeros(B, B)
P_causal = set()

for t in range(T):
    logits_t, A_t_block = model.forward(prompt + y[:t])

    # Expected walk change from current attention pattern
    expected_delta_R = A_t_block @ (I + R_prev)
    trigger = expected_delta_R.norm(p='fro')      # or walk variance over top-3

    if trigger > threshold:                        # structural change point
        for k in range(K):                         # K = 3–5 Langevin steps
            logits_t += η * ∇_{logits}[−H(softmax(logits_t))]
            logits_t += σ * randn_like(logits_t)
        P_causal.add(t)

    y_t = sample(softmax(logits_t))
    y.append(y_t)

    # Update walk with actual generated token
    R_t = A_t_block + A_t_block @ R_prev
    R_prev = R_t
```

The trigger uses the **expected** structural change (pre-sampling, from the current attention pattern). The Langevin gradient direction `∇[−H(softmax(logits))]` pushes toward sharper next-token distributions — tokens that themselves produce focused, decisive attention downstream.

`P_causal` — the set of positions where Langevin ran — is recorded as a byproduct of generation.

### Loss: Gradient Masked to the Rollout's Structural Change Points

**Option A — Binary mask from P_causal**:
```
L = −Σ_{t ∈ P_causal} A_i · log π_θ(y_t | y_{<t})
```

**Option B — Soft weighting from actual ||ΔR_t||_F**:
```
L = −Σ_t w_t · A_i · log π_θ(y_t | y_{<t})
    where w_t = ||ΔR_t||_F / Σ_s ||ΔR_s||_F
```

The soft version uses the **actual** walk change after y_t was generated — the realized structural impact of each token on the causal graph.

### The Co-Design

The rollout uses the **expected** walk change to decide where to deliberate. The loss uses the **actual** walk change to decide where to train. Both use the same metric (walk Frobenius change) at different stages:

```
Rollout trigger:  ||A_t_block · (I + R_{t-1})||_F     (pre-sampling, expected)
Loss weight:      ||R_t - R_{t-1}||_F                  (post-sampling, actual)
```

One metric, two uses. The rollout produces structurally diverse token choices at high-change positions; the loss trains exactly those positions. No separate signal, no separate computation — the walk is computed once during generation and consumed by both sides.

---

## The Co-Design Loop

```
Langevin at P_causal (high ||ΔR|| positions)
  → diverse token choices where causal structure would change
  → correct and wrong rollouts diverge specifically at P_causal
  → gradient concentrated at P_causal → policy improves at structurally important tokens
  → as policy improves, it produces cleaner causal structure
  → ||ΔR_t||_F becomes more sharply peaked at true decision points
  → P_causal detection becomes more precise
  → Langevin explores the decision manifold more accurately
  → ...
```

---

## As an Inference Algorithm

The rollout algorithm is self-contained. At inference time:

```python
def causal_generate(prompt, K=5, η=0.1, σ=0.05):
    y, R_prev = [], zeros(B, B)
    for t in range(T_max):
        logits_t, A_t_block = model.forward(prompt + y)
        trigger = (A_t_block @ (I + R_prev)).norm(p='fro')
        if trigger > threshold:
            for k in range(K):
                logits_t += η * ∇[−H(softmax(logits_t))]
                logits_t += σ * randn_like(logits_t)
        y_t = sample(softmax(logits_t))
        y.append(y_t)
        R_t = A_t_block + A_t_block @ R_prev
        R_prev = R_t
    return y
```

No stored maps, no prior batches, no reward signal. The model detects its own structural change points during generation and deliberates there. The training and inference algorithms are the **same algorithm**: at training time, σ is large (exploration); at inference, σ is small and η is large (exploitation/deliberation).

---

## Related Work

| Method | Loss modified? | Rollout modified? | Signal | Co-designed? |
|---|---|---|---|---|
| GRPO / PPO / REINFORCE++ | Yes — advantage weighting | No | None | No |
| Beyond-80/20 | Yes — mask low-entropy tokens | No | Output entropy (loss only) | No |
| W-GRPO v11 | Yes — walk gate | No | Differential walk δ_w (loss only) | No |
| TreeRL | Yes — process rewards from tree | Yes — branch at surprisal | Output surprisal | Partial (rollout→loss only) |
| GPO | No | Yes — fork at advantage argmax | Monte Carlo advantage | No (loss unchanged) |
| **PIVOT** | **Yes — ||ΔR|| soft/hard gate** | **Yes — Langevin at high ||ΔR|| positions** | **Temporal walk change (on-the-fly)** | **Yes — same metric, both sides** |

**Key distinction from Beyond-80/20**: Beyond-80/20 masks lowest-entropy tokens in the loss, independently of the rollout. This work uses walk-based structural change to guide the rollout (Langevin trigger), then gates the loss to the same positions. Signal is structural (multi-hop, temporal) rather than entropic (single-token output distribution).

**Key distinction from W-GRPO**: W-GRPO uses the differential walk δ_w = mean(walk|correct) − mean(walk|wrong), which requires reward-labeled contrast across rollouts of the same prompt. This work uses the walk change within a single rollout — no reward signal needed at generation time, no cross-rollout comparison. Applicable to any prompt, including all-wrong groups where δ_w degenerates.

**Key distinction from Sketch&Walk**: Sketch&Walk's walk R^k = R^{k-1} · A^k accumulates across transformer layers for a single next-token prediction (depth-wise). The temporal walk here accumulates across generation time steps (horizontal), capturing the evolving causal structure of the reasoning chain as tokens are produced.

---

## Algorithm Sketch (Full Training Step)

```python
def codesign_step(prompts, n=8):
    rollouts, causal_masks, walk_weights = [], [], []

    for prompt in prompts:
        y, P_causal, delta_R_norms = langevin_rollout(
            prompt, threshold=τ, K=K, η=η, σ=σ
        )
        rollouts.append(y)
        causal_masks.append(P_causal)
        walk_weights.append(delta_R_norms)    # ||ΔR_t||_F per token

    rewards = compute_reward(rollouts)
    advantages = grpo_advantages(rollouts, rewards)

    # Option A: binary mask
    loss = 0
    for i, (rollout, mask, adv) in enumerate(zip(rollouts, causal_masks, advantages)):
        for t in mask:
            loss -= adv * log_prob(rollout[t], rollout[:t], prompt[i])

    # Option B: soft walk weights
    # for i, (rollout, w, adv) in enumerate(zip(rollouts, walk_weights, advantages)):
    #     for t in range(len(rollout)):
    #         loss -= adv * w[t] * log_prob(rollout[t], rollout[:t], prompt[i])

    return loss / len(rollouts)
```

---

## Ablation Structure

| Config | Loss | Rollout | Loop? |
|---|---|---|---|
| GRPO baseline | Standard | Standard AR | No |
| Masked loss only (||ΔR|| gate) | Walk change gate | Standard AR | No |
| Langevin rollout only | Standard | Langevin at high ||ΔR|| | No |
| **Co-design (this work)** | **Walk change gate** | **Langevin at high ||ΔR||** | **Yes** |

---

## Open Questions

1. **Trigger metric choice**: `||ΔR_t||_F` (O(B²), single metric) vs walk variance over top-k (O(k·B²), stronger theoretical basis). Practical question: does the variance over top-3 provide enough additional signal over the Frobenius norm to justify k extra attention computations?

2. **Threshold τ**: what fraction of tokens should be identified as structural change points? Too many → gradient spreads again; too few → misses important positions. Hypothesis: ~15% of tokens, calibrated by the distribution of `||ΔR_t||_F` across rollout sequences.

3. **Walk curvature as alternative**: κ_t = ||R_t − 2·R_{t-1} + R_{t-2}||_F requires no extra computation and captures inflection points. May complement ||ΔR||_F for detecting qualitative transitions in the reasoning structure.

4. **Langevin gradient direction**: `∇[−H(softmax(logits_t))]` pushes toward sharper distributions. The walk gives the TRIGGER; entropy gradient gives the DIRECTION. These serve distinct roles and do not need to share the same signal. Alternative direction: zero-order gradient estimated from k=3 token samples and their downstream walk change.

5. **K and σ schedule**: should K and σ anneal during training? Early training: high σ, more exploration; late training: low σ, more exploitation. Could mirror temperature annealing in DAPO.

6. **Interaction with W-GRPO**: models trained with W-GRPO v11 have sharper walk geometry at causal positions. Does applying this co-design on top of a W-GRPO-trained model yield additional gains? Expected yes — cleaner walk dynamics → more precise ||ΔR|| peaks → better co-design loop.

7. **Block size sensitivity**: B = 128 (block_size = 8) is inherited from W-GRPO experiments. The temporal walk matmul cost scales O(B²) per step. Too-fine blocking increases cost; too-coarse loses resolution. May need separate ablation from W-GRPO's block_size.

---

## Implementation

### Phase 1: Loss Gating (Fully Wired)

`TemporalWalkComputer` in `verl/utils/walk_importance.py` computes block-level temporal walk from saved attention tensors. Called after rollout, before loss update.

`apply_pivot_advantage()` in `verl/trainer/ppo/core_algos.py` applies `||ΔR||`-weighted gating to advantages.

Training loop activation:
```yaml
algorithm:
  use_pivot: true
  pivot_mode: soft          # or binary
  pivot_alpha: 1.0
  pivot_threshold: 0.3
```

### Phase 2: Langevin Rollout via vLLM Patch

**Architecture**: The temporal walk is computed inside vLLM's decode loop by patching the vLLM attention layers and using a stateful per-request logits processor.

```
monkey_patch_model() [vLLMColocateWorkerExtension]
  └── patch_attention_layers_pivot(model)
        ├── registers forward pre-hook on every Attention layer
        └── hook: captures Q_new, K_new per layer → averages → _pivot_decode_state

PIVOTRolloutProcessor [one per generation request, attached as logits_processor]
  ├── reads Q_new, K_new from _pivot_decode_state using batch cursor
  ├── maintains shadow K history (response tokens only)
  ├── computes block-level attention → temporal walk C → ||ΔR_t||
  └── applies Langevin to logits if ||ΔR_t|| > threshold
```

**Signal scope**: the K history covers response-token keys only (from token 0 of the response). The temporal walk tracks causal structure evolution within the response — not cross-attention to the prompt. This is intentional: we're measuring whether the model's OWN reasoning structure changes at each step.

**Batch cursor assumption**: vLLM calls logits_processors in batch order (sequence 0 first, then 1, etc.) within each decode step. The cursor in `_pivot_decode_state` maps each logits_processor call to its batch index. This assumption holds for vLLM v1's sequential sampling loop.

Rollout activation:
```yaml
actor_rollout_ref:
  rollout:
    pivot:
      langevin_rollout: true    # enables Phase 2
      block_size: 8
      langevin_threshold: 0.3
      langevin_K: 3
      langevin_eta: 0.1
      langevin_sigma: 0.01
```

Attention patch activation (must also be set):
```yaml
actor_rollout_ref:
  rollout:
    pivot:
      langevin_rollout: true    # also enables attention hooks via monkey_patch_model
```

**Key files**:
- `verl/utils/vllm/pivot_patch.py` — `PIVOTRolloutProcessor`, attention hooks, `_pivot_decode_state`
- `verl/workers/rollout/vllm_rollout/utils.py` — `monkey_patch_model()` extended with `patch_attention_layers_pivot()`
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py` — `generate()` attaches `PIVOTRolloutProcessor` to `SamplingParams.logits_processors`
