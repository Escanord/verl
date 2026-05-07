# Walk-Weighted GRPO (W-GRPO)

W-GRPO extends GRPO by reweighting per-token advantages using attention hub scores derived from a Sketch-Determined Random Walk over the model's attention graphs.

## Motivation

Standard GRPO assigns a uniform advantage to every token in a response. W-GRPO hypothesizes that tokens which are structural hubs in the attention graph — attended to heavily by many other tokens across many layers — are more "important" to the model's computation, and should receive amplified gradient signal.

## Algorithm

### GRPO baseline

```
Â_i = (r_i - mean(r)) / std(r)   over group of n rollouts
```

Advantages are zero when all rollouts in a group have the same reward (i.e. all correct or all wrong).

### W-GRPO modification

```
Â_t_weighted = Â_t × (1 + α × w̃_t)
```

Where:
- `Â_t` is the standard GRPO advantage for token `t`
- `w̃_t` is the walk importance score for token `t` (ReLU'd z-score, in `[0, ∞)`)
- `α` is the weighting strength (default: 1.0)
- Setting `α=0` recovers standard GRPO

Since `w̃_t ≥ 0`, the weight `1 + α·w̃_t ≥ 1` always. Below-average-importance tokens are unaffected; above-average tokens are amplified. This never dampens or sign-flips the advantage.

## Walk Importance Score Computation

Computed during the actor's **log-prob forward pass** (full prefill over prompt+response), not during rollout generation.

### Step 1: Hook Q and K projections
Forward hooks capture `q_proj` and `k_proj` output at every transformer layer.
- Shape (packed, `use_remove_padding=True`): `(total_tokens, num_heads * head_dim)`

### Step 2: Block-pool Q and K
Mean-pool every `block_size=32` consecutive tokens → `(B, num_heads, num_blocks, head_dim)`

### Step 3: Hadamard sketch
Why not use attention logits directly: **flash attention** never materializes the `(T, T)` matrix — it's fused in CUDA. We reconstruct from Q and K.

Project each block's vector from `head_dim` → `hadamard_dim=64` via SRHT (Subsampled Randomized Hadamard Transform):
- Random sign flip `{-1, +1}`
- Fast Walsh-Hadamard Transform
- Subsample first `hadamard_dim` components

### Step 4: Block-level attention matrix
For each layer, per head:
```
A = softmax(Q_sketch @ K_sketch^T / sqrt(hadamard_dim))   # (B, num_blocks, num_blocks)
```
Average over heads → row-stochastic block attention matrix `W`.

### Step 5: Walk accumulation across layers
Starting from `R = W` at layer 0:
```
R^k = R^{k-1} @ W^k   (repeated walk_degree=4 times per layer)
```
`R[i, j]` = probability of a random walk landing on block `j` after starting at block `i`.
This is analogous to PageRank — blocks that many walks converge to get high scores.

### Step 6: Layer-average and column-sum
```
R_avg = mean(R^1, R^2, ..., R^L)        # average over layers
w_block[j] = sum_i R_avg[i, j]          # column sum = total walk mass landing on block j
```

**Why column sum measures "referenced by multiple queries":**
`W[i,j]` = how strongly token `i` (query) is coupled to token `j` (key). The column sum
`w[j] = Σ_i R_avg[i,j]` aggregates how much walk mass arrives at block `j` from all starting
positions — a PageRank-like score of how "central" block `j` is. High score = many other blocks
reference it through direct and indirect attention paths.

**agg_mode=response (WGRPO-v4):**
With `agg_mode=all`, prompt tokens dominate: `<|im_start|>`, the problem statement, `\n`
all attract heavy attention from the full sequence and score high regardless of reasoning content.
With `agg_mode=response`, only response blocks act as query rows in the column sum:
```
w_block[j] = sum_{i in response_blocks} R_avg[i, j]
```
A block scores high only if the model's own generated reasoning steps reference back to it.
This isolates within-reasoning hubs from trivial prompt anchors.

### Step 7: ReLU z-score normalization, scaled to [0, 1] (per sequence)
```
w_relu  = ReLU((w_block - mean(w_block)) / std(w_block))
w̃_block = w_relu / max(w_relu)
```
- Per-sequence normalization: scores are relative to that sequence's own distribution
- ReLU: only above-average blocks get nonzero scores; below-average → 0 (no effect)
- Divide by per-sequence max: most-attended block always gets score 1.0
- Range: `[0, 1]`, weight = `1 + α * w̃ ∈ [1, 1+α]`
- α is directly interpretable as the maximum amplification factor (default: 1.0 → max 2×)

### Step 8: Expand to token level
Each token in block `j` gets score `w̃_block[j]`. Zero outside response tokens.

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `block_size` | 32 | Tokens per block for pooling |
| `hadamard_dim` | 64 | Sketch projection dimension |
| `walk_degree` | 4 | Walk steps per layer |
| `walk_alpha` | 1.0 | Advantage amplification strength |

## Implementation

| File | Change |
|------|--------|
| `verl/utils/walk_importance.py` | `WalkImportanceComputer` class — hooks, walk computation, normalization, agg_mode |
| `verl/trainer/ppo/core_algos.py` | `apply_walk_weighted_advantage()` — applies weighting |
| `verl/trainer/ppo/ray_trainer.py` | Registers hooks, calls compute, logs walk metrics + top-k token strings, applies weighting |
| `verl/workers/config/actor.py` | `walk_importance: dict` field added to `FSDPActorConfig` |
| `verl/workers/actor/dp_actor.py` | `compute_walk_importance()` — reads config, runs hooks+forward, passes prompt_lens |

## Roadmap / Design Notes

### Why column sum = "referenced by multiple queries"
`W[i,j]` is the coupling strength between block `i` (query) and block `j` (key). The walk
propagates this: `R_avg[i,j]` = probability of a random walk starting at `i` arriving at `j`
via direct and indirect paths. The column sum is a PageRank-like hub score: blocks that many
walks converge to are structural hubs of the attention graph.

Crucially, since Q/K geometry is trained with causal masking, `Q[i]·K[j]` for `i < j` is
naturally small — the model's Q vectors learn to point toward earlier K vectors. So the walk
is implicitly approximately causal even without applying an explicit causal mask.

### The prompt contamination problem (motivation for v4)
With `agg_mode=all`, all positions act as queries in the column sum. Prompt tokens (problem
statement, `<|im_start|>`, separator tokens) are trivially referenced by all response tokens
simply by being early in the sequence. They dominate the column sum independent of reasoning.
Restricting to response-block queries (`agg_mode=response`) isolates within-reasoning hubs.

### Hadamard sketch is unnecessary (fix in v5)
The original motivation was that flash attention never materializes the (T,T) matrix. But
the walk computes its own attention from hooked Q and K. After block pooling with block_size=32,
a 4096-token sequence becomes 128 blocks — a 128×128 matrix, trivially small. Direct Q^T K
dot product is exact and has negligible cost. Removed in v5.

### Early-position bias in column sum (fix in v5)
With `agg_mode=all`, `w[j] = Σ_i R_avg[i,j]` sums over all num_blocks rows. Block j=0
(first block) has all num_blocks rows contributing; block j=num_blocks-1 (last) has only
1 row contributing. This creates a strong early-position bias independent of content.

With `agg_mode=causal`, only rows i >= j contribute (causal constraint: block i can only
reference block j if j appeared first). Normalized by valid_count[j] = (num_blocks - j):

    w[j] = (1 / (num_blocks - j)) * Σ_{i >= j} R_avg[i, j]

Now w[j] is the *average* walk mass from all blocks that can attend to j — a fair, position-
invariant measure of how strongly block j is referenced by its future context.

### What we still don't know
- Whether causal-normalized hub scores correlate with pivotal reasoning tokens
- Token logging (added in ray_trainer.py) will reveal this during v5 training

### Pre-breakthrough behavior of walk weights (v7 observation)
Before breakthrough, all rewards within a group are -1 (uniform), so GRPO advantages are exactly 0 (`std(r) = 0`). The weighted advantage `Â_t × (1 + α × w̃_t) = 0` regardless of `w̃_t` — walk weights and fork_boost have **zero effect** pre-breakthrough since they multiply a zero advantage.

**Observed in v7**: `mean_fork_block ≈ 0.15` (token ~5) pre-breakthrough, shifting to ~5.5 (token ~178) post-breakthrough. The early-position fork is noise, but it causes no gradient pollution since pg_loss = 0 throughout.

**The delayed breakthrough in v7 (step ~225 vs GRPO-v0 step ~155) is run-to-run variance**, not caused by fork_boost. Pre-breakthrough training dynamics are mechanically identical across all variants (pg_loss = 0, only KL active), so the only source of variation is data shuffle order. Breakthrough occurs when the model first encounters a solvable problem in a batch with mixed rewards — determined entirely by which problems appear early in the random shuffle. v7 started from scratch with a different shuffle; its ~70-step delay corresponds to ~35k additional samples, well within plausible variance over a 92k dataset.

**Implication**: no need to gate fork_boost on `no_contrast_frac`. The mechanism is naturally inert before breakthrough.

## Variants

| ID | Normalization | agg_mode | Notes |
|----|--------------|----------|-------|
| GRPO-v0 | — | — | Baseline GRPO, no walk weighting |
| WGRPO-v1 | z-score | all | First implementation; scores in `(-∞, +∞)`, below-average tokens dampened (weight < 1), can approach 0 for very negative scores. Sign-flip guard via `torch.where`. |
| WGRPO-v2 | ReLU(z-score) / max | all | Scores in `[0, 1]`; below-average tokens → 0 (weight=1, unaffected), most-attended block → 1.0 (weight=1+α). Bounds weight to `[1, 1+α]`, making α the max amplification. No sign-flip, no unbounded spikes. |
| WGRPO-v3 | z-score / abs_max | all | Scores in `[-1, 1]`; below-average tokens suppressed (weight→0), above-average amplified (weight→1+α). Clamp prevents sign flip. |
| WGRPO-v4 | ReLU(z-score) / max | response | Same normalization as v2. Key change: only response blocks act as query rows in column sum. Filters out prompt anchors; a block scores high only if the model's reasoning steps reference back to it. |
| WGRPO-v5 | ReLU(z-score) / max | causal | Two fixes over v2: (1) remove Hadamard sketch — direct Q^T K on block-pooled vectors, exact and cheap at block scale; (2) causal column sum normalized by valid count: `w[j] = mean_{i>=j} R_avg[i,j]`, removing early-position bias. |
| WGRPO-v6 | ReLU(z-score) / max (v5) + differential | causal | **Differential walk (Option A).** Uses v5 walk scores per rollout, then computes `delta_wi[t] = ReLU(mean_correct wi[t] - mean_wrong wi[t])` per prompt group. Structural tokens equally attended in both correct and wrong rollouts cancel out; only reward-correlated structural differences remain. Falls back to absolute walk for groups with uniform reward. |
| WGRPO-v7 | v6 + fork block amplification | causal | **Fork-block concentration (Option C, soft).** Adds per-group fork block: `fork = argmax_j delta_wi_block[j]`. Tokens in that block get walk weight × fork_boost (default 2×). Concentrates gradient at the structural decision point without generating additional rollouts. Full forked-generation version is future work. |
| WGRPO-v8 | v6 differential walk + hard fork generation | causal | **Hard fork (Option C, full).** Phase 1: n=8 rollouts + differential walk → find fork_tok (argmax block) per group. Phase 2 (only when no_contrast_frac < 1): construct `extended_prompt = original_prompt + correct_prefix[:fork_tok]`, generate n=8 new rollouts, compute GRPO advantages within Phase 2 groups independently, run a second actor update. Full log_prob changes on the suffix. Pre-breakthrough: Phase 2 skipped entirely. |
| WGRPO-v9 | v6 + top-k sampled fork blocks, soft boost only | causal | **Top-k fork sampling (soft only, no Phase 2).** Replaces v7's argmax with sampling k=3 blocks without replacement from `softmax(group_block_scores)`. Soft boost (walk weight × fork_boost) applied to all k blocks. No additional rollout generation — same cost as v7. Covers the full divergence region rather than locking onto one noisy peak. `walk_fork_k=3` (configurable; set to 1 to recover v7). |
| WGRPO-v10 | per-position z-score across rollouts | causal | **Group-normalized walk.** Replaces v6's group-level differential with per-rollout z-score at each token position: `w_norm[i,t] = (w[i,t] - mean_n(w[:,t])) / std_n(w[:,t])`. Directly analogous to GRPO's `(r_i - mean(r)) / std(r)`. No ReLU — below-mean walk is dampened (weight < 1), not clamped to 1. No explicit correct/wrong split — differential signal emerges from the z-score naturally. No fallback path needed. Weight: `(1 + α × w_norm[i,t]).clamp(min=0)`. |
| WGRPO-v11 | walk as direct process reward (pure gating) | causal | **Walk gates gradient completely.** `Â_t = A_grpo * delta_w[t]` — no additive offset, walk is pure multiplicative gate. Tokens where delta_w=0 receive zero gradient. **FAILED: never broke through (519 steps, differential_no_contrast_frac=1.0 throughout).** Death spiral: pure gating requires both mixed rewards AND nonzero differential walk simultaneously; pre-breakthrough A_grpo=0 means zero pg_loss, so model never learns to solve problems, so rewards stay uniform, so no contrast, so delta_w stays 0. The `(1+α*w)` offset in v1-v10 is essential — it preserves the GRPO signal during breakthrough. |

## Results (Qwen2.5-3B-Instruct, guru-RL-92k, 8×H200)

### MATH500

| ID | Steps | acc@1 | acc@16 (mean) | best@16 | maj@16 | Checkpoint |
|----|-------|-------|--------------|---------|--------|------------|
| GRPO-v0 | 400 | ~61.3% | 61.3% | 80.2% | 66.6% | `grpo/qwen2_5_3b/global_step_400` |
| WGRPO-v1 | 400 | 63.8% | 62.4% | **82.4%** | **69.1%** | `wgrpo/qwen2_5_3b/global_step_400` |
| WGRPO-v2 | 300 | ~63% | 60.8% | 81.1% | 67.0% | `wgrpo-v2/qwen2_5_3b/global_step_300` |
| WGRPO-v3 | 520 | ~61.6% | 61.4% | 79.9% | 66.8% | `wgrpo-v3/qwen2_5_3b/global_step_520` |
| WGRPO-v4 | 400 | ~61% | 60.9% | 80.8% | 66.4% | `wgrpo-v4/qwen2_5_3b/global_step_400` |
| WGRPO-v5 | 400 | ~61.3% | 61.3% | **81.2%** | 66.6% | `wgrpo-v5/qwen2_5_3b/global_step_400` |
| WGRPO-v6 | 345 (peak) | ~62.8% | **62.8%** | 82.0% | **69.8%** | `wgrpo-v6/qwen2_5_3b/global_step_340` |
| WGRPO-v7 | 390 (peak) | ~60% | 60.1% | 81.3% | 66.5% | `wgrpo-v7/qwen2_5_3b/global_step_400` |
| WGRPO-v9 | 430 (peak) | ~62.2% | 62.2% | 82.1% | 69.1% | `wgrpo-v9/qwen2_5_3b/global_step_430` |
| WGRPO-v10 | 360 (peak) | ~60.1% | 60.1% | 81.4% | 67.4% | `wgrpo-v10/qwen2_5_3b/global_step_360` |
| WGRPO-v6-block16 | 235 (peak) | ~59.4% | 59.4% | 82.4% | 67.2% | `wgrpo-v6-block16/qwen2_5_3b/global_step_235` |
| WGRPO-v6-block64 | 320 (peak) | ~61.3% | 61.3% | 80.6% | 67.3% | `wgrpo-v6-block64/qwen2_5_3b/global_step_320` |
| WGRPO-v11 | — | — | — | — | — | Never broke through (519 steps, pg_loss=0 throughout) |
| High-Ent-GRPO | 340 (peak) | N/A (eval @16 only) | 61.0% | 81.7% | 68.8% | `high-ent-grpo/qwen2_5_3b/global_step_340` |

### AIME

| ID | Steps | acc@16 (mean) | best@16 | maj@16 |
|----|-------|--------------|---------|--------|
| GRPO-v0 | 400 | 3.33% | 12.2% | 5.42% |
| WGRPO-v1 | 400 | 2.7% | 10.1% | 3.8% |
| WGRPO-v2 | 300 | 2.6% | 13.3% | 4.7% |
| WGRPO-v3 | 520 | 4.3% | 16.5% | 6.1% |
| WGRPO-v4 | 400 | 3.4% | 12.8% | 5.5% |
| WGRPO-v5 | 400 | 2.24% | 12.4% | 3.23% |
| WGRPO-v6 | 380 (peak) | **4.79%** | **16.9%** | **7.70%** |
| WGRPO-v7 | 390 (peak) | 4.04% | 15.4% | 7.41% |
| WGRPO-v9 | 480 (peak) | 3.54% | 14.5% | 6.4% |
| WGRPO-v10 | 305 (best@16) / 285 (maj@16) | 4.51% | 14.2% | 7.51% |
| WGRPO-v6-block16 | 240 (best@16) / 270 (maj@16) | ~2.2% | 13.1% | 5.09% |
| WGRPO-v6-block64 | 360 (best@16) | 4.22% | 15.9% | 6.65% |
| WGRPO-v11 | — | — | — | — | Never broke through |
| High-Ent-GRPO | 390 (peak) | 4.58% | 11.7% | 6.68% |

### Training notes
- WGRPO-v1 breakthrough (first nonzero advantage): step ~169 vs GRPO-v0 step ~155 (~14 steps later)
- After breakthrough, both converge to the same 60-64% acc@1 band
- WGRPO-v1: walk `importance_mean_nonzero_adv ≈ -0.10` throughout training — tokens carrying learning signal have slightly below-average walk scores, consistent with reasoning steps not being attention hubs; z-score dampens these tokens (weight < 1)
- WGRPO-v2: ReLU eliminates dampening; below-average tokens get weight = 1 (original GRPO signal), only hubs amplified; eval pending
- WGRPO-v6 vs v7: v6 (differential walk only) outperforms v7 (+ argmax fork boost) on all AIME metrics and MATH. v6 also ramps faster immediately post-breakthrough (+25 steps: 3.10% vs 1.93%) and reaches a higher eventual peak (4.79% vs 4.04% AIME mean@16). AIME best@16=17.5% at step 355 (new record, beats v3's 16.5%). Argmax fork boost in v7 appears to hurt: it concentrates gradient on a single noisy block, reducing gradient diversity relative to v6's smoother differential signal.
- WGRPO-v10 peak at step ~360, then regresses. MATH maj@16=67.4% is below v6's 69.8%; AIME best@16=14.2% and maj@16=7.51% are below v6's 16.9% / 7.70%. Group-normalized walk (per-position z-score across rollouts) underperforms v6's differential walk — removing the explicit correct/wrong split loses the reward-correlated positional signal. The z-score approach also allows below-mean walk tokens to be dampened (weight < 1), which may hurt tokens that are consistently important across all rollouts.
- WGRPO-v6-block16 (v6 with block_size=16 vs v6's block_size=32): breakthrough ~35 steps earlier (step ~165 vs ~215), but peak is lower on all metrics. MATH maj@16=67.2% vs v6's 69.8%; AIME best@16=13.1% vs v6's 16.9%; AIME maj@16=5.1% vs v6's 7.70%. The earlier breakthrough is a red herring — smaller block size gives finer position resolution but does not improve fork-position precision enough to overcome the noise introduced by pooling fewer tokens per block. block_size=32 remains the better choice for v6-style differential walk.
- WGRPO-v6-block64 (v6 with block_size=64 vs v6's block_size=32): MATH peak at step 320 (maj@16=67.3%, best@16=80.6%), AIME peak at step 360 (best@16=15.9%, maj@16=6.65%). MATH maj@16 matches block16 (67.3%) but is below v6 (69.8%). AIME best@16 at 15.9% is slightly below v6 (16.9%). Overall: block64 < v6 < no conclusion on monotonicity — block16 (67.2%) ≈ block64 (67.3%) < block32/v6 (69.8%) on MATH; for AIME, block64 (15.9%) > block16 (13.1%) but still below block32/v6 (16.9%). Block_size sensitivity curve is non-monotonic; block_size=32 is the empirical optimum across both metrics. Training crashed with OOM after step 400 (increasing response lengths); eval recovered via checkpoint sweep.
- WGRPO-v11 never broke through (stopped at step 519): `differential_no_contrast_frac=1.0` throughout. Pure gating (`Â_t = A_grpo * delta_w`) creates a death spiral — pre-breakthrough A_grpo=0 means pg_loss=0, model never improves, rewards stay uniform, no contrast, delta_w stays 0. The `(1 + α*w)` offset in all prior variants is essential: it preserves the GRPO signal when breakthrough first occurs. Fix: add a fallback constant (e.g. `Â_t = A_grpo * max(delta_w[t], ε)`) or mix with GRPO signal.
- WGRPO-v7 AIME peak at step ~360-390 (epoch ~2), then regresses to ~2.3% by step 475. MATH holds steady at 60-62% throughout; step 455 hit 62.0% mean@16. Best checkpoint for AIME: `global_step_400`; for MATH: `global_step_460`.
- High-Ent-GRPO (80/20 entropy masking, `entropy_top_ratio=0.2`): late breakthrough (~step 215 vs GRPO-v0 ~step 155), slower ramp. MATH peaks at step 340 (61.0% mean@16, 81.7% best@16, 68.8% maj@16); AIME peaks at step 390 (4.58% mean@16, 11.7% best@16, 6.68% maj@16). Overall comparable to GRPO-v0 on MATH, slightly below on AIME best@16 (11.7% vs 12.2%) and maj@16 (6.68% vs 5.42%). No clear benefit over GRPO-v0 within 400 steps; significantly below WGRPO-v6 on both axes. Stopped at step ~494. Note: ran with `ppo_micro_batch_size_per_gpu=16` (halved from 32 after OOM at step ~265 as longer responses began to strain backward memory).

## Mechanistic Hypothesis: Positional Bifurcation

*Added 2026-04-13. Speculative — empirical verification pending.*

### The claim

The performance gain from v6 may not come from the **semantic importance** of the walk scores. It may come from **positional bifurcation control**: identifying where in the token sequence correct and wrong generation trajectories diverge, and amplifying gradient at exactly those positions.

Under this framing, `δ[t]` is not "a weight for semantically important token t." It is a **problem-specific positional weight vector** — a soft mask over the response that identifies the trajectory forks for this particular prompt. The semantic content that emerges at those positions is a *consequence* of the positional pressure, not the cause.

### Why this is non-obvious

Standard credit assignment intuition says: find the tokens that *caused* the reward signal and upweight them. That is a semantic claim about token content.

The bifurcation hypothesis says instead: autoregressive generation is a trajectory through token space. Correct and wrong rollouts are trajectories that share the same starting point (the prompt) and diverge at specific absolute positions. The delta identifies those positions. By amplifying gradient at position t, you train the policy to commit to the right branch at that step. Whether the token at t is semantically "important" is irrelevant — what matters is that it is a *fork point* in the generation trajectory.

### Why the shared delta still works despite variable rollout lengths

Even though rollouts in a group have different lengths, the delta is broadcast uniformly to all of them. This is correct because the fork positions are a property of the **problem**, not of individual rollouts. "The key decision points for this AIME problem are at tokens 150–200 and 400–450" is a statement about the problem structure. All rollouts — short or long — share those positions (within their actual length, before their response mask zeros them out).

### What this implies about the walk computation

If positional bifurcation is the true mechanism, then the attention-walk computation may be doing less work than it appears. The walk contributes two things to v6:
1. **Position selection**: which t gets high δ[t]
2. **Graded weighting**: how strongly each t is amplified

The hypothesis predicts that (1) is the load-bearing component and (2) is secondary. The causal/structural properties of the walk (multi-hop attention paths, PageRank-like hub scoring) may be useful primarily as a *noise-robust position selector* — better than raw reward differences, but not because of the attention semantics per se.

### Empirical verification

The hypothesis makes clean, testable predictions. Proposed ablations in increasing order of definitiveness:

**Experiment A — KL-delta baseline (no walk, same positional structure)**

Replace the walk computation entirely with KL divergence of token logits between correct and wrong rollouts:

```python
δ_kl[t] = KL(softmax(logits_correct_mean[t]) || softmax(logits_wrong_mean[t]))
```

This requires no attention hooks — logits are already available from the rollout. The positional structure (which positions get high δ) is determined by reward-correlated token distribution divergence, not by attention walk. Normalization: same ReLU + max-norm as v6.

**Prediction**: if positional hypothesis is correct, KL-delta ≈ v6 on AIME. If walk structure matters intrinsically, KL-delta << v6.

**Experiment B — Random-position control (same sparsity, wrong positions)**

Take v6's delta tensor for each prompt group and randomly permute the position indices within the response. Same number of "highlighted" positions, same graded weights, but positional alignment destroyed.

**Prediction**: random-permuted << v6, and approximately = GRPO. This would confirm that *which positions* get amplified matters, not just the sparsity pattern.

**Experiment C — Uniform middle weighting (no computation)**

Use a fixed, domain-agnostic positional prior: δ[t] = 1 for t ∈ [0.25·L, 0.75·L], 0 elsewhere (middle half of each response). No walk, no reward contrast, no per-prompt adaptation.

**Prediction**: if any positional prior helps, uniform-middle > GRPO. How close it gets to v6 measures how much of v6's gain comes from the specific position selection vs. the general "middle of response is more important" prior.

**Experiment D — Log-prob contrast delta (reward signal, no walk)**

Replace walk with log-prob difference between correct and wrong rollouts at each position:

```python
δ_logprob[t] = ReLU(mean_correct_logprob[t] - mean_wrong_logprob[t])
```

Positions where correct rollouts assigned higher probability to their own tokens relative to wrong rollouts. This is a pure reward-alignment positional signal with no attention structure.

**Prediction**: δ_logprob should be highly correlated with v6's delta if the walk is just a proxy for reward-correlated position selection. Performance gap between δ_logprob and v6 isolates the marginal value of the attention walk.

### Decision tree for interpreting results

```
KL-delta ≈ v6?
  YES → positional mechanism confirmed, walk structure is noise-robust position selector
        → check if uniform-middle also ≈ v6 (if yes: even random positional prior helps)
  NO  → walk structure adds information beyond position selection
        → measure correlation between walk δ and KL δ at the position level
        → if high correlation: walk is good position selector; if low: semantic content matters
```

### Why this is a big claim

If confirmed, it would mean:
1. The attention walk is valuable primarily as a **position selector**, not as a semantic importance measure — the multi-hop causal structure is useful but incidental.
2. The key design insight of v6 is the **reward-contrastive positional alignment** across rollouts of the same prompt — not the walk scores themselves.
3. Much simpler alternatives (KL-delta, log-prob delta) might match v6 with far lower computational cost (no attention hooks, no block pooling, no walk accumulation).
4. The natural extension is real-time bifurcation detection during generation (PIVOT Phase 2) — intervening at fork positions as they occur, not weighting them post-hoc.

## Known Limitations

1. **Walk measures structural hubs, not reasoning importance**: attention hubs tend to be formatting/structural tokens, not the computational reasoning steps where correctness is determined.
2. **Full-sequence walk**: walk is computed over prompt+response together; prompt structure tokens can influence early response block scores.
3. **Flash attention workaround**: Hadamard sketch is an approximation; cannot use actual attention weights without disabling flash attention.
4. **Positional alignment assumption**: `δ[t]` averages walk scores across rollouts at the same absolute position, implicitly assuming position t corresponds to the same reasoning stage across rollouts. This holds approximately early in responses (shared prompt continuation style) but breaks down after rollouts diverge structurally. Semantic alignment (by reasoning step boundary, e.g. `\n\n` segmentation) would make this assumption explicit and correct where it fails.
