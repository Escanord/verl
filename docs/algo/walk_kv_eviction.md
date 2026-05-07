# Temporal Walk for KV Cache Token Eviction

## Core Idea

Standard KV eviction methods (H2O, SnapKV, PyramidKV) score past tokens by their **direct attention weight** — how much the current token attends to each past token. This is a one-hop, local signal.

The temporal walk generalizes this to **multi-hop structural importance**: a past token is important not just if the current token attends to it directly, but if many downstream tokens transitively depend on it through the causal attention graph.

At each generation step `t`, maintain a walk state `C` (accumulated multi-hop attention):

```
a_t   = attention row of token t           shape: (t,)
multi = a_t @ C                            multi-hop propagation through walk state
C[t, :t] = a_t + multi                    update: direct + indirect paths
```

The **walk importance score** of past token `j` at step `t` is:

```
score[j] = C[t, j]  (or sum over recent steps for stability)
```

Tokens with low `score[j]` have contributed little to the reasoning chain and are candidates for eviction.

The **walk change** at step `t`:

```
walk_change[t] = ||C[t, :] - C[t-1, :]||_2
```

Steps with high walk change are structural decision points — they reorganize the multi-hop dependency graph. Tokens at these positions are especially load-bearing.

---

## Why Multi-Hop Matters

Consider a chain: token A → token B → token C → token D.

- Token A is heavily attended by B, but D barely attends to A directly.
- Direct attention at step D: score[A] ≈ 0 → H2O would evict A.
- Walk at step D: A → B → C → D propagates, so score[A] is high → walk retains A.

For long reasoning chains (math proofs, code generation), intermediate steps set up variable bindings, partial conclusions, or proof hypotheses that are critical for later steps even though they're not directly attended to. The walk captures this.

---

## Comparison to Existing Methods

| Method | Signal | Multi-hop | Per-layer | Overhead |
|--------|--------|-----------|-----------|----------|
| H2O | Cumulative direct attention | No | Yes | O(T) per step |
| SnapKV | Attention from recent window | No | Yes | O(window × T) |
| StreamingLLM | Fixed recent + sink tokens | No | No | O(1) |
| PyramidKV | Layer-varying budget | No | Yes | O(T) per step |
| **Walk eviction** | Multi-hop propagation | **Yes** | Yes | O(T) per step |

The walk has the same asymptotic overhead as H2O (both maintain an O(T) score vector updated at each step), but captures transitivity that direct-attention methods miss.

---

## Implementation Sketch

Per layer `l`, maintain a walk state `C_l` of shape `(T,)` (importance scores for all past tokens). At each new token position `t`:

```python
a = attn_weights[l, t, :t]          # (t,) attention row, already computed
multi = a @ C_l[:t]                  # scalar: multi-hop mass
C_l[t] = a.sum()                     # direct contribution of t to future
C_l[:t] += a * (1 + multi)          # update past token importances
```

**Eviction decision**: when KV cache for layer `l` exceeds budget `K`, evict the `(T - K)` tokens with lowest `C_l` scores, excluding:
- The most recent `R` tokens (recency bias, similar to StreamingLLM's sink + recent)
- The first few tokens (attention sink tokens)

**Incremental cost**: one dot product `a @ C_l[:t]` per layer per step — same order as computing attention itself. No extra forward pass needed.

---

## Viability Assessment

**Strengths:**

1. **Zero extra compute at generation time.** Attention weights `a_t` are already computed; the walk update is a cheap O(T) operation on top.

2. **Principled transitivity.** The walk captures token importance that propagates through the reasoning chain — especially valuable for math/code where intermediate conclusions are referenced many steps later.

3. **Natural "structural pivot" detection.** High `walk_change[t]` identifies tokens that reorganize the dependency graph. These are exactly the tokens PIVOT uses as triggers for Langevin intervention — the same signal is load-bearing for both training and inference efficiency.

4. **Per-layer adaptivity.** Each layer develops different dependency patterns (early layers: syntactic/positional; late layers: semantic/logical). Per-layer walk scores naturally reflect this without requiring manual tuning.

**Weaknesses / Open Questions:**

1. **Walk state size.** `C_l` is O(T) per layer — for 32 layers and T=4096, that's 32 × 4096 floats ≈ 0.5 MB. Manageable, but adds memory pressure on top of the KV cache itself.

2. **Calibration needed.** The walk accumulates over the full sequence; early tokens naturally accumulate higher scores just by being seen more times. Need normalization (e.g., divide by the number of steps since the token was generated) to avoid positional bias.

3. **Single-layer vs. layer-averaged.** PIVOT-v1 used layer-averaged attention for the walk, which is richer but 32× more expensive to maintain. For eviction, per-layer walk is natural (each layer has its own KV cache to evict). The question is whether per-layer walk is sufficient or whether cross-layer aggregation is needed.

4. **Competes with simple baselines.** H2O already captures most of the "important past tokens" effect, and the marginal gain from multi-hop transitivity may be small in practice. The case for walk eviction is strongest on tasks with long reasoning chains where indirect dependencies dominate — math, code, multi-step planning.

5. **Dynamic eviction timing.** When to evict: at a fixed token budget, or adaptively when `walk_change[t]` is low (indicating no structural reorganization = safe to evict)? The adaptive approach is more principled but harder to implement in practice.

---

## Connection to PIVOT

The walk signal is the same one used in PIVOT-v1 for identifying training trigger positions. The eviction and training applications are complementary:

- **Training (PIVOT):** high `walk_change[t]` → Langevin intervention during rollout → policy learns to handle structural decision points
- **Inference (eviction):** low `C_l[j]` → token j is not load-bearing → safe to evict from KV cache

A model trained with PIVOT should produce better-calibrated walk signals at inference time, since PIVOT training incentivizes the model to concentrate structural changes at true decision points. This creates a **training-inference co-design loop**: PIVOT training sharpens the walk signal; sharper walk signal improves eviction quality at inference.

---

## Recommended Next Steps

1. **Implement and benchmark against H2O on LongBench / RULER** — measure whether multi-hop walk beats direct attention accumulation on tasks with long reasoning chains.

2. **Ablate normalization schemes** — raw cumulative score vs. recency-decayed vs. per-token-age normalized.

3. **Validate on math/code tasks specifically** — the strongest case for walk eviction is long chain-of-thought where intermediate conclusions matter.

4. **Check if walk importance ≈ gradient importance** — if walk score correlates with the gradient of the output with respect to each KV pair, that would provide theoretical justification that the walk is capturing the right signal.
