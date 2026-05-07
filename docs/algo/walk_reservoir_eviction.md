# KV Cache Eviction via Reservoir Sampling with Temporal Walk Opportunity Cost

## The Problem

As a language model generates tokens, its KV cache grows linearly. At some point the
cache exceeds the available memory budget and tokens must be evicted. The eviction
decision is: **which cached tokens can be removed without degrading the output?**

The naive answer is to evict whichever tokens the current query attends to least
(H2O, PyramidKV). This works reasonably well but has a structural blind spot:
**a token can be critical for future generation even if the current token barely
attends to it directly**, because reasoning chains are transitive. A variable defined
at position 50 might not be directly attended to at position 300, but position 200
used it, and position 300 depends on position 200's conclusion. Evicting position 50
breaks the chain silently.

This document describes a method that addresses both problems: **reservoir sampling
with temporal walk opportunity cost.**

---

## Two Complementary Signals

Eviction needs to answer two different questions simultaneously:

1. **Current relevance**: Does the current query need this block *right now*?
2. **Structural importance**: Would removing this block break something many future
   queries will depend on?

These are different signals. A block can score low on current relevance (the model
hasn't attended to it recently) while scoring high on structural importance (many past
queries routed their reasoning through it, and future queries will too). Direct-attention
methods conflate the two, treating low current relevance as grounds for eviction
regardless of structural role.

The method here uses:
- **Block-level attention scoring** for current relevance.
- **Temporal walk** for structural importance (opportunity cost).
- **Weighted reservoir sampling** to make the eviction decision under a fixed budget,
  combining both signals.

---

## Block-Level Scoring

Computing token-level attention scores for every cached token at every step is
expensive. Instead, tokens are grouped into non-overlapping **blocks** of `block_size`
tokens. Each block is represented by the mean of its key vectors.

**Block score at decode step** (current relevance):

```python
blk_k_mean = blk_k_sum / blk_k_count          # (B, H, Nk, D) block mean keys
attn_row = _block_score_decode(q, blk_k_mean)  # (B, Nk)
```

`_block_score_decode` computes scaled dot-products between the current query and each
block mean, applies softmax with high temperature (100) to preserve relative ordering
without saturation, sums over heads, and row-normalizes. The result is a `(B, Nk)`
vector where entry `k` measures how much the current query depends on block `k`.

The block mean cache (`blk_k_sum`, `blk_k_count`) is updated incrementally: each new
token adds its key vector to the running sum of the current open block. No recomputation
over the full cache.

---

## Temporal Walk: Structural Importance

### The Walk State

Per layer, maintain a vector `walk_C` of shape `(B, Nk)` — one score per cached block.
`walk_C[b, k]` accumulates **how much all tokens generated so far have depended on block
`k`, directly and transitively through the causal attention graph.**

At each decode step, after computing `attn_row`:

```python
multi  = (attn_row * walk_C).sum(dim=-1, keepdim=True)   # (B, 1) scalar per sequence
walk_C = walk_C + attn_row * (1 + multi)                  # (B, Nk) updated walk state
```

### Interpretation

- `attn_row * walk_C` measures: for each block `k`, how much attention is being directed
  to `k` right now, weighted by how structurally important `k` already is.
- `multi = (attn_row * walk_C).sum(-1)` is the total "multi-hop mass" at this step:
  the current token is routing attention through how much load-bearing structure?
- `walk_C += attn_row * (1 + multi)` updates each attended block by its direct
  contribution, amplified by the multi-hop context. If this step routes through
  heavily load-bearing blocks (`multi` large), all attended blocks get a larger boost.

When `walk_C = 0` (cold start), `multi = 0` and the update reduces to
`walk_C += attn_row` — identical to a plain cumulative attention sum. The walk is a
strictly richer generalization: it equals the cumulative sum when there is no prior
structural depth, and amplifies importance for blocks embedded in reasoning chains.

### Why This Captures Transitive Dependencies

Consider the chain A → B → C → D (each token attends to the previous):

- After step B: `walk_C[A] += attn_B[A] * 1` (no prior depth)
- After step C: `multi = attn_C[B] * walk_C[B]` (B was touched; multi > 0)
  `walk_C[A] += attn_C[A] * (1 + multi)` — A gets a transitive boost from C→B→A
- After step D: `multi = attn_D[C] * walk_C[C]` (even larger)
  `walk_C[A] += attn_D[A] * (1 + multi)` — A further boosted through C

Even if D barely attends to A directly (`attn_D[A] ≈ 0`), the chain propagates
importance back through the multi-hop term. H2O would score A near zero at step D
and evict it. The walk scores A high and retains it.

---

## Opportunity Cost: Why Current Relevance Alone Is Not Enough

Scoring blocks by current relevance only answers "what does this query need right now?"
A block might score low this step but be critical for every query over the next 500
tokens. Evicting it is irreversible.

**Opportunity cost** reframes eviction as a resource allocation problem: evicting block
`k` has a cost proportional to how much aggregate demand will be lost — not just demand
from the current query but from all future queries that would have depended on `k`.

The walk state `walk_C` is a direct estimate of this cost. It measures accumulated
multi-hop demand over all past steps. Under the assumption that future structural
dependencies resemble past ones (valid for structured reasoning tasks where the model
builds on prior steps), `walk_C[k]` is the best available estimate of how much future
queries will pay if block `k` is evicted now.

The opportunity cost boost is applied before sampling:

```python
col_norm    = walk_C / walk_C.sum(dim=-1, keepdim=True).clamp(min=1e-8)   # L1-normalize
sampling_row = attn_row * (1 + opp_weight * col_norm)                      # (B, Nk)
```

`opp_weight` controls the blend: at 0 the reservoir scores blocks only by current
relevance; at large values structural importance dominates. In practice 0.5–2.0 is a
reasonable starting range.

---

## Weighted Reservoir Sampling

Given `sampling_row` (B, Nk), a **weighted reservoir sampler** selects which `K` blocks
to retain under the fixed memory budget. Unlike a hard top-K, the reservoir introduces
controlled stochasticity: high-weight blocks are retained with high probability but not
deterministically. This has two benefits:

1. **Avoids degenerate collapse**: deterministic top-K can repeatedly retain the same
   set of "safe" early blocks while discarding everything that fluctuates. The reservoir
   explores the retention distribution.
2. **Forces robustness**: the model cannot rely on any specific block being guaranteed
   present; it must maintain outputs that are robust to stochastic context variation.

Some blocks are always forced to be retained regardless of their score:
- **Attention sinks**: the first 1–4 tokens that receive disproportionate attention
  regardless of content (known from StreamingLLM).
- **Recency window**: the most recent `R` blocks (the model attends to recent tokens
  most heavily; evicting them creates a hard discontinuity).

These forced-retention slots are subtracted from the budget before the reservoir runs.

---

## Full Decode Step

At each decode step the procedure is:

```
1. Compute attn_row = _block_score_decode(query, blk_k_mean)        (B, Nk)

2. Update walk state:
      multi   = (attn_row * walk_C).sum(-1, keepdim=True)            (B, 1)
      walk_C += attn_row * (1 + multi)                                (B, Nk)

3. Apply opportunity cost:
      col_norm     = walk_C / walk_C.sum(-1, keepdim=True)
      sampling_row = attn_row * (1 + opp_weight * col_norm)           (B, Nk)

4. Reservoir sample:
      allow_row = weighted_reservoir_sample(sampling_row, forced_mask) (B, Nk) bool

5. Evict:
      compact KV cache to kept blocks

6. Update cache stats (see Eviction Update section below)
```

Steps 2–3 add one dot product and two vector operations on top of the block scoring
that already runs at every step. The overhead is O(Nk) per layer — negligible.

---

## The Eviction Update Problem

After eviction, `walk_C` needs to reflect the compacted cache. This is subtle and the
right update is not obvious.

### Why a Simple Gather Is Insufficient

After evicting block E and keeping block B, the gather path is:

```python
walk_C = walk_C[:, keep_1d]
```

Block B's retained score includes mass from chains that ran *through* E. Example: a
prior step T attended to both E and B, with a large `multi` because E was load-bearing.
B's walk_C was boosted by `attn_T[B] * (1 + multi_T)` — the multi_T term reflected
E's structural role. After E is evicted, the chain query → E → B is broken. B's
walk_C is inflated by a path that no longer exists.

More importantly: future steps compute `multi = (attn_row * walk_C).sum(-1)`. With B
holding inflated stale mass, future `multi` values are artificially large, which
over-amplifies every subsequent update. **The inflation compounds step by step.**

### Why a Full Reset Defeats the Purpose

An alternative is to reset `walk_C` to `attn_row_post` (scores against the compacted
cache) — a clean slate. But the walk's value over simple cumulative attention is
precisely the accumulated structural history. A reset throws away everything the walk
has learned about which blocks are load-bearing, degrading it to a single-step signal
identical to the non-walk baseline.

### The Gather + Re-anchor Solution

After eviction, with the compacted cache available:

```python
# Step 1: gather history for surviving blocks
walk_C = walk_C[:, keep_1d]                                          # (B, Nk_kept)

# Step 2: compute attn_row_post — current query scored against post-eviction cache
attn_row_post = _block_score_decode(query, blk_k_mean_post)          # (B, Nk_kept)

# Step 3: re-anchor — one walk update using post-eviction structure
multi_post = (attn_row_post * walk_C).sum(dim=-1, keepdim=True)      # (B, 1)
walk_C     = walk_C + attn_row_post * (1 + multi_post)               # (B, Nk_kept)
layer_cache["walk_C"] = walk_C
```

This achieves both goals:
- **Preserve history**: surviving blocks retain their accumulated structural importance.
  Blocks that were genuinely load-bearing (high walk_C for real reasons) keep their
  scores. The history is the signal; discarding it is waste.
- **Correct the multi term**: the re-anchor step runs with `attn_row_post`, which is
  computed against the post-eviction graph. After this update, `walk_C` reflects one
  complete step in the post-eviction regime. The next step's `multi` will be computed
  against a state already anchored to the surviving blocks.
- **Natural self-correction**: blocks whose high walk_C was purely E-mediated will
  receive little boost from `attn_row_post` (future queries can't route through E), so
  their inflated scores dilute over subsequent steps rather than compounding.

The re-anchor costs exactly one additional `_block_score_decode` call (already computed
if `dynamic_rescore=True`) plus one walk update — the same cost as a normal step.

### Comparison

| Path | Walk state quality | Multi inflation | Trade-off |
|---|---|---|---|
| Bare gather | stale E-mass | compounds step by step | cheap but increasingly wrong |
| Full reset | history-free | none | loses all accumulated structure |
| **Gather + re-anchor** | history + post-eviction correction | corrected once | **recommended** |

---

## Prefill Seeding

During prefill the full block score matrix `(B, Nq_blocks, Nk_blocks)` is available.
Rather than cold-starting `walk_C = 0` at decode step 0, run a forward walk pass over
the prefill rows to seed the decode walk state:

```python
walk_C = torch.zeros(B, Nk_blocks, device=device)
for q in range(Nq_blocks):
    a_q   = block_scores[:, q, :]                         # (B, Nk), causally masked
    multi = (a_q * walk_C).sum(dim=-1, keepdim=True)
    walk_C = walk_C + a_q * (1 + multi)
layer_cache["walk_C"] = walk_C.detach()
```

This costs O(Nq × Nk) — same as computing the prefill block score matrix, no extra
key/query accesses.

After post-prefill eviction with dynamic rescoring, recompute the walk on the surviving
blocks and re-anchor before the first decode step:

```python
block_scores_post = _block_score_prefill(query_states, key_states_post, block_size)
walk_C_post = torch.zeros(B, Nk_kept, device=device)
for q in range(Nq_blocks):
    a_q   = block_scores_post[:, q, :]
    multi = (a_q * walk_C_post).sum(dim=-1, keepdim=True)
    walk_C_post = walk_C_post + a_q * (1 + multi)
layer_cache["walk_C"] = walk_C_post.detach()
```

The first decode step then starts with a walk state that reflects the full prefill
dependency structure over only the surviving blocks — no cold-start transient.

---

## Normalization and Age Bias

Walk scores accumulate over all steps, so blocks that entered the cache early
have had more steps to accumulate mass than recent blocks. This positional bias
is the same one present in cumulative attention methods.

Two mitigations:

**L1 normalization (already applied):** `col_norm = walk_C / walk_C.sum(-1, keepdim=True)`
before the opportunity cost boost means the signal is always relative. Absolute
magnitudes cancel; only relative rankings matter. This handles mild positional bias.

**Exponential decay (optional):**
```python
walk_C = decay * walk_C + attn_row * (1 + multi)    # decay ∈ (0, 1)
```
Trades long-range memory for recency sensitivity. At `decay = 1.0` (no decay) the walk
captures the full history. At `decay < 1` it forgets old paths. For long sequences with
many evictions, moderate decay (0.95–0.99) prevents early-token scores from dominating
indefinitely.

Starting point: no decay, rely on L1 normalization. Add decay only if empirical results
show recent blocks are being aggressively evicted due to early-block dominance.

---

## Prefill Opportunity Cost

During prefill, the same opportunity cost logic applies, but supply is computed directly
from the block score matrix rather than the walk accumulator:

```python
# Causal column sum: how much does each key block receive from all query blocks that
# can causally reach it (q >= k)?
supply = (block_scores * causal_mask[:, :, None]).sum(dim=1)   # (B, Nk)
block_scores_boosted = block_scores * (1 + opp_weight * supply / supply.sum(-1, kd=True))
```

If prefill seeding is active, the walk state computed during prefill (see above) can be
used as the prefill supply instead, providing a richer signal:

```python
# walk_C after prefill pass encodes full causal multi-hop structure — use it as supply
supply = walk_C_post
```

---

## Relationship to Existing Methods

| Method | Primary score | Opportunity cost | Multi-hop | Budget mechanism |
|---|---|---|---|---|
| H2O | Cumulative direct attn | None | No | Top-K |
| SnapKV | Recent-window attn | None | No | Top-K |
| StreamingLLM | Recency + sinks | None | No | Fixed window |
| PyramidKV | Direct attn, layer-varying budget | None | No | Top-K |
| This method | Block-level direct attn | Temporal walk (multi-hop) | **Yes** | Weighted reservoir |

Two key differences from all prior methods:

1. **Multi-hop opportunity cost**: prior methods evict based on direct attention only.
   The walk captures transitive dependencies that direct-attention methods miss.

2. **Reservoir sampling vs top-K**: reservoir sampling introduces controlled
   stochasticity in the retention decision. Top-K is deterministic and can collapse
   to always retaining the same early tokens; the reservoir maintains diversity in
   the retained context.

---

## Config Summary

| Field | Default | Effect |
|---|---|---|
| `kv_cache_eviction` | `False` | Enable physical compaction; all other features are inert without this. |
| `block_size` | `16` | Number of tokens per KV block. |
| `kv_cache_budget` | — | Maximum number of blocks to retain per layer. |
| `opportunity_cost_weight` | `0.0` | Walk opportunity cost blend weight. `0.0` = direct attention only. |
| `walk_age_decay` | `1.0` | Exponential decay on walk state. `1.0` = no decay. |
| `dynamic_rescore` | `False` | After eviction, recompute block means and re-anchor walk state. |
| `recency_window` | `4` | Number of most-recent blocks always retained. |
| `sink_blocks` | `1` | Number of initial blocks always retained (attention sinks). |
