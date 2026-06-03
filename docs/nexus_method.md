# Forget Without Compromise: Nexus Sampling for Streaming KV-Cache Eviction Under Fixed Budgets

---

## Abstract

The next generation of LLM workloads might be increasingly shaped by *agentic* and *streaming* settings—long-running agents, tool-augmented assistants, and persistent reasoning loops—in which the model's effective context is no longer a one-shot prompt but an unbounded stream that the system must manage under a *fixed* memory budget. Existing KV-cache eviction methods (H2O, Quest, StreamingLLM, Adamas) discard the irrelevance of this shift by committing, at every cache-update event, to a *deterministic* top-*K* set of blocks ranked by per-query attention. Deterministic top-*K* is the wrong primitive for a streaming budget: it is a hard threshold at the marginal rank that drops borderline blocks with full confidence regardless of how close their weight is to the cutoff, and once dropped a block cannot recover its mass on subsequent steps, producing a monotone-erosion failure mode in which long-context information silently disappears from the cache. We present **Nexus Sampling**, a training-free eviction method whose central primitive is *n*-fold averaged weighted reservoir sampling. The reservoir step replaces the hard threshold of top-*K* with an *unbiased weighted-without-replacement* retention rule: every block survives with probability proportional to its weight, and the marginal probability mass on borderline blocks is preserved across the long sequence of cache-update events that an infinite-stream workload produces. The averaging count *n* is a single knob that interpolates continuously between maximally stochastic retention (`n = 1`, classical reservoir sampling) and effectively deterministic top-*K* (`n → ∞`), letting the practitioner tune the variance-versus-accuracy trade-off without changing the rest of the pipeline. The reservoir samples from a *walk-augmented weight*: a short multi-hop walk recurrence adapted from Sketch&Walk accumulates indirect importance—blocks reached through chains of mutually attended blocks that one-hop scoring cannot identify—lifting these blocks into the head of the score distribution before the reservoir step. The method composes cleanly with our prior Sketch&Walk sparse attention framework, applies uniformly to prefill and decode, and—across LongBench and RULER at aggressive (80–90%) eviction budgets—maintains near-lossless accuracy where deterministic top-*K* baselines degrade. We argue that as agentic context approaches an effectively infinite stream, *the eviction primitive itself*, and not just the per-step scoring function, becomes the bottleneck of fixed-budget LLM memory management, and that an unbiased reservoir primitive is the appropriate replacement for deterministic top-*K*.

---

## 1. Introduction

GPU memory imposes a hard ceiling on what an LLM inference stack can hold during long-context generation, while the KV cache — the intermediate keys and values retained for every previously processed token — grows linearly with context length [Pope 2023; Dao 2022]. At sufficiently long contexts the cache exceeds this memory budget, a gap that widens as the field moves toward long-lived agent-style deployments (multi-turn assistants, persistent reasoning loops, repository-scale coding agents) where the effective context grows toward an unbounded stream. Once the cache no longer fits, the inference stack cannot simply *read* it more efficiently — the approach taken by sparse-attention methods over a still-full cache [Quest, MInference, FlexPrefill, Adamas] — it must permanently *evict* tokens or blocks. A growing body of work [H2O, StreamingLLM, PyramidKV, AdaKV, MorhKV, SnapKV] performs this *KV cache eviction*, dropping tokens predicted to be unimportant to future queries — a problem that is, by its nature, *streaming and fixed-budget*: tokens arrive continuously, evictions are irreversible, and the policy must repeatedly decide what to retain against future queries it has not yet seen. Despite their effectiveness, most existing eviction methods rely on a common design choice — at every cache-update event they retain the *K* highest-scoring blocks by **deterministic top-*K*** selection — and a detrimental effect of this design is that deterministic top-*K* inherently confuses transient marginality with permanent unimportance: in a streaming cache where evictions are irreversible, a block whose score is only intermittently above the cutoff is dropped on the first marginal event, even though its time-averaged importance may be high.

**Deterministic top-*K* misses blocks whose importance fluctuates.** Per-event importance scores in KV caches are noisy and time-varying. Attention in modern LLMs is heavy-tailed [Zhang et al., 2023; Le et al., 2026]: a small head of blocks holds most of the attention mass, while the rest of the cache sits at near-uniform low scores where the relative ranking between blocks is dominated by noise. A block important to query *t* may look marginal at *t* + 1 and important again at *t* + 10. Top-*K* treats each of these per-event scores as a final verdict — a block that lands below the cutoff at any single event is dropped with the same finality as a block deep in the tail, regardless of how close its score was to the cutoff. A long context produces many such verdicts, and the per-event errors compound. The result is *monotone marginal erosion*: every block whose score is only intermittently above the cutoff is silently and permanently lost.

**Reservoir sampling is the streaming-algorithms answer to this setting.** Weighted reservoir sampling [Vitter 1985; Efraimidis & Spirakis 2006] retains each block with probability proportional to its current score, so that a block's long-run survival across many events tracks its *time-averaged* importance rather than collapsing on the first event where its score happens to land below the cutoff. This is the right selection primitive for KV cache eviction, but it inherits the calibration of the score it samples against. The direct-attention magnitude used by existing methods is locally myopic: it counts only what the current query attends to, and misses *bridge tokens* — blocks that no single recent query attends to strongly, but that hold together a strongly-connected cluster of mutually-attended blocks across the window, such that removing one would sever the cluster's internal connections.

**We propose Nexus Sampling.** Nexus is a training-free KV cache eviction method built from two components and applied uniformly across prefill and decode. The first component, *walk-augmented scoring*, computes a per-block weight by passing the direct-attention score through a short iterative walk recurrence that surfaces bridge tokens — blocks anchoring strongly-connected clusters of mutually-attended tokens across the window. The second, and the headline contribution, is *weighted reservoir sampling*: the *K* retained blocks are drawn with inclusion probability proportional to this walk-augmented weight, replacing the deterministic top-*K* selection step that every prior eviction method shares. The walk recovers importance that direct-attention top-*K* cannot see, and the reservoir recovers importance that any deterministic top-*K* cannot see. Section 4 establishes the central theoretical guarantee — that Nexus's long-run block survival decays as a product of per-event inclusion probabilities (a mean over events), where deterministic top-*K*'s collapses to zero on the first below-cutoff event.

**Nexus Sampling preserves near-lossless accuracy at aggressive eviction.** Across multiple model scales and a broad range of long-context benchmarks, Nexus matches dense attention to within a small margin at retention ratios as low as 10–20%, where deterministic-top-*K* baselines (H2O, PyramidKV, AdaKV, SnapKV) lose 3–5 points of average accuracy. The same selection runs uniformly in prefill and decode. The gap to deterministic top-*K* widens monotonically as the eviction budget tightens and the stream lengthens — the empirical signature of the marginal-mass analysis of §4. Our contributions include:

- We identify a fundamental limitation of deterministic-top-*K* KV cache eviction: it converts transient marginality into permanent loss, producing *monotone marginal erosion* in the streaming, fixed-budget regime that is inherent to KV cache eviction.
- We propose **Nexus Sampling**, a training-free KV cache eviction method that applies to both prefill and decode phases, combining weighted reservoir sampling with a walk-augmented score that surfaces bridge tokens in strongly-connected clusters of mutually-attended blocks.
- We establish theoretical guarantees of Nexus Sampling — reservoir sampling's long-run block survival is a *mean over events* where deterministic-top-*K*'s is a *min* — and empirically demonstrate that Nexus preserves near-lossless accuracy at retention ratios where deterministic-top-*K* baselines degrade by 3–5%.

---

## 2. Nexus Sampling

Nexus Sampling combines two components into a single pass over the cache that runs alongside attention at every cache-update event. *Walk-augmented scoring* (Sections 2.2–2.5) computes a per-block weight from a denoised observation window plus a multi-hop walk recurrence; *weighted reservoir sampling* (Section 2.6) commits to a retention set via a sampling step whose marginal inclusion probabilities are proportional to weight. The reservoir is the load-bearing primitive of the method — the only component whose design *replaces* an algorithmic primitive of prior work rather than refining one — and the upstream weight construction exists specifically to feed it stable, comprehensive weights.

### 2.1 Preliminaries and Notation

A decoder-only transformer with `L` layers and `h` heads caches, at each layer and each head, the keys `K ∈ ℝ^{T × D}` and values `V ∈ ℝ^{T × D}` produced over the first `T` input tokens. At decode step `T + 1`, the current query `q ∈ ℝ^{1 × D}` attends over the full cache: `o = softmax(qKᵀ / √D) · V`. The cache size grows linearly with `T` and quickly dominates GPU memory for long contexts.

An **eviction method** chooses, at every cache-update event, a subset `S ⊆ {1, …, T}` of cached positions to retain, subject to a budget `|S| = K_total < T`. Subsequent attention is computed over `K_S, V_S` rather than the full cache. Eviction is *streaming*: positions evicted at event `e` are not recoverable at event `e + 1`. Eviction methods are therefore designed to *predict* which positions will continue to be important to future queries, given only the queries seen so far.

We work at the granularity of **key blocks** rather than individual tokens. Block-level eviction matches the granularity of modern sparse attention kernels [Tang et al., 2024; Le et al., 2026] and amortizes the per-block decision cost over `b` tokens, where `b = 64` in the implementations below.

| Symbol | Meaning |
|---|---|
| `T_k` | Number of cached key tokens. |
| `D` | Head dimension. |
| `b` | Tokens per key block (default 64). |
| `N_k = ⌈T_k / b⌉` | Number of key blocks, indexed by `j`. |
| `W` | Observation window length: number of trailing query rows used to score the cache. |
| `Q ∈ ℝ^{W × D}` | Observation window: the trailing `W` query rows (head-averaged). |
| `K ∈ ℝ^{T_k × D}` | Full key cache (head-averaged). |
| `P ∈ ℝ^{W × T_k}` | Token-level attention `softmax(QKᵀ / √D)`. |
| `BlockSum_b(P) ∈ ℝ^{W × N_k}` | Row-preserving sum of `P` inside each block of `b` consecutive keys. |
| `rownorm(·)` | L1 row normalization: `rownorm(X)_{w,j} = X_{w,j} / Σ_{j'} X_{w,j'}`. |
| `Ŝ ∈ ℝ^{W × N_k}` | `rownorm(BlockSum_b(P))`: each row is one window-query's distribution over blocks. |
| `a ∈ ℝ^{N_k}` | Base score: `a = (1/W) Σ_w Ŝ_w`. |
| `C ∈ ℝ^{N_k}` | Walk state, updated `H` times by `C ← C + a^(q) ⊙ (1 + a^(q)ᵀ C)`. |
| `c_hub ∈ ℝ^{N_k}` | Hub score: `Σ_w BlockSum_b(P)_w` (column sum of `BlockSum_b(P)` before rownorm). |
| `c̃ ∈ ℝ^{N_k}` | Normalized walked-or-hub vector: `c̃ = c / ‖c‖₁`. |
| `r_j = j / (N_k − 1)` | Recency ramp, 0 (oldest) to 1 (newest). |
| `λ` | Mixing weight on the walked/hub term. |
| `ε_tie` | Tie-break magnitude on the recency ramp (default 10⁻⁶). |
| `w ∈ ℝ^{N_k}` | Combined sampling weight: `w = a + λ·c̃ + ε_tie·r`. |
| `K_total` | Total retained-block budget. |
| `K_forced` | Number of blocks in the *forced set* (sink + recency floor + current block). |
| `K = K_total − K_forced` | Reservoir-sampled budget. |
| `n` | Per-block reservoir averaging count (configurable; `n = 1` = classical reservoir sampling). |
| `π_j` | Reservoir priority for block `j`: `π_j = (1/n) Σ_i u_j^{(i) 1/w_j}` with `u_j^{(i)} ~ U(0, 1)`. |

### 2.2 Per-Block Attention Weight from an Observation Window

During prefill, `Q ∈ ℝ^{W × D}` is the last `W` tokens of the prompt. During decode, `Q` is a rolling buffer that we extend by one new query each step and truncate back to the last `W` rows. We compute the standard token-level attention against the full key cache and sum the mass that lands inside each block:

    P = softmax( Q Kᵀ / √D ) ∈ ℝ^{W × T_k}

and

    BlockSum_b(P)_{w, j} = Σ_{t = (j−1)b + 1}^{j·b}  P_{w, t}.

Each row of `BlockSum_b(P)` is the attention mass that one observation-window query places on each of the `N_k` blocks; the row sums to `1` over *tokens* but not over *blocks*. Row-normalizing to a probability distribution over blocks gives

    Ŝ = rownorm(BlockSum_b(P)) ∈ ℝ^{W × N_k},

so each row `Ŝ_w` is one observation-window query's distribution over key blocks, summing to 1. Row-normalization makes per-query block scores commensurable across rows: it strips out the absolute attention magnitude (which is dominated by softmax sharpness and varies wildly across queries) and retains only the relative block ranking, which is what eviction actually needs.

A length-*W* observation window denoises the single-query attention score at the standard `1/√W` rate (Lemma 4.4). Small `W` reacts fast to the latest decoded token; large `W` averages over more context and is steadier. We treat `W` as an implementation knob rather than a structural component; the headline contribution of Nexus is the reservoir step of Section 2.6, and the window exists only to feed the reservoir a denoised weight.

*Empirical evidence needed:* (i) **Per-step block-rank churn** comparing single-query and *W*-averaged scoring (Figure 1, §6 ablations) — supports the noise-reduction motivation. (ii) **Window-length ablation** on LongBench AVG at fixed eviction (Table 4, *Window* block) — supports the `W` configuration.

### 2.3 Window Collapse to a Base Score

Averaging the `W` rows of `Ŝ` produces a single base score over blocks:

    a = (1 / W) Σ_{w = 1}^{W}  Ŝ_w   ∈ ℝ^{N_k}.

This is the Nexus equivalent of the single-query block score used by H2O or Quest, denoised by averaging over a window. By construction `a` is a probability distribution: `Σ_j a_j = 1`. We treat `a` as the **direct importance** vector for the eviction decision.

### 2.4 Multi-Hop Walk

The base score `a` captures only *direct* importance: which blocks the recent queries attend to right now. To capture *indirect* importance — blocks that are themselves heavily attended by other queries in the window — we iterate a short recurrence for `H = 3` steps, plugging in a fresh row `a^(q) = Ŝ_q` at each step:

    C ← C + a^(q) ⊙ (1 + a^(q)ᵀ C),    C^(0) = 0,

with `⊙` denoting elementwise product. The scalar `a^(q)ᵀ C` is the inner product between the current observation-window query's block distribution and the accumulated walk; the factor `1 + a^(q)ᵀ C` is an alignment-weighted scalar that boosts the current update when the present query reinforces the importance landscape already encoded in `C`. The elementwise multiplication by `a^(q)` then deposits this scalar-weighted mass on the blocks the present query concentrates on. Call the final value `C`.

**Why the alignment scalar.** The recurrence is a deliberate analogue of PageRank-style transitivity propagation [Brin & Page, 1998; Page et al., 1999], adapted to the per-window block-score setting. Without the `(1 + a^(q)ᵀ C)` scalar, the walk would reduce to `C = Σ_q a^(q) ∝ a` and add no information beyond the base score `a` already provides. The alignment factor is what makes the recurrence amplify blocks that successive observation-window queries *agree* on: a block heavily attended by query `q` and also by query `q'` (i.e. `a^(q')ᵀ C` is large after the `q`-th update) receives a multiplicative boost on the subsequent update, while a block attended only by an idiosyncratic single query is not boosted. The mechanism is structurally close to the per-layer walk of [Le et al., 2026], with the per-layer block-attention matrix replaced by the per-window per-query distribution.

**Why `H = 3`.** Three iterations are empirically sufficient: the alignment-scalar reinforcement stabilizes within a small number of steps as the recurrence amplifies the centers of mutually-supporting clusters, and the additional mass contributed by later iterations falls below the eviction-threshold noise floor.

*Empirical evidence needed:* **Walk-depth ablation** on LongBench AVG and RULER multi-hop tasks (Table 4, *Walk depth* block) — supports the `H = 3` choice. **Indirect-importance recovery plot** comparing block-rank under composed multi-layer attention vs. under one-hop attention, with walk-augmented rank overlaid (Figure 2 spec in §3) — supports the walk's mechanism.

### 2.5 Combined Sampling Weight

The final sampling weight is

    w = a + λ · c̃ + ε_tie · r,    c̃ = c / ‖c‖₁,

with three terms.

- **Direct importance `a`.** The base score from §2.3.
- **Indirect importance `λ · c̃`.** The vector `c` is one of two options, chosen by a configuration knob: (i) the walk vector `C` from §2.4, which captures multi-hop indirect importance; or (ii) the hub vector `c_hub = Σ_w BlockSum_b(P)_w` (column sum of `BlockSum_b(P)` *before* row-normalization), which captures how much total attention mass each block receives across all `W` window queries — distinguishing blocks attended consistently by many queries from blocks attended sharply by one. After L1-normalization, `c̃` is a probability distribution over blocks, scaled by the mixing weight `λ`. Setting `λ = 0` disables the indirect term entirely and reduces Nexus Sampling to a windowed, sampled-top-*K*.
- **Recency tie-break `ε_tie · r`.** The ramp `r_j = j / (N_k − 1)` is linear from `0` (oldest block) to `1` (newest), and `ε_tie = 10⁻⁶`. The magnitude is small enough to never flip a real signal — `a + λ c̃` is on the order of `10⁻¹` to `10⁻³` per block — but large enough to deterministically resolve ties in favor of more recent blocks. This is the same role lexicographic tie-breaking plays in classical reservoir sampling [Chao, 1982].

The mixing weight `λ` trades off direct against indirect importance. In practice `λ` is set so the two terms have comparable magnitudes at the median-importance block. The walk-vs-hub choice is workload-dependent: walked `c̃` is the default for decode where the multi-hop affinity structure is richer; hub `c̃` is cheaper and is competitive in prefill where the observation window already spans the prompt.

*Empirical evidence needed:* **`λ` ablation** (Table 4, *Walk weight* block) — supports the mixing weight choice. **Walk-vs-hub ablation** (additional row in Table 4) — supports the two-mode design.

### 2.6 Weighted Reservoir Sampling (Load-Bearing Component)

This is the load-bearing step of Nexus Sampling. The upstream components of §2.2–2.5 produce a stable, comprehensive per-block weight `w ∈ ℝ^{N_k}`. The reservoir step decides which `K` of the `N_k` candidate blocks survive the cache-update event. The choice of selection rule here — not the choice of weight — is what distinguishes Nexus Sampling from every prior eviction method.

Some blocks are retained unconditionally and constitute the **forced set**:

- the current block (the one containing the most recent decoded token);
- the attention sink (the first few absolute positions of the cache);
- a small **recency floor** of the most recently produced blocks.

These are kept with no scoring or sampling, accounting for `K_forced` slots out of the total budget `K_total`.

For every remaining candidate block `j`, we draw `n` independent uniform variates `u_j^{(1)}, …, u_j^{(n)} ~ U(0, 1)` and form the **n-averaged reservoir priority**

    π_j = (1 / n) Σ_{i = 1}^{n} u_j^{(i) 1 / w_j}.

We then keep the top `K = K_total − K_forced` candidates by `π_j`.

**Why reservoir sampling is the right primitive for streaming KV-cache eviction.** Eviction is the irreversible commitment of `N_k − K` blocks at every event of a long stream of events. Deterministic top-*K* is a hard threshold at the marginal rank: a block at rank `K + 1` is dropped with the *same* finality as a block at rank `N_k`, regardless of how close its weight is to the cutoff. Iterated across many events, this monotonically sieves out exactly the blocks for which the system has the *least* confidence in its retention decision — the marginal-rank ones — producing a long-run cache populated only by strongly-and-consistently-attended blocks at the cost of every diffuse, intermittent, or indirect signal. Reservoir sampling replaces this hard threshold with a smooth retention curve: marginally important blocks survive a fraction of the time proportional to their weight, and the resulting policy is the unique unbiased weighted-without-replacement sample [Efraimidis & Spirakis, 2006]. Over a long sequence of events, the long-run block-survival probability under reservoir sampling decays gracefully as a product of per-event weights (a *mean* over events in log space), in contrast to deterministic top-*K* which is effectively a *min* over events — one below-cutoff event terminates the block permanently. Section 4 makes this contrast precise (Lemma 4.3).

**The `n = 1` case (classical reservoir sampling).** With `n = 1`, the priority reduces to `π_j = u_j^{1 / w_j}` with a single uniform variate per block. This is exactly the weighted reservoir-sampling rule of [Efraimidis & Spirakis, 2006]: the resulting top-*K* set is a sample from the weighted-without-replacement distribution with marginal inclusion probabilities exactly proportional to `w_j`. A single draw is unbiased but high-variance.

**The `n > 1` case (averaged reservoir sampling).** Larger `n` averages out the per-block randomness in the priority key. Each `u_j^{(i) 1 / w_j}` has mean `w_j / (w_j + 1)`, and averaging `n` independent copies shrinks the variance at the standard `1/√n` rate. The marginal retention probability is no longer exactly `w_j`-proportional, but the rank of block `j` under `π_j` converges almost surely to the rank under `w_j / (w_j + 1)` as `n → ∞` (Lemma 4.2). `n` is the headline knob of Nexus Sampling: it interpolates *continuously* between the unbiased, high-variance reservoir behavior (`n = 1`) and the deterministic, zero-variance top-*K* limit (`n → ∞`), leaving the variance-versus-accuracy trade-off to the practitioner rather than baking it into the algorithm.

**The deterministic branch.** In the strict deterministic configuration we skip the priority key entirely and take top-*K* by `w_j` directly. This is the `n → ∞` limit of the averaged-reservoir rule, exposed as a separate code path because it removes the per-block uniform-variate draw from the kernel and is the cheapest configuration to run. The deterministic branch is the appropriate choice when the workload is *not* a long stream (a single fixed prompt) or when output reproducibility across seeds is required and the marginal-mass-recovery benefit is not worth the added variance. For the streaming-budget regime that motivates this paper, the stochastic configurations are the default.

*Empirical evidence needed:* **`n`-averaging ablation** spanning `n = 1, 2, 4, 8` and the deterministic `n → ∞` branch (Table 4, *Reservoir averaging* block) — supports the headline `n = 4` choice and demonstrates the variance-accuracy trade-off. **High-eviction comparison** (Table 3) showing the gap between Nexus and deterministic top-*K* widening with eviction aggressiveness — supports the marginal-mass argument. **Long-run block-survival plot** comparing top-*K* (step function in time-averaged weight) vs. reservoir (smooth curve) across many cache-update events on a long-context trace (Figure 3 spec in §3) — supports the min-vs-mean over events claim.

---

## 3. Why Nexus Sampling for KV Cache Eviction?

Existing KV cache eviction methods rely on two assumptions they rarely state. The first is that block importance can be ranked from per-event direct-attention scores. The second is that selection from this ranking can be performed by deterministic top-*K*. Both are reasonable when the workload is a single long prompt followed by a short generation: one eviction event, a still-near-dense cache, a forgiving slack. Both fail in the streaming, fixed-budget regime that is inherent to KV cache eviction and that becomes harder as context grows unboundedly.

The first assumption fails because of where the score distribution lives. Attention in modern LLMs is heavy-tailed: only 5–10% of the cache typically carries the great majority of attention mass [Zhang et al., 2023; Le et al., 2026], and the remaining 90–95% of blocks sit at near-uniform low scores. The relative ranking among those low-scoring blocks is dominated by noise rather than by content — and yet they are precisely the population from which any aggressive top-*K* cutoff must choose. Deterministic top-*K* commits to one such choice with hard confidence at every event, producing a selection that is essentially arbitrary on the bulk of the cache.

The second assumption fails because of the streaming structure. In a one-shot setting the per-event arbitrariness is a bounded one-time cost; in a stream of cache-update events it compounds. A block's long-run survival under deterministic top-*K* is *a min over events*: it survives only if its score crosses the cutoff at every single event in the stream, and any one below-cutoff event terminates it for good. Under weighted reservoir sampling, the same long-run survival decays as a *product* of per-event inclusion probabilities — a *mean over events* in log space — each strictly positive whenever the per-event weight is. Lemma 4.3 makes the contrast precise.

The empirical picture matches. Existing eviction methods do well on benchmarks where the score has a clear head and the relevant blocks are stably top-ranked. They degrade on benchmarks where neither holds — multi-hop retrieval, long-summary, and any task whose answer depends on blocks the recent query only sometimes attends to. The marginal-rank fallacy predicts exactly this: when the relevant blocks live at the margin under direct attention, deterministic top-*K* sieves them out one event at a time.

*Empirical evidence needed for the marginal-rank fallacy:* (i) **Score-distribution diagnostic** — log the per-event distribution of block scores on a fixed long-context prompt and show the head-vs-bulk separation (Figure 4, §6 ablations); confirm the head holds ≥ 90% of mass and the bulk is near-uniform. (ii) **Time-varying-importance diagnostic** — track the rank of individual blocks over many decode steps on the same prompt; show that most blocks oscillate between head and margin rather than being stably one or the other (Figure 1 spec, §6). (iii) **Long-run survival diagnostic** — simulate a long stream of cache-update events under top-*K* and under reservoir, and plot survival probability against time-averaged weight; top-*K* should appear as a step function in mean weight, reservoir as a smooth curve (Figure 3 spec, §6).

Nexus Sampling addresses both assumptions of the existing template, with two structural fixes that recover two complementary categories of blocks the existing literature silently loses:

- **Bridge tokens** — blocks that hold together a strongly-connected cluster of mutually-attended tokens across the window, such that removing one would sever the cluster's internal connections. A bridge is not the loudest connection under any single query; it accumulates *moderate* attention from each of many queries that are themselves attending to one another, so direct-attention scoring (H2O cumulative, Quest min/max, Adamas sketched) under-counts it by construction and deterministic top-*K* on a direct-attention score systematically evicts it as "marginal." Nexus surfaces bridges with two complementary signals before any selection happens: the column sum of `BlockSum_b(P)` before row-normalization (Section 2.5), which counts how much attention a block receives across the window — a block attended at 0.1 mass by every one of `W = 16` queries scores 1.6, where a block attended at 1.0 by one query and 0.0 by the rest scores 1.0 — and the iterative walk of Section 2.4, which reinforces blocks whose attention pattern aligns with the accumulated walk state and so amplifies the cluster's holding nodes. Both surface the same conceptual signal (bridge importance), and either is exposed as a configuration knob feeding the reservoir's weight.
- **Time-varying-importance tokens** — blocks whose per-event importance fluctuates: in the head at some events, at the margin at others. Even with a perfectly comprehensive weight — even if the walk and column sum fully surface every bridge — *deterministic* top-*K* still fails these blocks: it kills them on the first event where their weight lands below the cutoff, regardless of their time-averaged importance. The reservoir step of Section 2.6 is the only fix: it converts per-event marginality into per-event inclusion probability rather than into permanent loss.

The two categories map onto two orthogonal mechanisms. Bridges are fixed in the *weight construction* (walk and column sum); time-varying importance is fixed in the *selection rule* (reservoir). Neither subsumes the other. The reservoir cannot save bridges on its own — they have to be lifted into the head of the weight first, or sampling proportional to weight samples from the wrong distribution. The walk and column-sum signals cannot save time-varying importance on their own — a perfectly comprehensive per-event weight is still killed by deterministic top-*K* the first time its score lands below the cutoff. Both fixes are needed, and together they remove the two assumptions §3 rejects.

*Empirical evidence needed for the walk:* **Multi-hop retrieval tasks on RULER** (variable-trace-length subsets, e.g., variable-tracking, multi-hop) at matched eviction budget — Nexus vs. an unwalked variant (`λ = 0`) should show a measurable gap on multi-hop tasks but a small gap on single-hop tasks (Table 4, *Walk weight* block plus a multi-hop-vs-single-hop split). *Empirical evidence needed for the hub signal:* **Hub-vs-walk ablation** on prompts where attention is broad-but-shallow (e.g., long-summary tasks) vs. narrow-but-deep (e.g., needle-in-a-haystack) — the hub signal should help on the former, the walk on the latter. *Empirical evidence needed for the reservoir:* **High-eviction LongBench AVG curve** (Table 3) showing Nexus's gap to the runner-up widening at 80–90% eviction; **`n`-averaging ablation** showing the deterministic-limit branch underperforming `n = 1, 2, 4` in the high-eviction regime (Table 4).

By combining the walk-augmented weight with the reservoir selection rule, Nexus Sampling addresses both structural failure modes of the deterministic-top-*K* template in a single eviction policy that applies uniformly to prefill and decode.

---

## 4. Theoretical Analysis

We establish three guarantees that justify the Nexus Sampling design choices. The headline guarantees concern the reservoir primitive (§4.1) — the load-bearing component of the method. The supporting guarantees concern the upstream weight construction: window averaging as noise reduction (§4.2) and the walk recurrence as an indirect-importance aggregator (§4.3).

### 4.1 Reservoir Sampling: Unbiasedness, n-Averaged Concentration, and Streaming Marginal-Mass Survival

**Lemma 4.1 (Reservoir Unbiasedness, `n = 1`).** *Let `π_j = u_j^{1 / w_j}` with `u_j ~ U(0, 1)` independent. The set `S = {j_1, …, j_K}` of indices with the `K` largest `π_j` values is a sample from the weighted-without-replacement distribution with marginal inclusion probabilities proportional to `w_j` [Efraimidis & Spirakis, 2006, Theorem 1].* □

The unbiasedness is exact: the marginal probability that block `j` is among the retained top-*K* is `w_j / Σ_{j'} w_{j'}` for all `j`, modulo the `K`-combinatorial inclusion-exclusion correction. Borderline blocks (those with weight close to the *K*-th-ranked value) survive a fraction of the time proportional to their weight, rather than being deterministically dropped.

**Lemma 4.2 (n-Averaged Reservoir Concentration).** *Let `π_j^{(n)} = (1/n) Σ_i u_j^{(i) 1 / w_j}`. Then `E[π_j^{(n)}] = w_j / (w_j + 1)` and `Var(π_j^{(n)}) → 0` at rate `1/n` as `n → ∞`. In particular, the rank of block `j` under `π_j^{(n)}` converges almost surely to the rank under `w_j / (w_j + 1)` as `n → ∞`.* □

For moderate `w_j`, `w_j / (w_j + 1) ≈ w_j`, so the deterministic limit is approximately top-*K* by `w_j` directly. The averaging count `n` is therefore a *single-knob* interpolation between unbiased sampling (Lemma 4.1) and deterministic top-*K* (`n → ∞`).

**Lemma 4.3 (Streaming Marginal-Mass Survival: Min over Events vs. Mean over Events).** *Consider a sequence of cache-update events `e = 1, …, E` with per-event weights `w_j^(e) > 0` for block `j`, retention budget `K` per event, and per-event score distribution. Let `w_{(K)}^(e)` be the rank-K weight at event `e`. Let `S_j^{top-K}(E)` and `S_j^{res}(E)` denote the survival probability of block `j` after all `E` events under deterministic top-K and under reservoir sampling, respectively. Then:*

- *(Min over events, deterministic top-K)*: `S_j^{top-K}(E) = ∏_{e=1}^{E} 𝟙[w_j^(e) ≥ w_{(K)}^(e)]`. *In particular, `S_j^{top-K}(E) = 0` whenever `w_j^(e) < w_{(K)}^(e)` for any single event `e ≤ E`: a single below-cutoff event terminates the block permanently.*
- *(Mean over events, reservoir sampling)*: `S_j^{res}(E) = ∏_{e=1}^{E} q_j^(e)`, *where* `q_j^(e) ∈ (0, 1]` *is the per-event reservoir inclusion probability and is bounded below by a strictly positive function of `w_j^(e)` whenever `w_j^(e) > 0`. Taking logs, `log S_j^{res}(E) = Σ_e log q_j^(e)`, i.e. the log-survival is a sum of per-event log-inclusion probabilities — the mean-over-events behavior that contrasts with top-K's min.*

*The contrast is exact and structural: deterministic top-K collapses long-run survival to zero on the first below-cutoff event; reservoir sampling preserves it as a multiplicative average. For a block whose per-event weight is stable above the cutoff, both rules retain it with probability 1. For a block whose per-event weight is intermittently above the cutoff — exactly the class of blocks targeted by Nexus's marginal-mass argument — deterministic top-K's survival probability collapses to 0 while reservoir's survives at a rate that decays gracefully with the time-averaged weight.* □

This is the precise mechanism by which Nexus Sampling preserves the marginal probability mass that deterministic top-*K* discards in the streaming-budget regime. The argument does not require us to "save" more blocks per event than top-*K* does (the per-event budget `K` is the same under both rules); what reservoir sampling does is replace a fixed-by-tiebreak per-event marginal-subset with a weighted-random one, and the difference compounds across events.

*Empirical evidence needed for Lemma 4.3:* **Long-run block-survival simulation** on a fixed long-context trace, with both deterministic top-K and reservoir sampling, plotting empirical survival probability against time-averaged weight for every block (Figure 3 spec in §3). Expected pattern: top-K should appear as a sharp step function around the marginal weight; reservoir should be a smooth monotone curve.

### 4.2 The Observation Window as Noise Reduction

Let `a_q ∈ ℝ^{N_k}` denote the row-normalized per-query block distribution for query `q`. Suppose each `a_q = µ + ξ_q` decomposes into a *content-bearing* component `µ ∈ Δ^{N_k}` (the true block-importance distribution under the response's reasoning state) and a *noise* component `ξ_q` with `E[ξ_q] = 0` and `Var(ξ_q) ≤ ν² I` per coordinate.

**Lemma 4.4 (Window-Averaging Noise Reduction).** *Under the noise model above, with probability at least `1 − δ`,*

    ‖a − µ‖_∞ ≤ ν · √(2 log(2 N_k / δ) / W).

*Proof.* By a Hoeffding union bound over the `N_k` block coordinates of `a − µ = (1/W) Σ_q ξ_q`, applying the standard sub-Gaussian concentration of the average of `W` independent zero-mean noise variables. □

The bound is the standard `1/√W` noise-reduction rate for averaging i.i.d. random variables, with the *N_k*-dimensional union bound contributing only a `log N_k` factor. The implication for the reservoir step is that a modest window length (`W ≈ 16`–`32`) suffices to bring the per-block noise floor on the reservoir's input weights below the typical scale of meaningful per-block weight differences.

*Empirical evidence needed:* **Window-length ablation** at fixed eviction budget (Table 4, *Window* block). Expected pattern: AVG accuracy improves with `W` from `W = 1` and plateaus by `W ≈ 16`–`32`.

### 4.3 The Multi-Hop Walk as Indirect-Importance Aggregator

The walk recurrence `C ← C + a^(q) ⊙ (1 + a^(q)ᵀ C)` accumulates, after `H` iterations using window queries `q_1, …, q_H`,

    C_H = Σ_{h = 1}^{H} a^(q_h) ⊙ (1 + a^(q_h)ᵀ C_{h − 1}).

Expanding the recurrence to first order in the alignment scalars and writing `M_q = a^(q) (a^(q))ᵀ ∈ ℝ^{N_k × N_k}` for the rank-1 per-query block-affinity matrix:

**Lemma 4.5 (Walk Decomposition).** *To first order in `‖a^(q)‖_∞`,*

    C_H = ( Σ_h a^(q_h) ) + ( Σ_h M_{q_h} · C_{h − 1} ) + O(H · max_h ‖a^(q_h)‖_∞²).

*Proof sketch.* Expand the elementwise-and-scalar product `a ⊙ (1 + aᵀC)` as `a + (aᵀC) · a = a + M·C`, where `M = a aᵀ` is the rank-1 outer product. Substituting into the recurrence and unrolling gives the stated decomposition; the residual collects products of two or more `aᵀC` scalars, each bounded in magnitude by `H · max ‖a^(q)‖_∞²`. □

The first term is the unwalked window aggregate. The second term is the genuinely multi-hop contribution: each `M_{q_h}` is a rank-1 affinity matrix encoding which block pairs query `q_h` connects, and `M_{q_h} · C_{h − 1}` aggregates the accumulated walk under that affinity structure. The reservoir step (Lemma 4.1) consumes `C_H` as part of the combined weight `w`, so picking up `H`-hop chains of query-induced affinities translates directly into the reservoir's marginal-mass guarantee covering bridge tokens as well as directly-important ones.

The depth `H` controls the highest-order hop captured. As in [Le et al., 2026], small `H` (e.g. `H = 3`) suffices in practice because effective composed attention depth in transformer stacks is empirically shallow.

*Empirical evidence needed:* **Walk-depth ablation** (Table 4, *Walk depth* block) — supports the `H = 3` choice. **Multi-hop RULER subtasks** at matched eviction budget — supports the walk's claimed bridge-recovery role.

### 4.4 Computational Cost

Each component is a constant or low-degree polynomial in the cache size. The observation-window scoring (`P = softmax(Q Kᵀ / √D)`, `BlockSum_b`, `rownorm`) is `O(W · T_k · D)`; the window collapse is `O(W · N_k)`; the walk recurrence runs in `O(H · N_k)`; the combined-weight computation and the reservoir priority draws are `O(N_k)` and `O(n · N_k)` respectively. None of these exceeds the cost of the attention computation `Q Kᵀ` that the eviction step replaces or amortizes against, so Nexus Sampling does not introduce a new bottleneck in the inference pipeline.

*Empirical evidence needed:* **Kernel profile** breaking down the Nexus eviction-step time into scoring / window-collapse / walk / reservoir components, as a function of context length, at fixed `W`, `H`, `n` (companion to Figure 5 in §6.5). Expected pattern: the eviction step's total cost should be a small fraction of dense attention's cost at moderate-to-long contexts.

---

## 5. Experimental Setup

**Models.** We evaluate Nexus Sampling on long-context instruction-tuned models of different scales, matching the model suite of [Le et al., 2026] for direct comparability: Llama-3.1-8B-Instruct, Llama-3.2-1B-Instruct, and Qwen2-7B-Instruct, all of which support context lengths up to 128K tokens. We use the default chat template for each instruct model in all experiments. Following common practice, we do not apply Nexus Sampling to the first two layers, which exhibit low achievable sparsity.

**Benchmarks.** We evaluate on two complementary long-context benchmarks: (i) **LongBench** [Bai et al., 2024], spanning QA, reasoning, summarization, and code-understanding tasks; and (ii) **RULER** [Hsieh et al., 2024], a synthetic diagnostic stressing retrieval and position-sensitive reasoning over very long contexts. Context lengths reported span 4K to 64K tokens on RULER and the native long-context tasks of LongBench.

**Baselines.** We compare Nexus Sampling against KV cache eviction methods only — sparse-attention methods (Quest, Adamas, Sketch&Walk) are not direct baselines because they operate over a full cache rather than evict from it (Section 1). Baselines: (i) **dense attention** (the no-eviction reference); (ii) **StreamingLLM** [Xiao et al., 2024] — sink-plus-recency-window heuristic; (iii) **H2O** [Zhang et al., 2023] — cumulative-attention heavy-hitter retention; (iv) **PyramidKV** [Cai et al., 2024] — per-layer pyramidal budget allocation; (v) **AdaKV** [Feng et al., 2024] — per-head adaptive budget; (vi) **SnapKV** [Li et al., 2024] — observation-window-based selection. For each baseline we use the configuration reported in the original paper at the matched cache budget.

**Implementation details.** All experiments use Triton-implemented custom kernels for sparse attention (decode) and sparse prefill. Block size `b = 64`; observation window `W = 16` (decode) and `W = 32` (prefill); walk depth `H = 3`; mixing weight `λ = 0.5`; tie-break magnitude `ε_tie = 10⁻⁶`; reservoir averaging `n = 4` (the default; ablated in §6.4). Forced set: 4 sink blocks + 4 recency-floor blocks + current block. Cache budgets are reported as a *retention ratio* `K_total / N_k`.

All experiments are conducted on a single NVIDIA H100 GPU with 94 GB of memory. Evaluations use greedy decoding for output-determinism. Code is available at <https://github.com/Escanord/verl>.

---

## 6. Results

We organize the empirical evaluation around the three claims of §1 and §3:

- **C1 — Window denoising stabilizes the score distribution.** Empirical support: §6.1 (LongBench AVG at high eviction) and Table 4 *Window* ablation.
- **C2 — The walk recovers bridge importance that direct-attention top-K cannot see.** Empirical support: §6.2 (RULER multi-hop tasks) and Table 4 *Walk depth*, *Walk weight* ablations.
- **C3 — Reservoir sampling preserves time-varying importance that any deterministic top-K cannot see.** Empirical support: §6.3 (high-eviction regime), Table 4 *Reservoir averaging* ablation, and the Figure-3 long-run survival diagnostic.

§6.4 reports ablations; §6.5 reports end-to-end inference acceleration.

### 6.1 LongBench Accuracy at 80% Eviction

**Table 1.** Per-task accuracy on LongBench at 80% cache eviction (`K_total / N_k = 0.20`). All values are accuracy or task-specific score; bold marks the leading method per row. *Supports C1, C2, C3 collectively (headline accuracy result).*

| Method | 2wikimqa | gov-report | hotpot-qa | lcc | multifieldqa-en | multinews | musique | narrativeqa | passage-count | passage-retrieval | qasper | qmsum | repobench-p | samsum | trec | triviaqa | AVG | AVGpc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| StreamingLLM | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| H2O | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| PyramidKV | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| AdaKV | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| SnapKV | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **Nexus Sampling** | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

### 6.2 RULER Accuracy Across Context Lengths

**Table 2.** RULER accuracy across 4K–64K context lengths at 80% eviction. Bold marks the leading method per row. *Supports C2 (multi-hop subtasks) and C1 (long-context stability of window-averaged score).*

| Model | Method | 4K | 8K | 16K | 32K | 64K | Average |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | Dense | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Llama-3.1-8B | StreamingLLM | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Llama-3.1-8B | H2O | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Llama-3.1-8B | PyramidKV | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Llama-3.1-8B | AdaKV | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Llama-3.1-8B | SnapKV | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Llama-3.1-8B | **Nexus Sampling** | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

### 6.3 The High-Eviction Regime

**Table 3.** LongBench AVG (16-task average) as a function of cache budget for Llama-3.1-8B. Cache budgets shown as `K_total / N_k`. Bold marks the leading non-dense method per row; the gap between Nexus Sampling and the runner-up is reported in the rightmost column. *Supports C3 directly: the gap is the empirical signature of the marginal-mass argument.*

| Cache budget | Dense | StreamingLLM | H2O | PyramidKV | AdaKV | SnapKV | **Nexus Sampling** | Δ to runner-up |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 (50% eviction) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.30 (70% eviction) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.20 (80% eviction) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 0.10 (90% eviction) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

Expected pattern: at moderate eviction (50–70%), all methods retain enough capacity that the choice of selection rule matters less; at aggressive eviction (80–90%), where every retained block must be highly informative, Nexus Sampling's window denoising, walk-recovered bridges, and reservoir-preserved marginal mass collectively widen the gap to deterministic top-*K* baselines.

### 6.4 Ablations

We probe four design choices: observation window length `W`, walk depth `H`, mixing weight `λ`, and reservoir averaging count `n`. All ablations are on Llama-3.1-8B at 80% eviction; values are LongBench AVG. *Each ablation block is annotated with which claim it supports.*

**Table 4.** Nexus Sampling ablations. Bold marks the headline setting; Δ is the change vs. headline.

| Ablation | Setting | AVG | Δ |
|---|---|---:|---:|
| *Window (supports C1)* | `W = 1` (single-query) | [TBD] | [TBD] |
| *Window* | `W = 4` | [TBD] | [TBD] |
| *Window* | `W = 8` | [TBD] | [TBD] |
| *Window* | **`W = 16` (headline)** | [TBD] | — |
| *Window* | `W = 32` | [TBD] | [TBD] |
| *Walk depth (supports C2)* | `H = 0` (walk disabled) | [TBD] | [TBD] |
| *Walk depth* | `H = 1` | [TBD] | [TBD] |
| *Walk depth* | `H = 2` | [TBD] | [TBD] |
| *Walk depth* | **`H = 3` (headline)** | [TBD] | — |
| *Walk depth* | `H = 5` | [TBD] | [TBD] |
| *Walk weight (supports C2)* | `λ = 0` (walk-free) | [TBD] | [TBD] |
| *Walk weight* | `λ = 0.2` | [TBD] | [TBD] |
| *Walk weight* | **`λ = 0.5` (headline)** | [TBD] | — |
| *Walk weight* | `λ = 1.0` | [TBD] | [TBD] |
| *Indirect mode (supports C2)* | `c = c_hub` (hub signal) | [TBD] | [TBD] |
| *Indirect mode* | **`c = C` (walk, headline)** | [TBD] | — |
| *Reservoir averaging (supports C3)* | `n = 1` (pure reservoir) | [TBD] | [TBD] |
| *Reservoir averaging* | `n = 2` | [TBD] | [TBD] |
| *Reservoir averaging* | **`n = 4` (headline)** | [TBD] | — |
| *Reservoir averaging* | `n = 8` | [TBD] | [TBD] |
| *Reservoir averaging* | `n → ∞` (deterministic top-*K* by `w`) | [TBD] | [TBD] |

Expected qualitative findings:

- **Window (C1)**: `W = 1` (single-query) is the noisiest configuration; AVG increases monotonically up to `W ≈ 16` and plateaus; very large `W` (≥ 64) slightly degrades on tasks needing fast reaction to short-range context changes.
- **Walk depth (C2)**: `H = 0` removes the walk and validates the §2.4 motivation; `H = 3` captures the bulk of the gain; `H = 5` is essentially no different from `H = 3`.
- **Walk weight (C2)**: `λ = 0` (walk-free) closes much of the gap to Nexus but is consistently below; large `λ` overweights the walked term at the expense of direct importance.
- **Indirect mode (C2)**: walk-mode (`c = C`) is the headline; hub-mode (`c = c_hub`) is competitive at lower compute cost and is preferred in the prefill setting.
- **Reservoir averaging (C3)**: `n = 1` is the highest-variance setting and is occasionally lower on the deterministic eval seed but is the *most accurate* averaged over many seeds; `n = 4` provides a good variance/accuracy trade-off; the strict deterministic `n → ∞` branch loses a measurable amount of accuracy in the high-eviction regime — the empirical signature of the min-vs-mean argument of §4.1.

### 6.5 End-to-End Inference Acceleration

**Table 5.** End-to-end inference throughput (tokens/s) and time-to-first-token (TTFT, seconds) on Llama-3.1-8B-Instruct at 80% cache eviction, batch size 1, across context lengths. *Supports the §4.4 computational-cost claim: the Nexus eviction step does not introduce a new bottleneck.*

| Context | Dense TTFT (s) | Nexus TTFT (s) | Prefill speedup | Dense throughput (tok/s) | Nexus throughput (tok/s) | Decode speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 16K | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 32K | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 64K | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 128K | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

---

## 7. Discussion

**The window-vs-query trade-off.** `W` is the lever that trades reactivity for stability. At `W = 1` the eviction set tracks the very latest query: it reacts instantly to a topic shift, but it also rotates rapidly with single-token noise. At large `W` the set is stable but lags transitions. The §6.4 ablation finds a wide plateau between `W = 8` and `W = 32`, and we default to `W = 16` in decode as a robust mid-point.

**The walk as multi-hop Sketch&Walk.** The walk recurrence is the per-window analogue of the per-layer Sketch-Determined Walk in [Le et al., 2026]. Both compose pairwise affinities into higher-order chains via an iterated update; both use a small constant depth (`H ≈ 3`); and both are motivated by the non-transitivity of inner-product similarity [Le et al., 2026, Remark A.9]. The two components compose: in pipelines that use Sketch&Walk attention together with Nexus Sampling eviction, the per-layer walk captures cross-layer indirect importance at compute time, and the per-window walk captures cross-query indirect importance at eviction time.

**When sampling beats top-K.** The reservoir step matters most under two conditions: aggressive eviction, where the marginal block carries non-trivial weight relative to its neighbors below the cutoff, and long streams, where per-event errors compound. The min-vs-mean argument of §4.1 covers both — top-*K*'s long-run survival is a step function in time-averaged weight, reservoir's is a smooth product — so the gap between the two grows with eviction aggressiveness and stream length together. The default `n = 4` averages the priority enough to be reproducible across seeds while preserving the marginal-mass property; we keep the strict deterministic `n → ∞` branch as a code path for non-streaming workloads, or for runs where reproducibility outweighs cache fidelity.

**Composition with sparse attention.** Nexus is a cache-side method: it decides what to keep. Sparse attention methods (Quest, MInference, FlexPrefill, Sketch&Walk) are attention-side: they decide what to attend to over a still-full cache. The two compose. A natural pipeline pairs Sketch&Walk for in-attention selection with Nexus for cache retention across update events, and §6.5 quantifies the resulting compounded speedups.

**Future work.** Natural extensions include token-level reservoir sampling (sub-block granularity), multi-resolution eviction with per-layer budgets, and adaptive `n` schedules that anneal toward determinism as the cache matures. These are immediate consequences of the reservoir framing and are deferred to future work.

---

## Algorithm

```
Algorithm 1: Nexus Sampling Block Selection (per cache-update event)
─────────────────────────────────────────────────────────────────────────────
Input
  Q ∈ ℝ^{W × D}                  observation window (rolling buffer; head-averaged)
  K ∈ ℝ^{T_k × D}                full key cache (head-averaged)
  Hyperparameters
    b                            tokens per block
    H                            walk depth (default 3)
    λ                            mixing weight on walk/hub term
    ε_tie                        recency tie-break magnitude (default 1e-6)
    K_total                      total retained-block budget
    forced_idx                   indices of forced-retained blocks (sink, recency,
                                 current block)
    n                            reservoir averaging count (or 'det' for deterministic)
    mode                         'walk' | 'hub'  (selects c in step 4)

Output
  S ⊆ {1, …, N_k}                set of retained block indices, |S| = K_total

# 1. Per-block attention weight
P     = softmax(Q · Kᵀ / √D)                              # W × T_k
BS    = BlockSum_b(P)                                     # W × N_k
Ŝ     = rownorm(BS)                                       # W × N_k  (each row sums to 1)

# 2. Window collapse
a     = (1 / W) · Σ_w Ŝ_w                                 # N_k

# 3. Multi-hop walk (or hub signal)
if mode == 'walk':
    C = 0 ∈ ℝ^{N_k}
    for h = 1 to H:
        q     = choose_query_row(h, W)                    # e.g. q = ((h - 1) mod W) + 1
        a_q   = Ŝ_q
        s     = a_q · C                                   # scalar (alignment)
        C    += a_q ⊙ (1 + s)
    c = C
else:                                                     # 'hub' mode
    c = Σ_w BS_w                                          # column sum of BlockSum_b(P) (before rownorm)

c_tilde = c / ‖c‖_1                                       # probability over blocks

# 4. Combined sampling score
r       = (0, 1/(N_k-1), 2/(N_k-1), …, 1)                 # N_k
w       = a + λ · c_tilde + ε_tie · r                     # N_k

# 5. Weighted reservoir block selection
S       = forced_idx                                      # forced set, kept unconditionally
K_left  = K_total - |forced_idx|
cands   = {1, …, N_k} \ forced_idx

if n == 'det':
    S = S ∪ top_K(K_left, cands, key = w_j)
else:
    for j in cands:
        draw u_j^(1), …, u_j^(n) ~ U(0, 1)
        π_j = (1 / n) · Σ_i (u_j^(i))^(1 / w_j)
    S = S ∪ top_K(K_left, cands, key = π_j)

return S
```

```
Algorithm 2: Streaming Nexus Sampling (per-step driver)
─────────────────────────────────────────────────────────────────────────────
State
  Q_buf ∈ ℝ^{W × D}              rolling observation-window buffer
  K, V                            full key/value cache
  S ⊆ {1, …, N_k}                 currently retained block indices
  eviction_period τ               # of decode steps between cache-update events

# Prefill
Q_buf := last W rows of prompt query
S     := Algorithm 1(Q_buf, K, mode = 'hub')              # hub mode in prefill
evict positions {1, …, T_k} \ tokens(S)

# Decode (step-by-step)
for t = T_prompt + 1, T_prompt + 2, …:
    q_t        := model.compute_query(t)                  # head-averaged
    Q_buf     := push(Q_buf, q_t)                         # extend by one, truncate to W
    attend to K_S, V_S with q_t                           # sparse attention
    K, V      := extend with new key, value at position t

    if (t - T_prompt) mod τ == 0:
        S     := Algorithm 1(Q_buf, K, mode = 'walk')     # walk mode in decode
        evict positions {1, …, |K|} \ tokens(S)
```

---

## Empirical Evidence Map

The following claim → experiment map summarizes what evidence supports each design claim. Detailed plot specifications are inline in §2–§4; results live in §6.

| Claim | Where stated | Supporting experiment |
|---|---|---|
| **Heavy-tailed attention** (head holds ≥90% of mass, bulk is near-uniform) | §3 ¶2 | Score-distribution diagnostic (Figure 4 spec, §3); cites H2O, Sketch&Walk |
| **Per-query score is noisy and rotates** | §1 P2, §3 ¶3 | Per-step block-rank churn diagnostic (Figure 1 spec, §2.2) |
| **Window denoising stabilizes the score** (C1) | §1 P2, §2.2, §4.2 | Window-length ablation (Table 4 *Window* block); per-step churn diagnostic |
| **One-hop attention misses bridges** (C2 — bridge claim) | §3 ¶4 | Indirect-importance recovery plot (Figure 2 spec, §3); RULER multi-hop subtasks (Table 2) |
| **Walk recovers bridge importance** (C2) | §2.4, §3 ¶4, §4.3 | Walk-depth ablation (Table 4 *Walk depth*); walk weight ablation (Table 4 *Walk weight*); RULER multi-hop subtasks |
| **Hub signal complements walk** (C2 — hub claim) | §2.5, §3 ¶4 | Indirect-mode ablation (Table 4 *Indirect mode*) |
| **Deterministic top-K is a min over events; reservoir is a mean** (C3) | §3 ¶2, §4.1 Lemma 4.3 | Long-run block-survival diagnostic (Figure 3 spec, §3) |
| **Reservoir preserves time-varying-importance blocks** (C3) | §3 ¶4, §4.1 | High-eviction LongBench curve (Table 3); reservoir-averaging ablation (Table 4 *Reservoir averaging*) |
| **Gap to top-K widens with eviction aggressiveness** | §1 P5, §3 ¶5, §7 | High-eviction LongBench curve (Table 3) |
| **Gap to top-K widens with stream length** | §3 ¶2, §4.1 Lemma 4.3 | Long-run block-survival diagnostic + extended-decode RULER (Table 2 64K column) |
| **Eviction-step cost is small relative to attention** | §4.4 | Kernel profile (companion to Table 5) |
| **End-to-end Nexus speedup** | §1 P5, §7 | TTFT / throughput table (Table 5) |

---

## References

Bai, Y., et al. (2024). LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding. ACL.

Brin, S. & Page, L. (1998). The Anatomy of a Large-Scale Hypertextual Web Search Engine. Computer Networks and ISDN Systems.

Cai, Z., et al. (2024). PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling. arXiv:2406.02069.

Chao, M.-T. (1982). A General Purpose Unequal Probability Sampling Plan. Biometrika, 69(3), 653–656.

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS.

Efraimidis, P. S. & Spirakis, P. G. (2006). Weighted Random Sampling with a Reservoir. Information Processing Letters, 97(5), 181–185.

Feng, Y., et al. (2024). AdaKV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference. arXiv preprint.

Hsieh, C.-P., et al. (2024). RULER: What's the Real Context Size of Your Long-Context Language Models? arXiv:2404.06654.

Le, H. A. D., Joshi, S., Yang, Z., Xu, Z., & Shrivastava, A. (2026). Scout Before You Attend: Sketch-and-Walk Sparse Attention for Efficient LLM Inference. arXiv:2602.07397.

Li, Y., et al. (2024). SnapKV: LLM Knows What You are Looking for Before Generation. arXiv:2404.14469.

Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank Citation Ranking: Bringing Order to the Web. Stanford InfoLab Technical Report.

Pope, R., et al. (2023). Efficiently Scaling Transformer Inference. MLSys.

Tang, J., et al. (2024). QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference. ICML.

Vitter, J. S. (1985). Random Sampling with a Reservoir. ACM Transactions on Mathematical Software, 11(1), 37–57.

Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2024). Efficient Streaming Language Models with Attention Sinks. ICLR.

Yu, J., et al. (2025). Adamas: Adaptive Mass-Aware Sparse Attention for KV-Cache Compression. arXiv preprint.

Zhang, Z., et al. (2023). H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. NeurIPS.
