# Walk Co-Design Brainstorm: Generation as a New Inference Paradigm

*Unconstrained brainstorm on co-designing loss and rollout around the sparsity structure of reasoning. Emphasis on genuinely novel generation algorithms, not incremental modifications.*

---

## Naming Note: "Walk" vs. "Causal Centrality"

"Walk" names a *computation method* (random walk on attention graph). The underlying concept is broader: **which token positions are central hubs in this sequence's information flow graph** — positions that many other positions depend on. Calling it "walk" biases toward a specific implementation and a specific directionality (future refer-back / hub score).

Better framing: **causal centrality score** (or just **centrality score**) for the concept; the walk mechanism is one way to compute it. Other valid computations:
- Direct column sum of attention matrices (hub score, same thing as walk without the random-walk dynamics)
- Eigenvector centrality of the attention graph
- Value vector norm `‖v_t‖` per layer (how much t "broadcasts" into the residual stream)
- Attention entropy of incoming attention (how concentrated are references to t)

All measure "how central is this token in this sequence's information graph." The walk-based computation gives a specifically *differential* centrality when applied to correct vs. wrong rollouts (δ_w). But centrality from a single sequence — no reward, no prior batches — is just the hub score, and it's computable from any forward pass.

---

## The Core Reframe

The standard RL loop treats rollout and loss as two sequential phases: generate, then compute gradients. This imposes a hidden assumption: rollout is a black box that produces tokens, and the loss acts on those tokens after the fact.

Centrality scores break this assumption. A centrality score is computable from a single forward pass on any sequence — no reward, no prior batch, no comparison to other rollouts needed. It is available from the **current sequence itself**, immediately after generation. Once you accept this, you are no longer constrained to the generate-then-learn paradigm.

The ideas below span a spectrum from easy modifications to fundamentally new generation architectures.

---

## Tier 1: Modified Sampling — Same AR, Different Dynamics

### 1A. Langevin Dynamics at Causal Positions

Standard next-token sampling: draw from `softmax(logits / τ)` — same at every token, regardless of how much that token determines the outcome.

**δ_w-guided Langevin**: use the differential walk signal from W-GRPO (the same signal gating the loss in v11) to identify where to run stochastic gradient ascent in logit space *before* sampling. The key: δ_w is computed from the **current batch's own rollouts**, not from a prior batch.

#### During training — two phases within one step

```python
# Phase 1: standard rollout + δ_w computation
rollouts_std = generate(prompts, n=8)          # standard AR
rewards_std  = compute_reward(rollouts_std)
walk_std     = compute_walk_scores(rollouts_std)

# δ_w from THIS batch — reward-contrastive, prompt-specific
delta_w = compute_differential_walk(walk_std, rewards_std)
# delta_w[t] = ReLU(mean_correct_walk[t] - mean_wrong_walk[t])

# Phase 2: Langevin rollouts guided by this batch's δ_w
rollouts_lang = []
for prompt in prompts:
    y = []
    for t in range(T):
        logits_t = model.forward(prompt + y)
        if delta_w[t] > threshold:               # δ_w identifies causal position
            for k in range(K):                   # K = 3–5 Langevin steps
                logits_t += η * ∇_{logits}[delta_w_score(logits_t)]
                logits_t += σ * torch.randn_like(logits_t)
        y.append(sample(logits_t))
    rollouts_lang.append(y)

# Loss: all rollouts, same δ_w gates both Langevin AND the advantage weights
rewards_lang = compute_reward(rollouts_lang)
adv = grpo_advantages(rollouts_std + rollouts_lang, rewards_std + rewards_lang)
loss = policy_gradient_loss(rollouts_std + rollouts_lang, adv * delta_w)  # v11 gate
```

**The co-design closure**: the same δ_w signal flows in both directions simultaneously:
- **Loss ← δ_w**: gates gradient to causally important tokens (v11: `Â_t = A_i · δ_w[t]`)
- **Rollout ← δ_w**: triggers Langevin at causally important positions (Phase 2)
- **δ_w ← rollout**: sharper correct/wrong pairs from Langevin rollouts → sharper δ_w next step

This is the W-GRPO loop, not an addition to it.

#### At inference — the fundamental challenge and three solutions

δ_w requires a contrast (correct vs wrong). At inference, there is no ground-truth reward and typically only one generation. This is a real asymmetry, not a detail to paper over. Three solutions, in order of strength:

**The unifying principle**: δ_w always needs two reference points. At training, the contrast axis is reward. At inference, you need a substitute axis. The options differ only in what that axis is and what it costs.

| Contrast axis | Differential signal | Strength | Cost |
|---|---|---|---|
| Reward (training) | correct walk − wrong walk | Strongest | Free (already computed) |
| Self-consistency proxy | majority walk − minority walk | Strong | k=4 drafts |
| Temperature | greedy walk − stochastic walk | Moderate | k=2 drafts |
| Stored EMA | learned from training steps | Prompt-agnostic | Free (single pass) |

**Option A — self-consistency contrast (k-draft, closest to training):**
```
1. Generate k=4 drafts with standard sampling
2. Score by self-consistency:
     math   → do final answers agree? majority = "correct", minority = "wrong"
     reason → are intermediate steps internally consistent?
3. δ_w_proxy[t] = ReLU(mean_walk_majority[t] − mean_walk_minority[t])
4. Generate 1–2 final rollouts with Langevin at δ_w_proxy positions
5. Return best among all k+2 rollouts
```
This IS an inference algorithm. It subsumes best-of-k (already generating k drafts) and adds walk-guided Langevin exploration on top. Training and inference become structurally the same algorithm — W-GRPO with true reward at training time, W-GRPO with proxy reward at inference time. The test-time compute budget (k drafts + Langevin passes) directly trades off against answer quality.

**Option B — temperature contrast (reward-free, k=2 drafts):**
```
Draft A: greedy (τ ≈ 0)  — model's "committed" path
Draft B: stochastic (τ = 1.0) — model's exploratory path
δ_w_temp[t] = ReLU(walk_greedy[t] − walk_stochastic[t])
```
Positions where the committed path attends heavily but the stochastic path does not → positions where the model has strong structural dependencies in confident mode. Apply Langevin in a third pass. No reward, no self-consistency check, fully self-contained. Weaker contrast than Option A but requires only 2 drafts.

**Option C — stored δ_w EMA (single-pass, learned prior):**
```
# During training:
δ_w_ema[t] ← β · δ_w_ema[t] + (1−β) · δ_w_step[t]

# At inference:
apply Langevin at positions where δ_w_ema[t] > threshold
```
No drafts, no proxy reward. Uses the map learned during training of "where correct and wrong reasoning diverge for this problem type." Cheapest, but prompt-agnostic — the map does not adapt to the specific new prompt's causal structure.

#### What are we sampling from?

This is not just noise injection — it is sampling from a **tilted distribution**:
```
P_tilted(y_t | prefix) ∝ P(y_t | prefix) · exp(λ · δ_w[t])
```
Langevin is the correct sampler: the gradient pushes toward tokens that produce high differential walk structure (toward the token distribution of correct rollouts at this position); the noise prevents mode collapse. Temperature scaling cannot do this — temperature changes the sharpness of `P(y_t | prefix)` but cannot tilt the distribution toward δ_w.

#### Gradient options for the Langevin step

1. **δ_w gradient (primary)**: ∂(δ_w[t])/∂(logits_t) — ascending toward logit distributions that produce high differential walk at position t, i.e., toward the token distribution of correct rollouts. Requires backprop through the walk computation.

2. **Attention entropy gradient (single-pass fallback)**: ∂(−H(α_t))/∂(logits_t). No δ_w needed; backpropagable through the current forward pass only. Low entropy correlates with high hub score, which correlates with δ_w at causal positions.

3. **Zero-order lookahead**: sample k=5 candidate tokens, one extra forward step each, observe δ_w proxy at next position. Finite-difference. Accurate but costs 5 forward-pass fragments at causal positions.

#### Adaptive compute and training-inference symmetry

- Non-causal tokens (~85%): one forward pass, one sample — zero overhead
- Causal tokens (~15%): K Langevin steps

Same algorithm, same δ_w signal, two hyperparameter regimes:
- **Training**: high σ → exploration of the causal manifold → diverse pairs → sharper δ_w next step
- **Inference**: low σ, high η → deliberation before committing to each causal token

No train-test mismatch from the sampling algorithm itself.

#### Connection to diffusion language models

Diffusion LMs do iterative denoising over the entire sequence — but require full retraining. δ_w-guided Langevin is a surgical hybrid: standard AR for ~85% of tokens, local iterative refinement for ~15%, no architectural change. Diffusion's "spend more compute at generation time" property, applied only where δ_w says reasoning diverges.

#### Adaptive compute allocation

This is inference-time compute scaling at the right positions:
- Non-causal tokens (85%): single forward pass, single sample — zero overhead
- Causal tokens (15%): K additional gradient steps

With K=3–5 and 15% causal fraction, the overhead is ≈0.15 × K × (attention backward cost) per token — small in absolute terms, but spend entirely on positions where the model is making genuine reasoning decisions. Beam search spreads extra compute uniformly; CoT/o1-style thinking generates extra tokens; walk-Langevin allocates compute sub-tokenwise, at the right positions, without growing the context.

#### Training-inference symmetry

Same algorithm, same walk signal, two hyperparameter regimes:
- **Training rollout**: high noise (σ large) → exploration of the causal manifold → diverse correct/wrong pairs → sharper δ_w
- **Inference**: low noise (σ → 0), high gradient (η large) → exploitation → the model "deliberates" before committing to each causal token

No train-test distribution shift from the sampling algorithm. Best-of-N introduces mismatch (trains on single samples, infers with max over N). Chain-of-thought introduces mismatch if CoT is used only at inference. Walk-Langevin uses the same process both ways.

#### Connection to diffusion language models

Diffusion LMs (MDLM, D3PM, etc.) do iterative denoising over the entire sequence simultaneously. This requires retraining from scratch as a non-AR model.

Walk-Langevin is a surgical hybrid: standard AR for 85% of tokens, local iterative refinement for 15%. The diffusion model's "spend more compute at generation time" property, applied only at walk-identified positions, with no architectural change. It is diffusion embedded inside AR, activated by the walk.

---

### 1B. On-the-Fly Fork Detection via Attention Entropy

WGCR pre-computes the walk from prior batches to identify fork positions. But a walk proxy is available **during** generation at no extra cost.

As the model generates token t, attention weights `α_{t,i}` are already computed. Define:

```
causal_signal(t) = exp(−H(α_t)) = exp(Σ_i α_{t,i} log α_{t,i})
```

Low-entropy attention (`causal_signal` high): model attends to a few key positions — it's in decision mode. This is a walk-identified causal moment, detected in real time.

At detected causal moments, trigger any of:
- **Branch**: spawn m=2 parallel generations from this position (cheap since prefix is shared)
- **Langevin**: apply noise injection (idea 1A above)
- **Verify**: quick rollout simulation to check if this token choice leads to valid completion

**Why valuable**: fork detection is free (attention weights already computed during the forward pass). No pre-computed walk. No lookahead. The model's own attention geometry is the oracle. This enables WGCR-style rollout generation with zero additional compute for the detection step.

---

### 1C. Position-Adaptive Temperature with Live Walk Feedback

Walk-adaptive temperature:

```
τ_t = τ_base · exp(−α · causal_signal(t))
```

High causal signal → low temperature → sharp sampling at decision tokens.  
Low causal signal → high temperature → free exploration at filler tokens.

Inverted variant for training rollouts: high causal signal → HIGH temperature. You want diverse samples at positions that determine the outcome. During training: explore at causal positions. During inference: exploit at causal positions. Same signal, two modes.

---

## Tier 2: Non-Sequential Generation — Causal Skeleton First

### 2A. Causal-First Generation (CFG)

The walk identifies ~10–20% of token positions as causally important. These positions determine whether the solution is correct. The other 80–90% are elaboration and connective tissue.

What if we generate tokens in a different order?

**Algorithm**:
```
1. Predict likely causal positions P̂ from walk statistics on prior batches
   P̂ = top-k positions by E[δ_w[t]] over recent training steps

2. Generate causal tokens first — with more compute (beam, or multiple samples):
   y_{P̂_1}, y_{P̂_2}, ..., y_{P̂_k}

3. Condition on the causal skeleton, fill in non-causal positions:
   y_{non-causal} = fast_fill(prompt, causal_skeleton)
   (greedy AR, or a smaller/faster model)

4. Assemble final sequence preserving original token order
```

**Why it works**: if you get the causal tokens right, the filler follows almost deterministically. Diversity in filler tokens has near-zero effect on the final reward. All exploration budget concentrates on the ~15% of positions that decide success or failure.

**As inference algorithm**: this is a new decoding algorithm. At test time, generate the mathematical reasoning skeleton (key transformations, pivotal equations, deductions) first, then fill in the prose. The model's attention geometry identifies skeleton vs. filler — no human annotation, no separate model.

**Connection to planning**: CFG is hierarchical planning learned from attention patterns. Plan the key reasoning steps (causal tokens), then elaborate (filler tokens). The walk is an automatic planner.

---

### 2B. Causal Kernel Synthesis — Reasoning in Compressed Space

More extreme: identify the causal tokens of a correct solution to get a ~40-token "causal kernel" from an 800-token solution. Train in this compressed space.

```
Causal kernel: c* = y*[δ_w > threshold]   # ~40 tokens

Generation in kernel space:
  c_1, c_2, ..., c_n ~ π_kernel(c | prompt)   # n=50 is now cheap

Expansion back to full solution:
  y_i = expand(c_i, prompt)   # conditioned generation; near-deterministic
```

The RL training operates in kernel space — ~10× fewer tokens, ~10× cleaner credit assignment. The model learns two things simultaneously: how to synthesize the causal kernel (the key reasoning steps), and how to expand the kernel into a readable solution.

This is a new **two-stage architecture**: a Planner (generates causal kernel tokens) and a Writer (expands to full solution). The walk provides the supervision signal separating planner tokens from writer tokens — automatically discovered from attention geometry, without human labeling.

---

## Tier 3: Latent Space Generation — Beyond Token Perturbation

### 3A. Residual Stream Injection at Walk Positions

Token-level perturbation operates on the output distribution. Walk-level perturbation operates on the **internal computation** at the positions that matter.

At walk-identified layer-position pairs (l*, t*), inject perturbations directly into the residual stream:

```python
# During generation at token t* where δ_w[t*] > threshold:
# v* = direction in residual space most correlated with correct vs wrong rollouts
h_layer[l*][t*] += noise_scale * v*_perturbed
```

**Principled variant**: inject the **residual difference** between correct and wrong rollouts:
```
h^{l*}_{t*} += α · (mean_{correct}(h^{l*}_{t*}) − mean_{wrong}(h^{l*}_{t*}))
```

This is activation steering guided by the walk. The model's internal state at the critical position is being nudged toward the correct-trajectory manifold.

**Why this differs from logit perturbation**: there are cases where correct and wrong reasoning chains diverge in internal representations at step t*, but produce identical output tokens. The divergence is internal, not yet reflected in the next-token distribution. Token-level diversity misses this entirely. Residual stream injection can create genuine internal diversity at the right positions.

---

### 3B. Walk-Guided Iterative Refinement (Gibbs Sampling over Causal Tokens)

Treat generation as an iterative refinement process, not a single left-to-right pass.

```python
def masked_causal_refinement(prompt, n_steps=5):
    y = generate(prompt)                        # initial AR draft

    for step in range(n_steps):
        δ_w = compute_walk_scores(y)
        P_causal = top_k(δ_w, k=int(0.15 * len(y)))  # top 15% causal positions

        # Mask only causal positions and resample (Gibbs step)
        y_masked = mask(y, P_causal)
        y_new = infill(prompt, y_masked)        # infilling model fills causal positions

        if reward(y_new) >= reward(y):
            y = y_new                           # accept (Metropolis-Hastings style)

    return y
```

This is coordinate-descent / Gibbs sampling over the space of reasoning traces. The coordinates being sampled are the causally important tokens; the walk is the coordinate selector; infilling provides the proposal distribution.

**Why it finds better solutions**: in AR generation, an error at token t cascades — the model cannot revise its earlier decision. Masked causal refinement breaks this: the model gets multiple chances to fix individual causal tokens while holding others fixed. Common errors (arithmetic mistakes, wrong equation setups) can be corrected without regenerating the entire solution.

**As training rollout**: run MCR for each prompt; the final refined trajectory has much higher P(correct) than the initial draft. The loss operates on the refined trajectory. The model trains on better examples than it could produce directly — a form of self-improvement without an external oracle.

**As test-time inference**: the same algorithm works at deployment. Generate, refine at causal positions, generate, refine. The model allocates extra compute to positions where its attention says reasoning is happening.

---

## Tier 4: Rollout as a New Inference Algorithm

### 4A. Walk-Consistent Beam Search

Standard beam search scores beams by `log P(y_{1:t} | x)`.

Walk-consistent beam search adds a new scoring term:
```
score(y_{1:t}) = log P(y_{1:t} | x) + λ · walk_consistency(y_{1:t})

walk_consistency(y_{1:t}) = cosine_sim(
    δ_w_computed(y_{1:t}),             # walk profile of current beam
    δ_w_target                          # expected walk profile of correct solutions
)
```

Beams developing attention patterns consistent with correct solutions are scored higher — even when their token probability is equal. This breaks the `log P` tyranny in beam search by introducing an internal geometry scoring signal.

**At inference**: beam search is guided by "does this sequence look like it's following the causal structure of correct solutions?" rather than just "is this a probable continuation?". No external judge. No trained verifier. The model's own attention history is the quality signal.

---

### 4B. Walk-Guided Speculative Decoding

Standard speculative decoding: small draft model proposes k tokens; base model verifies all k in one parallel forward pass.

Walk-guided speculative decoding:

```
1. Draft model proposes tokens at ALL positions (fast, cheap)
2. Compute walk on draft to identify causal positions P*
3. Base model re-evaluates ONLY positions in P* (expensive compute only where it matters)
4. Accept non-causal positions from draft unconditionally (trust the draft)
5. At causal positions, use base model's distribution (never trust the draft)
```

**This is not just an efficiency trick**: it changes the generation. Non-causal tokens are always "fast-path" tokens from the draft. Causal tokens always get full base-model reasoning depth. The compute budget is allocated by the walk's sparsity map.

**Budget analysis**: if 15% of tokens are causal, base model is 10× cost of draft:
- Standard base model: 10x
- Walk-guided speculative: 0.15×10 + 0.85×1 ≈ 2.35x
- Full-quality at causal positions (unlike standard speculative which can accept wrong causal tokens)

---

### 4C. Walk as Value Function for Self-Play (No External Reward Model)

The walk IS a value function — no training needed.

In correct solutions, δ_w shows which tokens were causally important and how the attention geometry looks when reasoning is on track. In wrong solutions, the same positions look structurally different.

Replace binary outcome reward with a **continuous, dense, walk-based reward**:

```
r_walk(y) = −KL(δ_w(y) || δ_w_target)

where δ_w_target = E[δ_w(y*)] over prior correct solutions on similar prompts
```

This is a dense per-token reward: does each token's attention contribution look like a correct-solution token at this position?

**Self-reinforcing loop**: as the model learns reasoning with walk profiles matching correct solutions, it produces more correct solutions, which provides cleaner δ_w targets, which refines the reward signal. This is a self-supervised RL loop with no reward model, no PRM, no oracle.

**Relationship to PRIME**: PRIME trains an implicit PRM online from outcome labels to get dense token rewards. Walk-based reward achieves the same (dense token-level signal) from attention geometry alone — no separate reward model training, no separate forward passes.

---

## Tier 5: Cross-Rollout Learning — Causal Structure Transfer

### 5A. Causal Token Exchange

Given a correct rollout y* and a wrong rollout y⁻, the walk identifies the causally divergent positions. Create a hybrid:

```
y_hybrid = y⁻ with causal positions replaced by those from y*
```

If the hybrid is correct (often it is — only causal tokens were swapped), this directly shows that the specific causal tokens from y* are what determined success. Credit assignment is unambiguous: the swapped tokens are the ones that mattered.

**As rollout algorithm**: maintain a buffer of high-reward solutions. For each new prompt, at walk-identified causal positions, soft-mix the model distribution with the buffer's causal tokens:

```
P(y_t | prefix) = (1−β) · π_θ(y_t | prefix) + β · δ(y_t = y*_t)
```

The generated rollout is "seeded" with correct causal tokens from prior solutions. This is principled knowledge transfer: not KL regularization toward the old policy, but targeted injection of proven correct reasoning at positions where reasoning matters.

---

### 5B. Walk-Guided MCTS Without a Value Network

Use δ_w as a step-level value proxy for MCTS-style rollout expansion — no trained value network needed.

Standard MCTS needs a value function to estimate `V(state)`. Walk-guided MCTS uses:

```
V(prefix_{1:t}) ≈ walk_alignment(prefix_{1:t})
               = how much the prefix's walk profile matches δ_w_target
```

MCTS expansion preferentially branches at positions where the walk profile diverges from the correct-solution profile — these are the positions where the model's reasoning most needs to be corrected.

The self-reinforcing property: as training progresses, δ_w sharpens (correct/wrong attention patterns diverge more cleanly), making walk_alignment a more precise value estimate, making MCTS expansions more targeted.

---

## The Unified Picture

All these ideas use the same underlying structure:

```
Walk identifies:
  WHERE  — causal positions (sparse set, ~15% of tokens)
  WHAT   — causal direction (residual stream geometry at those positions)
  WHEN   — causal moments (detectable on-the-fly via attention entropy)

These three signals unlock:

  LOSS:         gradient only at WHERE
  TRAINING ROLLOUT: diversity at WHERE (Langevin, forking, Gibbs refinement)
  INFERENCE:    deep compute at WHERE, shallow at non-causal (speculative decoding)
  ARCHITECTURE: plan causal skeleton → fill non-causal (two-stage CFG)
  VALUE SIGNAL: reward based on walk-profile matching (no reward model)
  REFINEMENT:   Gibbs sampling over causal coordinates only
```

The walk is not a metric applied to a fixed generation process. It is an internal geometry of reasoning that the model already computes — and which can be made explicit to design every aspect of training, inference, and architecture.

---

## Priority Matrix

| Idea | Novelty | Impact | Implementation cost | Try first? |
|---|---|---|---|---|
| On-the-fly attention entropy fork detection (1B) | High | High | Low | Yes |
| Langevin dynamics at δ_w positions (1A) | High | Medium | Low | Yes |
| Walk-consistent beam search (4A) | Medium | High | Low | Yes |
| Causal token exchange / seeded rollout (5A) | Medium | High | Medium | Yes |
| Masked causal refinement / Gibbs (3B) | High | High | Medium | Soon |
| Walk-guided speculative decoding (4B) | High | Medium | Medium | Soon |
| Causal-first generation (2A) | Very High | Very High | Hard | Later |
| Residual stream injection (3A) | Very High | Unknown | Hard | Research |
| Walk as value function / self-play (4C) | Very High | Very High | Medium | Later |
| Causal kernel synthesis (2B) | Very High | Very High | Very Hard | Long-term |
