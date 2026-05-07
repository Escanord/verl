# Spectral Differential Walk (future direction)

## Core idea

For each prompt group in GRPO (n=8 rollouts, some correct/wrong), we already compute
`R_avg` (layer-averaged walk matrix, shape `num_blocks × num_blocks`) per rollout.

Define the **differential walk matrix**:

```
ΔR = mean(R_avg | correct rollouts) - mean(R_avg | wrong rollouts)
```

`ΔR[i,j]` = "when block i queries, how much more does it reference block j in correct
solutions vs wrong ones for this specific prompt?"

Taking the SVD:

```
ΔR = U Σ V^T
```

- `V[:,r]` (right): which key blocks are differentially referenced under mode r
- `U[:,r]` (left): which query blocks drive that differential referencing
- `σ_r`: how much of the correct-vs-wrong structural variance this mode explains

The column sum used in Option A (`δ[j] = Σ_i ΔR[i,j]`) is a marginal of this — it
collapses all query structure into one number per key block, discarding the directional
information the SVD preserves.

## What the leading singular vectors mean

**`V[:,0]`** (leading right singular vector):
Block j with high `V[j,0]` is more strongly referenced in correct solutions — but
specifically in the pattern jointly described with `U[:,0]`. Not "referenced more overall"
(that's the column sum), but "referenced more by the specific query pattern that most
distinguishes correct from wrong."

**`U[:,0]`** (leading left singular vector):
The query blocks that do the differential referencing. These are the blocks that "know
what to attend to" in correct reasoning but don't in wrong reasoning.

**The rank-1 outer product `σ_0 · U[:,0] V[:,0]^T`**: the single most important
computational pathway separating correct from wrong reasoning for this prompt.

## Multiple reasoning modes (rank-k)

```
ΔR ≈ Σ_{r=0}^{k-1} σ_r U[:,r] V[:,r]^T
```

Each rank-1 component is an independent "reasoning mode." For a math problem:

- **Mode 0** (σ_0 large): "correct solutions reference the problem constraint block
  from the equation-manipulation block" — the deduction the wrong solutions miss.
- **Mode 1** (σ_1 smaller): "correct solutions show self-referential loops between
  consecutive computation steps" — chain-of-thought integrity signal.
- **Mode 2+**: noise or artifacts.

SVD discovers this hierarchy without us labeling what matters.

## Per-rollout alignment score

```python
delta_R1 = sigma_0 * torch.outer(U[:,0], V[:,0])  # rank-1 approximation
alignment_k = (R_avg_k * delta_R1).sum()
```

Scalar per rollout: "how much does this rollout's walk structure look like the winning
structural pattern?"

- **Correct + high alignment**: model did the right structural thing and got reward.
- **Wrong + high alignment**: near-miss — correct structure but failed (arithmetic error?).
  Most informative training signal.
- **Wrong + low/negative alignment**: structurally wrong — the bad habit to unlearn.
- **Correct + low alignment**: got reward despite bad structure — lucky.

## Three integration levels for training

### Level 1: Drop-in replacement for token weights

Replace the absolute hub score with `V[:,0]` from ΔR as the shared token weight for
all rollouts in a group:

```python
R_mats = [compute_R_avg(layer_qk_k) for k in group]  # (n, num_blocks, num_blocks)
correct_mask = rewards > 0.5
delta_R = R_mats[correct_mask].mean(0) - R_mats[~correct_mask].mean(0)
_, _, Vt = torch.linalg.svd(delta_R)      # SVD of 128×128, microseconds
v0 = F.relu(Vt[0])                        # (num_blocks,) positive part
wi_k = expand_blocks_to_tokens(v0, ...)   # same weight for all k in group
```

Same advantage formula: `Â_final = Â_GRPO × (1 + α × w̃)`. Falls back to plain GRPO
when all rollouts correct or all wrong (ΔR ≈ 0 → w̃ → 0 → weight = 1).

### Level 2: Asymmetric weighting (more nuanced credit assignment)

| Rollout | Token aligns with V[:,0] | Meaning | Action |
|---------|--------------------------|---------|--------|
| Correct | yes | right structure + right reward | **strongly reinforce** |
| Correct | no  | lucky — wrong structure, right reward | weakly reinforce |
| Wrong   | yes | near-miss — right structure, wrong reward | weakly penalize |
| Wrong   | no  | bad structure + wrong reward | **strongly penalize** |

```python
for k in group:
    if rewards[k] > 0.5:
        directional_score = F.relu(v0)    # amplify correct-direction tokens
    else:
        directional_score = F.relu(-v0)   # amplify anti-correct-direction tokens
    wi_k = expand_blocks_to_tokens(directional_score, ...)
    A_final_k = A_GRPO_k * (1 + alpha * wi_k)
```

For wrong rollouts, amplifying anti-aligned tokens makes the negative GRPO gradient
larger at exactly the tokens where reasoning structure diverged — more precise unlearning.

### Level 3: Spectral fork for targeted rollout generation

```
Phase 1: Generate n/2 = 4 rollouts (standard)
         Compute R_avg per rollout
         Compute ΔR, SVD
         (i*, j*) = argmax |U[i,0]| × |V[j,0]|  ← spectral bottleneck
         fork_pos = i* × block_size

Phase 2: Generate n/2 = 4 rollouts forked from fork_pos
         Conditioning on: prompt + first fork_pos tokens of a correct Phase 1 rollout

Combine: 8 rollouts for GRPO advantage
```

The fork concentrates additional rollouts at the exact structural decision point.
As training progresses, fork_pos moves earlier in the response (toward the key deduction
step). No external supervision — walk tracks this automatically.

## Implementation requirements

New storage: keep `R_avg` per rollout instead of discarding after importance score.
Size: `n × num_blocks² = 8 × 128² = 131K` floats per sequence ≈ 0.5MB/seq in fp32.

Code changes:
- `verl/utils/walk_importance.py`: expose `R_avg` from `compute()` as a second return value
- `verl/workers/actor/dp_actor.py`: return `R_avg` tensors alongside importance scores
- `verl/trainer/ppo/ray_trainer.py`: group rollouts by prompt, compute ΔR + SVD per group

Level 3 additionally requires touching the vLLM rollout generation loop.

## Relationship to existing literature

- **DPO/RLHF**: compares chosen vs rejected at sequence level. This operates at
  sub-sequence structural level — which computational pathways differ.
- **Credit assignment (critics)**: assigns credit from temporal value estimates. This
  uses attention geometry directly — no learned value function.
- **Contrastive representation learning (SimCLR, CLIP)**: pairs are correct/wrong rollouts
  of same prompt; "representation" is the walk matrix; SVD finds the contrastive direction
  without learned parameters.

## Open questions

- Does `V[:,0]` correlate with human-interpretable reasoning steps (vs formatting)?
  Token logging (already in ray_trainer.py) would reveal this.
- How noisy is `ΔR` when `n_correct` or `n_wrong` is small (e.g., 1 vs 7)?
  May need to require both ≥ 2 before using spectral signal; fall back to absolute hub.
- Does rank-1 SVD suffice or do higher modes carry signal?
  Could measure via `σ_0 / Σ σ_r` (fraction of variance explained by leading mode).
- Level 2 asymmetric weighting: does penalizing anti-aligned wrong tokens help or hurt?
  Near-miss rollouts (wrong reward, correct structure) are ambiguous — suppressing them
  may remove useful gradient or may add noise.
