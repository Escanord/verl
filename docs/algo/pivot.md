# PIVOT — Experiment Tracking

*Pivot-point Identification Via On-the-fly walk Tracking.*

Algorithmic design and motivation: see [`codesign_loss_rollout.md`](codesign_loss_rollout.md).
Origin hypothesis: see W-GRPO ["Mechanistic Hypothesis: Positional Bifurcation"](wgrpo.md#mechanistic-hypothesis-positional-bifurcation).

---

## What PIVOT Does

Two co-designed components driven by the same signal — temporal walk change `||ΔR_t||`:

**Phase 1 — Loss gating**: after rollout, run an extra actor forward pass with Q/K hooks. Compute block-level temporal walk from the hooked attention. Gate GRPO advantages so gradient concentrates at structural change points.

**Phase 2 — Langevin rollout**: patch vLLM attention layers to capture Q/K during decode. Apply Langevin entropy-maximising noise to logits when `||ΔR_t|| > threshold` — pushing the model to explore at structural decision points during generation.

The co-design loop: Langevin explores at fork positions → more structurally diverse rollouts → cleaner contrastive gradient signal at the same positions → policy improves at forks → walk geometry sharpens → more precise trigger → ...

---

## Signal: Temporal Walk Change

At each response block `t`:

```
a_t   = layer-averaged block attention row of block t      (B, num_blocks)
multi = a_t @ C                                            multi-hop via walk state C
score_t = ||multi||_2                                      structural change magnitude
C[t, :] = a_t + multi                                      update walk state
```

Normalised within each response: z-score, ReLU (zero below-average), rescale to `[0,1]`.

**Difference from W-GRPO walk**: W-GRPO accumulates walk across transformer *layers* (depth) for a single forward pass. PIVOT accumulates across generation *time steps* (temporal) — tracking how the causal structure of the reasoning chain evolves token by token.

---

## Hyperparameters

### Phase 1 (loss gating)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pivot_mode` | `soft` | `soft`: `Â_t = A · (1 + alpha · score_t)` / `binary`: `Â_t = A · (score_t > threshold)` |
| `pivot_alpha` | 1.0 | Upweighting coefficient (soft mode). alpha=0 → standard GRPO |
| `pivot_threshold` | 0.3 | Binary mask cutoff (binary mode only) |
| `actor.pivot.block_size` | 32 | Tokens per block for Q/K pooling |
| `actor.pivot.norm_mode` | `relu_max` | Score normalisation: `relu_max` → `[0,1]`; `zscore_absmax` → `[-1,1]` |

### Phase 2 (Langevin rollout)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rollout.pivot.langevin_rollout` | `false` | Enable Phase 2 |
| `rollout.pivot.block_size` | 8 | Tokens per block inside vLLM decode (finer than Phase 1) |
| `rollout.pivot.langevin_threshold` | 0.3 | `||ΔR_t||` trigger threshold |
| `rollout.pivot.langevin_K` | 3 | Langevin steps per trigger |
| `rollout.pivot.langevin_eta` | 0.1 | Entropy gradient step size |
| `rollout.pivot.langevin_sigma` | 0.01 | Noise magnitude |

---

## Implementation

| File | What it does |
|------|-------------|
| `verl/utils/walk_importance.py` | `TemporalWalkComputer` — temporal walk from Q/K hooks; `apply_pivot_loss_weights()` |
| `verl/trainer/ppo/core_algos.py` | `apply_pivot_advantage()` — gates GRPO advantages |
| `verl/trainer/ppo/ray_trainer.py` | Calls `compute_pivot_scores`, applies advantage gating, logs metrics |
| `verl/workers/actor/dp_actor.py` | `compute_pivot_scores()` — extra forward pass with Q/K hooks |
| `verl/workers/config/actor.py` | `pivot: dict` field on `FSDPActorConfig` |
| `verl/workers/config/rollout.py` | `pivot: dict` field on `RolloutConfig` |
| `verl/utils/vllm/pivot_patch.py` | Attention hooks for vLLM decode; `PIVOTRolloutProcessor`; `PIVOTLangevinAdapter` (V1 engine-level logits processor) |
| `verl/workers/rollout/vllm_rollout/utils.py` | `monkey_patch_model()` → `patch_attention_layers_pivot()` |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | Registers `PIVOTLangevinAdapter` on `vllm_config.model_config.logits_processors` before engine creation |
| `examples/grpo_trainer/run_qwen2_5_3b_pivot.sh` | Launch script |

---

## PIVOT-v2

*Designed 2026-04-14. Implementation pending.*

### Motivation

PIVOT-v1's Phase 1 signal (temporal walk) is computed per-rollout independently — it finds positions where attention structure changes within a single rollout, with no reward conditioning. This is structurally similar to W-GRPO v1/v2 (per-rollout walk without differential), which underperformed v6's reward-conditioned differential walk. PIVOT-v2 makes the same upgrade: replace the per-rollout walk with a reward-conditioned representation divergence signal, and generalise Phase 2's trigger accordingly.

### Positional Bifurcation — Clarified

The position in "positional bifurcation" is the **token position index** `t ∈ {1,...,n}` (the RoPE coordinate), not a semantic step or attention-walk block. The claim:

> At certain position indices `t`, correct and wrong rollouts for the same prompt diverge in representation space. These fork positions are a property of the **problem**, shared across all rollouts for that prompt — not a property of individual rollout content.

`δ[t]` is a prompt-specific positional weight vector — a soft mask over the response that identifies trajectory forks for this particular prompt.

### Phase 1 — Loss Gating (Representation Divergence)

**Signal:** Hook the last-layer hidden state `h_t` during the **existing log-prob forward pass** (no extra forward pass). Per prompt group (same prompt, `n` rollouts):

```
v[t]           = mean(h_t | correct rollouts) - mean(h_t | wrong rollouts)
fork_score[t]  = ||v[t]||_2
```

Normalise within each response (relu_max to `[0,1]`) and gate advantages:

```
Â_t = A · (1 + alpha · fork_score[t])
```

**Key differences from v1:**
- No extra forward pass — hooks the existing log-prob pass (saves ~15s/step)
- Reward-conditioned cross-rollout comparison (v6-style differential, in representation space)
- Operates on residual stream, not attention walk

**Degeneracy:** when all rollouts have the same reward (`no_contrast` group), `v[t] = 0` and `fork_score[t] = 0` → falls back to standard GRPO, same as v6.

### Phase 2 — Training Rollout (ΔVar Trigger)

During generation, `n` rollouts for the same prompt run concurrently. At each position `t`, compute the **change in intra-group hidden state variance**:

```
Var[t]  = mean pairwise ||h_t^i - h_t^j||^2   over i,j in prompt group
ΔVar[t] = Var[t] - Var[t-1]
```

Trigger Langevin when `ΔVar[t] > threshold`. Langevin intervention unchanged.

**Why ΔVar not Var:** `Var[t]` accumulates divergence from all positions ≤ `t`. `ΔVar[t]` isolates divergence that happened specifically at position `t` — directly detecting the fork rather than its downstream consequence.

**Why this is position-focused:** the trigger fires when rollouts start diverging from each other at position `t`. This is a property of position `t` in the prompt's response geometry, not of any individual rollout's content. It requires no reward conditioning and is prompt-specific by construction.

**Implementation note:** requires grouped batch bookkeeping in vLLM — tracking which sequences share the same prompt and maintaining running `Var` across decode steps. Replaces the Q/K hooks and walk computation in vLLM entirely.

### Phase 2 — Inference-Time Rollout (Entropy Proxy)

At inference `n=1`, ΔVar is undefined. The trigger becomes:

```
H[P(·|context_t)] > threshold  →  fire Langevin
```

**The co-design hypothesis:** PIVOT-v2 training calibrates the model's intrinsic entropy to be higher at fork positions. The mechanism:

1. Phase 2 forces exploration (Langevin) at fork positions during training
2. The model is repeatedly exposed to diverse outcomes at fork positions
3. The model learns that at `t_fork`, small differences lead to large outcome variation
4. Post-training, the model's own entropy at fork positions is higher and better calibrated — not generic content-driven noise, but genuine structural uncertainty

**The transfer:** training converts `ΔVar[t]` (multi-rollout ground truth, training-only) into calibrated entropy (single-sequence, available at inference). The co-design loop closes: training shapes the model's uncertainty signal so inference can use it reliably.

**Validation required:** measure correlation between per-position entropy and `ΔVar[t]` on held-out prompts at a fixed checkpoint. Compare base model vs GRPO baseline vs PIVOT-v2. If PIVOT-v2 correlation is significantly higher, the co-design argument is confirmed. See [Validation](#validation) below.

### Implementation

| File | What it does |
|------|-------------|
| `verl/utils/walk_importance.py` | `RepresentationDivergenceComputer` — last-layer hook, group v[t], fork score |
| `verl/workers/actor/dp_actor.py` | `compute_fork_scores_v2()` — extra forward pass with last-layer hook, group divergence |
| `verl/workers/fsdp_workers.py` | `compute_fork_scores_v2` dispatch method |
| `verl/trainer/ppo/ray_trainer.py` | `pivot_version=2` branch; `_save_fork_profile()` writes profile after each step |
| `verl/utils/vllm/pivot_patch.py` | `PIVOTv2RolloutProcessor`; `PIVOTv2LangevinAdapter` (fork profile trigger + entropy fallback) |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | Registers v2 adapter when `pivot_version=2` |
| `verl/workers/rollout/vllm_rollout/utils.py` | Skips Q/K attention patching for v2 |
| `examples/grpo_trainer/run_qwen2_5_3b_pivot_v2.sh` | Launch script |

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `algorithm.pivot_version` | `1` | `1` = temporal walk; `2` = representation divergence |
| `algorithm.pivot_mode` | `soft` | Same as v1 |
| `algorithm.pivot_alpha` | `1.0` | Same as v1 |
| `algorithm.pivot_threshold` | `0.3` | Phase 1 gating threshold |
| `algorithm.pivot_fork_profile_path` | `/tmp/pivot_v2_fork_profile.npy` | Shared file for Phase 2 trigger |
| `actor.pivot.norm_mode` | `relu_max` | Fork score normalisation |
| `rollout.pivot.pivot_version` | `1` | Must match `algorithm.pivot_version` |
| `rollout.pivot.langevin_rollout` | `false` | Enable Phase 2 |
| `rollout.pivot.fork_profile_path` | — | Same path as `algorithm.pivot_fork_profile_path` |
| `rollout.pivot.langevin_threshold` | `0.3` | Fork profile trigger threshold |
| `rollout.pivot.entropy_threshold` | `2.0` | Entropy fallback trigger (inference / first step) |
| `rollout.pivot.langevin_K` | `3` | Langevin steps per trigger |
| `rollout.pivot.langevin_eta` | `0.1` | Entropy gradient step size |
| `rollout.pivot.langevin_sigma` | `0.01` | Noise magnitude |

### Comparison to v1

| | PIVOT-v1 | PIVOT-v2 |
|---|---|---|
| Phase 1 signal | Temporal walk (per-rollout, no reward cond.) | Representation divergence (cross-rollout, reward-conditioned) |
| Extra forward pass | Yes (~15s/step) | No (hooks existing log-prob pass) |
| Signal space | Attention walk | Residual stream |
| Phase 2 trigger | Temporal walk in vLLM (Q/K hooks) | ΔVar across concurrent rollouts |
| Inference trigger | Temporal walk (same as training) | Entropy (calibrated by training) |
| Inference co-design | No | Yes — training shapes entropy signal |

### Validation

**Entropy–ΔVar correlation experiment**

At a fixed checkpoint (e.g. step 300), on held-out prompts:
1. Generate `n=8` rollouts → compute `ΔVar[t]` per prompt group → ground truth fork positions
2. Generate `n=1` rollout → record entropy `H[P(·|context_t)]` per position
3. Compute Pearson/Spearman correlation between entropy and `ΔVar[t]` across positions and prompts

Expected ordering: base model < GRPO baseline < PIVOT-v2.

If PIVOT-v2 correlation is significantly higher than GRPO baseline, the entropy calibration hypothesis holds and the co-design argument is confirmed empirically.

### Future Work

**Attention sparsity at fork positions (post-confirmation)**

At fork positions, attention patterns change structurally. Once the entropy calibration is confirmed, this opens a further inference-time optimisation: dynamically adjust attention sparsity based on the entropy signal — denser attention computation at fork positions, sparser elsewhere. This would make fork-position awareness load-bearing for inference efficiency, completing the co-design picture.

---

## Variants

### PIVOT-v4 (mislabeled — walk advantage weighting only)

**What was intended**: group-mean Langevin rollout (`logits = (1-eta)*logits + eta*group_mean + sigma*noise`) triggered by delta_var, no walk, no entropy loss.

**What actually ran**: two bugs made v4 a no-op for the rollout:

1. **Wrong adapter**: `vllm_async_server.py` only routed `pivot_version == 2` to `PIVOTv2LangevinAdapter` (delta_var group-mean). All other versions fell to `PIVOTLangevinAdapter` (v1 Q/K-walk adapter). Fixed: `pivot_version >= 2` now uses v2 adapter.
2. **No attention hooks**: `PIVOTLangevinAdapter` reads `_pivot_decode_state.q_batch` written by `patch_attention_layers_pivot()`. That hook was never called in `vllm_async_server.py`, so `q_batch` stays `None` and every logits call returns unchanged. Complete no-op.

Additionally, `use_pivot=True` with `pivot_version=4` routed to walk-based `_compute_pivot_scores` (`TemporalWalkComputer`) — an unintended leftover from v1 that was never removed.

**What v4 gains actually came from**: walk-based temporal advantage weighting (`Â_t = A_t · (1 + pivot_score_t)`), not Langevin. This is the first clean measurement of walk advantage weighting in isolation.

---

### PIVOT-v4b (true v4)

**Design**: group-mean Langevin rollout in isolation. No walk, no entropy loss. Clean ablation.

- Rollout: `logits = (1-eta)*logits + eta*group_mean + sigma*noise` triggered at delta_var > threshold
- Loss: standard GRPO + KL (`entropy_coeff=0`, no `use_pivot`)
- Fixes applied: `PIVOTv2LangevinAdapter` now correctly loaded for `pivot_version >= 2`

Script: `run_qwen2_5_3b_pivot_v4b_from_grpo200.sh`

---

### PIVOT-v5a (co-designed loss)

**Design**: v4b rollout + trigger-gated entropy loss. No walk.

- Rollout: same group-mean Langevin as v4b
- Loss: entropy bonus weighted by delta_var at trigger positions: `entropy_coeff * agg_loss(entropy, mask=pivot_delta_vars * response_mask)`
- `pivot_delta_vars[t]` = actual delta_var value if group-mean fired at step t, else 0
- `entropy_coeff=0.05` safe because fires on ~1-5% of tokens (vs v3's ~80-90% via PG-gate)

Script: `run_qwen2_5_3b_pivot_v5a_from_grpo200.sh`

**Comparison matrix**:

| | Rollout | Advantage | Entropy loss |
|---|---|---|---|
| v4 (old) | walk (no-op) | walk-based (+) | none |
| v4b | group-mean Langevin ✓ | GRPO | none |
| v5a | group-mean Langevin ✓ | GRPO | trigger-gated ✓ |

---

## Results (Qwen2.5-3B-Instruct, guru-RL-92k, 8×H200)

### MATH500

| ID | Steps | acc@1 | acc@16 (mean) | best@16 | maj@16 | Checkpoint |
|----|-------|-------|---------------|---------|--------|------------|
| GRPO-v0 | 320 | 61.0% | 61.5% | 81.2% | 67.5% | `grpo/qwen2_5_3b/global_step_320` |
| WGRPO-v6 | 345 | ~62.8% | 62.8% | 82.0% | 69.8% | `wgrpo-v6/qwen2_5_3b/global_step_340` |
| PIVOT-v1 | 320 | 61.7% | 61.7% | 81.2% | 67.4% | `pivot/qwen2_5_3b/global_step_320` |
| PIVOT-v2 | 300 | ~59.7% | 59.7% | 81.7% | 67.3% | `pivot-v2-from-grpo200/qwen2_5_3b/global_step_100` |
| PIVOT-v3 | 25 (peak, stopped) | N/A | 46.3% | — | — | `pivot-v3-from-grpo200/qwen2_5_3b/global_step_20` |
| PIVOT-v4 (bugged) | 80 (peak MATH) | ~59.9% | 59.9% | 81.9% | 68.6% | `pivot-v4-from-grpo200/qwen2_5_3b/global_step_80` |
| PIVOT-v4b | 120 (eff 320, same as GRPO-v0) | — | 60.4% | 80.7% | **68.2%** | `pivot-v4b-from-grpo200/qwen2_5_3b/global_step_120` |
| PIVOT-v4b | 205 (peak MATH mean, eff 405) | — | **61.0%** | 80.6% | 67.0% | `pivot-v4b-from-grpo200/qwen2_5_3b/global_step_200` |
| PIVOT-v5a | 115 (in progress) | — | 57.9% | — | — | `pivot-v5a-from-grpo200/qwen2_5_3b/global_step_115` |
| PIVOT-v10a | 85 (peak MATH best@16) | — | 59.5% | **83.7%** | 68.5% | `pivot-v10a-from-grpo200/qwen2_5_3b/global_step_85` |
| PIVOT-v10a | 145 (peak AIME mean@16) | — | 60.1% | 82.0% | 68.6% | `pivot-v10a-from-grpo200/qwen2_5_3b/global_step_145` |
| PIVOT-v10a | 165 (peak MATH mean@16 + maj@16) | — | **61.2%** | 82.9% | **69.9%** | `pivot-v10a-from-grpo200/qwen2_5_3b/global_step_165` |
| PIVOT-v10a | 200 (stopped) | — | 60.5% | 83.2% | 68.6% | `pivot-v10a-from-grpo200/qwen2_5_3b/global_step_200` |
| PIVOT-v10b | 180 (peak mean) | — | **60.4%** | 81.3% | 68.6% | `pivot-v10b-from-grpo200/qwen2_5_3b/global_step_180` |
| PIVOT-v10b | 135 (peak maj) | — | 59.9% | 81.5% | **69.3%** | `pivot-v10b-from-grpo200/qwen2_5_3b/global_step_135` |
| PIVOT-v10b | 155 (peak best) | — | 59.5% | **82.6%** | 67.5% | `pivot-v10b-from-grpo200/qwen2_5_3b/global_step_160` |
| PIVOT-v10b | 200 (crashed) | — | 60.2% | 81.1% | 68.9% | `pivot-v10b-from-grpo200/qwen2_5_3b/global_step_200` |
| PIVOT-v12 | 145 (peak AIME best) | — | 59.5% | 81.6% | 67.7% | `pivot-v12-from-grpo200/qwen2_5_3b/global_step_145` |
| PIVOT-v12 | 190 (peak AIME mean) | — | 59.6% | 82.8% | 67.7% | `pivot-v12-from-grpo200/qwen2_5_3b/global_step_190` |
| PIVOT-v12 | 195 (peak MATH mean) | — | **60.5%** | 82.6% | 68.6% | `pivot-v12-from-grpo200/qwen2_5_3b/global_step_195` |
| PIVOT-v12 | 225 (peak MATH best + maj, stopped) | — | 60.3% | **83.7%** | **68.8%** | `pivot-v12-from-grpo200/qwen2_5_3b/global_step_225` |
| PIVOT-v13 | 220 (peak MATH, stopped) | — | **61.7%** | **83.5%** | **70.2%** | `pivot-v13-from-grpo200/qwen2_5_3b/global_step_220` |
| PIVOT-v15 | 80 (peak MATH mean+maj, stopped at 128) | — | 60.0% | 82.1% | 69.0% | `pivot-v15-from-grpo200/qwen2_5_3b/global_step_80` |
| PIVOT-v16b | 235 (peak MATH best@16) | — | 60.4% | **83.5%** | 68.7% | `pivot-v16b-from-grpo200/qwen2_5_3b/global_step_240` |
| PIVOT-v16b | 240 (peak MATH mean@16, stopped at 255) | — | **61.2%** | 82.1% | 68.8% | `pivot-v16b-from-grpo200/qwen2_5_3b/global_step_240` |
| PIVOT-v13d | 95 (peak MATH mean) | — | 59.8% | 81.5% | 67.9% | `pivot-v13d-from-grpo200/qwen2_5_3b/global_step_95` |
| PIVOT-v13d | 105 (peak MATH maj) | — | 59.3% | 81.1% | **68.3%** | `pivot-v13d-from-grpo200/qwen2_5_3b/global_step_105` |
| PIVOT-v13d | 120 (peak MATH best) | — | 59.4% | **82.9%** | 67.1% | `pivot-v13d-from-grpo200/qwen2_5_3b/global_step_120` |
| PIVOT-v13d | 130 (stopped) | — | 59.4% | 81.4% | 67.6% | `pivot-v13d-from-grpo200/qwen2_5_3b/global_step_130` |
| PIVOT-v14 | 160 (peak MATH best) | — | 59.9% | **83.2%** | 68.8% | `pivot-v14-from-grpo200/qwen2_5_3b/global_step_160` |
| PIVOT-v14 | 190 (peak MATH maj) | — | 60.1% | 83.0% | **69.2%** | `pivot-v14-from-grpo200/qwen2_5_3b/global_step_190` |
| PIVOT-v14 | 215 (peak MATH mean, stopped) | — | **60.3%** | 81.4% | 68.9% | `pivot-v14-from-grpo200/qwen2_5_3b/global_step_215` |

### AIME

| ID | Steps | acc@16 (mean) | best@16 | maj@16 |
|----|-------|---------------|---------|--------|
| GRPO-v0 | 320 | 1.8% | 11.3% | 2.4% |
| WGRPO-v6 | 380 | **4.79%** | **16.9%** | **7.70%** |
| PIVOT-v1 | 320 | 2.2% | 13.2% | 3.3% |
| PIVOT-v2 | 300 | 3.6% | 13.9% | 6.5% |
| PIVOT-v3 | 10–15 (peak, stopped) | 0.7% | — | — |
| PIVOT-v4 (bugged) | 140 (peak AIME) | 3.67% | 14.7% | 6.03% |
| PIVOT-v4b | 120 (eff 320, same as GRPO-v0) | 2.71% | 12.0% | 4.4% |
| PIVOT-v4b | 200 (peak AIME mean, eff 400) | **4.0%** | 12.6% | 7.2% |
| PIVOT-v4b | 240 (peak AIME maj, eff 440) | 3.9% | 13.4% | **7.3%** |
| PIVOT-v4b | 290 (peak AIME best, eff 490) | 2.7% | **14.3%** | 4.0% |
| PIVOT-v5a | 115 (in progress) | 1.72% | — | — |
| PIVOT-v10a | 85 (peak MATH best) | 2.32% | 10.9% | 4.3% |
| PIVOT-v10a | 145 (peak AIME mean) | **3.41%** | 12.8% | 6.4% |
| PIVOT-v10a | 165 (peak MATH maj@16) | 2.84% | 11.9% | 5.3% |
| PIVOT-v10a | 195 (peak AIME best) | 2.86% | **13.3%** | 5.6% |
| PIVOT-v10a | 200 (stopped) | 3.05% | 12.7% | 5.9% |
| PIVOT-v10b | 200 (peak AIME mean, crashed) | **3.41%** | 11.96% | **6.77%** |
| PIVOT-v10b | 170 (peak AIME best) | 3.02% | **13.48%** | 6.15% |
| PIVOT-v10b | 135 (peak MATH maj) | 3.28% | 11.04% | 6.57% |
| PIVOT-v12 | 145 (peak AIME best) | 2.68% | **13.40%** | 4.95% |
| PIVOT-v12 | 190 (peak AIME mean) | **3.26%** | 12.39% | 6.08% |
| PIVOT-v12 | 220 (peak AIME maj) | 3.12% | 12.68% | **6.65%** |
| PIVOT-v12 | 225 (stopped) | 3.02% | 13.28% | 5.75% |
| PIVOT-v13 | 195 (peak AIME best) | 2.81% | **13.69%** | 5.21% |
| PIVOT-v13 | 245 (peak AIME mean + maj, stopped at 249) | **3.18%** | 12.95% | **6.52%** |
| PIVOT-v15 | 95 (peak AIME mean, stopped at 128) | 2.71% | **13.88%** | 5.22% |
| PIVOT-v15 | 105 (peak AIME best, stopped at 128) | 2.58% | **13.88%** | 4.74% |
| PIVOT-v13d | 70 (peak AIME mean + maj) | **2.76%** | 12.07% | **5.66%** |
| PIVOT-v13d | 115 (peak AIME best) | 2.63% | **13.22%** | 4.92% |
| PIVOT-v13d | 130 (stopped) | 2.68% | 11.36% | 4.89% |
| PIVOT-v14 | 115 (peak AIME maj) | 2.97% | 12.2% | **5.95%** |
| PIVOT-v14 | 135 (peak AIME best) | 2.45% | **12.9%** | 4.42% |
| PIVOT-v14 | 175 (peak AIME mean, stopped) | **3.23%** | 12.3% | 5.79% |
| PIVOT-v15b | 180 (peak AIME mean, stopped at 200) | **3.46%** | 12.52% | 6.36% |
| PIVOT-v15b | 190 (peak AIME best, stopped at 200) | 3.26% | **13.41%** | 6.25% |
| PIVOT-v16b | 180 (peak AIME best) | 3.07% | **14.40%** | 5.83% |
| PIVOT-v16b | 245 (peak AIME mean + maj, stopped at 255) | **3.10%** | 13.83% | **6.00%** |
| PIVOT-v15b | 195 (peak AIME maj, stopped at 200) | 3.23% | 12.71% | **6.41%** |
| PIVOT-v16 | 115 (peak AIME best+maj, stopped at 200) | 2.89% | **12.36%** | **5.74%** |
| PIVOT-v16 | 190 (peak AIME mean, stopped at 200) | **2.92%** | 10.36% | 5.61% |
| PIVOT-v16b | 180 (peak AIME mean+best, in progress) | **3.07%** | **14.40%** | 5.83% |
| PIVOT-v16b | 140 (peak AIME maj, in progress) | 2.81% | 11.26% | **5.86%** |
| High-Ent-GRPO-from-grpo200 | 225 (in progress) | **3.39%** | 12.96% | **7.14%** |

---

## Results (Qwen3-4B-Base, guru-RL-92k, 8×H200)

Trained from scratch (no SFT prior). `train_batch_size=1024`, `n=8`, `kl_loss_coef=0.001`. Resumed from step 20 (batch_size=512) with increased batch size. Script: `run_qwen3_4b_base_grpo.sh`. Stopped at step 220.

### MATH500

| ID | Steps | acc@16 (mean) | best@16 | maj@16 | Checkpoint |
|----|-------|---------------|---------|--------|------------|
| GRPO-Qwen3-4B-Base | 170 (peak best+maj) | 78.3% | **89.5%** | **82.2%** | `grpo/qwen3_4b_base/global_step_140` |
| GRPO-Qwen3-4B-Base | 220 (peak mean, stopped) | **78.8%** | 89.1% | 82.1% | `grpo/qwen3_4b_base/global_step_220` |

### AIME

| ID | Steps | acc@16 (mean) | best@16 | maj@16 |
|----|-------|---------------|---------|--------|
| GRPO-Qwen3-4B-Base | 155 (peak best) | 10.3% | **23.8%** | 12.1% |
| GRPO-Qwen3-4B-Base | 210 (peak mean+maj, stopped at 220) | **11.2%** | 21.4% | **13.5%** |

### Key observation: Qwen3-4B-Base GRPO vs Qwen2.5-3B-Instruct GRPO

| Metric | Qwen2.5-3B-Instruct GRPO-v0 | Qwen3-4B-Base GRPO (step 220) | Δ |
|--------|----------------------------|-------------------------------|---|
| MATH mean@16 | 61.5% | **78.8%** | +17.3pp |
| MATH best@16 | 81.2% | **89.5%** | +8.3pp |
| MATH maj@16 | 67.5% | **82.2%** | +14.7pp |
| AIME mean@16 | 1.8% | **11.2%** | **+6×** |
| AIME best@16 | 11.3% | **23.8%** | **+2.1×** |
| AIME maj@16 | 2.4% | **13.5%** | **+5.6×** |

The 4B base model substantially outperforms the 3B instruct baseline on both benchmarks. AIME best@16=23.8% is the all-time high across all runs. The base model's higher plasticity (no SFT prior locking in response patterns) combined with the larger batch size (1024 vs 512) likely drives the improvement.

### High-Ent-GRPO on Qwen3-4B-Base (entropy_top_ratio=0.2)

High-entropy GRPO applies loss only to the top-20% highest-entropy tokens per step (Wang et al., NeurIPS 2025). Trained from scratch, same config as plain GRPO above. Stopped at step 204.

#### MATH500

| ID | Steps | acc@16 (mean) | best@16 | maj@16 |
|----|-------|---------------|---------|--------|
| High-Ent-GRPO-Qwen3-4B-Base | 195 (peak mean+maj) | **75.0%** | 87.9% | **78.9%** |
| High-Ent-GRPO-Qwen3-4B-Base | 145 (peak best) | 73.3% | **88.0%** | 77.4% |

#### AIME

| ID | Steps | acc@16 (mean) | best@16 | maj@16 |
|----|-------|---------------|---------|--------|
| High-Ent-GRPO-Qwen3-4B-Base | 200 (peak mean) | **7.66%** | 15.32% | 9.48% |
| High-Ent-GRPO-Qwen3-4B-Base | 155 (peak best) | 7.34% | **15.70%** | 13.11% |
| High-Ent-GRPO-Qwen3-4B-Base | 160 (peak maj) | 7.42% | 14.96% | **10.13%** |

#### High-Ent-GRPO vs plain GRPO (Qwen3-4B-Base)

| Metric | GRPO (step 220) | High-Ent-GRPO (peak) | Δ |
|--------|-----------------|----------------------|---|
| MATH mean@16 | **78.8%** | 75.0% | −3.8pp |
| MATH best@16 | **89.5%** | 88.0% | −1.5pp |
| MATH maj@16 | **82.2%** | 78.9% | −3.3pp |
| AIME mean@16 | **11.2%** | 7.66% | −3.5pp |
| AIME best@16 | **23.8%** | 15.70% | −8.1pp |
| AIME maj@16 | **13.5%** | 10.13% | −3.4pp |

**Result**: High-entropy token masking hurts on Qwen3-4B-Base across all metrics. Restricting loss to the top-20% entropy tokens removes signal from the ~80% of tokens where the model is making committed (low-entropy) predictions — which still carry reward-correlated gradient. On a base model that starts from scratch with high entropy everywhere, this masking likely discards too much useful gradient early in training.

### Key observation: Langevin helps AIME, not MATH500

Comparing v4b (group-mean Langevin rollout) vs GRPO-v0 at the same effective training step (320):

| Metric | GRPO-v0 | v4b (Langevin) | Δ |
|--------|---------|----------------|---|
| MATH500 mean@16 | 61.5% | 60.4% | −1.1pp (≈ noise) |
| MATH500 maj@16 | 67.5% | 68.2% | +0.7pp (≈ noise) |
| AIME mean@16 | 1.80% | 2.71% | **+51%** |
| AIME maj@16 | 2.4% | 4.4% | **+83%** |

At v4b's peak AIME (step 200, eff 400): AIME mean@16 = 4.0% vs GRPO's 1.80% (**+122%**, >2×). AIME best@16 peaks at 14.3% (step 290); AIME maj@16 peaks at 7.3% (step 240).

**Interpretation**: MATH500 is saturated at ~60–62% for this model/data scale — neither GRPO nor any PIVOT variant pushes past it meaningfully. The signal is on AIME, where Langevin rollout diversity produces a real lift. The mechanism: AIME problems have genuine decision forks where diverse rollouts matter; MATH500 problems at this difficulty level are largely solved or unsolvable regardless of rollout diversity.

**v5a vs v4b**: Adding the token-disagreement-gated entropy loss (v5a) does not improve over plain Langevin (v4b). v5a MATH500 tracks ~1–2pp below v4b throughout; AIME is still in progress. The entropy bonus raised training entropy (0.35 → 0.80 nats peak, stabilising at ~0.35) but did not convert to benchmark gains.

---

## Training Observations

### PIVOT-v2 (2026-04-15, run steps 200–345 total / N=0–145 from GRPO-200 seed)

**Config**: seeded from GRPO-v0 step 200, `entropy_threshold=3.5`, `langevin_threshold=0.005`, `entropy_coeff=0`, `kl_loss_coef=0.001`, `max_response_length=4096`. Restarted from N=100 (total step 300) with fixes — see below.

**Starting point**: GRPO-v0 step 200 evaluated at MATH500 mean@16=10.2%, AIME mean@16=0.2%, consistent with being at/near the breakthrough boundary.

#### Three Training Phases

**Phase 1 — Breakthrough (N=0–20, total 200–220)**
- MATH500 leaps 10.2% → 51.0%, AIME 0.2% → 1.3% in 20 PIVOT steps.
- Entropy rises from 0.29 to peak 0.53 nats (N=15), then begins declining.
- Fork profile fraction rises 1.8% → 11% as model starts producing varied outputs.
- Step time ~380s (92s gen, 26s ref, 38s adv, 79s update).

**Phase 2 — Climbing (N=20–85, total 220–285)**
- MATH500 51% → 59%, AIME 1.3% → 3.4%. Peak AIME at N=85: best@16=13.9%, maj@16=6.2%.
- Entropy declines 0.51 → 0.14 nats (no `entropy_coeff` to arrest decay).
- Clip ratio rises slowly 0.2% → 1.3%. Response length 677 → 994 tokens.
- Langevin fires at a stable ~20% of token positions throughout.

**Phase 3 — Entropy Collapse and Plateau (N=85–145, total 285–345)**
- MATH500 plateaus 58–60% (noisy). AIME plateaus 2.9–3.6% (noisy).
- **Root cause — entropy collapse**: entropy 0.14 → 0.075 nats. All 8 rollouts per prompt converge to identical outputs → GRPO advantages → 0 → gradient signal vanishes.
- **KL drift**: `actor/kl_loss` 0.074 → 0.128. `kl_loss_coef=0.001` too weak to prevent model from drifting away from reference policy.
- **Length inflation**: Langevin fires at near-deterministic positions (high `fork_score` for collapsed-entropy rollouts), generating confused longer sequences. Response length 994 → 1215 tokens; clip_ratio 1.3% → 6.4%.
- Langevin pivot_frac is constant throughout (~20%) — collapse amplifies Langevin's effect, but Langevin itself is not the cause.

#### Diagnostics Summary

| Metric | N=1 | N=20 | N=85 | N=145 |
|--------|-----|------|------|-------|
| `actor/entropy` | 0.292 | 0.509 | 0.138 | 0.075 |
| `response_length/mean` | 833 | 677 | 994 | 1215 |
| `response_length/clip_ratio` | 0.68% | 0.17% | 1.34% | 6.40% |
| `actor/kl_loss` | 0.0004 | 0.059 | 0.074 | 0.128 |
| `pivot/frac_above_threshold` | 1.8% | 11.1% | 20.7% | 18.6% |

#### Restart from N=100 (2026-04-17)

Training restarted from checkpoint N=100 (total step 300) with:
- `entropy_coeff`: 0 → **0.01** (primary fix: arrest entropy collapse)
- `kl_loss_coef`: 0.001 → **0.005** (stronger KL regularization)
- `langevin_threshold`: 0.005 → **0.02** (reduce firing rate from ~20% to ~5–8%)
- `ppo_micro_batch_size_per_gpu`: 8 → **16** (memory efficiency; no effect on gradient update)

#### Step-by-step Metrics (N = steps from GRPO-200; total = N + 200)

| N | Total | MATH mean@16 | MATH best@16 | MATH maj@16 | AIME mean@16 | AIME best@16 | AIME maj@16 |
|---|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 200 | 10.2% | 29.8% | 9.2% | 0.2% | 1.4% | 0.0% |
| 5 | 205 | 22.9% | 56.1% | 23.8% | 0.6% | 4.6% | 0.1% |
| 10 | 210 | 42.3% | 76.3% | 51.9% | 0.7% | 5.7% | 0.4% |
| 15 | 215 | 48.2% | 78.8% | 60.0% | 0.8% | 5.3% | 0.9% |
| 20 | 220 | 51.0% | 79.3% | 62.8% | 1.3% | 7.9% | 2.1% |
| 30 | 230 | 53.2% | 80.2% | 63.3% | 2.2% | 10.5% | 3.6% |
| 40 | 240 | 53.2% | 81.2% | 63.1% | 2.5% | 10.4% | 4.7% |
| 50 | 250 | 56.4% | 82.1% | 67.0% | 2.9% | 13.1% | 5.2% |
| 60 | 260 | 57.6% | 81.6% | 67.8% | 3.3% | 12.0% | 6.0% |
| 70 | 270 | 57.9% | 82.3% | 66.7% | 2.9% | 11.7% | 5.3% |
| 80 | 280 | 58.8% | 82.1% | 66.7% | 3.2% | 11.9% | 5.5% |
| **85** | **285** | 58.8% | 81.8% | 67.4% | 3.4% | **13.9%** | 6.2% |
| 90 | 290 | 59.6% | 81.3% | 67.6% | 3.0% | 12.8% | 5.4% |
| **100** | **300** | **59.7%** | 81.7% | 67.3% | **3.6%** | **13.9%** | **6.5%** |
| 110 | 310 | 59.2% | 80.9% | 67.5% | 3.3% | 12.6% | 6.2% |
| 120 | 320 | 59.5% | 81.3% | 67.3% | 2.9% | 11.1% | 5.6% |
| 130 | 330 | 58.6% | 81.2% | 66.8% | 3.1% | 11.4% | 5.8% |
| 145 | 345 | 58.2% | 79.5% | 65.7% | 3.2% | 11.6% | 5.5% |

**Peak**: MATH500 59.7% mean@16 @ step 300 (N=100); AIME 3.6% / 13.9% best@16 / 6.5% maj@16 @ step 300.

**vs GRPO-v0 at step 320**: MATH mean@16 59.5% vs 61.5% (−2pp); AIME mean@16 2.9% vs 1.8% (+61%), AIME maj@16 5.6% vs 2.4% (+133%). PIVOT-v2 clearly outperforms GRPO-v0 on AIME despite entropy collapse. The entropy_coeff fix (restarted N=100) is the critical test for whether MATH500 can also exceed GRPO-v0.

### PIVOT-v12 (Langevin-consistent GRPO numerator, 2026-04-24, 227 steps from GRPO-200 seed, stopped)

**Config**: `pivot_version=4`, `langevin_rollout=True`, `delta_var_threshold=5.0`, `entropy_threshold=8.0`, `langevin_K=1`, `langevin_top_k=20`, `langevin_eta=0.1`, `langevin_sigma=0.01`, `lan_grpo_coeff=1.0`, `entropy_cap=0.8`, `kl_loss_coef=0.005`. No internalize_mode / no KL imitation loss. Seeded from GRPO-200.

**Design**: Pure RL — at trigger positions, replace `log π_current(a_t)` with `log p_lan_current(a_t)` in the GRPO numerator:
```
p_lan_current(t) = (1-η)·π_current(t) + η·softmax(group_mean_logits(t))
L_trigger(t) = adv · (log p_lan_current(a_t) - log π_old(a_t))
L_non-trigger(t) = adv · (log π_current(a_t) - log π_old(a_t))   [standard GRPO]
```
Gradient flows through `π_current` only; `group_mean_logits` is detached. No explicit KL target imitation.

**Val metrics (selected steps, eff = 200 + step)**:

| step | eff | MATH mean@16 | MATH best@16 | MATH maj@16 | AIME mean@16 | AIME best@16 | AIME maj@16 |
|------|-----|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 200 | ~10% | ~30% | ~9% | ~0.1% | ~1.4% | ~0% |
| 30 | 230 | 52.9% | 82.1% | 64.0% | 1.48% | 8.46% | 2.75% |
| 60 | 260 | 58.4% | 82.6% | 68.2% | 2.73% | 11.15% | 4.99% |
| 90 | 290 | 59.4% | 82.9% | 68.0% | 3.10% | 12.02% | 5.66% |
| **100** | **300** | 59.4% | 82.1% | 68.5% | 3.07% | **13.28%** | 5.71% |
| **145** | **345** | 59.5% | 81.6% | 67.7% | 2.68% | **13.40%** | 4.95% |
| **190** | **390** | 59.6% | 82.8% | 67.7% | **3.26%** | 12.39% | 6.08% |
| **195** | **395** | **60.5%** | 82.6% | 68.6% | 2.76% | 11.43% | 4.94% |
| 220 | 420 | 60.2% | 83.5% | 68.7% | 3.12% | 12.68% | **6.65%** |
| 225 | 425 | 60.3% | **83.7%** | **68.8%** | 3.02% | 13.28% | 5.75% |

**Training dynamics**:
- Trigger fraction: 17% → 44% (rising as entropy compresses, same feedback loop as v10a)
- Entropy: 0.30 → brief peak 0.40 (step 10) → compressed to 0.15–0.18 floor by step 50
- KL loss: 3.35 → 10 (slow drift; `kl_loss_coef=0.005` too weak to hold it)
- Step time: ~340s early, settling to ~285s after breakthrough

**Key results**:
- MATH mean@16 peak: **60.5% @ step 195** — matching v4b's peak (61.0%)
- MATH best@16 peak: **83.7% @ step 225** — matching v10a's best
- MATH maj@16 peak: **68.8% @ step 225** — between v4b (68.2%) and v10a (69.9%)
- AIME mean@16 peak: **3.26% @ step 190** — below v4b's 4.0% but above v10a's 3.41%
- AIME best@16 peak: **13.4% @ step 145** — comparable to v4b (14.3%)

**vs v10a (split_kl_bipolar) at matched steps**:
| Metric | v10a | v12 | Δ |
|--------|------|-----|---|
| MATH mean@16 peak | 61.2% | 60.5% | −0.7pp |
| MATH maj@16 peak | 69.9% | 68.8% | −1.1pp |
| AIME mean@16 peak | 3.41% | **3.26%** | −0.15pp |
| AIME mean@16 @ step 190 | 3.20% | **3.26%** | +0.06pp |

v12 slightly trails v10a on MATH (no bipolar compression helping peaked predictions) but is comparable on AIME. The entropy compression still occurred (same ΔVar feedback loop) despite the absence of an explicit bipolar repulsion loss — entropy fell due to the kl_loss drift compressing the distribution toward the reference, and the lan_grpo gradient concentrating probability at trigger tokens.

**vs v4b (Langevin-only, no internalization)**:
- AIME mean@16 peak: v12 3.26% vs v4b 4.0% — v4b still leads. The lan_grpo numerator replacement did not recover v4b's AIME generalization edge, suggesting the bottleneck is the entropy compression (shared between v12 and v10a) rather than the implicit SFT channel specifically.

**Verdict**: v12's Langevin-consistent numerator (log p_lan_current instead of log π_current) produces training comparable to v10a but without the explicit KL imitation loss. MATH performance is slightly below v10a (no bipolar compression benefit); AIME is marginally better than v10a but still below v4b. The entropy compression from ΔVar feedback remains the dominant limiting factor across all internalization variants. Stopped at step 227.

### PIVOT-v10a (split_kl_bipolar internalization, 2026-04-23, 204 steps from GRPO-200 seed, stopped)

**Config**: `pivot_version=4`, `langevin_rollout=True`, `delta_var_threshold=5.0`, `entropy_threshold=8.0`, `langevin_K=1`, `langevin_top_k=20`, `langevin_eta=0.1`, `langevin_sigma=0.01`, `internalize_mode=split_kl_bipolar`, `internalize_coeff=0.05`, `lan_grpo_coeff=1.0`, `entropy_cap=0.8`, `kl_loss_coef=0.005`. Seeded from GRPO-200.

**Internalization design — split_kl_bipolar**: All n=8 rollouts enter the internalization loss, signed by their advantage:
```
cm(t) = clamp(winner_mean(t) - loser_mean(t), ±|winner_mean(t)|)
p_lan_b(t) = softmax((1-eta)*logits_b(t) + eta*cm(t))
L = sum_{b, t:trigger} adv(b) * KL(p_lan_b || pi_theta, top-20)
```
Winners (adv > 0) are attracted toward p_lan_b; losers (adv < 0) are repelled from p_lan_b. The 7:1 loser:winner ratio at most steps means repulsion dominates, causing global entropy to fall.

**Val metrics (selected steps, eff = 200 + step)**:

| step | eff | MATH mean@16 | MATH best@16 | MATH maj@16 | AIME mean@16 | AIME best@16 | AIME maj@16 |
|------|-----|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 200 | 10.0% | 29.9% | 8.9% | 0.13% | 1.4% | 0.0% |
| 25 | 225 | 50.0% | 79.8% | 60.5% | 1.30% | 7.5% | 2.1% |
| 60 | 260 | 58.9% | 83.5% | 68.8% | 2.14% | 10.2% | 4.3% |
| **85** | **285** | 59.5% | **83.7%** | 68.5% | 2.32% | 10.9% | 4.3% |
| 120 | 320 | 59.2% | 80.9% | 67.9% | 3.02% | 11.8% | 6.5% |
| **145** | **345** | 60.1% | 82.0% | 68.6% | **3.41%** | 12.8% | 6.4% |
| **165** | **365** | **61.2%** | 82.9% | **69.9%** | 2.84% | 11.9% | 5.3% |
| 185 | 385 | 61.2% | 82.8% | 69.3% | 2.79% | 11.3% | 5.7% |
| 195 | 395 | 60.1% | 81.8% | 68.6% | 2.86% | **13.3%** | 5.6% |
| 200 | 400 | 60.5% | 83.2% | 68.6% | 3.05% | 12.7% | 5.9% |

**Entropy trajectory**:

| step | entropy (nats) |
|------|----------------|
| 1 | 0.357 |
| 21 | 0.423 (brief peak, early rollout diversity) |
| 41 | 0.256 |
| 61 | 0.216 |
| 81 | 0.196 |
| 101 | 0.190 |
| 141 | 0.187 |
| 181 | 0.185 |
| 204 | 0.186 (stopped) |

**Key results**:
- MATH maj@16 = **69.9% @ step 165**, surpassing v4b's peak of 68.2%. Best MATH maj@16 of any variant so far.
- MATH best@16 = **83.7% @ step 85**, highest of any variant.
- AIME mean@16 peak = 3.41% @ step 145 — below v4b's 4.0% @ step 200 by −0.6pp.
- AIME mean@16 @ step 200 = 3.05% vs v4b's 4.0% — consistent gap throughout.
- AIME best@16 peak = 13.3%, comparable to v4b's 14.3%.

**Entropy compression — mechanism**: Bipolar repulsion (7:1 loser:winner at most steps) outvotes winner attraction, compressing the distribution. Entropy fell monotonically from 0.36 to a floor of ~0.18–0.19 nats by step 100, roughly half the baseline. This sharpening does not harm MATH (which benefits from more committed predictions at high-confidence positions) but may limit AIME exploration.

**AIME generalization gap — implicit SFT channel**: v10a's AIME mean@16 lags v4b (3.05% vs 4.0% at step 200) despite stronger MATH performance. The likely mechanism is the internalization loss acting as an implicit supervised fine-tuning channel on Langevin-perturbed tokens (see [v12 design section](#langevin-consistent-grpo-v12-design)): at trigger positions, the KL loss pushes `π_current` to reproduce p_lan_b's token distribution regardless of advantage sign — behavioral cloning toward a training-distribution target. v4b has no internalization and shows better AIME generalization. The bipolar repulsion further specializes the distribution toward seen fork patterns. Future fix: v12's Langevin-consistent numerator (`log p_lan_current` instead of `log π_current`) would remove this implicit SFT channel while preserving the directed gradient.

**vs v4b (Langevin-only baseline)**:

| Metric | v4b peak | v10a peak | Δ |
|--------|---------|-----------|---|
| MATH mean@16 | 61.0% | 61.2% | +0.2pp (noise) |
| MATH best@16 | 82.4% | **83.7%** | **+1.3pp** |
| MATH maj@16 | 68.2% | **69.9%** | **+1.7pp** |
| AIME mean@16 | **4.0%** | 3.41% | **−0.6pp** |
| AIME best@16 | **14.3%** | 13.3% | **−1.0pp** |

**Verdict**: bipolar internalization improves MATH best@16 and maj@16 (sharper distributions commit better to correct tokens) but trades off AIME generalization (implicit SFT channel + entropy compression limits out-of-distribution exploration). Training stopped at step 204.

### PIVOT-v10b (contrastive_kl_bipolar internalization, 2026-04-23, 200 steps from GRPO-200 seed, crashed)

**Config**: `pivot_version=4`, `langevin_rollout=True`, `delta_var_threshold=5.0`, `entropy_threshold=8.0`, `langevin_K=1`, `langevin_top_k=20`, `langevin_eta=0.1`, `langevin_sigma=0.01`, `internalize_mode=contrastive_kl_bipolar`, `internalize_coeff=0.05`, `kl_loss_coef=0.005`. Seeded from GRPO-200. Crashed at step 200 (vLLM engine core died).

**Internalization design — contrastive_kl_bipolar**: GRPO-style soft advantage-weighted contrastive mean, with bipolar loss over all rollouts:
```
cm(t) = (1/n) * sum_i adv_i * logits_i(t)   (soft adv-weighted direction)
p_lan_b(t) = softmax((1-eta)*logits_b(t) + eta*cm(t))
L = sum_{b, t:trigger} adv(b) * KL(p_lan_b || pi_theta, top-20)
```
vs v10a (split_kl_bipolar): same bipolar loss structure, but cm is the soft adv-weighted sum rather than the hard winner_mean − loser_mean split. The soft weighting preserves advantage magnitude information; the hard split treats all winners equally.

**Val metrics (selected steps, eff = 200 + step)**:

| step | eff | MATH mean@16 | MATH best@16 | MATH maj@16 | AIME mean@16 | AIME best@16 | AIME maj@16 |
|------|-----|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 200 | 10.21% | 30.19% | 8.93% | 0.05% | 0.54% | 0.00% |
| 5 | 205 | 21.73% | 55.35% | 21.99% | 0.26% | 1.98% | 0.02% |
| 10 | 210 | 36.93% | 71.86% | 43.69% | 0.39% | 3.27% | 0.06% |
| 15 | 215 | 44.81% | 76.51% | 55.26% | 0.68% | 4.56% | 0.52% |
| 20 | 220 | 49.43% | 78.26% | 60.20% | 1.15% | 6.31% | 1.73% |
| 25 | 225 | 52.30% | 81.70% | 63.23% | 1.41% | 7.57% | 2.37% |
| 30 | 230 | 53.70% | 80.73% | 64.85% | 1.69% | 8.95% | 3.27% |
| 50 | 250 | 55.61% | 80.95% | 65.58% | 2.37% | 10.41% | 4.70% |
| 75 | 275 | 58.60% | 81.58% | 67.64% | 2.76% | 11.88% | 5.28% |
| 100 | 300 | 59.00% | 81.29% | 67.61% | 2.94% | 13.32% | 5.38% |
| **135** | **335** | 59.92% | 81.51% | **69.34%** | 3.28% | 11.04% | 6.57% |
| **155** | **355** | 59.52% | **82.62%** | 67.54% | 2.94% | 11.91% | — |
| **170** | **370** | 59.96% | 81.06% | 68.37% | 3.02% | **13.48%** | 6.15% |
| **180** | **380** | **60.36%** | 81.31% | 68.60% | 3.07% | 12.36% | 5.58% |
| **200** | **400** | 60.16% | 81.06% | 68.86% | **3.41%** | 11.96% | **6.77%** |

**Entropy trajectory**:

| step | entropy (nats) |
|------|----------------|
| 1 | 0.363 |
| 7 | 0.500 (brief peak) |
| 24 | 0.361 |
| 50 | 0.300 |
| 75 | 0.193 |
| 100 | 0.183 |
| 150 | 0.177 |
| 200 | 0.175 (crashed) |

**Key results**:
- Fastest early breakthrough of any variant: MATH mean@16 10% → 45% in 15 steps, 10% → 60% in ~130 steps.
- MATH maj@16 = **69.3% @ step 135**, matching v10a's 69.9% — second-best maj@16 of any variant.
- AIME mean@16 peak = **3.41% @ step 200** (last step before crash) — still trending up.
- AIME best@16 peak = **13.48% @ step 170**.
- Plateau at same ceiling as all other variants: MATH mean@16 ~59–60%, AIME mean@16 ~2.5–3.4%.

**vs v10a (split_kl_bipolar)**:

| Metric | v10a peak | v10b peak | Δ |
|--------|-----------|-----------|---|
| MATH mean@16 | 61.2% | 60.4% | −0.8pp |
| MATH best@16 | 83.7% | 82.6% | −1.1pp |
| MATH maj@16 | **69.9%** | 69.3% | −0.6pp |
| AIME mean@16 | 3.41% | **3.41%** | 0 |
| AIME best@16 | 13.3% | **13.48%** | +0.2pp |
| AIME maj@16 | — | **6.77%** | — |

v10b slightly trails v10a on MATH (hard split with winner_mean − loser_mean marginally sharper than soft adv-weighted cm). AIME is essentially tied. The soft contrastive direction does not provide a clear advantage over hard split on these benchmarks.

**Same ceiling, same pattern**: entropy compressed to ~0.18 nats by step 75 (same as v10a's 0.18 nats). MATH plateaued at steps 75–200 (same region as v10a). Confirms the ceiling is a model/data capacity limit, not a variant-specific failure mode.

**Crash**: vLLM engine core died at step 200. Training was still slowly improving (AIME at new high, MATH marginally up). Run not resumed.

### PIVOT-v4b (true Langevin ablation, 2026-04-20, 300 steps from GRPO-200 seed, stopped)

**Config**: `pivot_version=4`, `langevin_rollout=True`, `delta_var_threshold=5.0`, `entropy_threshold=8.0`, `langevin_K=1`, `langevin_top_k=20`, `langevin_eta=0.1`, `langevin_sigma=0.01`, `entropy_coeff=0`, `kl_loss_coef=0.001`. No walk, no entropy loss. Resumed from step 180 after crash; JSONL for steps 0–175 was overwritten on resume.

**Val metrics (selected steps, eff = 200 + step)**:

| step | eff | MATH mean@16 | MATH best@16 | MATH maj@16 | AIME mean@16 | AIME best@16 | AIME maj@16 |
|------|-----|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 200 | 9.6% | 29.6% | 8.0% | 0.13% | 1.2% | 0.0% |
| 30 | 230 | 53.5% | 81.4% | 64.0% | 1.67% | 9.2% | 3.1% |
| 75 | 275 | 58.9% | 81.0% | 67.0% | 2.42% | 10.9% | 4.9% |
| 115 | 315 | 60.4% | 80.9% | **68.2%** | 3.26% | 12.3% | 5.5% |
| 120 | 320 | 60.4% | 80.7% | **68.2%** | 2.71% | 12.0% | 4.4% |
| 135 | 335 | 60.6% | 82.4% | 68.1% | 3.3% | 13.7% | 5.9% |
| 175 | 375 | 60.4% | 80.2% | 67.2% | 3.20% | 13.0% | 5.1% |
| **200** | **400** | **61.0%** | 80.1% | 67.4% | **4.0%** | 12.6% | **7.2%** |
| 215 | 415 | 60.2% | 80.0% | 66.3% | 3.6% | **14.3%** | 6.5% |
| 240 | 440 | 59.9% | 79.9% | 65.9% | 3.9% | 13.4% | **7.3%** |
| 290 | 490 | 60.0% | 79.5% | 66.2% | 2.7% | **14.3%** | 4.0% |
| 295 | 495 | 60.2% | 79.9% | 66.3% | 2.8% | 13.2% | 4.4% |

**vs GRPO-v0 at step 320** (same effective training compute, step 120):

| Metric | GRPO-v0 | v4b (Langevin) | Δ |
|--------|---------|----------------|---|
| MATH500 mean@16 | 61.5% | 60.4% | −1.1pp (noise) |
| MATH500 maj@16 | 67.5% | 68.2% | +0.7pp (noise) |
| AIME mean@16 | 1.80% | 2.71% | **+51%** |
| AIME maj@16 | 2.4% | 4.4% | **+83%** |

**Peak AIME vs GRPO-v0** (step 200, eff 400): AIME mean@16 = 4.0% vs GRPO's 1.80% (**+122%**, >2×).

**Interpretation**: Langevin rollout diversity does not move MATH500 (saturated ~61%, MATH maj@16 peaks early at step 120 and then slowly regresses). Clear, sustained lift on AIME — harder problems where diverse rollout trajectories matter. AIME mean@16 keeps improving past step 120 (2.71% → 4.0%) while MATH plateaus. Entropy collapse (no `entropy_coeff`) does not prevent AIME improvement, suggesting AIME gains come from early-stage diversity before entropy collapses.

### PIVOT-v5a (co-designed entropy loss, 2026-04-20, in progress from GRPO-200 seed)

**Config**: v4b rollout + token-disagreement-gated entropy bonus. `entropy_coeff=0.05`, `entropy_loss_cap=0.01` (total entropy loss contribution capped). Gate: positions where ≥2 of 8 rollouts diverge AND group has non-zero GRPO advantages.

**Training stability**: entropy peaked at 0.84 nats (step 16), stabilised to 0.36 nats by step 117. `entropy_loss_cap` prevents runaway — contribution stays ≤ 0.0098 throughout.

**Val metrics vs v4b (selected steps)**:

| step | v5a MATH@16 | v4b MATH@16 | v5a AIME@16 | v4b AIME@16 |
|------|:-----------:|:-----------:|:-----------:|:-----------:|
| 5 | 22.0% | 20.0% | 0.21% | 0.34% |
| 25 | 51.1% | 50.8% | 0.83% | 0.83% |
| 50 | 54.5% | 56.9% | 1.12% | 2.08% |
| 75 | 56.8% | 58.9% | 1.51% | 2.42% |
| 100 | 57.0% | 59.7% | 1.72% | 2.73% |
| 115 | 57.9% | 60.4% | 1.72% | 3.26% |

**Verdict (provisional, run still in progress)**: entropy bonus does not improve over plain Langevin (v4b). v5a tracks ~1–2pp below v4b on MATH500 and lags on AIME. The entropy bonus raised training entropy transiently but the extra diversity did not convert to benchmark gains. Run continuing to confirm whether the gap narrows late-stage.

### PIVOT-v4 (bugged, 2026-04-19, stopped at step 189, seeded from GRPO-200)

**What ran**: temporal walk advantage weighting (`Â_t = A_t · (1 + pivot_score_t)`) only — Langevin was a complete no-op due to two bugs (wrong adapter routing + missing attention hooks, documented in [PIVOT-v4 variant](#pivot-v4-mislabeled--walk-advantage-weighting-only)). Effectively the first clean ablation of Phase 1 temporal walk in isolation.

**Peak**: MATH maj@16=68.6% at step 80 (total step 280 from scratch); AIME best@16=14.7% / mean@16=3.67% / maj@16=6.03% at step 140. Peaked early (step 80–140) then regressed to ~67% MATH / ~12% AIME by step 189.

**vs v6**: MATH maj@16=68.6% vs v6's 69.8% (−1.2pp); AIME best@16=14.7% vs v6's 16.9% (−2.2pp). Temporal walk advantage weighting alone is close to v6's differential walk but consistently below it — reward-contrastive differential (v6) adds meaningful signal over per-rollout temporal walk (v4 Phase 1).

**vs PIVOT-v2**: MATH comparable (68.6% vs 67.3%); AIME v4 slightly better (14.7% vs 13.9% best@16). But v2 has Langevin rollout (Phase 2) active, so v4's Phase 1 walk signal alone roughly matches v2's combined Phase 1+2 — suggests Phase 2 is not adding much in v2 either.

### PIVOT-v3 (2026-04-19, stopped at step ~58, seeded from GRPO-200)

**Config vs v2**: added `entropy_coeff=0.001` (gated on pg-active tokens), `pivot_version=2`, `pivot_mode=soft`, `pivot_threshold=0.005`, `pivot_alpha=1.0`, `langevin_K=1`, `langevin_top_k=20`, `langevin_eta=0.1`, `langevin_sigma=0.01`, `delta_var_threshold=5.0`, `entropy_threshold=8.0`.

**Result**: clear regression. MATH500 peaked at 46.3% at step 25, then monotonically declined to 30.9% by step 55. AIME peaked at 0.7% (steps 10–15) then dropped to 0.1%. Step time doubled from ~343s → 690s as Langevin fired more aggressively. **Stopped at step ~58.**

| step | reward | pg_loss | MATH@16 | AIME@16 |
|------|--------|---------|---------|---------|
| 5 | -0.985 | -0.0035 | 20.1% | 0.3% |
| 10 | -0.950 | -0.0056 | 38.7% | 0.7% |
| 15 | -0.942 | -0.0079 | 43.5% | 0.7% |
| 20 | -0.930 | -0.0104 | 45.3% | 0.5% |
| **25** | **-0.923** | **-0.0084** | **46.3%** | **0.4%** |
| 30 | -0.917 | -0.0086 | 45.8% | 0.5% |
| 35 | -0.924 | -0.0122 | 42.8% | 0.2% |
| 40 | -0.928 | -0.0144 | 39.1% | 0.2% |
| 45 | -0.930 | -0.0181 | 33.8% | 0.1% |
| 50 | -0.909 | -0.0167 | 33.7% | 0.1% |
| 55 | -0.901 | -0.0175 | 30.9% | 0.1% |

**Diagnosis**: `pg_loss` was negative throughout (growing more negative over time), meaning the entropy bonus and Langevin perturbations pushed the policy away from high-reward outputs rather than toward them. The `entropy_coeff=0.001` entropy loss appears to be too large relative to the pg signal at this pre/near-breakthrough stage — it dominates and drives the policy toward higher-entropy (more uniform) outputs, collapsing reward. The Langevin noise further destabilizes rollouts, inflating step time without benefit.

**Root cause vs v2**: v2 had no `entropy_coeff` and a weaker Langevin (`entropy_threshold=3.5`, no `langevin_top_k`). v3's combination of entropy bonus + stronger Langevin at the critical breakthrough phase actively harms the policy. Entropy bonus should only be applied *after* breakthrough, not before.

### PIVOT-v1 (2026-04-13, completed, step 320)

- Implementation sanity (step 1): `pivot/score_mean=0.211`, `pivot/score_std=0.282`, `pivot/frac_above_threshold=0.322`
- ~32% of response tokens exceed the 0.3 threshold — consistent with the design target of ~15-30% structural change positions
- `timing_s/pivot_scores≈13s` per step — adds ~7% overhead vs standard GRPO (vs gen cost of ~35s)
- All 8 vLLM servers confirm: `PIVOT: registered PIVOTLangevinAdapter as V1 engine-level logits processor`
- vLLM V1 compatibility: Phase 2 cannot use `SamplingParams(logits_processors=[...])` — V1 removed per-request logits processors for CUDA graph compatibility. Fix: `PIVOTLangevinAdapter(AdapterLogitsProcessor)` registered at engine init via `vllm_config.model_config.logits_processors`.
- Pre-breakthrough (steps 1–110): all rewards=-1, advantages=0. Langevin fires but has zero loss effect.
- **MATH500** breakthrough at step ~110, ramps sharply to ~63% by step 200, then **plateaus** with no further improvement to step 320. Final: **61.7% acc@16** (mean) — on par with GRPO-v0 (61.5%).
- **AIME** completely flat at ~2–3% across all 320 steps. No late-stage kick-in (unlike W-GRPO v6). Temporal walk + Langevin does not produce AIME gains.
- OOM at step 262 (CUDA fragmentation, 30MB short): fixed with `torch.cuda.empty_cache()` before actor update loop + `ppo_micro_batch_size_per_gpu` 32→16.
- **Verdict**: PIVOT-v1 ≈ GRPO on both benchmarks. Temporal walk (per-rollout, no reward conditioning) is a weak Phase 1 signal — no reward-contrastive differential. Motivates PIVOT-v2's representation divergence approach.

#### Step-by-step metrics (score/mean@16, reward ∈ [-1,1]; acc = (score+1)/2)

| Step | AIME reward | AIME acc% | MATH500 reward | MATH500 acc% |
|------|-------------|-----------|----------------|--------------|
| 0 | -1.0000 | 0.0% | -1.0000 | 0.0% |
| 5 | -1.0000 | 0.0% | -1.0000 | 0.0% |
| 10 | -1.0000 | 0.0% | -1.0000 | 0.0% |
| 15 | -1.0000 | 0.0% | -1.0000 | 0.0% |
| 20 | -1.0000 | 0.0% | -1.0000 | 0.0% |
| 25 | -1.0000 | 0.0% | -1.0000 | 0.0% |
| 30 | -1.0000 | 0.0% | -1.0000 | 0.0% |
| 35 | -1.0000 | 0.0% | -0.9995 | 0.0% |
| 40 | -1.0000 | 0.0% | -1.0000 | 0.0% |
| 45 | -1.0000 | 0.0% | -1.0000 | 0.0% |
| 50 | -1.0000 | 0.0% | -1.0000 | 0.0% |
| 55 | -1.0000 | 0.0% | -1.0000 | 0.0% |
| 60 | -1.0000 | 0.0% | -1.0000 | 0.0% |
| 65 | -1.0000 | 0.0% | — | — |
| 70 | -1.0000 | 0.0% | -1.0000 | 0.0% |
| 75 | -1.0000 | 0.0% | -1.0000 | 0.0% |
| 80 | -1.0000 | 0.0% | -1.0000 | 0.0% |
| 85 | -1.0000 | 0.0% | -0.9998 | 0.0% |
| 90 | -1.0000 | 0.0% | -0.9998 | 0.0% |
| 95 | -1.0000 | 0.0% | -1.0000 | 0.0% |
| 100 | -1.0000 | 0.0% | -1.0000 | 0.0% |
| 105 | -1.0000 | 0.0% | -0.9972 | 0.1% |
| 110 | -1.0000 | 0.0% | -0.9550 | 2.3% |
| **115** | -0.9943 | **0.3%** | -0.6228 | **18.9%** |
| 120 | -0.9839 | 0.8% | -0.2973 | 35.1% |
| 125 | -0.9646 | 1.8% | 0.0410 | 52.0% |
| 130 | -0.9604 | 2.0% | 0.1212 | 56.1% |
| 135 | -0.9516 | 2.4% | 0.1668 | 58.3% |
| 140 | -0.9464 | 2.7% | 0.1795 | 59.0% |
| 145 | -0.9552 | 2.2% | 0.1782 | 58.9% |
| 150 | -0.9417 | 2.9% | 0.1842 | 59.2% |
| 155 | -0.9505 | 2.5% | 0.1993 | 60.0% |
| 160 | -0.9458 | 2.7% | 0.1925 | 59.6% |
| 165 | -0.9469 | 2.7% | 0.2208 | 61.0% |
| 170 | -0.9505 | 2.5% | 0.2208 | 61.0% |
| 175 | -0.9458 | 2.7% | 0.2065 | 60.3% |
| 180 | -0.9432 | 2.8% | 0.2238 | 61.2% |
| 185 | -0.9422 | 2.9% | 0.2392 | 62.0% |
| 190 | -0.9490 | 2.6% | 0.2278 | 61.4% |
| 195 | -0.9417 | 2.9% | 0.2240 | 61.2% |
| 200 | -0.9448 | 2.8% | 0.2240 | 61.2% |
| 205 | -0.9469 | 2.7% | 0.2218 | 61.1% |
| 210 | -0.9490 | 2.6% | 0.2132 | 60.7% |
| 215 | -0.9479 | 2.6% | 0.2125 | 60.6% |
| 220 | -0.9469 | 2.7% | 0.2240 | 61.2% |
| 225 | -0.9500 | 2.5% | 0.2137 | 60.7% |
| 230 | -0.9479 | 2.6% | 0.2330 | 61.7% |
| 235 | -0.9411 | 2.9% | 0.2410 | 62.1% |
| 240 | -0.9464 | 2.7% | 0.2200 | 61.0% |
| 245 | -0.9490 | 2.6% | 0.2190 | 61.0% |
| 250 | -0.9484 | 2.6% | 0.2527 | 62.6% |
| 255 | -0.9536 | 2.3% | 0.2445 | 62.2% |
| 260 | -0.9432 | 2.8% | 0.2397 | 62.0% |
| 265 | -0.9521 | 2.4% | 0.2432 | 62.2% |
| 270 | -0.9464 | 2.7% | 0.2572 | 62.9% |
| 275 | -0.9490 | 2.6% | 0.2565 | 62.8% |
| 280 | -0.9542 | 2.3% | 0.2475 | 62.4% |
| 285 | -0.9521 | 2.4% | 0.2467 | 62.3% |
| 290 | -0.9547 | 2.3% | 0.2405 | 62.0% |
| **295** | -0.9557 | 2.2% | **0.2597** | **63.0%** |
| 300 | -0.9599 | 2.0% | 0.2355 | 61.8% |
| 305 | -0.9563 | 2.2% | 0.2430 | 62.2% |
| 310 | -0.9500 | 2.5% | 0.2352 | 61.8% |
| 315 | -0.9547 | 2.3% | 0.2440 | 62.2% |
| 320 | -0.9552 | 2.2% | 0.2330 | 61.7% |

**Peak**: MATH500 63.0% @ step 295 (plateau since ~step 165); AIME 2.9% @ steps 150/185/195 (noise floor, never improves).

---

### PIVOT-v13d (correct-rollout group mean + adaptive threshold, 2026-04-25, 130 steps from GRPO-200 seed, stopped)

**Config**: `pivot_version=4`, `langevin_rollout=True`, `delta_var_threshold=9.0`, `trig_percentile=90`, `langevin_K=1`, `langevin_top_k=20`, `langevin_eta=0.1`, `langevin_sigma=0.01`, `lan_grpo_coeff=1.0`, `lan_grpo_correct_is=True`, `lan_grpo_correct_mu=True`, `kl_loss_coef=0.005`. Seeded from GRPO-200.

**Design changes vs v13**: Uses `mu_correct` (mean logits over correct/positive-advantage rollouts only) instead of full-group mean for the IS denominator mixing target. Adaptive threshold (`trig_percentile=90`) keeps trig_frac ≈ 10% throughout. Rollout Langevin still uses full-group mean.

**Val metrics (selected steps)**:

| step | MATH mean@16 | MATH best@16 | MATH maj@16 | AIME mean@16 | AIME best@16 | AIME maj@16 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 30 | 51.7% | 80.6% | 62.6% | 1.51% | 7.33% | 2.14% |
| 70 | 58.1% | 80.8% | 66.8% | **2.76%** | 12.07% | **5.66%** |
| **95** | **59.8%** | 81.5% | 67.9% | 2.47% | 10.39% | 4.65% |
| **105** | 59.3% | 81.1% | **68.3%** | 2.21% | 9.78% | 4.80% |
| **115** | 59.6% | 81.4% | 66.8% | 2.63% | **13.22%** | 4.92% |
| **120** | 59.4% | **82.9%** | 67.1% | 2.50% | 11.83% | 4.55% |
| 130 | 59.4% | 81.4% | 67.6% | 2.68% | 11.36% | 4.89% |

**Training dynamics**:
- Trig fraction: **9.5–10.3% throughout** — adaptive threshold working perfectly
- Entropy: 0.30 → compressed to 0.16 floor by step 50 (same as v12, despite stable trig_frac)
- KL loss: ~0.001 → ~0.009 (very low and stable, unlike v12's 3→10 drift)

**Key finding — entropy compression is not from trig_frac feedback**: v13d has stable 10% trig_frac throughout yet entropy still compresses to 0.16, same as v12 (where trig_frac grew from 17% to 44%). The compression is intrinsic to the lan_grpo gradient concentrating probability at trigger tokens, not a feedback artifact of growing trigger coverage.

**AIME underperforms v12 and v13**: AIME mean@16 peak 2.76% @ step 70, well below v12 (3.26%) and v13 (3.18%). The `mu_correct` target (winners only) is less effective than full-group mean for AIME generalization — restricting the mixing target to correct rollouts removes the contrastive diversity signal that makes the group mean useful. Stopped at step 130.

**vs v12**:
| Metric | v12 peak | v13d peak | Δ |
|--------|---------|----------|---|
| MATH mean@16 | **60.5%** | 59.8% | −0.7pp |
| MATH maj@16 | **68.8%** | 68.3% | −0.5pp |
| AIME mean@16 | **3.26%** | 2.76% | **−0.5pp** |
| AIME best@16 | **13.40%** | 13.22% | −0.18pp |

v13d is uniformly below v12. Using correct-rollout-only mean hurts both MATH and AIME. Full-group mean (v12) is the better mixing target.

---

### PIVOT-v13 (2026-04-23 to 2026-04-25, stopped at step 249, seeded from GRPO-200)

**Config**: `pivot_version=4` (group-mean Langevin), `delta_var_threshold=9.0`, `langevin_rollout=True`, `lan_grpo_coeff=1.0`, `lan_grpo_correct_is=True`, `entropy_cap=0.8`, `n_rollouts_per_prompt=8`, `kl_loss_coef=0.005`. No codesign or winning_kl internalize loss — pure RL at Langevin trigger positions with IS correction.

**Key differences from v12**: uses `lan_grpo_correct_is=True` (importance-sampling correction on Langevin-blended log-probs) and raises `delta_var_threshold` from 5.0 → 9.0 (tighter trigger, ~47% trig_frac at convergence). No internalize mode beyond the built-in IS-corrected lan_grpo gradient.

**Training behavior**:
- Entropy: fast collapse 0.32 → 0.18 (steps 1–50), then stable ~0.17–0.18 nats. No entropy explosion.
- trig_frac: converges to ~47–49% of positions at threshold=9.0 (roughly half the response).
- pg_loss: ~0.19–0.23 (healthy, positive gradient).
- ppo_kl: −5.4 to −5.8 (model systematically drifted from reference — expected with kl_loss_coef=0.005 on a long run).
- kl_loss: stable ~0.037.

**Val metrics**:

| Step | MATH mean@16 | MATH best@16 | MATH maj@16 | AIME mean@16 | AIME best@16 | AIME maj@16 |
|------|-------------|-------------|-------------|-------------|-------------|-------------|
| 220 (peak MATH) | **61.7%** | **83.5%** | **70.2%** | 2.5% | 10.6% | 5.5% |
| 195 (peak AIME best) | 60.0% | 82.2% | 68.6% | 2.8% | **13.7%** | 5.2% |
| 245 (peak AIME mean/maj) | 60.0% | 82.5% | 68.2% | **3.2%** | 13.0% | **6.5%** |

**Summary**: v13's MATH peak (61.7%) matches WGRPO-v6 and beats all prior PIVOT variants. AIME mean@16 (3.18%) is comparable to v12 (3.26%) but below v4b's peak (4.0%). MATH maj@16 of 70.2% is the best across all runs to date (v4b=68.2%, v10a=69.9%, WGRPO=68.6%).

---

### PIVOT-v14 (2026-04-25, stopped at step 215, seeded from GRPO-200)

**Config**: `pivot_version=4`, `delta_var_threshold=9.0`, `langevin_rollout=True`, `lan_grpo_coeff=1.0`, `lan_grpo_correct_is=True`, `fork_advantage_alpha=0.5`, `entropy_cap=0.8`, `n_rollouts_per_prompt=8`, `kl_loss_coef=0.005`. No codesign loss. Fork-weighted advantage amplifies the GRPO advantage 1.5× at trigger positions (`A_t_fork = A_i * 1.5`).

**Key differences from v13**: adds `fork_advantage_alpha=0.5` — GRPO advantages at trigger positions (delta_var > threshold) are scaled by `(1 + 0.5) = 1.5×`. IS correction unchanged.

**Training behavior**:
- Entropy: collapse 0.32 → 0.17 (steps 1–65), stable ~0.16–0.18 nats thereafter. Slightly faster collapse than v13b.
- MATH mean@16: rapid ascent to ~59% by step 55, then slow climb to 60.3% by step 215.
- AIME: noisy but trending upward; early maj@16 peak (5.95% @ step 115) notably higher than v13/v13b at the same steps.

**Val metrics**:

| Step | MATH mean@16 | MATH best@16 | MATH maj@16 | AIME mean@16 | AIME best@16 | AIME maj@16 |
|------|-------------|-------------|-------------|-------------|-------------|-------------|
| 115 (peak AIME maj) | 59.4% | 82.2% | 67.8% | 2.97% | 12.2% | **5.95%** |
| 135 (peak AIME best) | 59.6% | 82.8% | 68.5% | 2.45% | **12.9%** | 4.42% |
| 160 (peak MATH best) | 59.9% | **83.2%** | 68.8% | 2.58% | 11.5% | 4.87% |
| 175 (peak AIME mean) | 59.3% | 82.1% | 67.0% | **3.23%** | 12.3% | 5.79% |
| 190 (peak MATH maj) | 60.1% | 83.0% | **69.2%** | 2.73% | 10.5% | 5.59% |
| 215 (peak MATH mean, stopped) | **60.3%** | 81.4% | 68.9% | 2.11% | 11.2% | 4.02% |

**Key results**:
- Peak MATH mean@16: **60.3% @ step 215**
- Peak MATH best@16: **83.2% @ step 160**
- Peak MATH maj@16: **69.2% @ step 190**
- Peak AIME mean@16: **3.23% @ step 175**
- Peak AIME best@16: **12.9% @ step 135**
- Peak AIME maj@16: **5.95% @ step 115**

**vs v13**:

| Metric | v13 | v14 | Δ |
|--------|-----|-----|---|
| MATH mean@16 peak | **61.7%** | 60.3% | −1.4pp |
| MATH maj@16 peak | **70.2%** | 69.2% | −1.0pp |
| AIME mean@16 peak | **3.18%** | 3.23% | +0.05pp |
| AIME maj@16 peak | **6.52%** | 5.95% | −0.57pp |
| AIME maj@16 @ step 90 | 4.88% | **5.70%** | +0.82pp |

Fork-weighted advantage (alpha=0.5) did not improve MATH or AIME over v13. MATH peaks are 1-1.4pp below v13's best (which broke 61% and 70% maj). The early AIME maj@16 trajectory (steps 75-100) is modestly better than v13/v13b, suggesting fork weighting does concentrate gradient at useful positions early in training, but the effect dissipates as entropy compresses.

**Verdict**: No net gain over v13. The fork advantage amplification is theoretically motivated but the magnitude (alpha=0.5) is either too small or the capacity ceiling masks any benefit. Stopped at step 215.

---

### PIVOT-v13c (2026-04-25, stopped at step 133, seeded from GRPO-200)

**Config**: `pivot_version=4`, `delta_var_threshold=9.0`, `trig_percentile=90` (adaptive threshold), `langevin_rollout=True`, `lan_grpo_coeff=1.0`, `lan_grpo_correct_is=True`, `entropy_cap=0.8`, `n_rollouts_per_prompt=8`, `kl_loss_coef=0.005`. No `lan_grpo_correct_mu`. No codesign loss.

**Key differences from v13**: adds `trig_percentile=90` (adaptive threshold that keeps trig_frac ≈ 10% throughout training, vs the fixed `delta_var_threshold=9.0` which caused trig_frac to drift from ~10% to ~40% as delta_var shrinks with training). Both rollout and actor buffer the last 2000 delta_var samples to compute the running 90th-percentile threshold; fallback to `delta_var_threshold=9.0` for the first 200 samples.

**Training behavior**:
- Response length: 837 → 954 tokens (step 1 → 133), steady growth, no plateau.
- trig_frac: held near 10% throughout by adaptive threshold (vs v13's 47-49%).
- Entropy: stable, no collapse or explosion.

**Val metrics**:

| Step | MATH mean@16 | MATH best@16 | MATH maj@16 | AIME mean@16 | AIME best@16 | AIME maj@16 |
|------|-------------|-------------|-------------|-------------|-------------|-------------|
| 115 (peak AIME mean+maj) | 59.6% | 81.1% | 68.1% | **2.81%** | 11.3% | **5.79%** |
| 95 (peak AIME best) | 59.7% | 81.9% | 67.9% | 2.40% | **12.3%** | 4.18% |
| 125 (peak M500 mean) | **60.3%** | **83.2%** | **68.9%** | 2.45% | 10.6% | 5.05% |
| 130 (last val) | 59.6% | 82.2% | 67.3% | 2.08% | 11.8% | 4.11% |

**Key results** (peaks, 133 steps):
- Peak MATH mean@16: **60.3% @ step 125**
- Peak MATH best@16: **83.2% @ step 125**
- Peak MATH maj@16: **68.9% @ step 125**
- Peak AIME mean@16: **2.81% @ step 115**
- Peak AIME best@16: **12.3% @ step 95**
- Peak AIME maj@16: **5.79% @ step 115**
- mean/best ratio (AIME): 0.176–0.253 across val checkpoints

**vs v13 at comparable steps (~130)**:
- MATH mean: comparable (v13c 60.3% vs v13 ~60.0% at step 130)
- AIME mean: v13c 2.81% peak vs v13 ~3.0% at step 130 — slightly below, but v13 ran longer
- AIME maj: v13c 5.79% peak vs v13 ~5.5% at comparable step — v13c marginally better

**Context — training-inference Langevin mismatch**: During training, group-mean Langevin provides a directional push toward correct tokens; the IS denominator (`log p_lan_old`) is much larger than `log π_old` at correct-token positions, so the PPO clipping ratio `π_current / p_lan_old << 1`. When this ratio falls below `1-ε`, the PPO clip zeroes the gradient at exactly the positions where learning signal is most needed. This is the root cause of PIVOT's lower mean/best ratio (~0.20–0.25) vs High-Ent-GRPO (~0.33) at comparable training depth. v13c does not address this; v15 and v15b are designed to fix it.

**Verdict**: Adaptive threshold (`trig_percentile=90`) successfully stabilises trig_frac at ~10% and produces healthy training. MATH performance (60.3%) matches v13 at comparable steps. AIME mean/best ratio is the key gap vs High-Ent-GRPO. Stopped at step 133.

---

### PIVOT-v15 (2026-04-25 to 2026-04-26, stopped at step 128, seeded from GRPO-200)

**Config**: v13 base + `lan_grpo_direct_coeff=1.0` + `lan_grpo_correct_mu=True` + `trig_percentile=90`. Motivation: in v13 the IS-corrected denominator `p_lan_old` inflated the reference at heavily-perturbed positions, pushing `r = π/p_lan_old << 1` into the PPO lower clip region and zeroing the gradient exactly where Langevin changed the distribution most. v15 adds a parallel direct GRPO term using the original `π_old` as denominator (ratio ≈ 1, never clipped) to ensure fork positions always receive a gradient.

**Key differences from v13**: `trig_percentile=90` fires only on the top-10% delta_var positions (vs v13's absolute threshold hitting ~47%). The direct GRPO term adds `λ·adv·(log π - log π_old)` at those positions.

**Training behavior**:
- trig_frac: ~9–10% (top-10% percentile trigger working as designed — much tighter than v13's 47%)
- pg_loss: ~0.04 (much lower than v13's ~0.21 — 10% coverage vs 47% means less total gradient mass)
- ppo_kl: ~−1.4 to −1.5 (much less policy drift from ref than v13's −5.5 — consistent with tighter trigger)
- entropy: stable ~0.16–0.17 nats

**Val metrics (stopped at step 128)**:

| Step | MATH mean@16 | MATH best@16 | MATH maj@16 | AIME mean@16 | AIME best@16 | AIME maj@16 |
|------|-------------|-------------|-------------|-------------|-------------|-------------|
| 80 (peak MATH) | 60.0% | 82.1% | 69.0% | 2.0% | 10.2% | 3.9% |
| 95 (peak AIME mean) | 59.7% | 83.5% | 68.8% | **2.7%** | 13.3% | 5.2% |
| 105 (peak AIME best) | 58.3% | 82.0% | 67.2% | 2.6% | **13.9%** | 4.7% |

**Summary**: v15 underperforms v13 at equal steps — MATH mean@16 peaks at 60.0% vs v13's 61.7%; AIME mean@16 peaks at 2.71% vs v13's 3.18%. The tighter 10% trigger (via `trig_percentile=90`) reduces policy drift but also reduces gradient signal strength. The direct GRPO term did not compensate. Stopped early at step 128 — trajectory showed no acceleration over v13.

---

### PIVOT-v16b (2026-04-26 to 2026-04-27, stopped at step 255, seeded from GRPO-200)

**Config**: IS addon (`lan_grpo_is_addon_coeff=1.0`) + entropy top-ratio masking (`entropy_top_ratio=0.2`, `entropy_coeff=0`). The IS addon adds a clipped-PPO term using `p_lan_old` as denominator on top of the base GRPO loss (base loss unpatched). The 80/20 masking restricts PG updates to the top-20% highest-entropy tokens per step — adapted from Wang et al. NeurIPS 2025.

**Key behaviors**:
- `high_ent_token_frac`: exactly 0.200 every step — mask working as specified
- Entropy: rose from 0.36 → 0.47 (steps 1–20), then declined to 0.17–0.19. Slower collapse than v13 (which hit 0.17 by step 50); v16b maintained ~0.19 at step 255
- ppo_kl ≈ 0 throughout — the 80/20 masking keeps per-step policy updates very conservative; policy barely drifts in KL terms
- kl_loss: stable ~0.10–0.12 (higher than v13's 0.037 — consistent with more cautious updates accumulating over more steps)
- trig_frac: ~9–10% (corrected after crash fix; IS addon applies to top-10% delta_var positions)
- Crashed at step 10 on first run due to inhomogeneous shape in `reduce_metrics` for `pivot/lan_grpo_is_addon_loss` / `pivot/lan_grpo_is_clipfrac`. Fix: always log `float("nan")` defaults before the conditional block.

**Val metrics:**

| Step | MATH mean@16 | MATH best@16 | MATH maj@16 | AIME mean@16 | AIME best@16 | AIME maj@16 |
|------|-------------|-------------|-------------|-------------|-------------|-------------|
| 180 (peak AIME best) | 60.0% | 82.2% | 68.1% | 3.07% | **14.40%** | 5.83% |
| 235 (peak MATH best) | 60.4% | **83.5%** | 68.7% | 2.76% | 13.50% | 5.15% |
| 240 (peak MATH mean) | **61.2%** | 82.1% | 68.8% | 2.94% | 13.67% | 5.47% |
| 245 (peak AIME mean + maj) | 60.3% | 82.5% | 68.0% | **3.10%** | 13.83% | **6.00%** |

**Summary**: AIME best@16 of **14.4%** at step 180 is the highest across all PIVOT variants (beats v4b's 14.3%). MATH mean@16 peaks at 61.2% (matches v13). AIME mean@16 (3.10%) is comparable to v13 (3.18%) and slightly below v15b (3.46%). The 80/20 entropy masking delivered on its premise — entropy stayed higher for longer — but the final AIME mean ceiling matches rather than exceeds previous variants.


---

### PIVOT-v15b (blended IS denominator, 2026-04-26, stopped at step 200, seeded from GRPO-200)

**Config**: v13d + `lan_grpo_denom_blend=0.5`. Single change: replaces v13d's full IS denominator with a geometric blend:
```
log_denom[trig] = (1-w)·log_p_lan_old + w·log_π_old    w=0.5
                = log(√(p_lan_old · π_old))
```
All other hyperparameters identical to v13d: `lan_grpo_correct_is=True`, `lan_grpo_correct_mu=True`, `trig_percentile=90` (top-10% ΔVar positions fire, ~10% trig_frac), `entropy_cap=0.8`, `langevin_K=1`, `langevin_eta=0.1`, `langevin_sigma=0.01`, `langevin_top_k=20`. Single-term loss — no double-counting (contrast with v15 which added a parallel direct term). Seeded from GRPO step-200 checkpoint.

#### Val metrics (all steps)

| Step | MATH mean@16 | MATH best@16 | MATH maj@16 | AIME mean@16 | AIME best@16 | AIME maj@16 |
|------|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| 5 | 20.7% | 52.0% | 21.7% | 0.44% | 3.31% | 0.04% |
| 10 | 35.1% | 68.7% | 41.3% | 0.62% | 4.18% | 0.52% |
| 15 | 40.2% | 72.5% | 48.3% | 0.52% | 3.94% | 0.28% |
| 20 | 48.3% | 78.8% | 58.6% | 1.07% | 5.85% | 1.62% |
| 25 | 51.2% | 78.8% | 61.7% | 1.64% | 8.47% | 2.69% |
| 30 | 52.9% | 80.8% | 62.2% | 1.77% | 9.00% | 3.07% |
| 35 | 54.8% | 81.3% | 65.4% | 1.98% | 8.39% | 3.75% |
| 40 | 56.6% | 82.0% | 66.7% | 2.16% | 10.89% | 4.09% |
| 45 | 57.4% | 81.8% | 66.8% | 2.32% | 11.24% | 4.27% |
| 50 | 57.8% | 82.4% | 68.0% | 2.06% | 10.57% | 3.68% |
| 55 | 58.2% | 82.6% | 67.4% | 2.45% | 13.68% | 4.25% |
| 60 | 58.9% | 82.6% | 67.8% | 2.53% | 12.16% | 5.00% |
| 65 | 58.5% | 80.8% | 67.0% | 2.68% | 10.85% | 4.88% |
| 70 | 59.4% | 82.3% | 68.2% | 2.16% | 10.00% | 4.18% |
| 75 | 58.8% | 82.1% | 68.1% | 2.32% | 11.93% | 4.40% |
| 80 | 59.4% | 82.5% | 68.2% | 2.71% | 10.88% | 5.14% |
| 85 | 59.6% | 82.4% | 68.3% | 2.73% | 12.14% | 5.21% |
| 90 | **60.6%** | 82.5% | **69.1%** | 2.60% | 12.58% | 5.02% |
| 95 | 59.2% | 82.2% | 67.4% | 2.53% | 10.53% | 5.08% |
| 100 | 60.1% | 82.1% | 68.2% | 2.60% | 11.65% | 4.60% |
| 105 | 59.6% | 82.3% | 68.2% | 2.76% | 10.68% | 5.04% |
| 110 | 59.7% | 82.8% | 67.5% | 2.58% | 10.25% | 5.06% |
| 115 | 59.4% | 81.1% | 67.7% | 2.81% | 11.91% | 5.69% |
| 120 | 59.2% | 81.9% | 67.5% | 2.53% | 10.98% | 4.78% |
| 125 | 58.8% | 81.8% | 66.7% | 2.86% | 9.96% | 5.46% |
| 130 | 60.1% | 82.7% | 68.5% | 3.05% | 12.10% | 5.94% |
| 135 | 59.8% | 82.0% | 67.8% | 2.89% | 11.25% | 5.41% |
| 140 | 59.6% | 82.0% | 66.8% | 3.31% | 12.53% | 5.92% |
| 145 | 59.3% | 80.9% | 67.8% | 3.10% | 11.27% | 6.18% |
| 150 | 59.8% | 82.2% | 66.9% | 2.89% | 11.12% | 5.55% |
| 155 | 59.5% | 81.2% | 66.7% | 2.76% | 11.93% | 5.06% |
| 160 | 59.2% | 80.9% | 67.0% | 2.86% | 11.76% | 5.09% |
| 165 | 59.5% | 81.3% | 67.2% | 2.79% | 12.53% | 5.03% |
| 170 | 59.6% | 82.2% | 67.2% | 2.71% | 10.21% | 5.10% |
| 175 | 59.2% | 82.0% | 67.3% | 2.99% | 11.91% | 5.49% |
| **180** | 59.8% | 81.7% | 66.7% | **3.46%** | 12.52% | 6.36% |
| 185 | 59.6% | 81.1% | 66.6% | 2.63% | 11.33% | 4.82% |
| **190** | 60.0% | 82.3% | 67.6% | 3.26% | **13.41%** | 6.25% |
| **195** | 60.1% | 81.7% | **68.1%** | 3.23% | 12.71% | **6.41%** |
| 200 | 59.8% | 81.2% | 67.1% | 3.18% | 11.81% | 5.75% |

#### Training dynamics

| Step | score | entropy | pg_loss | clip↑ | clip↓ | kl_loss | trig_frac | resp_len |
|------|-------|---------|---------|-------|-------|---------|-----------|---------|
| 1 | −0.989 | 0.295 | 0.0058 | 0.04% | 0.27% | 0.0004 | 10.1% | 850 |
| 5 | — (val) | — | — | — | — | — | — | — |
| 10 | −0.960 | 0.364 | 0.0103 | 0.18% | 0.65% | 0.0131 | 9.6% | 808 |
| 20 | −0.911 | 0.319 | 0.0240 | 0.42% | 1.53% | 0.0246 | 9.6% | 789 |
| 30 | −0.878 | 0.236 | 0.0347 | 0.63% | 1.87% | 0.0238 | 9.8% | 877 |
| 40 | −0.875 | 0.209 | 0.0339 | 0.56% | 2.02% | 0.0289 | 9.4% | 864 |
| 50 | −0.846 | 0.188 | 0.0385 | 0.77% | 2.29% | 0.0280 | 9.7% | 902 |
| 60 | −0.837 | 0.181 | 0.0424 | 0.85% | 2.25% | 0.0308 | 10.1% | 940 |
| 70 | −0.839 | 0.175 | 0.0425 | 0.77% | 2.17% | 0.0319 | 9.8% | 946 |
| 80 | −0.831 | 0.173 | 0.0469 | 0.85% | 2.62% | 0.0331 | 10.3% | 961 |
| 90 | −0.829 | 0.159 | 0.0390 | 0.89% | 2.15% | 0.0343 | 9.9% | 977 |
| 100 | −0.834 | 0.171 | 0.0388 | 0.76% | 2.10% | 0.0363 | 9.6% | 920 |
| 110 | −0.815 | 0.160 | 0.0408 | 0.85% | 2.11% | 0.0377 | 9.5% | 950 |
| 120 | −0.833 | 0.159 | 0.0406 | 0.84% | 1.90% | 0.0374 | 9.6% | 948 |
| 130 | −0.809 | 0.168 | 0.0420 | 0.86% | 2.25% | 0.0378 | 9.8% | 941 |
| 140 | −0.807 | 0.162 | 0.0446 | 0.95% | 2.40% | 0.0377 | 9.5% | 953 |
| 150 | −0.802 | 0.145 | 0.0423 | 0.84% | 2.21% | 0.0407 | 10.2% | 970 |
| 160 | −0.799 | 0.148 | 0.0408 | 0.90% | 2.21% | 0.0428 | 9.4% | 943 |
| 170 | −0.803 | 0.148 | 0.0439 | 0.92% | 2.29% | 0.0423 | 10.0% | 986 |
| 180 | −0.792 | 0.146 | 0.0494 | 0.97% | 2.47% | 0.0430 | 9.9% | 968 |
| 190 | −0.811 | 0.162 | 0.0432 | 0.86% | 2.12% | 0.0405 | 9.7% | 969 |
| 200 | −0.825 | 0.139 | 0.0453 | 0.92% | 2.33% | 0.0419 | 10.4% | 1036 |

#### Entropy trajectory

| Step | entropy (nats) |
|------|---------------|
| 1 | 0.295 |
| 8 | 0.367 (peak) |
| 20 | 0.319 |
| 30 | 0.236 |
| 50 | 0.188 |
| 80 | 0.173 |
| 100 | 0.171 |
| 130 | 0.168 |
| 160 | 0.148 |
| 180 | 0.146 |
| 200 | 0.139 |

#### Training phases

**Phase 1 — Immediate breakthrough (steps 1–20, seeded from GRPO-200):**
MATH leaps 0% → 48% in 20 steps. AIME 0% → 1.07%. Entropy peaks at 0.367 nats (step 8) — the model briefly expands its distribution on contact with the new loss signal, then begins compressing. KL loss rises from near-zero to 0.025.

**Phase 2 — Rapid climb (steps 20–90):**
MATH 48% → 61% (peak at step 90). AIME 1.07% → 2.73% (step 85). Entropy compresses 0.32 → 0.16 nats as the model concentrates probability at high-reward tokens. trig_frac stabilises at ~9.5–10.5% — the `trig_percentile=90` adaptive threshold holds this constant by design. clip↓ (lower PPO clip fraction) stabilises at ~2.0–2.6%, consistently 2–3× the upper clip — evidence that the blended IS ratio still trends slightly below 1 at trigger positions, but far less severely than v13d's full denominator would produce.

**Phase 3 — MATH plateau, AIME late climbing (steps 90–200):**
MATH plateaus at 59–60.5% (same ceiling as all variants at this model/data scale). AIME continues noisy improvement: 2.60% (step 100) → 3.46% (step 180, peak). KL loss drifts 0.034 → 0.043 (slow, manageable). Entropy continues slow compression 0.17 → 0.14 nats — no collapse event. Response length grows 920 → 1036 tokens; length clip rises from 0.9% to 2.8% by step 200 (mild).

**Key v13d comparison**: v13d's clip↓ at trigger positions would be catastrophic (ratio `π_current/p_lan_old << 1` → near-100% lower clip at triggers). v15b's blend halves the IS displacement → clip↓ stays at 2–3% → gradient flows at fork positions throughout training.

#### Key results

- **Best AIME mean@16 of all PIVOT variants**: 3.46% @ step 180 (eff 380 from scratch)
- **AIME best@16 peak**: 13.41% @ step 190
- **AIME maj@16 peak**: 6.41% @ step 195
- **MATH mean@16 peak**: 60.6% @ step 90 (early; plateaus thereafter)
- **vs v13d** (same config minus blend): AIME mean 2.76% → 3.46% (+25%); AIME maj 5.66% → 6.41% (+13%)
- **vs v16** (additive IS addon, same trigger): AIME mean 2.92% → 3.46% (+19%)
- **vs High-Ent-GRPO** (225 steps): 3.39% → 3.46% (marginal edge, fewer steps)

#### Langevin noise ablation (inference-time, v15b step 180)

| Config | AIME mean@16 | Interpretation |
|--------|:-----------:|----------------|
| No Langevin (pure π) | 2.97% | baseline |
| σ=0, η=0.1 (gradient only) | 2.81% | ∇H alone hurts |
| η=0, σ=0.01 (noise only) | 3.07% | noise alone helps slightly |
| σ=0.1, η=0.1 (more noise) | 3.26% | more noise, diminishing returns |
| **σ=0.01, η=0.1 (baseline)** | **3.57%** | gradient + noise synergistic |
| thr=6.5 (fires ~top-35%) | 3.23% | too many positions |
| thr=8.0 (fires ~top-20%) | 3.57% | sweet spot |
| thr=9.5 (fires ~top-5%) | 2.84% | too few positions |
| K=2 (two Langevin steps) | 3.20% | overshoots |
| η=0.3 (stronger gradient push) | 3.15% | overshoots |

Pure gradient Langevin (σ=0) is worse than no Langevin. The noise term is load-bearing: it diffuses `p_lan_old` around its ∇H attractor, lowering the peak of the IS denominator and keeping PPO ratios out of the clip region. The gradient alone creates a maximally adversarial IS problem; small noise regularises the IS correction numerically. See [IS Correction Geometry](#is-correction-geometry-why-v15b-works) for the theoretical analysis.

---

## IS Correction Geometry: Why v15b Works

### The Goldilocks Structure

All IS variants face the same tension: correct for the Langevin sampling bias vs stay within PPO's trust region. The blend parameter `w` in v15b's denominator traces a Goldilocks curve:

| Variant | w | Denominator | Ratio at trigger | Effect |
|---------|---|-------------|-----------------|--------|
| v4b | 1.0 | `π_old` | `π_current / π_old ≈ 1` | Unclipped, but biased (ignores Langevin shift) |
| v13d | 0.0 | `p_lan_old` | `π_current / p_lan_old << 1` | Unbiased, but clipped to zero at fork positions |
| **v15b** | **0.5** | `√(p_lan_old · π_old)` | closer to 1 | Partial IS, rarely clipped |

v13d fixes the bias but destroys the signal: the PPO lower clip (`r < 1-ε`) fires hardest at exactly the positions where Langevin moved the distribution most — the fork positions where learning signal is most needed. v15b's blend keeps the ratio within the trust region while still acknowledging the distributional shift.

### Geometric Mean as Bhattacharyya Interpolation

The blended denominator `√(p_lan · π_old)` is the geometric mean of the two distributions. In information geometry this is the midpoint on the statistical manifold under the Fisher metric — equivalently, the distribution that minimises the sum of squared Hellinger distances to both endpoints:

```
√(p_lan · π) = argmin_q  [H²(q, p_lan) + H²(q, π)]
```

where `H²(p,q) = ½ · ∫(√p - √q)²` is the squared Hellinger distance. This is principled, not arbitrary: it is the minimum-variance unbiased interpolation between the two reference distributions.

### Effective Trust Region Widening

For small Langevin perturbations `p_lan ≈ π(1 + δ)`, the blend approximation gives:

```
√(p_lan · π) ≈ π · (p_lan/π)^0.5 ≈ π · (1 + δ/2)
```

The PPO ratio in v15b becomes `π_current / [π · (1 + δ/2)]`, compared with v13d's `π_current / [π · (1 + δ)]`. Taking the square root approximately halves the IS displacement at each trigger position. Since the PPO lower clip fires when `r < 1-ε`, and v13d's denominator inflates the denominator by `(1+δ)` vs v15b's `(1+δ/2)`, **v15b's effective trust region at trigger positions is roughly twice as wide as v13d's for the same clip parameter ε**. This is exactly the expansion needed to keep fork positions unclipped while Langevin's perturbation magnitude is moderate (η=0.1).

### Noise as IS Regularization

The noise ablation reveals a deeper role for σ: pure gradient Langevin (σ=0) is strictly worse than no Langevin (2.81% vs 2.97%). The deterministic ∇H push creates a point-mass IS problem — `p_lan_old` is concentrated exactly where ∇H pointed, making the IS ratio `π_current / p_lan_old` near-zero at those tokens. Small noise (σ=0.01) diffuses `p_lan_old` around its ∇H attractor, lowering the peak of the IS denominator and keeping ratios out of the clip region. The noise is not only exploration diversity — it is numerically stabilising the IS correction. This explains the synergy between ∇H and ε that neither term achieves alone.

### Why v16's Clean Separation Fails

v16 separates the objective:
```
total_loss = GRPO(π_current / π_old)  +  λ · clip-PPO(π_current / p_lan_old)
```
The main GRPO term has ratio ≈ 1 (trust region intact). But the IS addon's ratio `π_current / p_lan_old` faces the same clipping problem as v13d — at fork positions where Langevin moved the distribution, the IS addon clips out. When the IS addon clips, you fall back to plain GRPO at that position, which ignores the sampling bias. The clean architectural separation preserves the main loss's trust region but doesn't solve the IS correction's clipping problem; v15b's single blended term is more effective precisely because it changes the geometry of the denominator rather than adding a second term with the same problematic ratio.

---

## Langevin-Consistent GRPO (v12 design)

*Designed 2026-04-23. Implementation pending.*

### The Off-Policy Mismatch Problem

In all current PIVOT variants, the standard GRPO policy gradient at trigger positions is:

```
adv · log π_current(a_t^lan)
```

But `a_t^lan` was sampled from `p_lan_rollout` (Langevin applied to `π_old`), not from `π`. This creates an **implicit SFT channel**: when `adv > 0`, the gradient pushes `π_current` to directly reproduce Langevin's token choice — behavioral cloning toward p_lan, reward-conditioned but still p_lan-shaped. This is the same SFT-like character as the explicit KL internalization loss, just more subtle.

### The Fix: Langevin-Consistent Numerator

At trigger positions, replace `log π_current(a_t^lan)` with `log p_lan_current(a_t^lan)`, where `p_lan_current` is the Langevin step applied to the **current model's logits** at position t:

| Position | Token source | Numerator | Denominator |
|---|---|---|---|
| Non-trigger | `π_old` | `log π_current(a_t)` | `log π_old(a_t)` |
| Trigger | `p_lan_rollout` | `log p_lan_current(a_t^lan)` | `log π_old(a_t^lan)` |

The denominator stays `π_old` throughout — it serves as the **trust region anchor** for the PPO clip, preventing the current policy from drifting too far from the rollout policy. No change needed there.

### Why This Is the Right Numerator

With `log p_lan_current` as the numerator:
- Gradient flows through the Langevin step back into the model weights
- Trains `π_current` to be a **good initialization for Langevin**: after the Langevin step, the distribution naturally concentrates on the rewarding token
- The model is NOT learning to directly mimic `a_t^lan` — it is learning to be positioned such that Langevin finds the right answer
- No reference distribution used as a target: p_lan is recomputed from the current model, not a frozen external distribution

### Self-Calibrating Property

As the model improves at trigger position t via this gradient:
- `π_current` shifts so `p_lan_current` concentrates on the rewarding token
- delta_var at t drops (less rollout disagreement once π is more consistent)
- Langevin stops firing at t — the model has internalized that fork
- Langevin migrates to the next unresolved fork

This is "teaching the model when to fire Langevin" through the reward signal alone, without any explicit curriculum or reference distribution.

### Engineering Requirements

1. **Store `μ_win`** (the winning mean logits used as Langevin target) per trigger position per group during rollout — must be added to the rollout metadata passed back to the actor.
2. **Differentiable Langevin step**: `p_lan = (1-eta)*softmax(logits) + eta*softmax(μ_win) + sigma*noise`. Gradient through `softmax(logits)` is straightforward; `μ_win` is treated as a constant (no-grad).
3. **No change to denominator**: `old_log_prob` stays as computed by the current actor log-prob pass.

### Is This Important?

Yes. Two reasons:

**Theoretical**: it closes the loop on GRPO purity. The core claim of PIVOT is that fork-position interventions are reward-driven. The implicit SFT channel contradicts that claim — it makes trigger positions behave like supervised positions regardless of the advantage sign. The fix makes the claim rigorous.

**Empirical**: the generalization gap observed in v10a/v10b (reward improving, AIME plateauing) is consistent with the implicit SFT channel causing training-distribution memorization at trigger positions. v4b (no internalization, no implicit SFT) shows better AIME generalization than v6c (with internalization). The Langevin-consistent numerator may recover v4b's generalization while adding the directed gradient of internalization.

---

## Open Questions / Roadmap

### Does Phase 2 help?

The critical ablation is PIVOT with Phase 2 disabled (loss gating only) vs full PIVOT. If the co-design loop matters, full PIVOT should outperform on AIME. If Phase 1 captures most of the benefit, Phase 2 adds little.

### Does either phase help vs W-GRPO v6?

PIVOT uses a different walk than W-GRPO: temporal (generation time) vs depth (layer depth). The temporal walk measures whether the current token's choice changes the causal structure — more directly tied to decisiveness. But v6's differential walk is reward-contrastive and may be stronger for that reason. Need direct comparison at 300-400 steps.

### Threshold sensitivity

`langevin_threshold=0.3` and `pivot_threshold=0.3` are inherited from design intuition, not calibrated. The right threshold is the one that fires at true decision points while not misfiring on connective tissue. Possible calibration: target 15-20% trigger rate (current: 32%).

### Block size

Phase 1 uses `block_size=32` (inherited from W-GRPO); Phase 2 uses `block_size=8` (finer, since we're making per-step decisions during decode). The right Phase 1 block_size for temporal walk may differ from W-GRPO's spatial walk — temporal walk is already token-sequential, so finer blocking may work better.

### Positional bifurcation

If the positional bifurcation hypothesis (wgrpo.md) is correct, the temporal walk is doing useful work as a *position selector*, not just as a semantic weight. The clearest test: replace the walk trigger with a uniform "middle of response" prior and check if it degrades. PIVOT's temporal walk should be strictly better at finding actual fork positions.

### Phase 2 signal quality

The current Phase 2 uses a simplified temporal walk inside vLLM — single-layer Q/K (not layer-averaged), simplified block pooling. The Phase 1 walk is more carefully computed (full layer averaging, exact block pooling). A future variant could align the two signal computations more closely.

### Rollout Langevin design: group-mean is empirically better

**Finding (2026-04-22).** v4b (group-mean rollout Langevin) outperforms v6c (entropy-gradient rollout Langevin + winning_kl internalization) by a large margin: AIME mean@16 peak 4.0% vs 2.8%, Math500 61.0% vs 59.4%.

The two rollout Langevin designs are:

| Design | Formula | Signal |
|--------|---------|--------|
| Entropy gradient (v2/v3 fallback) | `logits += eta * grad_H + sigma * eps` where `grad_H = -p*(log_p+H)` | Per-token entropy maximization, no cross-rollout info |
| **Group-mean (v4b, recommended)** | `logits += eta * (group_mean - logits) + noise_scale * eps` | Pulls each rollout toward the concurrent group's mean; cross-rollout signal available immediately |

The group-mean approach is strictly more informative: it uses the actual distribution of concurrent rollouts at this position, whereas the entropy gradient has no knowledge of what other rollouts are doing.

**Consistency with the internalization loss.** The internalization loss (winning_kl, split_kl, contrastive_kl, bipolar variants) also operates on the group's logit distribution — it computes `cm` from cross-rollout aggregates. The group-mean rollout Langevin is structurally aligned with these losses: both say "use the group's aggregate logit structure to guide this rollout." The entropy-gradient rollout says "explore independently" and is incoherent with the group-level loss signal.

**Recommendation for future variants**: always use `pivot_version=4` (group-mean rollout Langevin). The entropy-gradient fallback (triggered when `pivot_version != 4` or `_group_mean_logits is None`) is a weaker baseline. v10a and v10b already use `pivot_version=4`.

**Remaining gap.** The group-mean rollout is unweighted (all n rollouts contribute equally). The ideal rollout Langevin would weight rollouts by their eventual advantages — pulling each rollout toward the winning direction rather than the neutral group mean. This is impossible in a single pass (advantages require completed rollouts), but is worth exploring via two-pass rollout or lagged-batch approximations where feasible.

---

## Experiment Plan (Paper)

### Framing

PIVOT's contribution is a **plug-in module** that improves any GRPO-family method by adding variance-triggered Langevin correction at inference time, paired with an internalization loss that caches the correction into the weights. The paper story is:

> PIVOT + DAPO > DAPO — PIVOT improves the strongest existing GRPO baseline, demonstrating orthogonality with DAPO's gradient-level improvements.

DAPO's contributions (Clip-Higher, dynamic sampling, token-level PG normalization) operate on the policy gradient computation. PIVOT's contributions operate on *where* in the sequence to intervene and *how* to guide the distribution there. They are non-overlapping.

---

### Baselines

| Baseline | Role | Reference |
|----------|------|-----------|
| **GRPO** | Direct ancestor / minimum bar | DeepSeekMath, arXiv 2402.03300 |
| **DAPO** | Strongest GRPO variant, entropy control | ByteDance, arXiv 2503.14476 |
| **VinePPO** | Best token-level credit assignment competitor | ICML 2025, arXiv 2410.01679 |
| **COLD Decoding** | Langevin-for-generation reference (cite, not run) | NeurIPS 2022, arXiv 2202.11705 |
| **ReST-MCTS\*** | Inference-time + training co-design reference (cite) | NeurIPS 2024, arXiv 2406.03816 |
| **Qwen2.5-Math** | Pretrained model baseline (zero-shot floor) | Alibaba, arXiv 2409.12122 |

---

### Priority 1 — Existential (run first)

**Matched-compute GRPO comparison.** Without this, all PIVOT gains could be explained by more training steps.

| Run | Config | Purpose |
|-----|--------|---------|
| GRPO continued | GRPO from step-200, run 70–100 more steps | Matched-compute baseline |
| PIVOT-v6c-nodecay | Already running | PIVOT at same compute |

If PIVOT-v6c-nodecay (step 60–100 from GRPO-200) > GRPO (step 60–100 more from GRPO-200) on AIME, the core claim holds.

---

### Priority 2 — Core Paper Experiments

Full experiment matrix on Qwen2.5 (3B and 7B+):

| Method | MATH500 | AIME acc@16 | AIME maj@16 |
|--------|---------|-------------|-------------|
| Qwen2.5 base (no RL) | — | — | — |
| GRPO | — | — | — |
| DAPO | — | — | — |
| VinePPO | — | — | — |
| GRPO + PIVOT | — | — | — |
| **DAPO + PIVOT** | — | — | — |

**Implementation of DAPO + PIVOT**: enable DAPO's flags (`clip_higher`, `dynamic_sampling`, `token_level_pg_loss`) alongside PIVOT's (`langevin_rollout=True`, `internalize_mode=winning_kl`, `internalize_coeff=0.1`). No code changes needed — config-only.

Run script: `run_qwen2_5_3b_dapo_pivot_from_scratch.sh` (to be created when cluster is free).

---

### Priority 3 — Ablations

These validate the theory claims and answer reviewer questions.

| Ablation | What it tests |
|----------|---------------|
| **Random trigger vs variance trigger** | Is the variance signal doing real work, or does any ~5% position subset produce the same gain? Replace `delta_var > threshold` with random Bernoulli(p) at the same mean firing rate. |
| **Internalization on vs off** | Does the KL loss matter beyond Langevin rollout diversity? Compare v4b (Langevin only, no internalize) vs v6c-nodecay (Langevin + internalize). This is already partially done — v4b vs v6c results exist. |
| **Trigger firing rate** | Sweep `delta_var_threshold` to get 1%, 5%, 20% firing rates. Expected: 5–10% is optimal; too sparse misses real forks, too dense fires on noise. |
| **Langevin K steps** | K=1 vs K=3. Does multi-step MCMC on the energy surface help? |

---

### Priority 4 — Theory Validation

**Variance = gradient significance.** At a fixed checkpoint, on held-out prompts:

1. Generate n=8 rollouts → compute `ΔVar[t]` per position (ground truth fork positions).
2. Compute `|∇_{logits_t} E[r]|` via advantage-weighted group mean logit magnitude (the energy gradient estimate).
3. Measure Spearman correlation between `ΔVar[t]` and `|g_t|` across positions and prompts.

If correlation is high, this formalizes the variance-trigger as a gradient significance test. Compare base model vs GRPO baseline vs PIVOT — if PIVOT's correlation is higher, the training calibrates the variance signal.

Also run the entropy–ΔVar correlation experiment from [PIVOT-v2 Validation](#validation) to confirm inference-time trigger calibration.

---

### Evaluation Protocol

- **MATH500**: acc@16 (mean), maj@16, best@16
- **AIME** (2024 + 2025): acc@16 (mean), maj@16, best@16
- **AMC** (optional, intermediate difficulty between MATH500 and AIME)
- Report at matched effective compute (total steps from pretrained base, not from GRPO seed)
- All runs: Qwen2.5-3B and Qwen2.5-7B minimum; 14B if time allows

---

### Entropy-Neutral Contrastive Direction (v8a/v8b/v9b)

**Observation.** The root cause of entropy inflation in v8a (and to a lesser extent v8b) is that the contrastive direction `δ = contrastive_mean` has non-zero expectation under the current policy:

```
ΔH ≈ -E_p[δ] = -Σ_v p(v) · contrastive_mean(v)
```

When `E_p[δ] < 0` (contrastive direction suppresses the most likely tokens), entropy rises.

**Fix.** Project `contrastive_mean` onto the entropy-neutral subspace by subtracting its expectation under the group-mean policy:

```python
p_group = softmax(mean(logits_i over group))          # (len_trig, vocab)
contrastive_mean ← contrastive_mean - E_{p_group}[contrastive_mean]
                 = contrastive_mean - (p_group * contrastive_mean).sum(-1, keepdim=True)
```

After projection, `E_p[contrastive_mean] = 0`, so `ΔH ≈ 0` to first order. The direction only **redistributes** probability mass among tokens at trigger positions — pure credit assignment without entropy side-effects.

**Why this is theoretically clean.** Since adding a constant to logits leaves softmax unchanged, projecting out the mean under `p` is the natural way to remove the entropy-changing component while preserving the full redistributive signal. This is equivalent to working in the tangent space of the probability simplex at `p_group`.

**Applied to:**
- `split_kl` (v8a): projection applied after `winner_mean - loser_mean` clamping
- `contrastive_kl` (v8b): projection applied after adv-weighted sum
- `thermostat_kl` (v9b): projection applied after thermostat blend

**Paper note.** If experiments confirm that entropy stays stable with the projection while learning speed is maintained, this becomes a methodological contribution: *the entropy-neutral contrastive direction is the canonical form of the internalization signal*. It separates the credit assignment effect (where to put mass) from the entropy effect (how much total uncertainty to have), which is a principled design choice rather than a heuristic fix.

---

### Key Risk

DAPO's Clip-Higher already prevents entropy collapse, which is one mechanism by which PIVOT helps. If DAPO + PIVOT shows smaller gains over DAPO than GRPO + PIVOT shows over GRPO, it means some of PIVOT's benefit overlaps with DAPO's entropy control. The unique remaining contribution would be the variance-triggered credit assignment — which is still novel and worth reporting, but the story shifts slightly toward "PIVOT = better credit assignment at decision points" rather than "PIVOT = better entropy + credit assignment."

---

## Qwen3-4B-Base Experiments (2026-04-29)

### Overview

First PIVOT runs on Qwen3-4B-Base (no instruction tuning), using Guru-RL dataset. All runs start from scratch (no GRPO seed). Training batch size 1024, n=8 rollouts per prompt, max_response_length=4096.

Baseline: plain GRPO on Qwen3-4B-Base.

**GRPO baseline val metrics**:

| step | MATH mean@16 | AIME mean@16 | resp (mean) |
|------|:---:|:---:|:---:|
| 75  | 11.9% | 0.91% | 1294 |
| 80  | 30.4% | 2.47% | 1116 |
| 85  | 51.2% | 5.10% | 1197 |
| 90  | 64.3% | 6.77% | 1200 |
| 95  | 70.7% | 7.76% | 1358 |
| 100 | 73.0% | 7.73% | 1474 |
| 110 | 74.6% | 9.53% | 1771 |
| 120 | 75.1% | 9.32% | 1794 |
| 130 | 75.9% | 10.16% | 1888 |
| 145 | 77.3% | 9.40% | 1950 |

GRPO takes ~75 steps to escape the near-zero accuracy dead zone (response length still ~1300 tokens throughout — no collapse).

---

### PIVOT-v15d on Qwen3-4B-Base

**Config**: `pivot_version=2`, entropy trigger (`entropy_trigger_only=True`, `trig_percentile=80`, `entropy_threshold=1.5`), `langevin_K=1`, `langevin_top_k=20`, `langevin_eta=0.1`, `langevin_sigma=0.01`, `lan_grpo_coeff=1.0`, `lan_grpo_correct_is=True`, `lan_grpo_correct_mu=True`, `lan_grpo_denom_blend=0.5`, `entropy_cap=0.8`, `n_rollouts_per_prompt=8`, `kl_loss_coef=0.005`.

Two sub-runs documented below.

#### v15d — original run (no sink gate, steps 1–97)

No `langevin_min_trigger_position`. Langevin fires from token 0.

**Val metrics**:

| step | MATH mean@16 | AIME mean@16 | resp |
|------|:---:|:---:|:---:|
| 25  | 11.4% | 0.08% | 813 |
| 30  | 25.0% | 0.26% | 267 |
| 35  | 26.3% | 0.49% | 95 |
| 45  | 29.4% | 1.93% | 73 |
| 50  | 32.9% | 2.14% | 51 |
| 55  | 35.8% | 0.23% | 67 |
| 70  | 47.6% | 0.52% | 163 |
| 80  | 53.7% | 2.50% | 345 |
| 85  | 54.9% | 2.58% | 384 |
| 90  | 54.6% | 3.15% | 481 |
| 95  | 57.3% | 3.98% | 565 |

**Shortcut collapse observation**: Langevin fired at structural/format decision tokens in the first ~50 positions of a response. Because those early positions are high-entropy (format branching), the entropy trigger fires there, Langevin pushes logits toward high-entropy directions, and the IS weight upweights short correct answers. By step 25 responses were already collapsing (resp=813), by step 30 collapsed to 267 tokens. Accuracy still improved (bootstrap effect) but plateaued at ~57% MATH / 4% AIME by step 97 — well below GRPO's 77% / 9.4% at step 145.

The bootstrap mechanism: short correct answers exist in the data, Langevin makes the model stumble onto them earlier via entropy perturbation at structural positions, IS correction amplifies these correct short-answer rollouts, the model learns to produce them. This gives a ~50-step head start over GRPO on MATH but prevents the model from ever learning long-form reasoning chains needed for hard problems.

#### v15d — restart from step 20 with sink gate (langevin_min_trigger_position=40)

Resumed from `global_step_20` checkpoint. Added `langevin_min_trigger_position=40` to block Langevin from firing on the first 40 response tokens.

**Val metrics (new run only)**:

| step | MATH mean@16 | AIME mean@16 | resp |
|------|:---:|:---:|:---:|
| 25  | 5.0%  | 0.03% | 1046 |
| 30  | 20.3% | 0.10% | 948  |
| 35  | 24.8% | 0.49% | 731  |
| 40  | 24.4% | 0.57% | 107  |
| 45  | 25.4% | 0.73% | 153  |
| 50  | 26.3% | 0.55% | 99   |
| 60  | 27.1% | 1.48% | 105  |

**Result**: The 40-token gate delayed collapse by ~10 steps (resp still ~950 at step 30, then crashed at step 35–40) but did not prevent it. After collapse, trig_frac jumped from 0.06–0.15 to 0.27–0.30, and the model stabilized at ~27% MATH with response lengths of 50–160 tokens. No recovery observed.

**Conclusion**: 40 tokens is too short. Typical collapsed responses are 50–150 tokens; positions 41+ are still eligible for Langevin, so the bootstrap mechanism still fires. A larger gate (300+ tokens) is needed to fully block Langevin from reinforcing shortcut answers.

---

### PIVOT-v15e on Qwen3-4B-Base

**Config**: v15d + `lan_grpo_restrict_to_trigger=True` (policy gradient flows only at triggered positions, ~20% of tokens). `langevin_min_trigger_position=40`.

**Val metrics**:

| step | MATH mean@16 | AIME mean@16 | resp |
|------|:---:|:---:|:---:|
| 25  | 0.3% | 0.03% | 1033 |
| 60  | 1.1% | 0.00% | 1179 |
| 65  | 19.6% | 0.23% | 330  |
| 70  | 23.6% | 0.39% | 53   |
| 75  | 25.2% | 0.29% | 15   |
| 80  | 24.9% | 1.17% | 16   |
| 95  | 26.0% | 0.08% | 18   |
| 110 | 26.7% | 0.16% | 25   |
| 115 | 27.2% | 0.03% | 44   |

**Diagnosis**: Dead zone for 60 steps (0.3–1.1% MATH), then sudden collapse at step 65 (resp: 1179 → 330 → 15). After collapse, responses stabilized at 15–44 tokens — too short for `restrict_to_trigger` to find any meaningful gradient signal (trig_frac ≈ 0.20 on 15-token responses = ~3 trigger positions per sample). Model completely unable to recover. Plateaued at ~27% MATH, worse than plain GRPO at the same step count.

**Why `restrict_to_trigger` failed here**: The flag makes sense as a variance reduction technique when responses are long — filtering PG to the 20% most fork-like positions removes gradient noise from mundane tokens. But once responses collapse to 15 tokens, restricting PG to 3 positions/sample essentially kills the gradient. The collapse trap is permanent because there is no path back to long responses.

**Final training state (run stopped at step 119)**: Training steps 116–119 confirmed no recovery — resp=15–44 tokens, entropy=0.09–0.12, gnorm=1.3–1.8 (high norm on near-degenerate short outputs), score/mean=-0.74 to -0.76. Notably, PIVOT's best@16 (45–49%) far exceeded its mean@16 (25–27%), confirming the model outputs near-random short guesses rather than reliable solutions.

**Comparison vs GRPO at step 119**: GRPO had score/mean=-0.47, entropy=0.05, resp=1880 tokens, MATH mean@16=76%, AIME mean@16=9.4%. PIVOT's shortcut collapse left it ~50% below GRPO on MATH and ~9% below on AIME despite a 5× head start in early accuracy.

---

### Key Finding: Langevin Bootstrap via Shortcut Answers

The dominant behavior in PIVOT on Qwen3-4B-Base from scratch is a **Langevin shortcut bootstrap**:

1. Early in training (steps 1–30), the model generates mostly near-random outputs (score ≈ −1.0).
2. The entropy trigger fires at high-entropy positions, which are concentrated in the first 40–100 response tokens (structural format decisions: whether to `<think>`, how to open the answer).
3. Langevin perturbation at these positions occasionally pushes the model toward a short correct answer format.
4. IS correction upweights these short-correct rollouts in the GRPO objective.
5. The model rapidly learns to produce short correct answers (3–10× faster than GRPO at escaping near-zero accuracy).
6. However, these short answers generalize poorly to hard problems (AIME requires multi-step reasoning, not short answers).
7. The bootstrap ceiling is ~27–57% MATH (depending on gate size) vs GRPO's eventual 77%.

**Mitigation**: Block Langevin on short-answer positions with `langevin_min_trigger_position`. A gate of 40 tokens was insufficient (still collapsed). **300 tokens is the planned next threshold** (v15f, v15g) — any collapsed response shorter than 300 tokens will not trigger Langevin at all, breaking the reinforcement loop.

**Alternative**: The shortcut bootstrap may still be net-positive if the model later recovers to long reasoning. v15d-original shows partial recovery (resp grew from 51 → 565 tokens over steps 50–95), but the ceiling remained well below GRPO. The open question is whether a longer run (200+ steps) would eventually match GRPO or stay below.

---

### GRPO-Qwen3-4B-Base (2026-04-28, trained from scratch)

**Config**: `train_batch_size=1024` (stepped up from 512 at step 20), `n=8`, `kl_loss_coef=0.001`, `max_response_length=4096`, `gpu_memory_utilization=0.4`, `ppo_micro_batch_size_per_gpu=16`. Vanilla GRPO, no PIVOT/Langevin. Script: `run_qwen3_4b_base_grpo.sh`. Stopped at step 220.

**Training trajectory**:

| Step | MATH mean@16 | MATH best@16 | MATH maj@16 | AIME mean@16 | AIME best@16 | AIME maj@16 | entropy | resp_len |
|------|-------------|-------------|------------|-------------|-------------|------------|---------|----------|
| 5    | 0.1%  | 1.3%  | 0.0% | 0.0% | 0.0%  | 0.0% | 0.74 | 1177 |
| 45   | 0.4%  | 3.4%  | 0.0% | 0.0% | 0.0%  | 0.0% | 0.98 | 1249 |
| 70   | 3.1%  | 26.6% | 0.0% | 0.1% | 1.1%  | 0.0% | —    | 1356 |
| 75   | 11.9% | 58.0% | 5.1% | 0.9% | 6.1%  | 0.3% | —    | 1248 |
| 80   | 30.4% | 78.5% | 47.2%| 2.5% | 12.0% | 2.8% | —    | 1097 |
| 85   | 51.2% | 85.7% | 70.7%| 5.1% | 15.8% | 8.7% | —    | 1187 |
| 100  | 73.0% | 88.9% | 79.5%| 7.7% | 20.2% | 10.9%| —    | 1510 |
| 125  | 75.6% | 89.1% | 80.1%|**10.1%**|**23.3%**|12.2%|0.027|1869|
| 155  | 76.8% | 88.5% | 80.3%| 10.3%|**23.8%**|12.1%|0.028|2086|
| 170  | 78.3% |**89.5%**|**82.2%**|10.0%|19.8%|**13.5%**|0.028|2094|
| 210  | 78.8% | 88.7% | 81.8%|**11.2%**|21.4%|**13.5%**|0.024|2102|
| **220**  | **78.8%** | 89.1% | 82.1%| 11.1%| 21.1%| 13.4%|0.023|2149|

**Phase transition**: Score ≈ −1.0 for steps 1–70 (base model not yet producing correctly-formatted answers). Sharp transition at steps 70–85: score −1.0 → −0.55, MATH 3% → 51% in 15 steps. Response length dipped at step 80 (1356 → 1097) as the model switched from free-form to structured answer format, then rebounded and grew monotonically to ~2150 tokens.

**Entropy collapse**: Started at ~0.8 nats, fell to ~0.023 by step 220. No `entropy_coeff` to arrest decay — consistent with PIVOT-v2 observation that base GRPO leads to entropy collapse. Despite collapse, performance kept improving slowly (unlike 3B instruct where collapse caused plateau at ~62% MATH). Base model may have more headroom.

**Response length / clip ratio**: ~20% of responses were hitting the 4096 token cap by step 220. Raising `max_response_length` past 4096 would likely improve AIME further (hard problems need more reasoning steps).

---

### PIVOT-v17b on Qwen3-4B-Base (2026-05-02, stopped at step 137)

**Config**: Same as v17c but resumed from `pivot-v17/qwen3_4b_base/global_step_20`. Script: `run_qwen3_4b_base_pivot_v17b.sh`.

Key hyperparameters: `pivot_version=2`, `langevin_rollout=True`, `entropy_trigger_only=True`, `trig_percentile=65`, `entropy_threshold=0.4`, `langevin_K=1`, `langevin_top_k=20`, `langevin_eta=0.1`, `langevin_sigma=0.01`, `langevin_min_trigger_position=200`, `langevin_momentum=0.5`, `langevin_feedback=True`, `langevin_exploit_ratio=0.5`, `lan_grpo_coeff=1.0`, `lan_grpo_correct_is=True`, `lan_grpo_correct_mu=True`, `lan_grpo_denom_blend=0.5`, `entropy_cap=0.8`, `lan_grpo_restrict_to_trigger=False`.

**Training trajectory**:

| step | MATH mean@16 | MATH best@16 | AIME mean@16 | AIME best@16 | resp | entropy |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 25   | 0.8%  | 7.3%  | 0.00% | 0.00%  | 1106 | 0.406 |
| 35   | 14.0% | 54.5% | 0.08% | 0.82%  |  944 | 0.449 |
| 40   | 26.9% | 62.8% | 0.47% | 3.75%  |  166 | 0.444 |
| 50   | 41.4% | 75.1% | 0.55% | 4.05%  |  140 | 0.274 |
| 60   | 47.9% | 75.6% | 0.70% | 5.38%  |  189 | 0.218 |
| 70   | 54.5% | 81.1% | 2.29% | 12.28% |  219 | 0.160 |
| 75   | 57.9% | 83.2% | 3.02% | 12.82% |  564 | 0.145 |
| 80   | 60.2% | 84.4% | 3.78% | 13.64% |  450 | 0.130 |
| 85   | 61.5% | 85.5% | 4.04% | 13.69% |  865 | 0.117 |
| 90   | 63.9% | 85.8% | 4.48% | 13.80% |  878 | 0.108 |
| 95   | 65.6% | 85.4% | 5.70% | 15.39% |  877 | 0.099 |
| 100  | 66.2% | 86.6% | 5.62% | 11.99% |  889 | 0.097 |
| 110  | 67.5% | 86.3% | 6.04% | 14.26% |  965 | 0.083 |
| 120  | 68.8% | 86.5% | 6.04% | 13.96% |  751 | 0.068 |
| 130  | 70.3% | 87.6% | 6.77% | 11.97% | 1054 | 0.066 |
| **135**  | **70.7%** | **86.4%** | **7.06%** | **13.34%** | 987 | 0.066 |

**Final train state (step 137)**: score/mean=−0.575, resp=1007, entropy=0.063, trig_frac=3.6%, pg_loss=0.023.

**Behavior**: Same Langevin shortcut bootstrap as other v17 variants — resp collapsed from ~1100 to ~140 tokens between steps 35–50, then partially recovered to 550–1050 by step 75+. Entropy collapsed to 0.063 by step 137 (lower than v17c's 0.088 at the same stage). Response length capped at ~1000 (cf. GRPO's 2000+), consistent with Langevin entropy perturbation blocking long-chain commitment.

**Comparison vs GRPO at step 137**: GRPO had MATH mean@16=76.2%, AIME mean@16=9.5%, resp=1933. v17b reached 70.7% MATH and 7.1% AIME — still ~6% below GRPO on MATH and ~2.4% below on AIME. The Langevin bootstrap gave a ~35-step head start (v17b at 70% MATH by step 135 vs GRPO's ~65% at step 100), but GRPO's unconstrained resp_len growth allowed it to overtake by step ~110.

**Conclusion**: v17b confirms the PIVOT response-length ceiling (~1000 tokens) as the binding constraint. Both HEG and PIVOT variants cap at ~1000 tokens while GRPO grows to 2000+; this is the mechanism by which GRPO overtakes. Stopped at step 137.

**Summary**: GRPO on Qwen3-4B-Base from scratch achieves MATH mean@16=78.8% and AIME mean@16=11.2% at step 220 — substantially better than the 3B instruct GRPO baseline (61.5% / 1.8%). AIME best@16=23.8% is the all-time high across all runs. The larger batch size (1024) was critical: at batch=512 (steps 1–20) the gradient signal was too sparse; the switch to 1024 at step 20 drove rapid learning.

---

### PIVOT-v18 on Qwen3-4B-Base (asymmetric IS + target-band G)

**Config**: Two fixes over v17c targeting the collapse pathology. Script: `run_qwen3_4b_base_pivot_v18.sh`.

1. **Asymmetric IS correction (loss side).** Negative-advantage trigger positions use `min(log_p_lan, log_π_old)` as the denominator, so the effective ratio recovers to standard GRPO strength for wrong committed paths. Positive-advantage positions keep the v17c geometric-mean blend unchanged. Enabled via `lan_grpo_asym_is=True`.
2. **Target-band G signal (rollout side).** Replaces the v17c `median(H_past) − H_current` feedback with `H_current − α · H_first`, where `H_first` is the entropy at the trajectory's first trigger. G commits when H > α·H_first, stops near the target, reverses if overshot. Enabled via `langevin_alpha_target=0.7` (allowing 30% entropy decay from trajectory start before reversing).

Key hyperparameters (new/changed vs v17b): `lan_grpo_asym_is=True`, `langevin_feedback=True`, `langevin_alpha_target=0.7`, `langevin_exploit_ratio=0.6`, `langevin_momentum=0.7`. Everything else inherited from v17c: `pivot_version=2`, `entropy_trigger_only=True`, `trig_percentile=85`, `entropy_threshold=0.4`, `langevin_K=1`, `langevin_top_k=20`, `langevin_eta=0.1`, `langevin_sigma=0.01`, `langevin_min_trigger_position=200`, `lan_grpo_coeff=1.0`, `entropy_cap=0.8`.

**Behavior**: Fixed the wrong-path IS-gating pathology of v17c but did not on its own suppress length collapse — the Langevin shortcut mechanism still fires because `min_trigger_position=200` is well within the collapsed-length regime. Established the "target-band G" formulation used by all subsequent versions.

---

### PIVOT-v18b on Qwen3-4B-Base (CUDA-graph capture, algo-equivalent to v18)

**Config**: Same algorithm as v18 with `lan_use_cuda_graph=True` routing rollouts through `PIVOTv18bLangevinAdapter`. Script: `run_qwen3_4b_base_pivot_v18b.sh`.

Changes are purely implementation-level:
- G state lives fully on GPU (no per-proc G tensors).
- Langevin perturbation kernel is CUDA-graph-captured for reduced CPU overhead.
- G is updated **before** graph replay so Langevin uses the freshest momentum.
- `entropy_threshold` is fixed from config (the adaptive `trig_percentile` path is not wired into the graph; each proc still uses its own `proc.entropy_threshold` for the trigger decision, but the graph itself uses the config-fixed threshold).

**Behavior**: Bit-for-bit reproduces v18's training trajectory with ~2× lower Langevin wall-clock overhead. No algorithmic difference expected; used as the throughput baseline for v18-family variants.

---

### PIVOT-v18c on Qwen3-4B-Base (v18 + length penalty)

**Config**: v18 with an additional length penalty on rollout scores. Script: `run_qwen3_4b_base_pivot_v18c.sh`.

    penalty = lp_coef · (min(resp_len / lp_min_len, 1.0) − 1.0)

Added to group scores **before** GRPO normalization so short responses score worse than long ones within the same prompt group. Initial config: `length_penalty_coef=2.0`, `length_penalty_min_len=1000`; the coefficient was later swept to 0.5. Additionally reverts `langevin_min_trigger_position` back to 200 (pumping it to 800/1000 in earlier explorations had zero effect on the timing of collapse — always step 29–31).

Key hyperparameters (new/changed vs v18): `+algorithm.length_penalty_coef=2.0`, `+algorithm.length_penalty_min_len=1000`, `langevin_min_trigger_position=200`.

**Behavior**: On 4B, the length penalty prevents the sharp step-29 collapse but a slower drift toward short outputs still emerges by step 60–80. On 1.7B, the length penalty is more effective and the run reaches useful MATH accuracy. Established the length-penalty formulation later carried over to `drift_method.md §3.4` (subsequently superseded by KL-in-reward in v22).

---

### PIVOT-v18d on Qwen3-4B-Base (v18c ablation: top-K = 150)

**Config**: v18c with `langevin_top_k=150` (vs v18c's 20). All other hyperparameters identical. Script: `run_qwen3_4b_base_pivot_v18d.sh`.

**Purpose**: Test whether allowing Langevin to reach lower-probability "reasoning pivot" tokens ("revisit", "wait", "actually") — outside the top-20 but within the plausible vocabulary at trigger positions — improves outcome.

**Behavior**: Widening the top-K subspace increased perturbation into off-manifold tokens and worsened length collapse. Confirmed the "head, not tail" §3.2 story from `drift_method.md`: perturbation should be restricted to a small top-K subspace to remain on the coherent-continuation manifold.

---

### PIVOT-v19 (DRIFT-v19) on Qwen3-4B-Base (EOS mask + window-based adaptive t_min, length penalty removed)

**Config**: v18c structure with two surgical changes targeting length collapse. Script: `run_qwen3_4b_base_pivot_v19.sh`.

1. **EOS-class token mask on the Langevin top-K subspace.** Langevin can no longer push probability toward `</answer>`, `<|im_end|>`, EOS — closing the shortcut-to-termination pathway at the perturbation level.
2. **Adaptive `t_min = clamp(median(recent_lengths) − W_min, floor, cap)`** with `W_min=400, floor=400, cap=2500`. Anchors on a guaranteed Langevin firing window `W_min` on a median-length rollout; at the current Qwen3 length regime (median ~1000–1500) this places `t_min` in [600, 1100], replicating the empirical 1.7B fix where pumping `t_min ~ 800` stopped collapse.

The v18c length penalty is **removed** — the goal is to verify that (1) + (2) prevent collapse on their own.

**Behavior on 4B**: Catastrophic collapse at step 35–50 (length 1478 → 151), plateau at length 30–100 with MATH ~30% AIME ~0% for ~200 steps, then spontaneous recovery (length back to 465 by step 315, MATH 62%). Diagnostic: the window-based `t_min` **drops with length**, so Langevin keeps firing in the collapsed regime and reinforces the short-answer mode. This mismatch motivated v20's monotonic-peak formulation.

**Behavior on 1.7B**: v19 works well — no collapse, comparable or better ceiling than v18c.

---

### PIVOT-v20 (DRIFT-v20) on Qwen3-4B-Base (monotonic-peak t_min ratchet)

**Config**: v19's EOS mask + a **monotonic-peak** t_min replacing v19's window-based adaptive rule. Script: `run_qwen3_4b_base_pivot_v20.sh`.

    t_min(step) = clamp(α · peak_length, floor, cap)
    peak_length = max(peak_length, current_median_length)     # monotonic, never decreases

With `α=0.6`, `floor=200`, `cap=3200`. Key insight: v19's window-based `t_min` drops with length and lets Langevin keep firing in the collapsed regime; v20's monotonic-peak `t_min` **ratchets up** with length, so if the median length dips below `α · peak`, Langevin auto-disables — the policy can then recover via plain GRPO without further Langevin-driven exploration of the shortcut.

Key hyperparameters (new vs v19): `langevin_min_trigger_position_mode=peak`, `langevin_peak_alpha=0.6`, `langevin_t_min_floor=200`, `langevin_t_min_cap=3200`. Inherits v19's `langevin_mask_eos=True`.

**Behavior**: Solved v19's 4B catastrophic collapse. The peak-ratchet t_min is now inherited by all subsequent versions and is the mechanism referenced in `drift_method.md §3.1`.

---

### PIVOT-v21 (DRIFT-v21) on Qwen3-4B-Base (soft-IS multiplier — clean per-token weight)

**Config**: v20 with a **soft-IS multiplier** replacing the v18-family blended-denominator IS correction. Script: `run_qwen3_4b_base_pivot_v21.sh`.

Derived directly from the unbiased policy gradient `∇J = E_{a∼q}[ (π_θ/q) · A · ∇log π_θ(a) ]` with `q = π_lan_old` at triggers. Factoring `(π_θ/q) = (π_θ/π_old) · (π_old/q)` lets PPO's ratio stay at `r = π_θ/π_old` (trust region intact) and pushes the off-policy correction `(π_old/q)` onto the advantage as a per-token weight:

    w_t = min(1, π_old(a_t) / π_lan_old(a_t))    at trigger positions
        = 1                                        elsewhere
    A'_t = A_t · w_t

Equivalent to multiplying the per-token PG loss by `w_t`. The cap at 1 keeps the estimator variance-bounded and unbiased in the rate-inflation regime (where Langevin raises the committed token's probability) — precisely the regime that drives length collapse. See `drift_method.md §3.5` for the derivation.

Key hyperparameters (new/changed vs v20): `lan_grpo_soft_is=True`, `lan_grpo_correct_is=False` (old denominator patch off), `lan_grpo_denom_blend` dropped, `lan_grpo_asym_is` dropped, `lan_grpo_correct_mu` dropped.

**Behavior**: v21 **collapsed harder than v18+** on 4B. Diagnostic: v20's peak-ratchet t_min disables Langevin post-collapse (correct behavior), but that means v21's IS correction only acts at trigger positions and becomes a no-op once the policy has already found the short-answer shortcut. Plain GRPO with a length-independent ±1 reward then has a degenerate local optimum at "guess the answer template directly" whenever the model's reasoning ability is below its short-guess success rate. This failure mode motivated v22's reward-level shaping.

---

### PIVOT-v22 (DRIFT-v22) on Qwen3-4B-Base and Qwen3-8B-Base (KL-in-reward mean-KL shaping — current design)

**Config**: v21 + **InstructGPT-style per-rollout KL-in-reward shaping** on top of the v21 soft-IS multiplier. Scripts: `run_qwen3_4b_base_pivot_v22.sh`, `run_qwen3_8b_base_pivot_v22.sh`, `run_qwen3_1p7b_base_pivot_v22.sh`.

Reward shaping applied per rollout **before** advantage normalization:

    r'^{i,j}  =  r^{i,j}  −  β_KL · (1 / L^{i,j}) · Σ_t KL̂( π_θ(· | s_t) ‖ π_ref(· | s_t) )

- **Mean-KL, not sum-KL.** Long natural rollouts have low per-token KL from base and receive negligible penalty; short shortcut rollouts have high per-token KL (the collapsed policy is OOD relative to the natural reasoning manifold) and receive heavy penalty. Sum-of-KL (verl's default) would bias the wrong direction because it grows linearly with length.
- **Applied at reward level, not loss level.** Because within-group mean-subtraction removes any additive shift common to a group, an in-loss KL only anchors global drift and cannot reshape within-group ranking. The in-reward form pushes short-collapsed correct rollouts below long-correct ones in advantage rank — a within-group signal the policy gradient uses to prefer full reasoning.

The v21 soft-IS multiplier stays on. The mechanisms compose: the soft-IS handles off-policy gradient at Langevin-firing positions; the KL-shaped reward handles the global GRPO objective. Diagnostic per-length-stratum reward logging (`diag/short_lt200_*`, `diag/mid_200_800_*`, `diag/long_ge800_*`) is added at every train step to verify the collapse mechanism.

Key hyperparameters (new/changed vs v21): `algorithm.use_kl_in_reward=True`, `+algorithm.kl_in_reward_mode=rollout_mean`, `algorithm.kl_penalty=low_var_kl`, `algorithm.kl_ctrl.kl_coef=0.2` (4B) / `2.0` (8B) / `0.2` (1.7B). All v20/v21 mechanisms retained: peak-ratchet t_min, EOS mask, soft-IS multiplier, target-band G with `α_target=0.7`, `entropy_cap=0.8`, top-K=20, clip-higher (0.2, 0.28), in-loss `kl_loss_coef=0.001`.

**Behavior on Qwen3-4B-Base (guru_rl, ran to step 229)**:

| Metric | Peak | Step |
|---|---|---|
| MATH mean@16 | **79.42%** | 195 |
| MATH best@16 | **90.08%** | 80 |
| AIME24 mean@16 | **12.81%** | 190 |
| AIME24 best@16 | **24.40%** | 205 |

Length collapse fully suppressed; response length remains in the natural regime (~1000+ tokens) throughout training. Entropy trajectory is a controlled explosion-then-plateau pattern (peak ~4.7 near step 109, settles into a healthy plateau ~2.0–2.5) rather than the collapse-to-zero pattern of v17b/v21.

**Behavior on Qwen3-1.7B-Base (73 steps logged)**: MATH mean@16 = 16.12% (step 65), AIME24 mean@16 = 0.44% (step 50). Same non-collapse trajectory as 4B, on a smaller-model scale.

**Behavior on Qwen3-8B-Base**: `kl_coef=2.0` is the tuned coefficient (0.2 was insufficient to prevent length collapse on 8B given the base model's stronger prior). Confirms KL-in-reward is a scale-general defense: the coefficient scales with the base policy's confidence, but the mechanism itself does not require length-target retuning.

**Conclusion**: v22 is the current DRIFT design. The switch from v18c's length-target penalty (`λ, L_target`) to per-rollout mean-KL removes both hyperparameters and gives a scale-free length-collapse defense whose "target" is implicitly the base policy's own length distribution. See `drift_method.md §3.4` for the paper-prose treatment.

---

### PIVOT-v22b on Qwen3-4B-Base (ablation: v22 without positional guard)

**Config**: v22 with the positional guard **removed** — `langevin_min_trigger_position=0` (mode=static). All other hyperparameters identical to v22 (KL-in-reward + v21 soft-IS + everything else). Script: `run_qwen3_4b_base_pivot_v22b.sh`.

**Purpose**: Test whether the v20 peak-ratchet `t_min` positional guard is load-bearing for collapse prevention in v22, or whether v22's KL-in-reward reward shaping alone is sufficient.

**Hypothesis**: If KL-in-reward is the dominant defense, v22b should still avoid collapse. If v22b collapses but v22 does not, `t_min` is doing meaningful work even with KL shaping in place.

**Behavior**: [pending analysis — trained but not yet fully written up]. Documented here as the ablation companion to v22 for the paper's §6.4 ablation section.
