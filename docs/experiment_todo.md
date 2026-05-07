# Experiment TODO for the DRIFT paper

Tracking what's needed to complete the experiment section. Priorities reflect
"what a NeurIPS reviewer would expect" given the current state of the paper.

## Already in flight / done

- [x] AIME24, AIME25, OlympiadBench, GPQA prepped as parquets
- [x] `eval_via_verl.sh` + per-method scripts (`eval_grpo_4b.sh`, etc.)
- [x] `aggregate_eval.py` produces per-method-model summary JSONL + markdown
- [x] Patches landed: agent_loop passes `rollout_log_probs`; reward manager
      surfaces `response_length`, `entropy`; reward fn bubbles them through
- [x] **GRPO 4B sweep running now** (PID 175819, ~14 min/ckpt, 8 ckpts ≈ 2h)
- [x] AIME mean@16, AIME best@16, Math@16 logged for the active training runs
      (v18b 1.7B, v18c 1.7B/4B) — usable for the bootstrap-curve figure

---

## Critical path (paper depends on these)

### Eval sweeps — fill in the §6 main table

For each (method × size), produce the per-method summary JSONL with all 4
benchmarks at multiple step boundaries (`STEPS_1P7B="20 40 60 80 100 120 140 160"`,
`STEPS_4B` similar but capped by what's available).

- [x] **GRPO × 4B** — running now (PID 175819)
- [ ] **GRPO × 1.7B** — `bash eval_grpo_1p7b.sh`
- [ ] **HEG × 4B** — `bash eval_heg_4b.sh`
- [ ] **HEG × 1.7B** — `bash eval_heg_1p7b.sh`
- [ ] **DRIFT (v18c) × 4B** — `bash eval_v18c_4b.sh`
- [ ] **DRIFT (v18b) × 1.7B** — `bash eval_v18b_1p7b.sh`

Each sweep saturates 8 GPUs (~2h). Sequential. Total wall time ≈ 12h for all 6.

### Empirical mechanism studies (defend §3 design choices)

- [ ] **Study 2: reward variance over training** (no compute, parses existing JSONLs)
      Run: `python3 examples/empirical/reward_variance.py --jsonls GRPO=... HEG=... DRIFT=... --out reward_variance_4b.json`
      Produces the bootstrap-curve panel in §1/§4 + the entropy-stability panel in §3.3.
- [ ] **Study 1: rollout diversity** (inference, ~10 min/ckpt × 6 ckpts)
      Run: `bash examples/empirical/run_diversity_sweep.sh`
      Defends §3.2 "diffusion spreads n rollouts into distinct continuations".
- [ ] **Study 3: branch-point entropy time-series** (single-GPU inference, ~5 min/prompt)
      Run: `python3 examples/empirical/entropy_time_series.py --ckpt /path/to/v18b_1p7b/step120/actor/hf_merged ...`
      Defends §3.3.2 "entropy maintenance vs reduction".

### Multiple seeds for headline DRIFT 4B run

- [ ] **Seed 2** of `v18c 4B` — same starting checkpoint, same hyperparameters,
      different rollout/init seed. Even one extra seed lets us report mean ± range.
- [ ] **Seed 3** if time permits (NeurIPS expects ≥3 seeds).

The bootstrap acceleration claim (45-step head start) is the most variance-prone
because it depends on early-training stochasticity. Two extra seeds is the
minimum to argue the bootstrap is not an outlier.

---

## Optional (paper is stronger with each, but not blocking)

Each item below addresses a specific reviewer-attack vector. Listed in priority
order: items earlier in the list have higher score impact.

### 1. σ = 0 ablation — defends §3.2 "diffusion breaks the dead zone"

- [ ] One DRIFT 4B training run with `langevin_sigma=0` from the same starting
      checkpoint as v18c 4B. Predict: collapsed group diversity, no reward
      variance, fails to escape dead zone.

This is the strongest single ablation in the paper — it directly defends the
§3.2 claim that the diffusion term (not just the drift) is what produces the
n-rollout symmetry breaking.

### 2. Continue training the DRIFT runs to plateau

- [ ] **v18c 4B → step 150+** — currently at step 85. The "convergence at equal
      ceiling" claim on 4B is empirically unproven without this. GRPO plateaus
      at ~step 145 with Math ≈ 0.79; we need v18c at the same horizon.
- [ ] **v18b 1.7B → step 200** — currently at step 151. Confirm whether the
      step 150 uptick (AIME mean 0.057, AIME best 0.194) is real signal or
      single-step noise.

These are the cheapest big wins (just continue existing runs) and they settle
the asymptotic-comparison claim that anchors §6.2.

### 3. Larger model scale — closes the "where's 7B?" attack

- [ ] **DRIFT × Qwen3-7B-Base** (or Qwen2.5-7B-Base if more standard) — even a
      partial run to step 80–100 would establish that the bootstrap behavior
      generalizes. Same compute as 4B but ~1.5× slower per step.

The most common reviewer comment for a 2026 NeurIPS submission on
RL-for-reasoning will be "where's 7B?" — DAPO, DeepSeek-R1, the entropy paper
all use 7B+. This is the highest-leverage missing experiment.

### 4. DAPO head-to-head — closes the orthogonality claim

- [ ] DAPO baseline run on Qwen3-4B-Base, same data, same step count.
      Currently we cite DAPO and claim orthogonality but have no head-to-head
      numbers. A reviewer will ask.
- [ ] (Stretch) **DAPO + DRIFT** combined run — directly verifies the
      "orthogonal, can be combined" claim in the conclusion.

### 5. Theorem 1 empirical hook

- [ ] Compute fraction of GRPO groups with non-zero reward variance over
      training, three lines (GRPO/HEG/DRIFT). Theorem 1 predicts DRIFT's
      group-discovery probability is amplified n-fold by group sampling;
      this is the empirical hook.
      *Already covered by Empirical Study 2 above — just needs the right axis.*

### 6. Entropy-reduction signal ablation — defends §3.3.2

- [ ] One DRIFT 4B run replacing the maintenance signal `s_t = H_t − α·H_first`
      with the reduction signal `s_t = -(H_t − α·H_first)`. Predict: rapid
      entropy collapse and reinforcement of the failure mode the dead-zone
      regime is supposed to be in.

This is the most interesting design choice in the paper — the ablation closes
the case that maintenance, not reduction, is what makes the SPSA feedback work.

### 7. Additional benchmarks in the val sweep

- [ ] **MATH-500** (we already have `test_math500.parquet`) — add to
      `data.val_files` in `eval_via_verl.sh`. Currently saturating on DRIFT,
      but worth reporting for completeness against the entropy paper's table.
- [ ] **AMC** — not prepped, ~30 lines of HF-loading code.
- [ ] **Minerva-Math** — already prepped (`test_minerva_math.parquet`), just
      add to `eval_via_verl.sh` `val_files`. Decision deferred until DRIFT >
      baselines on the harder benchmarks.

### 8. Smaller mechanism-section ablations

- [ ] **Top-K vs full vocab perturbation** — defends §3.2 "head, not tail".
      One additional run with K=∞ on 4B; predict derailment from incoherent tokens.
- [ ] **t_min sweep** ({200, 400, 800}) — shows the length-collapse trade-off
      is real. We already have v18 (mintrig=200) vs v18c (mintrig=800) data on
      4B that may suffice.
- [ ] **α_target sweep** ({0.3, 0.5, 0.7}) — shows the maintenance fraction
      matters.
- [ ] **No length penalty** (`λ_len = 0`) — already partially have this from
      v18 vs v18c on 4B; could just use v18 trajectory as the ablation.

---

## Reviewer-friendliness checklist (mostly text, easy to add)

- [ ] Reproducibility detail in §5: tokenizer / chat template, exact decoding
      params, prompt template.
- [ ] Code release plan — anonymized GitHub link for the appendix.
- [ ] Compute disclosure — H200 hours per run, total compute used. Required
      for NeurIPS Code of Ethics / compute-disclosure question.
- [ ] Limitation statement — model scale (only 1.7B/4B), seed count, data
      source (single math distribution).
- [ ] Broader impact statement — RL for math reasoning, not safety-relevant.

---

## Honest reviewer-score estimate (for prioritization)

Anchored from the §1–6 paper as drafted, with critical path complete:

- Critical path only: borderline reject (3). Reviewers will flag the missing
  σ=0 ablation, lack of seeds, and 4B-only scale.
- Critical path + Optional 1 (σ=0) + Optional 2 (continue to plateau): 4
  (borderline accept).
- Critical path + Optional 1, 2, 3 (7B), 4 (DAPO): 5 (accept).
- All Optional items: 5–6 (clear accept). The mechanism story is then fully
  defended on multiple axes.
