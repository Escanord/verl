# Empirical studies for the DRIFT paper

Three end-to-end scripts producing structured outputs that can be parsed into
figures for the experiment / motivation section of the paper.

| Study | Script | Compute | What it defends |
|---|---|---|---|
| 1. Rollout diversity within a GRPO group | `rollout_diversity.py` | inference, ~10 min/checkpoint | §3.2 "diffusion spreads n rollouts into distinct continuations" |
| 2. Reward variance over training | `reward_variance.py` | none (JSONL parse) | §1 dead-zone framing; "DRIFT escapes earlier" |
| 3. Branch-point entropy time-series | `entropy_time_series.py` | single-GPU inference, ~5 min/prompt | §3.3 entropy maintenance vs reduction |

All three are run *post-hoc* on saved checkpoints / training logs.  No new
training runs required.

---

## 1. Rollout diversity (`rollout_diversity.py`)

Sample n=8 rollouts per prompt at temperature 1.0 from a saved checkpoint, then
compute pairwise diversity (token-level Levenshtein distance and 1 - Jaccard
similarity) within each n-rollout group.  Compare distributions across
GRPO/HEG/DRIFT checkpoints.

```bash
# Per-checkpoint invocation (one GPU is enough for 1.7B/4B with TP=1):
python rollout_diversity.py \
    --ckpt /path/to/grpo_4b_step60/actor/hf_merged \
    --prompts /storage/.../data/guru_rl/test_aime.parquet \
    --n_prompts 30 --n 8 \
    --tag grpo_4b_step60 \
    --out_dir ./diversity_out
```

Run once each for GRPO/HEG/DRIFT at a comparable training step (e.g., the step
where each method has Math@16 ≈ 0.5).  The output JSON contains per-prompt
pair-wise distances; plot a histogram or violin per method to defend the
"σ·ζ_t breaks symmetry across the n rollouts" argument.

**Bonus**: also run with `--ckpt` pointing at the *base* Qwen3-Base model — the
diversity at random init is the floor; methods are valuable insofar as they
preserve diversity above this floor while improving accuracy.

## 2. Reward variance over training (`reward_variance.py`)

Parses the per-step training JSONLs we already have (no new inference).  For
each method, extracts step-level critic statistics:

- `score.std`, `score.gap` — within-batch reward variance
- `advantages.std` — variance of the GRPO advantage (the actual gradient driver)
- `pg_loss`, `grad_norm` — gradient signal magnitude
- `entropy`, `response_length`, `trig_frac` — diagnostics

```bash
python reward_variance.py \
    --jsonls \
        GRPO=/storage/.../grpo/.../qwen3_4b_base_grpo.jsonl \
        HEG=/storage/.../high-ent-grpo/.../qwen3_4b_base_high_ent_grpo.jsonl \
        DRIFT=/storage/.../pivot-v18c/.../qwen3_4b_base_pivot_v18c.jsonl \
    --out reward_variance_4b.json
```

Plot from the resulting JSON:

- **Headline panel**: fraction of steps with `score.std > ε` over time, three
  lines.  Shows GRPO's dead zone visually.
- **Secondary panel**: `pg_loss` magnitude vs step, three lines.  Same story.
- **Diagnostic panel**: `entropy` vs step.  Validates the §3.3 entropy stability
  claim — DRIFT entropy decays gracefully, HEG explodes or saturates.

## 3. Branch-point entropy time-series (`entropy_time_series.py`)

Generate two rollouts on the *same* prompt from the *same* DRIFT checkpoint:
one with Langevin perturbation enabled, one without.  Log per-token entropy
H_t and the trigger mask.

```bash
python entropy_time_series.py \
    --ckpt /storage/.../pivot-v18b/qwen3_1p7b_base/global_step_120/actor/hf_merged \
    --prompts /storage/.../data/guru_rl/test_aime.parquet \
    --n_prompts 5 --max_new_tokens 1500 \
    --tag drift_v18b_step120 \
    --out_dir ./entropy_out
```

Plot from the resulting JSONL:

- For each prompt, plot H_t vs token index, two lines (vanilla vs DRIFT).  Mark
  trigger positions on the DRIFT line.  Add a horizontal reference at
  `α_target · H_first`.
- Aggregate across the 5 prompts: scatter (trigger index, H_t) for both
  conditions to show the maintenance vs decay pattern.

This is the visual proof of §3.3.2's claim that the maintenance feedback keeps
the trajectory perturbable at downstream branch points.

---

## Sweep helper (`run_diversity_sweep.sh`)

Convenience driver that runs `rollout_diversity.py` over a fixed set of
(method × size × step) checkpoints and produces six output JSONs.  Reads the
same `RUNS` table as the eval scripts; merges the results into a single
plot-ready CSV.

```bash
bash run_diversity_sweep.sh
```
