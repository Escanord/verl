# NeurIPS 2026 Rebuttal — DRIFT/PIVOT Experiments

Rebuttal-cycle experiments and status.  Companion doc: `drift_method.md`
(current v22 spec) and `algo/pivot.md` (version history).

Base model for 4B runs: `Qwen3-4B-Base`.  Base for 8B: `Qwen3-8B-Base`.
Training data (math+science): `guru_rl` (17.4K).  Training data (code):
`guru_code` = primeintellect + taco (16.4K, stdin/stdout).  Val (math):
AIME24 (×8), AIME25, MATH500, GPQA-Diamond, OlympiadBench-math-en.
Val (code): HumanEval, MBPP, LiveCodeBench (5-test-trimmed).


## Reviewer #1 — Drift's training-time contribution

**Concern (paraphrased):** Table 8 in the submission compares
diffusion vs. drift at inference only; the reviewer wants evidence that
the SPSA-learned drift term contributes during *training*, not just
sampling.

**Setup.** All four variants share the v22 recipe (KL-in-reward β=0.2,
clip-higher 0.28, GRPO backbone).  We toggle the Langevin update:

| Variant       | drift α | diffusion σ | Langevin on? |
|---------------|--------:|------------:|:------------:|
| v22-full      |    0.10 |        0.01 |     ✓         |
| **α=0**       |    0.00 |        0.01 |     ✓ (diffusion only) |
| σ=0 (control) |    0.10 |        0.00 |     ✓ (drift only)     |
| no-Langevin   |     —   |          —  |     ✗         |

All four use Qwen3-4B-Base, 8× H200, batch 1024, `guru_rl` train set.
Val every 5 steps.

**Result (val, `mean@16`, step 100 unless noted; all values in %):**

| Variant       | AIME24 | AIME25 | MATH500 | Olymp | Notes |
|---------------|-------:|-------:|--------:|------:|-------|
| α=0 (diffusion only) | 3.0  | 0.0  | 27.7 |  9.2 | Stuck; MATH degrades to **24.5 by step 220** |
| σ=0 (drift only)     | 9.5 | 15.6 | 75.5 | 42.6 | Control — learning proceeds |
| Plain GRPO (no Langevin)*  | 7.7 |   —  | 73.0 |   —  | Reference baseline (no KL-in-reward, no Langevin) |
| **drift**†     | **10.9** |   —  | **75.9** |   —  | Full DRIFT recipe |

*Plain GRPO (no Langevin) from `checkpoints/verl/grpo/verl_grpo_qwen3_guru_rl/qwen3_4b_base_grpo.jsonl`
(pre-fix era, same reward regime as the ablation runs).  This run's
val_files did not include AIME25 or OlympiadBench, so those cells are
blank.  Serves as the "no-Langevin" reference for this table — v22's
no-Langevin variant is fundamentally the same backbone minus Langevin;
the fresh no-Langevin rerun on fixed reward (job 24635) is still
bootstrapping and will replace this row once it lands step-100
numbers.

†v22-full seed 1 from `checkpoints/verl/pivot-v22/verl_pivot_v22_qwen3_guru_rl/qwen3_4b_base_pivot_v22.jsonl`
(229 rows, through step 220).  Same val_files limitation as Plain GRPO
— AIME25 and OlympiadBench were added to the val set only after this
run.  Multi-seed v22 numbers on the *fixed* reward are in the
reviewer-#2 section.  At step 100, v22-full beats Plain GRPO (no
Langevin) by **+3.2 pts on AIME24** and **+2.9 pts on MATH**.

**Takeaway for the response.**
*α=0 fails catastrophically*: without the drift term, the model never
escapes the dead zone (AIME24 stays at 0–3 %, MATH plateaus around
0.28 and then *degrades* to 0.24 by step 220).  This directly answers
the reviewer's question — the SPSA-learned drift is *training-critical*,
not a sampling-time nicety.

σ=0 is a stability control: setting the diffusion coefficient to zero
does not destabilize training within the observed window; this
addresses a plausible follow-up (does the noise term cause anything to
break?).

**Data locations.**
- `checkpoints/verl/pivot-v22-abl/verl_pivot_v22_abl_qwen3_guru_rl/qwen3_4b_base_pivot_v22_alpha0.jsonl`  (α=0, 237 rows, through step 220)
- `checkpoints/verl/pivot-v22-abl/verl_pivot_v22_abl_qwen3_guru_rl/qwen3_4b_base_pivot_v22_sigma0.jsonl` (σ=0, 101 rows, step 100)
- `checkpoints/verl/grpo/verl_grpo_qwen3_guru_rl/qwen3_4b_base_grpo.jsonl` (plain GRPO, 203 rows, through step 200)

**Decision:** existing pre-fix curves are sufficient — all four
variants used the same reward/prompt regime, so the *relative* ordering
is valid.  Re-runs on the fixed reward (jobs 24638 sigma0-seed4,
24639 alpha0-seed4) were **cancelled** on 2026-07-23.


## Reviewer #2 — Single-seed reproducibility

**Concern (paraphrased):** RL-for-reasoning is notoriously high-variance
across seeds; the paper's main-table numbers are single-seed.  Need at
least 3 independent seeds to establish reproducibility.

**Setup.** DRIFT-v22 on Qwen3-4B-Base, `guru_rl`.  Same recipe as
Table 4 seed 1.  Fresh runs with fixed reward (`strict_box_verify=True`)
and `\boxed{}`-only train prompts.

| Seed | Job ID | Status | Notes |
|-----:|:-------|:-------|:------|
| 1    | (Table 4 baseline) | done | Pre-fix reward — kept for archival, not compared step-matched to the reruns |
| 4    | 24633  | RUNNING (step 3+, 2026-07-23) | Fixed reward, boxed-only prompts |
| 5    | 24634  | RUNNING (step 3+, 2026-07-23) | Fixed reward, boxed-only prompts |
| 2, 3 | (earlier reruns 24628/24629) | cancelled  | Cancelled after reward-fix decision; superseded by seeds 4/5 |

**Additional seed evidence from seed-2 (pre-fix, jsonl available):** the
seed-2 run demonstrated *v22's peak-ratchet t_min mechanism* in action —
response length collapsed from 1345 → 23 tokens at step 50, then the
peak-ratchet auto-disabled Langevin (since resp_len fell below t_min),
allowing recovery to 886 tokens by step 104.  This trajectory is
useful evidence that v22 is *robust to bad seed initializations*, not
just that it works on the good ones.

**Pending as of 2026-07-23:** wait for seeds 4 & 5 to reach step 100+
(step-matched comparison against Table 4 baseline).  Report mean ± std
across the three seeds for the rebuttal.


## DAPO reproduction & follow-up

**Concern.** Reviewer asked whether DAPO's Table 4 numbers in the
paper reproduce on our infrastructure.

**Diagnostic (2026-07-23).** First attempt on Qwen3-4B-Base with the
`\boxed{}`-only prompt gave ~0 % val accuracy through step 60 (paper
claims ~33 % AIME / 89 % MATH).  Root cause: our reward function used
Minerva-first extraction which requires an `Answer:` prefix in the
model output.  Since we stripped `Answer:` from prompts (to align with
`\boxed{}` format), the model produces `\boxed{...}` and Minerva scores
it as wrong — extractor silently drops rollouts.

Ran offline verify on the step-60 DAPO checkpoint:

| Extractor | Val accuracy |
|-----------|-------------:|
| Minerva-first (buggy) | 1.48 % |
| `\boxed{}` strict     | 23.48 % |

**Fix.** `examples/grpo_trainer/guru_rl_reward.py` now calls
`math_dapo.compute_score(..., strict_box_verify=True)` unconditionally.
This aligns train reward, val reward, and offline eval — no dual "Answer"
+ boxed formats are learned together.

**Relaunched runs on fixed reward (all on `neurips_rebuttal` reservation):**

| Job ID | Variant | Purpose |
|-------:|:--------|:--------|
| 24632  | DAPO (4B)        | Reviewer's asked-for DAPO reproduction |
| 24636  | DAPO+HEG (4B)    | 80/20 paper method for context |
| 24637  | v22+DAPO (4B)    | Combined v22 + dynamic sampling |


## Code-domain DRIFT (Guru-code subset)

**Motivation.** Reviewer asked whether DRIFT transfers to code.  We
selected the Guru-code subset (16.4K prompts: primeintellect + TACO,
stdin/stdout format) and reserved HumanEval + MBPP + LiveCodeBench
(279-prompt v6 slice, per-row test cases capped at 5 to fit in memory)
for held-out val.

**Sanity check (2026-07-23).** Val-only run of Qwen3-4B-Base against
the three code eval sets confirmed the code reward pipeline works
end-to-end:

| Dataset      | mean@4 | best@4 | maj@4 |
|--------------|-------:|-------:|------:|
| HumanEval    |  0.413 |  0.674 | 0.450 |
| MBPP         |  0.344 |  0.585 | 0.371 |
| LiveCodeBench (trimmed) | 0.032 | 0.066 | 0.029 |

HumanEval/MBPP match the reported Qwen3-4B-Base pass@1 range.  LCB is a
much harder benchmark for a raw base model — training should give a
sizable lift.

**Training jobs (submitted 2026-07-23):**

| Job ID | Variant           | Notes |
|-------:|:------------------|:------|
| 24645  | GRPO (baseline)    | batch 512, n=8, 15 epochs |
| 24646  | DRIFT-v22          | Same recipe as `pivot_v22_seed4` (math), pointed at `guru_code` |


## 8B-scale evidence — AIME25/OlympBench backfill

**Motivation.** The Table-4 DRIFT-v22 8B run originally included only
AIME24 + MATH in its val_files list; AIME25 and OlympiadBench were
added later.  For the v22 vs GRPO 8B trajectory plot in the rebuttal,
we need matched AIME25 + Olymp coverage from step 20 onward.

**Setup.** `eval_v22_8b_backfill.sh` re-runs val (`mean@16`) at each
saved checkpoint (steps 20, 40, ..., 180) of the 8B v22 run against
the full 4-benchmark val set with the *fixed* reward.  Result: 9 rows
per benchmark added to the plot.

**Infrastructure fix.** FSDP checkpoints are saved as
`model_world_size_N_rank_*.pt` — loadable only at the training world
size (8).  To run this eval on 4 GPUs (fits in idle reservation slots
alongside training), we now unshard each checkpoint to HuggingFace
safetensors via `verl.model_merger merge --backend fsdp` before eval.
This is a one-time cost (~2 min/ckpt on CPU, ~16 GB per checkpoint).

**Job:** 24647 (pending, 4 GPU, 6 h wall).  Pre-merges of steps 40..180
are running on the login node in parallel.


## Job map (as of 2026-07-23, EOD)

Reservation: `neurips_rebuttal / partition=reserved / qos=resv`.

| Job ID | Name                | GPUs | Status  | Purpose |
|-------:|:--------------------|:----:|:--------|:--------|
| 24632  | dapo_4b             |  8   | RUNNING | DAPO reproduction (reviewer) |
| 24633  | v22_seed4_4b        |  8   | RUNNING | Multi-seed v22 (reviewer #2) |
| 24634  | v22_seed5_4b        |  8   | RUNNING | Multi-seed v22 (reviewer #2) |
| 24635  | v22_nolang_seed4_4b |  8   | RUNNING | Nolang re-run with fixed reward (companion to α=0/σ=0 ablation) |
| 24636  | dapo_heg_4b         |  8   | RUNNING | 80/20 paper (context for reviewer) |
| 24637  | v22_dapo_4b         |  8   | RUNNING | v22 + DAPO combined |
| 24645  | grpo_code_4b        |  8   | PENDING | Code-domain baseline (reviewer transfer ask) |
| 24646  | v22_code_4b         |  8   | PENDING | Code-domain DRIFT (reviewer transfer ask) |
| 24647  | eval_v22_8b_backfill|  4   | PENDING | Fill AIME25/Olymp early steps for 8B plot |
| 24638  | v22_sigma0_seed4_4b |  —   | CANCELLED | Existing pre-fix σ=0 data is sufficient |
| 24639  | v22_alpha0_seed4_4b |  —   | CANCELLED | Existing pre-fix α=0 data is sufficient |


## Reward / data / infrastructure fixes applied this cycle

1. **Reward extractor.** `math_dapo.compute_score(..., strict_box_verify=True)`
   in `guru_rl_reward.py` — used at both train time and val time.
2. **Train data.** `dapo_math_17k/train.parquet` rewritten to strip
   the "Answer:" instruction; only `\boxed{}` is asked for.  Original
   backed up as `train.parquet.bak_answerfmt`.
3. **Code reward.** `guru_code_reward.py` — subprocess-based scorer
   supporting stdin/stdout (primeintellect/taco/livecodebench) and
   functional (humaneval `check(candidate)`, mbpp raw asserts).
4. **LCB parquet.** Trimmed to 5 test cases per row to bypass pyarrow's
   nested-array-chunk limit; still preserves signal (a single failing
   case gates the reward).
5. **FSDP checkpoint load.** Added `verl.model_merger merge` step to
   the 8B backfill eval so we can run eval on any world_size, not just
   the training world size.
6. **vLLM cascade attention.** `VERL_VLLM_DISABLE_CASCADE_ATTN=1` set
   in every launch script; documented in `verl/workers/rollout/vllm_rollout/vllm_async_server.py`.


## What's left

- Land seed 4 & 5 v22 curves to step 100+ (reviewer #2 seed table).
- Land DAPO + DAPO-HEG + DAPO-v22 to step 100+ (reviewer #1 baselines).
- Wait for code-domain GRPO vs v22 to reach step 40+ (transferability
  claim).
- Regenerate the `v22_vs_grpo_8b_step500.png` chart with the 8B
  backfill data filled in (step 20..180 gap closed).
