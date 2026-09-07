# Running the 3 DRIFT/heg 4B scripts on a fresh machine

This doc reproduces, from scratch on a new machine, three Qwen3-4B-Base RL runs:

| # | combo | run script (`examples/grpo_trainer/`) | writes to (`checkpoints/verl/`) |
|---|---|---|---|
| 1 | **DAPO + heg** | `run_qwen3_4b_base_dapo_heg.sh` | `dapo/qwen3_4b_base_heg` |
| 2 | **GRPO + heg + mx_cap2 (union)** | `run_qwen3_4b_base_pivot_v22_neffmax_cap2_heg_union.sh` | `pivot-v22/qwen3_4b_base_neffmax_cap2_heg_union` |
| 3 | **DAPO + heg + mx_cap2 (union)** | `run_qwen3_4b_base_pivot_v22_neffmax_cap2_heg_union_dapo.sh` | `pivot-v22-dapo/qwen3_4b_base_neffmax_cap2_heg_union` |

- **heg** = High-Entropy GRPO: policy-gradient masked to the top-20 %-entropy tokens (`entropy_top_ratio=0.2`).
- **mx_cap2** = DRIFT-v22: N_eff-band Langevin trigger (fire iff `2 ≤ 1/Σp² ≤ 8`), soft-IS advantage reweight `w = min(2, π_old/π_lan_old)` (`lan_grpo_softis_wmax=2.0`), noise off (`langevin_exploit_ratio=1.0`).
- **union** = gradient acts on (top-entropy **∪** Langevin-trigger) tokens instead of their intersection (`entropy_top_union_trigger=True`). Script 1 has no DRIFT, so no union flag.

> ⚠️ **These runs require the *customized* verl in this repo — NOT upstream verl.** The DRIFT/heg logic lives in patched files (see §2). `pip install verl` from PyPI will **not** work: it lacks `pivot_version`, `langevin_rollout`, `entropy_top_ratio`, `entropy_top_union_trigger`, etc.

---

## 0. Hardware & software baseline (what these were validated on)

- **GPUs:** 8 × NVIDIA H200 (80 GB works too; needs ≥ ~70 GB/GPU for the 20 k-response DAPO runs). Single node, 8 GPUs.
- **CUDA:** 12.9 runtime (12.8 torch build). Driver must support CUDA 12.x.
- **OS:** Linux x86_64.
- **Key package versions** (full pin list in `docs/requirements-freeze-4b-heg-drift.txt`):
  - python **3.12.0**, torch **2.8.0+cu128**, vllm **0.11.0**, ray **2.54.0**,
    transformers **4.56.1**, flash-attn **2.8.1** (cu12/torch2.8), datasets 4.8.4,
    tensordict 0.10.0, hydra-core 1.3.2, omegaconf 2.3.0, liger_kernel 0.7.0.

---

## 1. Get the code (the customized verl fork)

Copy **this entire `verl/` directory** to the new machine (it is a self-contained verl 0.8.0.dev
checkout with the DRIFT/heg patches; it has no git remote, so clone-from-GitHub is not an option —
transfer the tree itself):

```bash
# from the source machine
rsync -av --exclude='.git' --exclude='__pycache__' \
      /home/escanord/duy/verl/  NEWHOST:/path/to/verl/
```

It already contains: the 3 run scripts, the reward function
(`examples/grpo_trainer/guru_rl_reward.py`), the flash-attn wheel
(`flash_attn-2.8.1+cu12torch2.8...whl`), and the patched source (§2).

## 2. Custom source files (do not overwrite with upstream)

The DRIFT/heg behaviour is implemented in these files — they must be the patched versions from this repo:

```
verl/trainer/ppo/core_algos.py            # get_global_entropy_top_mask (heg)
verl/workers/actor/dp_actor.py            # heg masking + soft-IS reweight + entropy_top_union_trigger
verl/workers/config/actor.py              # actor config: entropy_top_ratio, entropy_top_union_trigger, pivot.*
verl/workers/config/rollout.py            # rollout pivot config
verl/workers/rollout/vllm_rollout/vllm_async_server.py   # Langevin rollout (pivot_version=2)
verl/workers/rollout/vllm_rollout/utils.py               # Langevin helpers
verl/trainer/ppo/ray_trainer.py           # plumbing
```

## 3. Create the Python environment

```bash
# Miniconda/conda assumed. Create a py3.12 env named 'verl':
conda create -y -n verl python=3.12.0
conda activate verl

# Install CUDA 12.8 torch stack first, then the rest:
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install the local flash-attn wheel shipped in the repo (matches torch2.8/cu12/py312):
pip install /path/to/verl/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# Install verl (editable) + its deps:
cd /path/to/verl
pip install -e .

# If any versions drift, pin exactly from the captured freeze:
pip install -r docs/requirements-freeze-4b-heg-drift.txt
```

> The freeze file lists `flash_attn @ file:///home/escanord/duy/verl/flash_attn-...whl` — that path
> is machine-specific; install the wheel by its new path as shown above rather than from the freeze line.

## 4. Get the model — Qwen3-4B-Base

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-4B-Base --local-dir /path/to/models/Qwen3-4B-Base
```
(It is the **pretrained base** model — `model_type: qwen3`, `architectures: Qwen3ForCausalLM`.
Not the instruct `Qwen3-4B`.)

## 5. Get the data — guru_rl

Copy the `guru_rl` parquet set (these exact files; ~40 MB total). The 3 scripts each read
`train.parquet` for training and the 5 math/knowledge test parquets for validation:

```
data/guru_rl/train.parquet                          # RL training prompts (~37 MB)
data/guru_rl/test_aime.parquet                      # AIME24 (repeated)
data/guru_rl/test_aime25.parquet                    # AIME25
data/guru_rl/test_math500.parquet                   # MATH500
data/guru_rl/test_gpqa_diamond.parquet              # GPQA-Diamond
data/guru_rl/test_olympiadbench_math_en.parquet     # OlympiadBench-math-en
```
```bash
rsync -av /home/escanord/duy/data/guru_rl/  NEWHOST:/path/to/data/guru_rl/
```
Schema per row: `data_source`, `prompt` (chat list), `reward_model.ground_truth`, `extra_info`.
The reward is `examples/grpo_trainer/guru_rl_reward.py` (`compute_score`, binary ±1) — already in the repo.

---

## 6. Point the scripts at the new machine's paths

Each script has 3 machine-specific locations. Two are **env-overridable**, one must be **edited**:

| what | how | default in script |
|---|---|---|
| **DATA_DIR** | `export DATA_DIR=/path/to/data/guru_rl` | `/home/escanord/duy/data/guru_rl` |
| **MODEL** | `export MODEL=/path/to/models/Qwen3-4B-Base` | `/home/escanord/duy/checkpoints/models/Qwen3-4B-Base` |
| **conda activation** | edit the `source .../miniconda3/etc/profile.d/conda.sh` line | `/home/escanord/duy/venv-vault/miniconda3/...` |
| **CKPT_DIR + `VERL_FILE_LOGGER_ROOT`** | edit near the top of each script | `/home/escanord/duy/checkpoints/verl/...` |

So the minimal edit per script is: fix the `source .../conda.sh` line, and change the
`CKPT_DIR=` and `export VERL_FILE_LOGGER_ROOT=` prefixes to your checkpoints root.
`DATA_DIR`/`MODEL` you can just export before running (they use `${VAR:-default}`).

Metrics/checkpoints land under `VERL_FILE_LOGGER_ROOT`:
`<VERL_FILE_LOGGER_ROOT>/<project_name>/<experiment_name>.jsonl` for eval metrics, and
`CKPT_DIR/global_step_*/` for checkpoints (save/test_freq = 5).

---

## 7. Run

### Direct (single 8-GPU node, no SLURM)
```bash
conda activate verl
cd /path/to/verl
export DATA_DIR=/path/to/data/guru_rl
export MODEL=/path/to/models/Qwen3-4B-Base
# vLLM stability flags the scripts already export are fine; run any of:
bash examples/grpo_trainer/run_qwen3_4b_base_dapo_heg.sh
bash examples/grpo_trainer/run_qwen3_4b_base_pivot_v22_neffmax_cap2_heg_union.sh
bash examples/grpo_trainer/run_qwen3_4b_base_pivot_v22_neffmax_cap2_heg_union_dapo.sh
```

### SLURM
Reference sbatch files (edit `--account`/`--partition`/paths for the new cluster):
```
slurm/verl/dapo_heg_4b_long.sbatch
slurm/verl/v22_neffmax_cap2_heg_union_4b.sbatch
slurm/verl/v22_neffmax_cap2_heg_union_dapo_4b.sbatch
```
Each is a single node, `--gres=gpu:8`, 7-day walltime, and calls the matching run script.
`resume_mode=auto` (verl default) resumes from the latest `CKPT_DIR/global_step_*` on requeue.

---

## 8. Sanity checks before a long run

1. **Env import:** `python -c "import verl, vllm, torch, flash_attn; print('ok')"`.
2. **Custom flags present:** `grep -q entropy_top_union_trigger verl/verl/workers/actor/dp_actor.py && echo patched`.
3. **Data loads / val concatenates:** the run reaches "Filtering prompts…" then vLLM init without a
   `datasets` feature-alignment error (all 5 val parquets share schema — they do for guru_rl).
4. **First eval @ step 5** writes a row with keys
   `val-core/math__aime25/acc/mean@16`, `.../math__math/...`, `.../mcq__gpqa_diamond/...`, etc.
5. **DRIFT firing (scripts 2 & 3 only):** the vLLM worker logs `PIVOT-v2 [summary] … trigger_frac=…`
   — confirms the Langevin trigger is active.

## 9. Expected cost

- Script 1 & 3 (DAPO base, 20 480 resp + filter_groups): **hours per step**, ~525 total steps.
- Script 2 (GRPO base, 16 384 resp): faster, but still a multi-day run to a useful depth.
- Budget one 8-GPU node per script; they are independent and can run concurrently on separate nodes.
