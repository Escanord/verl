#!/usr/bin/env bash
# Sanity check: run val-only on fresh Qwen3-4B-Base against HumanEval + MBPP
# using the new guru_code_reward.py.  Verifies end-to-end that:
#   - the code parquets load with the correct data_source column
#   - the code reward function fires per rollout without crashing
#   - the val-core/<data_source>/acc/mean@16 metrics land in the log
# Purpose: catch any wiring bug before submitting multi-day training runs.
#
# Runs on 4 GPUs (half a node) — small footprint since it's val-only.
set -x

DATA_DIR="${DATA_DIR:-/home/escanord/duy/data/guru_code}"
MODEL="${MODEL:-/home/escanord/duy/checkpoints/models/Qwen3-4B-Base}"
REWARD_FN="$(dirname "$0")/../grpo_trainer/guru_code_reward.py"
OUT_DIR="${OUT_DIR:-/home/escanord/duy/checkpoints/verl/eval_out/sanity_code_4b}"

val_files="['$DATA_DIR/test_humaneval.parquet','$DATA_DIR/test_mbpp.parquet','$DATA_DIR/test_livecodebench.parquet']"
# train file required by scaffold but never consumed (val_only=True)
train_files="$DATA_DIR/train.parquet"

mkdir -p "$OUT_DIR"

source /home/escanord/duy/venv-vault/miniconda3/etc/profile.d/conda.sh
conda activate verl

export VERL_FILE_LOGGER_ROOT="$OUT_DIR"
export VERL_VLLM_DISABLE_CASCADE_ATTN=1
export VERL_VLLM_CUDAGRAPH_MODE=PIECEWISE
export HF_HOME=$HOME/duy/.cache/huggingface

python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.nccl_timeout=1800 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=64 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.custom_reward_function.path="$REWARD_FN" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","file"]' \
    trainer.project_name='sanity_code_4b' \
    trainer.experiment_name='sanity_code_4b_base' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.val_before_train=True \
    ++trainer.val_only=True \
    trainer.total_epochs=1 \
    trainer.default_local_dir="$OUT_DIR" \
    ray_kwargs.ray_init.num_cpus=16 \
    "$@"
