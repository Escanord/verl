#!/usr/bin/env bash
# PIVOT-v2: Representation-divergence-driven fork detection + Langevin rollout.
#
# Key differences from PIVOT-v1:
#
#   Phase 1 (loss gating):
#     v1 — temporal walk (per-rollout, Q/K hooks across all layers, extra fwd pass)
#     v2 — representation divergence (cross-rollout, reward-conditioned)
#            v[t] = mean(h_t|correct) - mean(h_t|wrong), fork_score[t] = ||v[t]||
#            hooks last-layer hidden states, no extra forward pass
#
#   Phase 2 (Langevin trigger):
#     v1 — temporal walk computed online inside vLLM (Q/K hooks during decode)
#     v2 — fork score profile from Phase 1 saved to disk after each step;
#            vLLM loads it as position-level trigger at next generation batch;
#            fallback to entropy threshold (inference / first step)
#
# To run Phase 1 only (no Langevin), remove the
# actor_rollout_ref.rollout.pivot.langevin_rollout=True line.
#
# See docs/algo/pivot.md (PIVOT-v2 section) for full design rationale.
set -x

DATA_DIR="${DATA_DIR:-/storage/workspace/server-1/duy/data/guru_rl}"
MODEL="${MODEL:-/storage/workspace/server-1/duy/checkpoints/models/Qwen2.5-3B-Instruct}"
REWARD_FN="$(dirname "$0")/guru_rl_reward.py"

train_files="$DATA_DIR/train.parquet"
val_files="['$DATA_DIR/test_aime.parquet','$DATA_DIR/test_math500.parquet']"

CKPT_DIR=/storage/workspace/server-1/duy/checkpoints/verl/pivot-v2/qwen2_5_3b

# Shared path for fork profile (trainer writes, vLLM servers read)
FORK_PROFILE_PATH=/tmp/pivot_v2_fork_profile.npy

source /storage/workspace/server-1/duy/venv-vault/miniconda3/etc/profile.d/conda.sh
conda activate verl

VERL_FILE_LOGGER_ROOT=/storage/workspace/server-1/duy/checkpoints/verl/pivot-v2 \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.use_pivot=True \
    +algorithm.pivot_version=2 \
    +algorithm.pivot_mode=soft \
    +algorithm.pivot_threshold=0.3 \
    +algorithm.pivot_alpha=1.0 \
    +algorithm.pivot_fork_profile_path="$FORK_PROFILE_PATH" \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.pivot.norm_mode=relu_max \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    +actor_rollout_ref.rollout.pivot.pivot_version=2 \
    +actor_rollout_ref.rollout.pivot.langevin_rollout=True \
    +actor_rollout_ref.rollout.pivot.fork_profile_path="$FORK_PROFILE_PATH" \
    +actor_rollout_ref.rollout.pivot.langevin_threshold=0.3 \
    +actor_rollout_ref.rollout.pivot.entropy_threshold=2.0 \
    +actor_rollout_ref.rollout.pivot.langevin_K=3 \
    +actor_rollout_ref.rollout.pivot.langevin_eta=0.1 \
    +actor_rollout_ref.rollout.pivot.langevin_sigma=0.01 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path="$REWARD_FN" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","file"]' \
    trainer.project_name='verl_pivot_v2_guru_rl' \
    trainer.experiment_name='qwen2_5_3b_pivot_v2' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    trainer.default_local_dir="$CKPT_DIR" \
    "$@"
