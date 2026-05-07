#!/usr/bin/env bash
# WGRPO-v7: Fork-block gradient concentration on top of v6 differential walk.
#
# Builds on v6 (differential walk) with an additional step:
#
#   fork_block = argmax_{j} mean_{group} delta_wi_block[j]
#   tokens in fork_block get walk weight × fork_boost (default 2.0)
#
# The fork block is the structural decision point: the response block where
# correct and wrong solutions diverge most in their attention structure.
# Amplifying gradient there concentrates learning signal at the pivotal
# reasoning step without generating additional rollouts.
#
# This is the "soft fork" version of Option C. The full forked-generation
# version (generating additional rollouts from the fork point, conditioning
# on a correct prefix) is the next direction — see
# docs/algo/spectral_differential_walk.md for the design.
#
# See docs/algo/wgrpo.md for the variants table.
set -x

DATA_DIR="${DATA_DIR:-/storage/workspace/server-1/duy/data/guru_rl}"
MODEL="${MODEL:-/storage/workspace/server-1/duy/checkpoints/models/Qwen2.5-3B-Instruct}"
REWARD_FN="$(dirname "$0")/guru_rl_reward.py"

train_files="$DATA_DIR/train.parquet"
val_files="['$DATA_DIR/test_aime.parquet','$DATA_DIR/test_math500.parquet']"

CKPT_DIR=/storage/workspace/server-1/duy/checkpoints/verl/wgrpo-v7/qwen2_5_3b

source /storage/workspace/server-1/duy/venv-vault/miniconda3/etc/profile.d/conda.sh
conda activate verl

VERL_FILE_LOGGER_ROOT=/storage/workspace/server-1/duy/checkpoints/verl/wgrpo-v7 \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.use_walk_weighted_advantage=True \
    +algorithm.walk_alpha=1.0 \
    +algorithm.walk_differential=True \
    +algorithm.walk_fork_boost=True \
    +algorithm.walk_fork_boost_factor=2.0 \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.walk_importance.block_size=32 \
    +actor_rollout_ref.actor.walk_importance.walk_degree=4 \
    +actor_rollout_ref.actor.walk_importance.norm_mode=relu_max \
    +actor_rollout_ref.actor.walk_importance.agg_mode=causal \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path="$REWARD_FN" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","file"]' \
    trainer.project_name='verl_wgrpo_guru_rl' \
    trainer.experiment_name='qwen2_5_3b_wgrpo_v7' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    trainer.default_local_dir="$CKPT_DIR" \
    "$@"
