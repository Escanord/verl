#!/usr/bin/env bash
# DAPO + High-Entropy 80/20 on guru-RL-92k with Qwen2.5-3B-Instruct on 8x H200.
# Matches the paper's exact DAPO config with entropy_top_ratio=0.2 added.
#
# vs GRPO baseline (run_qwen2_5_3b_guru_rl.sh):
#   - reward_manager=dapo  (clips overlong; token-level reward shaping)
#   - filter_groups        (drops all-correct / all-wrong batches)
#   - zero KL              (no use_kl_loss, no use_kl_in_reward)
#   - asymmetric clipping  (clip_ratio_low=0.2, clip_ratio_high=0.28, c=10)
#   - entropy_top_ratio=0.2 (top 20% highest-entropy tokens only)
#
# To run the plain DAPO baseline (no entropy filter), remove entropy_top_ratio.
#
# Reference: "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective
# Reinforcement Learning for LLM Reasoning", Wang et al., NeurIPS 2025.
# https://arxiv.org/abs/2506.01939
set -x

DATA_DIR="${DATA_DIR:-$HOME/duy/data/guru_rl}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
REWARD_FN="$(dirname "$0")/guru_rl_reward.py"

train_files="$DATA_DIR/train.parquet"
val_files="['$DATA_DIR/test_aime.parquet','$DATA_DIR/test_math500.parquet']"

source ~/duy/venv-vault/miniconda3/etc/profile.d/conda.sh
conda activate verl

VERL_FILE_LOGGER_ROOT=/storage/workspace/server-1/duy/checkpoints/verl/high_ent_dapo \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=acc \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.entropy_top_ratio=0.2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.reward_manager.name=dapo \
    reward.custom_reward_function.path="$REWARD_FN" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","file"]' \
    trainer.project_name='verl_high_ent_dapo_guru_rl' \
    trainer.experiment_name='qwen2_5_3b_high_ent_dapo' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    trainer.default_local_dir=/storage/workspace/server-1/duy/checkpoints/verl/high_ent_dapo/qwen2_5_3b \
    "$@"
