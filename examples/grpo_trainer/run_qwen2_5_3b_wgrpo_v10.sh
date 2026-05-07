#!/usr/bin/env bash
# WGRPO-v10: Group-normalized walk — per-position z-score across rollouts.
#
# Replaces v6's group-level differential (mean_correct - mean_wrong) with
# per-rollout z-score normalization at each token position across the n rollouts
# sharing a prompt — directly analogous to GRPO's advantage normalization:
#
#   w_norm[i, t] = (w[i, t] - mean_n(w[:, t])) / std_n(w[:, t])
#
# Key differences from v6:
#   - Per-rollout: each rollout gets its own weight at each position, not the
#     group mean. A correct rollout that strongly attended to a hub gets more
#     amplification than one that only weakly attended there.
#   - No ReLU: below-mean walk tokens are dampened (weight < 1), not clamped
#     to 1. Analogous to GRPO keeping negative advantages.
#   - No explicit correct/wrong split: the differential signal emerges from the
#     z-score — positions where correct rollouts attend more get positive z-scores
#     for those rollouts and negative for wrong rollouts.
#   - No fallback path: no_contrast groups are handled automatically (advantages
#     are zero anyway when all rewards are equal).
#   - clamp(min=0) in apply_walk_weighted_advantage prevents advantage sign-flip.
#
# Final weight: (1 + alpha * w_norm[i, t]).clamp(min=0)
#
# See docs/algo/wgrpo.md for variants table.
set -x

DATA_DIR="${DATA_DIR:-/storage/workspace/server-1/duy/data/guru_rl}"
MODEL="${MODEL:-/storage/workspace/server-1/duy/checkpoints/models/Qwen2.5-3B-Instruct}"
REWARD_FN="$(dirname "$0")/guru_rl_reward.py"

train_files="$DATA_DIR/train.parquet"
val_files="['$DATA_DIR/test_aime.parquet','$DATA_DIR/test_math500.parquet']"

CKPT_DIR=/storage/workspace/server-1/duy/checkpoints/verl/wgrpo-v10/qwen2_5_3b

source /storage/workspace/server-1/duy/venv-vault/miniconda3/etc/profile.d/conda.sh
conda activate verl

VERL_FILE_LOGGER_ROOT=/storage/workspace/server-1/duy/checkpoints/verl/wgrpo-v10 \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.use_walk_weighted_advantage=True \
    +algorithm.walk_alpha=1.0 \
    +algorithm.walk_group_norm=True \
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
    trainer.experiment_name='qwen2_5_3b_wgrpo_v10' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    trainer.default_local_dir="$CKPT_DIR" \
    "$@"
