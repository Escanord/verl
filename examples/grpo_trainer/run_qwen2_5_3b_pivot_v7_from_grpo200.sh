#!/usr/bin/env bash
# PIVOT-v7 seeded from GRPO step-200 checkpoint.
#
# Fixes v6 entropy collapse.  v6 pushed π_θ toward the group-mean Langevin
# distribution at trigger positions, which is a diffuse target (it blends all
# rollouts including losing ones) and creates a feedback loop:
#
#   diffuse π_θ → diverse rollout samples → high ΔVar → Langevin fires →
#   group mean is diffuse → internalize toward diffuse target → diffuse π_θ
#
# v7 breaks the loop by changing the internalization target from the group-mean
# Langevin distribution to the winning rollout's own token at each trigger position:
#
#   L_intern = Σ_{b: adv(b)>0} Σ_{t: trigger} adv(b,t) * (−log π_θ(responses[b,t]))
#
# This is advantage-weighted NLL restricted to branch points.  On successful
# internalization π_θ sharpens toward the correct token at position t, which
# reduces ΔVar in the next rollout → Langevin fires less → loop terminates.
#
# Rollout (v4b, unchanged):
#   logits_new[i] = (1 - eta) * logits[i] + eta * group_mean + sigma * noise
#   triggered by delta_var > threshold
set -x

DATA_DIR="${DATA_DIR:-/storage/workspace/server-1/duy/data/guru_rl}"
MODEL="${MODEL:-/storage/workspace/server-1/duy/checkpoints/verl/grpo/qwen2_5_3b/global_step_200/actor/huggingface_merged}"
REWARD_FN="$(dirname "$0")/guru_rl_reward.py"

train_files="$DATA_DIR/train.parquet"
val_files="['$DATA_DIR/test_aime.parquet','$DATA_DIR/test_math500.parquet']"

CKPT_DIR=/storage/workspace/server-1/duy/checkpoints/verl/pivot-v7-from-grpo200/qwen2_5_3b

source /storage/workspace/server-1/duy/venv-vault/miniconda3/etc/profile.d/conda.sh
conda activate verl

export VERL_FILE_LOGGER_ROOT=/storage/workspace/server-1/duy/checkpoints/verl/pivot-v7-from-grpo200
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    +actor_rollout_ref.rollout.pivot.pivot_version=4 \
    +actor_rollout_ref.rollout.pivot.langevin_rollout=True \
    +actor_rollout_ref.rollout.pivot.delta_var_threshold=5.0 \
    +actor_rollout_ref.rollout.pivot.entropy_threshold=8.0 \
    +actor_rollout_ref.rollout.pivot.langevin_K=1 \
    +actor_rollout_ref.rollout.pivot.langevin_top_k=20 \
    +actor_rollout_ref.rollout.pivot.langevin_eta=0.1 \
    +actor_rollout_ref.rollout.pivot.langevin_sigma=0.01 \
    +actor_rollout_ref.actor.pivot.internalize_coeff=0.001 \
    +actor_rollout_ref.actor.pivot.entropy_cap=1.5 \
    +actor_rollout_ref.actor.pivot.decay_steps=150 \
    +actor_rollout_ref.actor.pivot.n_rollouts_per_prompt=8 \
    +actor_rollout_ref.actor.pivot.delta_var_threshold=5.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path="$REWARD_FN" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","file"]' \
    trainer.project_name='verl_pivot_v7_guru_rl' \
    trainer.experiment_name='qwen2_5_3b_pivot_v7_from_grpo200' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=3 \
    trainer.default_local_dir="$CKPT_DIR" \
    "$@"
