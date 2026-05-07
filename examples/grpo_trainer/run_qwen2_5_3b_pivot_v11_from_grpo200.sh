#!/usr/bin/env bash
# PIVOT-v11 seeded from GRPO step-200 checkpoint.
#
# Co-design loss: calibrate delta_var to per-position sensitivity.
#
#   sensitivity(t) = Pearson(pairwise_symKL_at_t, pairwise_|adv_i - adv_j|)
#
#   L(t) = (1 - 2*sensitivity(t)) * delta_var(t)
#
# High sensitivity → maximize delta_var (real fork: make model more uncertain here).
# Low sensitivity  → minimize delta_var (spurious fork: make model more confident here).
#
# No synthetic Langevin target. No cm direction. Gradient flows through
# delta_var = _grp_trig[:,t,:].var(dim=0).mean() — a differentiable group property.
#
# Closes the co-design loop: model learns to express calibrated uncertainty at
# genuinely consequential positions, making the delta_var trigger increasingly
# informative over training.
set -x

DATA_DIR="${DATA_DIR:-/storage/workspace/server-1/duy/data/guru_rl}"
MODEL="${MODEL:-/storage/workspace/server-1/duy/checkpoints/verl/grpo/qwen2_5_3b/global_step_200/actor/huggingface_merged}"
REWARD_FN="$(dirname "$0")/guru_rl_reward.py"

train_files="$DATA_DIR/train.parquet"
val_files="['$DATA_DIR/test_aime.parquet','$DATA_DIR/test_math500.parquet']"

CKPT_DIR=/storage/workspace/server-1/duy/checkpoints/verl/pivot-v11-from-grpo200/qwen2_5_3b

source /storage/workspace/server-1/duy/venv-vault/miniconda3/etc/profile.d/conda.sh
conda activate verl

export VERL_FILE_LOGGER_ROOT=/storage/workspace/server-1/duy/checkpoints/verl/pivot-v11-from-grpo200
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
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
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
    +actor_rollout_ref.rollout.pivot.delta_var_threshold=50.0 \
    +actor_rollout_ref.rollout.pivot.entropy_threshold=8.0 \
    +actor_rollout_ref.rollout.pivot.langevin_K=1 \
    +actor_rollout_ref.rollout.pivot.langevin_top_k=20 \
    +actor_rollout_ref.rollout.pivot.langevin_eta=0.1 \
    +actor_rollout_ref.rollout.pivot.langevin_sigma=0.01 \
    +actor_rollout_ref.actor.pivot.internalize_coeff=0.01 \
    +actor_rollout_ref.actor.pivot.internalize_mode=codesign \
    +actor_rollout_ref.actor.pivot.lan_grpo_coeff=1.0 \
    +actor_rollout_ref.actor.pivot.codesign_loss_clip=0.8 \
    +actor_rollout_ref.actor.pivot.entropy_cap=0.8 \
    +actor_rollout_ref.actor.pivot.n_rollouts_per_prompt=8 \
    +actor_rollout_ref.actor.pivot.delta_var_threshold=50.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path="$REWARD_FN" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","file"]' \
    trainer.project_name='verl_pivot_v11_guru_rl' \
    trainer.experiment_name='qwen2_5_3b_pivot_v11_from_grpo200' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    trainer.total_epochs=3 \
    trainer.default_local_dir="$CKPT_DIR" \
    "$@"
