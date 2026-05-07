#!/usr/bin/env bash
# PIVOT-v2 seeded from GRPO step-200 checkpoint.
#
# Motivation: step-200 is ~90 steps past breakthrough (~step 110), deep in the
# fast-learning phase. The model has learned the Answer: output format, has
# sufficient correct rollouts for Phase 1 (representation divergence signal),
# and generation entropy at routine positions is low enough that
# entropy_threshold=2.0 fires only at genuine branch points.
#
# step-120 was rejected: only 10 steps past breakthrough, weights still
# essentially identical to base model, 0% val accuracy, 45% clip ratio.
#
# v2 calibration (from run 1, steps 1-57):
#   - langevin_threshold=0.3 was 300x too high vs actual fork scores (~0.001).
#     Lowered to 0.005 so the fork profile path actually fires (top ~5-10% of
#     structurally identified branch points).
#   - entropy_threshold=2.0 was causing Langevin to fire on ~0.13% of positions
#     per token, corrupting ~78% of sequences to max-length (Poisson: 1-e^-1.5).
#     Raised to 3.5 (model max observed entropy ~2.64 nats) to prevent fallback
#     corruption while keeping it as a safety net.
#   - max_response_length: 4096→8192 to allow the model to actually solve hard
#     problems once length inflation is fixed.
#
# See docs/algo/pivot.md for full design rationale.
set -x

DATA_DIR="${DATA_DIR:-/storage/workspace/server-1/duy/data/guru_rl}"
MODEL="${MODEL:-/storage/workspace/server-1/duy/checkpoints/verl/grpo/qwen2_5_3b/global_step_200/actor/huggingface_merged}"
REWARD_FN="$(dirname "$0")/guru_rl_reward.py"

train_files="$DATA_DIR/train.parquet"
val_files="['$DATA_DIR/test_aime.parquet','$DATA_DIR/test_math500.parquet']"

CKPT_DIR=/storage/workspace/server-1/duy/checkpoints/verl/pivot-v2-from-grpo200/qwen2_5_3b

source /storage/workspace/server-1/duy/venv-vault/miniconda3/etc/profile.d/conda.sh
conda activate verl

VERL_FILE_LOGGER_ROOT=/storage/workspace/server-1/duy/checkpoints/verl/pivot-v2-from-grpo200 \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.use_pivot=True \
    +algorithm.pivot_version=2 \
    +algorithm.pivot_mode=soft \
    +algorithm.pivot_threshold=0.005 \
    +algorithm.pivot_alpha=1.0 \
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
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.pivot.norm_mode=relu_max \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    +actor_rollout_ref.rollout.pivot.pivot_version=2 \
    +actor_rollout_ref.rollout.pivot.langevin_rollout=True \
    +actor_rollout_ref.rollout.pivot.delta_var_threshold=5.0 \
    +actor_rollout_ref.rollout.pivot.entropy_threshold=8.0 \
    +actor_rollout_ref.rollout.pivot.langevin_K=1 \
    +actor_rollout_ref.rollout.pivot.langevin_top_k=20 \
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
    trainer.experiment_name='qwen2_5_3b_pivot_v2_from_grpo200' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=3 \
    trainer.default_local_dir="$CKPT_DIR" \
    "$@"
