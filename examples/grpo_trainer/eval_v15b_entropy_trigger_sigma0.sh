#!/usr/bin/env bash
# Evaluate v15b step 180: pure-gradient Langevin (σ=0, no noise).
# Isolates the entropy gradient push from the stochastic noise term.
# ε~N(0,σ²) is zeroed out: logits_new = logits + η·∇H(p) only.
#
# Contrast with:
#   eval_v15b_entropy_trigger.sh       — σ=0.01 (baseline, 3.57% mean@16)
#   eval_v15b_entropy_trigger_sigma01.sh — σ=0.10 (more noise)
#   eval_v15b_no_langevin.sh           — pure π (2.97% mean@16)
set -x

DATA_DIR="${DATA_DIR:-/storage/workspace/server-1/duy/data/guru_rl}"
BASE_MODEL="${BASE_MODEL:-/storage/workspace/server-1/duy/checkpoints/verl/grpo/qwen2_5_3b/global_step_200/actor/huggingface_merged}"
REWARD_FN="$(dirname "$0")/guru_rl_reward.py"

CKPT_DIR=/storage/workspace/server-1/duy/checkpoints/verl/pivot-v15b-from-grpo200/qwen2_5_3b
STEP="${1:-global_step_180}"
RESUME_PATH="$CKPT_DIR/$STEP"
shift 2>/dev/null || true

val_files="['$DATA_DIR/test_aime.parquet','$DATA_DIR/test_math500.parquet']"

source /storage/workspace/server-1/duy/venv-vault/miniconda3/etc/profile.d/conda.sh
conda activate verl

export VERL_FILE_LOGGER_ROOT=/storage/workspace/server-1/duy/checkpoints/verl/eval-v15b-entropy-trigger-sigma0

echo "==> Evaluating v15b checkpoint (entropy trigger σ=0, pure gradient): $RESUME_PATH"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$val_files" \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$BASE_MODEL" \
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
    +actor_rollout_ref.rollout.pivot.entropy_threshold=8.0 \
    +actor_rollout_ref.rollout.pivot.entropy_trigger_only=True \
    +actor_rollout_ref.rollout.pivot.langevin_K=1 \
    +actor_rollout_ref.rollout.pivot.langevin_top_k=20 \
    +actor_rollout_ref.rollout.pivot.langevin_eta=0.1 \
    +actor_rollout_ref.rollout.pivot.langevin_sigma=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path="$REWARD_FN" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","file"]' \
    trainer.project_name='verl_eval_v15b_entropy_trigger_sigma0' \
    trainer.experiment_name="eval_v15b_entropy_trigger_sigma0_${STEP}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$RESUME_PATH" \
    trainer.default_local_dir=/storage/workspace/server-1/duy/checkpoints/verl/eval-v15b-entropy-trigger-sigma0 \
    "$@"
