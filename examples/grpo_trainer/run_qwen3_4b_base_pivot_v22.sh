#!/usr/bin/env bash
# PIVOT-v22 (DRIFT-v22) on Qwen3-4B-Base.
#
# v22 adds InstructGPT-style KL-from-base reward shaping on top of v21's
# soft IS multiplier.  Rationale: v21 collapsed harder than v18+ because
#   (a) the IS correction only acts at trigger positions, but post-collapse
#       (resp_len < t_min) there are no triggers and v21 becomes a no-op;
#   (b) plain GRPO with a length-independent reward has a degenerate
#       local optimum at "guess the answer template directly" once the
#       model's reasoning ability is below its short-guess success rate.
#
# v22 reward shaping (per rollout, before advantage):
#   r'_i  =  r_correct_i  -  beta · mean_t( KL(π_θ(·|s_t) || π_ref(·|s_t)) )
#
# - Per-rollout MEAN KL (not sum):  long natural rollouts have low per-token
#   KL from base, get minimal penalty.  Short shortcut rollouts have high
#   per-token KL from base (the policy is OOD relative to the natural
#   reasoning manifold), get heavy penalty.  Sum-of-KL (verl default) would
#   actually bias the wrong direction since long rollouts have more tokens.
# - Applied at REWARD level (enters advantage) rather than as a separate
#   loss term — this changes the GRPO objective, not just the gradient
#   magnitude.  The existing kl_loss_coef=0.001 was at the loss level,
#   which is why it couldn't prevent collapse.
#
# v21's per-trigger soft IS multiplier stays on.  Both mechanisms compose:
# the IS multiplier handles off-policy gradient at Langevin-firing positions;
# the KL-shaped reward handles the global GRPO objective.
#
# Diagnostic logging added to verify the collapse-mechanism hypothesis:
# `diag/short_lt200_*`, `diag/mid_200_800_*`, `diag/long_ge800_*` track
# per-length-stratum reward distribution at every train step.  The
# hypothesis predicts that during the catastrophic phase (v21 steps 40-45)
# we'd see short_reward_mean cross above long_reward_mean — that's when
# GRPO discovers the shortcut.
#
# Trains from scratch (no resume flags).
set -x

DATA_DIR="${DATA_DIR:-/home/escanord/duy/data/guru_rl}"
MODEL="${MODEL:-/home/escanord/duy/checkpoints/models/Qwen3-4B-Base}"
REWARD_FN="$(dirname "$0")/guru_rl_reward.py"

train_files="$DATA_DIR/train.parquet"
val_files="['$DATA_DIR/test_aime.parquet','$DATA_DIR/test_math500.parquet']"

CKPT_DIR=/home/escanord/duy/checkpoints/verl/pivot-v22/qwen3_4b_base

source /home/escanord/duy/venv-vault/miniconda3/etc/profile.d/conda.sh
conda activate verl

export VERL_FILE_LOGGER_ROOT=/home/escanord/duy/checkpoints/verl/pivot-v22
export VLLM_FLASH_ATTN_VERSION=2
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HOME=$HOME/duy/.cache/huggingface
export HF_DATASETS_CACHE=$HOME/duy/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=$HOME/duy/.cache/huggingface/hub
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=True \
    +algorithm.kl_in_reward_mode=rollout_mean \
    algorithm.kl_penalty=low_var_kl \
    algorithm.kl_ctrl.kl_coef=0.2 \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=1024 \
    data.val_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    +actor_rollout_ref.rollout.pivot.pivot_version=2 \
    +actor_rollout_ref.rollout.pivot.langevin_rollout=True \
    +actor_rollout_ref.rollout.pivot.entropy_trigger_only=True \
    +actor_rollout_ref.rollout.pivot.trig_percentile=85 \
    +actor_rollout_ref.rollout.pivot.entropy_threshold=0.4 \
    +actor_rollout_ref.rollout.pivot.langevin_K=1 \
    +actor_rollout_ref.rollout.pivot.langevin_top_k=20 \
    +actor_rollout_ref.rollout.pivot.langevin_eta=0.1 \
    +actor_rollout_ref.rollout.pivot.langevin_sigma=0.01 \
    +actor_rollout_ref.rollout.pivot.langevin_min_trigger_position=800 \
    +actor_rollout_ref.rollout.pivot.langevin_min_trigger_position_mode=peak \
    +actor_rollout_ref.rollout.pivot.langevin_peak_alpha=0.6 \
    +actor_rollout_ref.rollout.pivot.langevin_t_min_floor=200 \
    +actor_rollout_ref.rollout.pivot.langevin_t_min_cap=3200 \
    +actor_rollout_ref.rollout.pivot.langevin_mask_eos=True \
    +actor_rollout_ref.rollout.pivot.langevin_momentum=0.7 \
    +actor_rollout_ref.rollout.pivot.langevin_feedback=True \
    +actor_rollout_ref.rollout.pivot.langevin_exploit_ratio=0.6 \
    +actor_rollout_ref.rollout.pivot.langevin_alpha_target=0.7 \
    +actor_rollout_ref.actor.pivot.pivot_version=2 \
    +actor_rollout_ref.actor.pivot.lan_grpo_coeff=1.0 \
    +actor_rollout_ref.actor.pivot.lan_grpo_soft_is=True \
    +actor_rollout_ref.actor.pivot.entropy_cap=0.8 \
    +actor_rollout_ref.actor.pivot.n_rollouts_per_prompt=8 \
    +actor_rollout_ref.actor.pivot.entropy_trigger_only=True \
    +actor_rollout_ref.actor.pivot.trig_percentile=85 \
    +actor_rollout_ref.actor.pivot.lan_grpo_restrict_to_trigger=False \
    +actor_rollout_ref.actor.pivot.langevin_eta=0.1 \
    +actor_rollout_ref.actor.pivot.langevin_top_k=20 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.custom_reward_function.path="$REWARD_FN" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","file"]' \
    trainer.project_name='verl_pivot_v22_qwen3_guru_rl' \
    trainer.experiment_name='qwen3_4b_base_pivot_v22' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    trainer.total_epochs=15 \
    trainer.default_local_dir="$CKPT_DIR" \
    ray_kwargs.ray_init.num_cpus=32 \
    "$@"
