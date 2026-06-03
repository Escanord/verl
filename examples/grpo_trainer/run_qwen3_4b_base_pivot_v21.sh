#!/usr/bin/env bash
# PIVOT-v21 (DRIFT-v21) on Qwen3-4B-Base.
#
# v21 replaces the v18+ blended-denominator IS correction with a clean per-token
# soft-IS multiplier, derived directly from the unbiased policy gradient:
#
#   ∇J = E_{a~q}[ (π_θ/q) · A · ∇log π_θ(a) ]    (q = π_lan_old at triggers)
#
# Factoring (π_θ/q) = (π_θ/π_old)·(π_old/q) lets us keep PPO's ratio at
# r = π_θ/π_old (trust region intact) and push the off-policy correction
# (π_old/q) onto the advantage as a per-token weight:
#
#   w_t = min(1, π_old(a_t)/π_lan_old(a_t))    at trigger positions
#       = 1                                     elsewhere
#   A'_t = A_t · w_t
#
# Equivalent to multiplying the per-token pg loss by w_t (pg = -A·ratio).
#
# What changed vs v20:
#   - lan_grpo_soft_is=True       (new v21 mode)
#   - lan_grpo_correct_is=False   (old denominator patch off)
#   - lan_grpo_denom_blend dropped (was 0.5 — the √-blend trust-region hack)
#   - lan_grpo_asym_is dropped     (was True — asymmetric IS branch)
#   - lan_grpo_correct_mu dropped  (was dead code)
#
# Everything else (peak-ratchet t_min from v20, EOS mask, feedback Langevin,
# trig_percentile, entropy_threshold, momentum, exploit_ratio) is unchanged.
#
# Why this should fix collapse:
#   v18+ blend gives effective IS weight √(π_old/π_lan_old) — half the bias is
#   still on the floor. v21 applies the full IS weight (capped at 1 for
#   variance), so the gradient direction matches the true policy gradient of
#   the policy we're trying to optimize.
#
# Trains from scratch (no resume flags).
set -x

DATA_DIR="${DATA_DIR:-/home/escanord/duy/data/guru_rl}"
MODEL="${MODEL:-/home/escanord/duy/checkpoints/models/Qwen3-4B-Base}"
REWARD_FN="$(dirname "$0")/guru_rl_reward.py"

train_files="$DATA_DIR/train.parquet"
val_files="['$DATA_DIR/test_aime.parquet','$DATA_DIR/test_math500.parquet']"

CKPT_DIR=/home/escanord/duy/checkpoints/verl/pivot-v21/qwen3_4b_base

source /home/escanord/duy/venv-vault/miniconda3/etc/profile.d/conda.sh
conda activate verl

export VERL_FILE_LOGGER_ROOT=/home/escanord/duy/checkpoints/verl/pivot-v21
export VLLM_FLASH_ATTN_VERSION=2
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# Point HuggingFace caches at the user-owned home mount.  The cluster's
# default /shared/huggingface/datasets/ is read-only for non-root users on
# compute nodes; without this override the datasets library can't acquire its
# lock file and parquet loading fails with PermissionError.
export HF_HOME=$HOME/duy/.cache/huggingface
export HF_DATASETS_CACHE=$HOME/duy/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=$HOME/duy/.cache/huggingface/hub
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
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
    trainer.project_name='verl_pivot_v21_qwen3_guru_rl' \
    trainer.experiment_name='qwen3_4b_base_pivot_v21' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    trainer.total_epochs=15 \
    trainer.default_local_dir="$CKPT_DIR" \
    ray_kwargs.ray_init.num_cpus=32 \
    "$@"
