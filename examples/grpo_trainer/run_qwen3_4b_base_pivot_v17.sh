#!/usr/bin/env bash
# PIVOT-v17 on Qwen3-4B-Base.
#
# Replaces the entropy-gradient drift with an adaptive G-guided noise direction,
# learned online from per-step entropy feedback:
#
#   At trigger position t:
#     ε_t  = α·(G/‖G‖) + (1-α)·randn_unit   (α=0.5: half exploit, half explore)
#     logits_t += σ·ε_t                       (no ∇H drift)
#
#   At position t+1 (observing outcome):
#     signal = H_baseline − H_{t+1}           (positive = more confident after trigger)
#     G ← γ·G + (1-γ)·signal·ε_t            (γ=0.5: fast adaptation)
#
# G accumulates the SPSA-style gradient estimate of V w.r.t. logit noise direction,
# using next-step entropy as the on-the-fly value proxy. When ε pushed toward a
# more coherent continuation (lower H), G reinforces that direction; when ε made
# things worse, G reverses. Bidirectional, self-correcting.
#
# vs v15f/v15h:
#   - No ∇H drift (removes entropy-maximisation bias)
#   - G replaces blind momentum (adaptive direction from feedback, not fixed ∇H)
#   - γ=0.5 instead of 0.9 (fast adaptation, 2-step effective window)
#   - min_trigger_position=200 (shorter sink, allows shortcut-answer breakout)
#   - lan_grpo_restrict_to_trigger=False (all positions contribute to GRPO loss)
#   - Per-step entropy buffer reset (already in code, keeps calibration tight)
set -x

DATA_DIR="${DATA_DIR:-/storage/workspace/server-1/duy/data/guru_rl}"
MODEL="${MODEL:-/storage/workspace/server-1/duy/checkpoints/models/Qwen3-4B-Base}"
REWARD_FN="$(dirname "$0")/guru_rl_reward.py"

train_files="$DATA_DIR/train.parquet"
val_files="['$DATA_DIR/test_aime.parquet','$DATA_DIR/test_math500.parquet']"

CKPT_DIR=/storage/workspace/server-1/duy/checkpoints/verl/pivot-v17/qwen3_4b_base

source /storage/workspace/server-1/duy/venv-vault/miniconda3/etc/profile.d/conda.sh
conda activate verl

export VERL_FILE_LOGGER_ROOT=/storage/workspace/server-1/duy/checkpoints/verl/pivot-v17
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
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
    +actor_rollout_ref.rollout.pivot.trig_percentile=65 \
    +actor_rollout_ref.rollout.pivot.entropy_threshold=0.4 \
    +actor_rollout_ref.rollout.pivot.langevin_K=1 \
    +actor_rollout_ref.rollout.pivot.langevin_top_k=20 \
    +actor_rollout_ref.rollout.pivot.langevin_eta=0.1 \
    +actor_rollout_ref.rollout.pivot.langevin_sigma=0.01 \
    +actor_rollout_ref.rollout.pivot.langevin_min_trigger_position=200 \
    +actor_rollout_ref.rollout.pivot.langevin_momentum=0.5 \
    +actor_rollout_ref.rollout.pivot.langevin_feedback=True \
    +actor_rollout_ref.rollout.pivot.langevin_exploit_ratio=0.5 \
    +actor_rollout_ref.actor.pivot.pivot_version=2 \
    +actor_rollout_ref.actor.pivot.lan_grpo_coeff=1.0 \
    +actor_rollout_ref.actor.pivot.lan_grpo_correct_is=True \
    +actor_rollout_ref.actor.pivot.lan_grpo_correct_mu=True \
    +actor_rollout_ref.actor.pivot.lan_grpo_denom_blend=0.5 \
    +actor_rollout_ref.actor.pivot.entropy_cap=0.8 \
    +actor_rollout_ref.actor.pivot.n_rollouts_per_prompt=8 \
    +actor_rollout_ref.actor.pivot.entropy_trigger_only=True \
    +actor_rollout_ref.actor.pivot.trig_percentile=65 \
    +actor_rollout_ref.actor.pivot.lan_grpo_restrict_to_trigger=False \
    +actor_rollout_ref.actor.pivot.langevin_eta=0.1 \
    +actor_rollout_ref.actor.pivot.langevin_top_k=20 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path="$REWARD_FN" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","file"]' \
    trainer.project_name='verl_pivot_v17_qwen3_guru_rl' \
    trainer.experiment_name='qwen3_4b_base_pivot_v17' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    trainer.total_epochs=15 \
    trainer.default_local_dir="$CKPT_DIR" \
    "$@"
