#!/usr/bin/env bash
# PIVOT-v15c trained from scratch (Qwen2.5-3B-Instruct base).
#
# v15b + trig_percentile=80 so IS addon fires on ~20% of response positions.
#
# v15b uses trig_percentile=90 → top-10% delta_var positions trigger (~10%).
# v15c uses trig_percentile=80 → top-20% delta_var positions trigger (~20%).
#
# Mirrors v16c (same 20% trigger budget) but uses the blended IS denominator
# instead of the additive IS addon:
#   r = π_current / sqrt(p_lan_old · π_old)   [blend=0.5, single-term loss]
# vs v16c:
#   r_IS = π_current / p_lan_old               [additive addon + dual-clip]
#
# Direct comparison v15c vs v16c isolates blend vs dual-clip at equal trigger
# budget. All other hyper-parameters identical to v15b.
set -x

DATA_DIR="${DATA_DIR:-/storage/workspace/server-1/duy/data/guru_rl}"
MODEL="${MODEL:-/storage/workspace/server-1/duy/checkpoints/models/Qwen2.5-3B-Instruct}"
REWARD_FN="$(dirname "$0")/guru_rl_reward.py"

train_files="$DATA_DIR/train.parquet"
val_files="['$DATA_DIR/test_aime.parquet','$DATA_DIR/test_math500.parquet']"

CKPT_DIR=/storage/workspace/server-1/duy/checkpoints/verl/pivot-v15c/qwen2_5_3b

source /storage/workspace/server-1/duy/venv-vault/miniconda3/etc/profile.d/conda.sh
conda activate verl

export VERL_FILE_LOGGER_ROOT=/storage/workspace/server-1/duy/checkpoints/verl/pivot-v15c
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
    +actor_rollout_ref.rollout.pivot.delta_var_threshold=9.0 \
    +actor_rollout_ref.rollout.pivot.trig_percentile=80 \
    +actor_rollout_ref.rollout.pivot.entropy_threshold=8.0 \
    +actor_rollout_ref.rollout.pivot.langevin_K=1 \
    +actor_rollout_ref.rollout.pivot.langevin_top_k=20 \
    +actor_rollout_ref.rollout.pivot.langevin_eta=0.1 \
    +actor_rollout_ref.rollout.pivot.langevin_sigma=0.01 \
    +actor_rollout_ref.actor.pivot.lan_grpo_coeff=1.0 \
    +actor_rollout_ref.actor.pivot.lan_grpo_correct_is=True \
    +actor_rollout_ref.actor.pivot.lan_grpo_correct_mu=True \
    +actor_rollout_ref.actor.pivot.lan_grpo_denom_blend=0.5 \
    +actor_rollout_ref.actor.pivot.entropy_cap=0.8 \
    +actor_rollout_ref.actor.pivot.n_rollouts_per_prompt=8 \
    +actor_rollout_ref.actor.pivot.delta_var_threshold=9.0 \
    +actor_rollout_ref.actor.pivot.trig_percentile=80 \
    +actor_rollout_ref.actor.pivot.langevin_eta=0.1 \
    +actor_rollout_ref.actor.pivot.langevin_top_k=20 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path="$REWARD_FN" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","file"]' \
    trainer.project_name='verl_pivot_v15c_guru_rl' \
    trainer.experiment_name='qwen2_5_3b_pivot_v15c' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    trainer.total_epochs=15 \
    trainer.default_local_dir="$CKPT_DIR" \
    "$@"
