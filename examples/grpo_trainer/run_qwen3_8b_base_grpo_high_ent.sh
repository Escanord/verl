#!/usr/bin/env bash
# High-Entropy GRPO (top-20% entropy tokens only) on Qwen3-8B-Base.
#
# Reference: "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective
# Reinforcement Learning for LLM Reasoning", Wang et al., NeurIPS 2025.
#
# Matches the GRPO 8B baseline scale (train_bsz=1024, n=8, light KL=0.001,
# max_response=4096), with the single algorithmic change of restricting
# policy-gradient updates to the top-20% highest-entropy tokens via
# +actor.entropy_top_ratio=0.2.
set -x

DATA_DIR="${DATA_DIR:-/home/escanord/duy/data/guru_rl}"
MODEL="${MODEL:-/home/escanord/duy/checkpoints/models/Qwen3-8B-Base}"
REWARD_FN="$(dirname "$0")/guru_rl_reward.py"

train_files="$DATA_DIR/train.parquet"
val_files="['$DATA_DIR/test_aime.parquet','$DATA_DIR/test_aime25.parquet','$DATA_DIR/test_math500.parquet','$DATA_DIR/test_gpqa_diamond.parquet','$DATA_DIR/test_olympiadbench_math_en.parquet']"

CKPT_DIR=/home/escanord/duy/checkpoints/verl/high-ent-grpo/qwen3_8b_base

source /home/escanord/duy/venv-vault/miniconda3/etc/profile.d/conda.sh
conda activate verl

export VERL_FILE_LOGGER_ROOT=/home/escanord/duy/checkpoints/verl/high-ent-grpo
export VERL_VLLM_DISABLE_CASCADE_ATTN=1
export VERL_VLLM_CUDAGRAPH_MODE=PIECEWISE
export VLLM_FLASH_ATTN_VERSION=2
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    +actor_rollout_ref.actor.entropy_top_ratio=0.2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.custom_reward_function.path="$REWARD_FN" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","file"]' \
    trainer.project_name='verl_high_ent_grpo_qwen3_guru_rl' \
    trainer.experiment_name='qwen3_8b_base_grpo_high_ent' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    trainer.total_epochs=15 \
    trainer.default_local_dir="$CKPT_DIR" \
    ray_kwargs.ray_init.num_cpus=32 \
    "$@"
