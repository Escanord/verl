#!/usr/bin/env bash
# Diagnostic: run val-only on DAPO 4B step-60 checkpoint, score with BOTH
# is_correct_minerva (needs "Answer:" prefix) and is_correct_strict_box
# (needs \boxed{}) side-by-side, and log a tail of each rollout so we can
# see what format the model is actually producing.
set -x

DATA_DIR="${DATA_DIR:-/home/escanord/duy/data/dapo_math_17k}"
MODEL="${MODEL:-/home/escanord/duy/checkpoints/models/Qwen3-4B-Base}"
CKPT="${CKPT:-/home/escanord/duy/checkpoints/verl/dapo/qwen3_4b_base/global_step_60}"
REWARD_FN="$(dirname "$0")/../grpo_trainer/dapo_4b_diagnostic_reward.py"
OUT_DIR="${OUT_DIR:-/home/escanord/duy/checkpoints/verl/eval_out/dapo_4b_diag_step60}"

# Only MATH-500 for speed
val_files="['$DATA_DIR/test_math500.parquet']"
train_files="$DATA_DIR/train.parquet"

mkdir -p "$OUT_DIR"

source /home/escanord/duy/venv-vault/miniconda3/etc/profile.d/conda.sh
conda activate verl

export VERL_FILE_LOGGER_ROOT="$OUT_DIR"
export VERL_VLLM_DISABLE_CASCADE_ATTN=1
export VERL_VLLM_CUDAGRAPH_MODE=PIECEWISE
export HF_HOME=$HOME/duy/.cache/huggingface
# Diagnostic logger config
export DAPO_DIAG_LOG_EVERY=1
export DAPO_DIAG_TAIL_CHARS=800

python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.nccl_timeout=3600 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=32 \
    data.val_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=20480 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=False \
    +reward.reward_kwargs.overlong_buffer_cfg.len=4096 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.max_resp_len=20480 \
    reward.custom_reward_function.path="$REWARD_FN" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","file"]' \
    trainer.project_name='dapo_4b_diagnostic' \
    trainer.experiment_name='dapo_4b_step60_diag' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.val_before_train=True \
    ++trainer.val_only=True \
    trainer.total_epochs=1 \
    trainer.default_local_dir="$OUT_DIR" \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$CKPT" \
    ray_kwargs.ray_init.num_cpus=32 \
    "$@"
