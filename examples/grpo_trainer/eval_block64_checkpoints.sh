#!/usr/bin/env bash
# Evaluate all wgrpo-v6-block64 checkpoints (steps 120-400) for val metrics.
set -e

DATA_DIR="${DATA_DIR:-/storage/workspace/server-1/duy/data/guru_rl}"
CKPT_DIR=/storage/workspace/server-1/duy/checkpoints/verl/wgrpo-v6-block64/qwen2_5_3b
MODEL=/storage/workspace/server-1/duy/checkpoints/models/Qwen2.5-3B-Instruct
REWARD_FN="$(dirname "$0")/guru_rl_reward.py"
OUT_DIR=/storage/workspace/server-1/duy/checkpoints/verl/wgrpo-v6-block64/eval_results

mkdir -p "$OUT_DIR"

val_files="['$DATA_DIR/test_aime.parquet','$DATA_DIR/test_math500.parquet']"

source /storage/workspace/server-1/duy/venv-vault/miniconda3/etc/profile.d/conda.sh
conda activate verl

for STEP in global_step_280 global_step_300 global_step_320 global_step_340 \
            global_step_360 global_step_380 global_step_400; do

    RESUME_PATH="$CKPT_DIR/$STEP"
    if [ ! -d "$RESUME_PATH" ]; then
        echo "==> Skipping $STEP (not found)"
        continue
    fi

    echo "==> Evaluating $STEP ..."
    export VERL_FILE_LOGGER_ROOT="$OUT_DIR"

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
        actor_rollout_ref.model.path=$MODEL \
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
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.val_kwargs.n=16 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        reward.custom_reward_function.path="$REWARD_FN" \
        reward.custom_reward_function.name=compute_score \
        trainer.critic_warmup=0 \
        trainer.logger='["console","file"]' \
        trainer.project_name='verl_wgrpo_guru_rl' \
        trainer.experiment_name="block64_eval_${STEP}" \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.val_only=True \
        trainer.val_before_train=True \
        trainer.resume_mode=resume_path \
        trainer.resume_from_path="$RESUME_PATH" \
        trainer.default_local_dir="$CKPT_DIR" \
        && echo "==> Done $STEP" || echo "==> FAILED $STEP"

    echo "---"
done

echo "All evals complete. Results in $OUT_DIR"
