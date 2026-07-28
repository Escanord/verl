#!/usr/bin/env bash
# Backfill GRPO-8B-Base val on AIME24/AIME25/MATH-500/OlympBench.
#
# The original GRPO 8B run was launched with val_files that excluded
# AIME25 and OlympBench until step ~485 (val_files config was updated
# at resume).  This script evaluates each checkpoint in step
# 20..480 (the missing range) on the full benchmark set so we can plot
# v22 vs GRPO step-matched out to step 500.
#
# Submit:  sbatch /home/escanord/duy/slurm/verl/eval_grpo_8b_backfill.sbatch
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-/home/escanord/duy/data/guru_rl}"
MODEL="${MODEL:-/home/escanord/duy/checkpoints/models/Qwen3-8B-Base}"
CKPT_ROOT="${CKPT_ROOT:-/home/escanord/duy/checkpoints/verl/grpo/qwen3_8b_base}"
EVAL_OUT_ROOT="${EVAL_OUT_ROOT:-/home/escanord/duy/checkpoints/verl/eval_out/grpo_8b_backfill}"
REWARD_FN="$(dirname "${SCRIPT_DIR}")/grpo_trainer/guru_rl_reward.py"

N_GPUS="${N_GPUS:-8}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
N_VAL="${N_VAL:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"

STEPS="${STEPS:-20 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320 340 360 380 400 420 440 460 480}"

mkdir -p "${EVAL_OUT_ROOT}"

val_files="['$DATA_DIR/test_aime.parquet','$DATA_DIR/test_aime25.parquet','$DATA_DIR/test_math500.parquet','$DATA_DIR/test_olympiadbench_math_en.parquet']"
train_files="$DATA_DIR/train.parquet"

source /home/escanord/duy/venv-vault/miniconda3/etc/profile.d/conda.sh
conda activate verl

export VERL_VLLM_DISABLE_CASCADE_ATTN=1
export VERL_VLLM_CUDAGRAPH_MODE=PIECEWISE
export HF_HOME="${HF_HOME:-$HOME/duy/.cache/huggingface}"

for step in ${STEPS}; do
    ckpt_dir="${CKPT_ROOT}/global_step_${step}"
    if [[ ! -d "${ckpt_dir}" ]]; then
        echo "[skip] step ${step}: checkpoint not found at ${ckpt_dir}"
        continue
    fi
    tag="grpo_8b_step${step}"
    out_dir="${EVAL_OUT_ROOT}/${tag}"
    done_marker="${out_dir}/done.txt"
    if [[ -f "${done_marker}" ]]; then
        echo "[have] ${tag}"
        continue
    fi
    mkdir -p "${out_dir}"
    echo "[eval] ${tag}  ckpt=${ckpt_dir}"

    export VERL_FILE_LOGGER_ROOT="${out_dir}"

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
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
        actor_rollout_ref.rollout.enable_prefix_caching=False \
        actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
        actor_rollout_ref.rollout.max_num_seqs=512 \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.val_kwargs.n="${N_VAL}" \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        reward.custom_reward_function.path="$REWARD_FN" \
        reward.custom_reward_function.name=compute_score \
        trainer.critic_warmup=0 \
        trainer.logger='["console","file"]' \
        trainer.project_name='grpo_8b_backfill' \
        trainer.experiment_name="${tag}" \
        trainer.n_gpus_per_node="${N_GPUS}" \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=1 \
        trainer.val_before_train=True \
        ++trainer.val_only=True \
        trainer.total_epochs=1 \
        trainer.default_local_dir="${out_dir}" \
        trainer.resume_mode=resume_path \
        trainer.resume_from_path="${ckpt_dir}" \
        ray_kwargs.ray_init.num_cpus=32 \
        2>&1 | tee "${out_dir}/eval.log"

    touch "${done_marker}"
done

echo "===================================================================="
echo "Done. Results under: ${EVAL_OUT_ROOT}"
echo "Per-step val metrics in <step_dir>/eval.log (grep 'val-core/')"
echo "===================================================================="
