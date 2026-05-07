#!/usr/bin/env bash
# Run a single-checkpoint, multi-benchmark eval through verl's distributed
# rollout/scoring infrastructure (Ray + FSDP + vLLM).  All 4 benchmarks
# (AIME24, AIME25, OlympiadBench, GPQA) are evaluated in one verl run by
# passing them all in data.val_files; verl logs per-benchmark val metrics
# under val-core/<data_source>/acc/{mean@16,best@16/mean}.
#
# Args (positional):
#   $1  = run tag (e.g. v18b_1p7b)
#   $2  = base model path                (e.g. /.../models/Qwen3-1.7B-Base)
#   $3  = resume_from_path checkpoint    (e.g. /.../pivot-v18b/qwen3_1p7b_base/global_step_120)
#   $4  = local_dir for this eval        (a fresh dir; verl writes its eval JSONL here)
#
# Reads:
#   DATA_DIR        defaults to /storage/workspace/server-1/duy/data/guru_rl
#   N_GPUS          defaults to 8
#   ROLLOUT_TP      defaults to 1   (per-rollout TP; data-parallel = N_GPUS / ROLLOUT_TP)
#   N_VAL           defaults to 16  (samples per problem)
#   GPU_MEM_UTIL    defaults to 0.6
set -x

TAG="${1:?usage: eval_via_verl.sh <tag> <model> <ckpt> <out_dir>}"
MODEL="${2:?model path required}"
CKPT="${3:?ckpt path required}"
OUT_DIR="${4:?out dir required}"

DATA_DIR="${DATA_DIR:-/storage/workspace/server-1/duy/data/guru_rl}"
N_GPUS="${N_GPUS:-8}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
N_VAL="${N_VAL:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"
REWARD_FN="$(dirname "$0")/../grpo_trainer/guru_rl_reward.py"

# All eval parquets in one shot.  GPQA uses mcq__ data_source which dispatches
# to the MCQ scorer in guru_rl_reward.py.  Minerva intentionally excluded.
val_files="['$DATA_DIR/test_aime.parquet','$DATA_DIR/test_aime25.parquet','$DATA_DIR/test_olympiadbench_math_en.parquet','$DATA_DIR/test_gpqa_diamond.parquet']"

# A train file is required by the trainer scaffold even when total_epochs=0;
# we point at the existing train parquet but it is never consumed.
train_files="$DATA_DIR/train.parquet"

mkdir -p "${OUT_DIR}"

source /storage/workspace/server-1/duy/venv-vault/miniconda3/etc/profile.d/conda.sh
conda activate verl

export VERL_FILE_LOGGER_ROOT="${OUT_DIR}"
export VLLM_FLASH_ATTN_VERSION=2
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n="${N_VAL}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.custom_reward_function.path="$REWARD_FN" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","file"]' \
    trainer.project_name='drift_eval' \
    trainer.experiment_name="${TAG}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.val_before_train=True \
    ++trainer.val_only=True \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${OUT_DIR}" \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="${CKPT}" \
    "${@:5}"
