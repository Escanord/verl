#!/usr/bin/env bash
# Vanilla GRPO on Qwen3-4B-Base — CODE domain (guru_code) instead of math.
# Clone of run_qwen3_4b_base_grpo_rebuttal.sh with:
#   - DATA_DIR   -> data/guru_code (train = primeintellect+taco stdin/stdout)
#   - REWARD_FN  -> guru_code_reward.py (subprocess exec + timeout, binary +-1)
#   - val_files  -> HumanEval(164) + MBPP(500) + LiveCodeBench-trimmed(279)
#   - project/ckpt/experiment isolated so it never mixes with math grpo jsonls
# Same recipe otherwise: batch 1024, n=8, val n=16, 16k resp, clip 0.2/0.2,
# no use_kl_in_reward, low_var kl_loss 1e-3, save/test_freq=5.
# NOTE: code reward spawns one python subprocess PER test case (<=50), timeout
# only (no fs/net sandbox) — accepted risk on shared node per launch decision.
set -x

DATA_DIR="${DATA_DIR:-/home/escanord/duy/data/guru_code}"
MODEL="${MODEL:-/home/escanord/duy/checkpoints/models/Qwen3-4B-Base}"
REWARD_FN="$(dirname "$0")/guru_code_reward.py"

train_files="$DATA_DIR/train.parquet"
val_files="['$DATA_DIR/test_humaneval.parquet','$DATA_DIR/test_mbpp.parquet','$DATA_DIR/test_livecodebench.parquet']"

CKPT_DIR=/home/escanord/duy/checkpoints/verl/grpo-code/qwen3_4b_base_grpo_code

source /home/escanord/duy/venv-vault/miniconda3/etc/profile.d/conda.sh
conda activate verl

export VERL_FILE_LOGGER_ROOT=/home/escanord/duy/checkpoints/verl/grpo-code
export VERL_VLLM_DISABLE_CASCADE_ATTN=1
export VERL_VLLM_CUDAGRAPH_MODE=PIECEWISE
export VLLM_FLASH_ATTN_VERSION=2
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HOME=$HOME/duy/.cache/huggingface
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=1024 \
    data.val_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path="$REWARD_FN" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","file"]' \
    trainer.project_name='verl_grpo_qwen3_guru_code' \
    trainer.experiment_name="qwen3_4b_base_grpo_code_${SLURM_JOB_ID:-manual}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    trainer.total_epochs=15 \
    trainer.default_local_dir="$CKPT_DIR" \
    ray_kwargs.ray_init.num_cpus=32 \
    "$@"
