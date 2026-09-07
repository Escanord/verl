#!/usr/bin/env bash
# DAPO + heg on Qwen3-4B-Base (NO DRIFT/Langevin) — guru_rl.
# Clone of run_qwen3_4b_base_dapo.sh + heg top-20%-entropy masking:
#   +actor_rollout_ref.actor.entropy_top_ratio=0.2
# (No pivot/langevin knobs, so union/intersection is moot — heg simply masks
#  policy-gradient to the top-20%-entropy tokens.) Paper-scale DAPO base: 20480
# resp, batch 512, filter_groups gen 3x, clip 0.2/0.28/c=10, token-mean loss,
# dapo reward_manager w/ overlong 4096, n=16, NO KL. Default data = guru_rl so it
# is directly comparable to the dapo / dapo+mx_cap2 4B runs. Fresh run.
set -x

DATA_DIR="${DATA_DIR:-/home/escanord/duy/data/guru_rl}"
MODEL="${MODEL:-/home/escanord/duy/checkpoints/models/Qwen3-4B-Base}"
REWARD_FN="$(dirname "$0")/guru_rl_reward.py"

train_files="$DATA_DIR/train.parquet"
val_files="['$DATA_DIR/test_aime.parquet','$DATA_DIR/test_aime25.parquet','$DATA_DIR/test_math500.parquet','$DATA_DIR/test_gpqa_diamond.parquet','$DATA_DIR/test_olympiadbench_math_en.parquet']"

CKPT_DIR=/home/escanord/duy/checkpoints/verl/dapo/qwen3_4b_base_heg

source /home/escanord/duy/venv-vault/miniconda3/etc/profile.d/conda.sh
conda activate verl

export VERL_FILE_LOGGER_ROOT=/home/escanord/duy/checkpoints/verl/dapo
export VERL_VLLM_DISABLE_CASCADE_ATTN=1
export VERL_VLLM_CUDAGRAPH_MODE=PIECEWISE
export VLLM_FLASH_ATTN_VERSION=2
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HOME=$HOME/duy/.cache/huggingface
export HF_DATASETS_CACHE=$HOME/duy/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=$HOME/duy/.cache/huggingface/hub
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

max_prompt_length=2048
max_response_length=20480
train_prompt_bsz=512
gen_prompt_bsz=$((train_prompt_bsz * 3))
ppo_mini_bsz=32

python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.nccl_timeout=3600 \
    algorithm.adv_estimator=grpo \
    +algorithm.filter_groups.enable=True \
    +algorithm.filter_groups.metric=acc \
    +algorithm.filter_groups.max_num_gen_batches=10 \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=${train_prompt_bsz} \
    +data.gen_batch_size=${gen_prompt_bsz} \
    data.val_batch_size=256 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path="$MODEL" \
    +actor_rollout_ref.model.override_config.attention_dropout=0.0 \
    +actor_rollout_ref.model.override_config.embd_pdrop=0.0 \
    +actor_rollout_ref.model.override_config.resid_pdrop=0.0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_bsz} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    +actor_rollout_ref.actor.entropy_top_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward.reward_kwargs.overlong_buffer_cfg.len=4096 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    reward.custom_reward_function.path="$REWARD_FN" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","file"]' \
    trainer.project_name='verl_dapo_qwen3_guru_rl' \
    trainer.experiment_name="qwen3_4b_base_dapo_heg_${SLURM_JOB_ID:-manual}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    trainer.total_epochs=15 \
    trainer.default_local_dir="$CKPT_DIR" \
    ray_kwargs.ray_init.num_cpus=32 \
    "$@"
