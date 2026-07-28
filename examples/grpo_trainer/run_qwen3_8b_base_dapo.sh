#!/usr/bin/env bash
# DAPO on Qwen3-8B-Base — paper-scale config matching Wang et al. 80/20 §5.2.
#
# Paper (arXiv 2506.01939) §5.2 verbatim:
#   "we apply the same hyperparameters as recommended by DAPO: for clip-higher,
#    ε_high = 0.28, ε_low = 0.2; for overlong reward shaping, the maximum
#    response length is 20480 and the cache length is 4096. Furthermore, we use
#    a training batch size of 512 and a mini-batch size of 32 in verl's
#    configuration, resulting in 16 gradient steps per training batch, with a
#    learning rate of 1e-6 and no learning rate warmup or scheduling.
#    Importantly, the training process excludes both KL divergence loss and
#    entropy loss."
#
#   DAPO objective (Eq. 4) carries the dynamic-sampling constraint
#   0 < |{o_i correct}| < G  → filter_groups=True.
#   gen_batch_size = 3× train_batch_size (overprovision for filter_groups).
set -x

DATA_DIR="${DATA_DIR:-/home/escanord/duy/data/dapo_math_17k}"
MODEL="${MODEL:-/home/escanord/duy/checkpoints/models/Qwen3-8B-Base}"
REWARD_FN="$(dirname "$0")/guru_rl_reward.py"

train_files="$DATA_DIR/train.parquet"
val_files="['$DATA_DIR/test_aime.parquet','$DATA_DIR/test_aime25.parquet','$DATA_DIR/test_math500.parquet','$DATA_DIR/test_gpqa_diamond.parquet','$DATA_DIR/test_olympiadbench_math_en.parquet']"

CKPT_DIR=/home/escanord/duy/checkpoints/verl/dapo/qwen3_8b_base

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
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
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
    trainer.experiment_name='qwen3_8b_base_dapo' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    trainer.total_epochs=15 \
    trainer.default_local_dir="$CKPT_DIR" \
    ray_kwargs.ray_init.num_cpus=32 \
    "$@"
