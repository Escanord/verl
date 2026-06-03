#!/usr/bin/env bash
# Inference-time DRIFT component ablation on a fixed checkpoint.
#
# Runs `eval_via_verl.sh` on AIME24 only with different rollout-time pivot
# configurations on the same trained checkpoint, decomposing how much each
# DRIFT component (drift G, diffusion sigma, top-K, etc.) contributes when
# layered on top of the trained policy at inference time.
#
# All conditions share the same checkpoint, prompts (AIME24), and val_kwargs.
# Only the rollout-time pivot configuration differs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TAG_BASE="${TAG_BASE:-ablation_v18b_step160}"
MODEL=/storage/workspace/server-1/duy/checkpoints/models/Qwen3-1.7B-Base
CKPT=/storage/workspace/server-1/duy/checkpoints/verl/pivot-v18b/qwen3_1p7b_base/global_step_160
OUT_ROOT="${OUT_ROOT:-/storage/workspace/server-1/duy/checkpoints/verl/empirical/pivot_ablation}"
N_GPUS="${N_GPUS:-8}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
N_VAL="${N_VAL:-16}"
DATA_DIR="${DATA_DIR:-/storage/workspace/server-1/duy/data/guru_rl}"

mkdir -p "${OUT_ROOT}"

# AIME24 only for speed.
AIME_ONLY="data.val_files=['${DATA_DIR}/test_aime.parquet']"

# Shared DRIFT pivot config — same as training run, used by drift_full / alpha_0 / sigma_0.
PIVOT_BASE=(
    "+actor_rollout_ref.rollout.pivot.pivot_version=2"
    "+actor_rollout_ref.rollout.pivot.langevin_rollout=True"
    "+actor_rollout_ref.rollout.pivot.entropy_trigger_only=True"
    "+actor_rollout_ref.rollout.pivot.trig_percentile=75"
    "+actor_rollout_ref.rollout.pivot.entropy_threshold=0.4"
    "+actor_rollout_ref.rollout.pivot.langevin_K=1"
    "+actor_rollout_ref.rollout.pivot.langevin_top_k=20"
    "+actor_rollout_ref.rollout.pivot.langevin_eta=0.1"
    "+actor_rollout_ref.rollout.pivot.langevin_min_trigger_position=800"
    "+actor_rollout_ref.rollout.pivot.langevin_momentum=0.7"
    "+actor_rollout_ref.rollout.pivot.langevin_feedback=True"
    "+actor_rollout_ref.rollout.pivot.langevin_alpha_target=0.7"
    "+actor_rollout_ref.rollout.pivot.lan_use_cuda_graph=True"
)

run_condition() {
    local cond_tag="$1"; shift
    local extra=("$@")
    local tag="${TAG_BASE}_${cond_tag}"
    local out_dir="${OUT_ROOT}/${cond_tag}"
    if [[ -f "${out_dir}/drift_eval/${tag}.jsonl" ]] && \
       grep -q "val-core" "${out_dir}/drift_eval/${tag}.jsonl" 2>/dev/null; then
        echo "[have] ${cond_tag}"
        return
    fi
    echo "[run]  ${cond_tag}"
    N_GPUS="${N_GPUS}" ROLLOUT_TP="${ROLLOUT_TP}" N_VAL="${N_VAL}" \
        bash "${SCRIPT_DIR}/eval_via_verl.sh" \
        "${tag}" "${MODEL}" "${CKPT}" "${out_dir}" \
        "${AIME_ONLY}" \
        "${extra[@]}" \
        2>&1 | tee "${OUT_ROOT}/${cond_tag}_console.log"
}

# 1. vanilla — no pivot at inference (langevin_rollout=False is the default).
run_condition vanilla

# 2. drift_full — full DRIFT inference (matches training config).
run_condition drift_full \
    "${PIVOT_BASE[@]}" \
    "+actor_rollout_ref.rollout.pivot.langevin_sigma=0.01" \
    "+actor_rollout_ref.rollout.pivot.langevin_exploit_ratio=0.6"

# 3. alpha_0 — no learned drift mixing (pure random sphere + diffusion).
run_condition alpha_0 \
    "${PIVOT_BASE[@]}" \
    "+actor_rollout_ref.rollout.pivot.langevin_sigma=0.01" \
    "+actor_rollout_ref.rollout.pivot.langevin_exploit_ratio=0.0"

# 4. sigma_0 — no Gaussian diffusion (drift + random sphere only).
run_condition sigma_0 \
    "${PIVOT_BASE[@]}" \
    "+actor_rollout_ref.rollout.pivot.langevin_sigma=0.0" \
    "+actor_rollout_ref.rollout.pivot.langevin_exploit_ratio=0.6"

echo
echo "Done.  Aggregate AIME24 mean@16 per condition:"
echo "  ls ${OUT_ROOT}/*/drift_eval/*.jsonl"
