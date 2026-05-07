#!/usr/bin/env bash
# Evaluate GRPO baseline on Qwen3-4B-Base.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TAG_PREFIX="grpo_4b"
MODEL=/storage/workspace/server-1/duy/checkpoints/models/Qwen3-4B-Base
CKPT_ROOT=/storage/workspace/server-1/duy/checkpoints/verl/grpo/qwen3_4b_base

EVAL_OUT_ROOT="${EVAL_OUT_ROOT:-/storage/workspace/server-1/duy/checkpoints/verl/eval_out}"
N_GPUS="${N_GPUS:-8}"
ROLLOUT_TP="${ROLLOUT_TP:-2}"
N_VAL="${N_VAL:-16}"
STEPS="${STEPS_4B:-20 40 60 80 100 120 140 160}"

mkdir -p "${EVAL_OUT_ROOT}"

METHOD_RUN="grpo"
SIZE="4b"

for step in ${STEPS}; do
    ckpt_dir="${CKPT_ROOT}/global_step_${step}"
    if [[ ! -d "${ckpt_dir}" ]]; then
        echo "[skip] ${TAG_PREFIX} step ${step}: checkpoint not found"
        continue
    fi
    tag="${TAG_PREFIX}_step${step}"
    out_dir="${EVAL_OUT_ROOT}/${tag}"
    jsonl="${out_dir}/drift_eval/${tag}.jsonl"
    if [[ -f "${jsonl}" ]] && grep -q "val-core" "${jsonl}" 2>/dev/null; then
        echo "[have] ${tag}"
        continue
    fi
    echo "[eval] ${tag}"
    N_GPUS="${N_GPUS}" ROLLOUT_TP="${ROLLOUT_TP}" N_VAL="${N_VAL}" \
        bash "${SCRIPT_DIR}/eval_via_verl.sh" "${tag}" "${MODEL}" "${ckpt_dir}" "${out_dir}" \
        2>&1 | tee "${EVAL_OUT_ROOT}/${tag}_console.log"
done

SUMMARY_FILE="${EVAL_OUT_ROOT}/${METHOD_RUN}_${SIZE}_summary.jsonl"
echo "[aggregate] ${SUMMARY_FILE}"
python3 "${SCRIPT_DIR}/aggregate_eval.py" \
    --root "${EVAL_OUT_ROOT}" --method "${METHOD_RUN}" --size "${SIZE}" \
    --format jsonl --out "${SUMMARY_FILE}" || echo "[warn] aggregation skipped"
