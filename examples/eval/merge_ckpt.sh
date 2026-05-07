#!/usr/bin/env bash
# Merge a verl FSDP checkpoint into a single HF-format directory that vLLM can load.
#
# Usage:
#     bash merge_ckpt.sh <ckpt_dir> [out_dir]
#
# Example:
#     bash merge_ckpt.sh /storage/.../pivot-v18b/qwen3_1p7b_base/global_step_120
#
# The merged model is written to <ckpt_dir>/actor/hf_merged/ by default
# (or to [out_dir] if supplied). The merge is idempotent: if the target
# already contains a config.json + safetensors, the merge is skipped.
set -euo pipefail

CKPT_DIR="${1:?usage: merge_ckpt.sh <ckpt_dir> [out_dir]}"
OUT_DIR="${2:-${CKPT_DIR}/actor/hf_merged}"

if [[ ! -d "${CKPT_DIR}/actor" ]]; then
    echo "Error: ${CKPT_DIR}/actor not found"
    exit 1
fi

if [[ -f "${OUT_DIR}/config.json" ]] && ls "${OUT_DIR}"/*.safetensors >/dev/null 2>&1; then
    echo "Already merged: ${OUT_DIR}"
    exit 0
fi

mkdir -p "${OUT_DIR}"

source /storage/workspace/server-1/duy/venv-vault/miniconda3/etc/profile.d/conda.sh
conda activate verl

echo "Merging FSDP shards in ${CKPT_DIR}/actor -> ${OUT_DIR}"
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${CKPT_DIR}/actor" \
    --target_dir "${OUT_DIR}"

echo "Done: ${OUT_DIR}"
