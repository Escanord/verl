#!/usr/bin/env bash
# Run rollout_diversity.py for each (method × size × step) and write per-run
# diversity__<tag>.json under DIVERSITY_OUT.
#
# Usage:
#     bash run_diversity_sweep.sh
#     N_PROMPTS=50 bash run_diversity_sweep.sh
#
# Env knobs:
#     N_PROMPTS  number of prompts (default 30)
#     N          rollouts per prompt (default 8)
#     TP         vLLM tensor-parallel size (default 1)
#     PROMPTS    parquet path (default test_aime.parquet)
#     DIVERSITY_OUT  output directory
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DIVERSITY_OUT="${DIVERSITY_OUT:-/storage/workspace/server-1/duy/checkpoints/verl/diversity_out}"
PROMPTS="${PROMPTS:-/storage/workspace/server-1/duy/data/guru_rl/test_aime.parquet}"
N_PROMPTS="${N_PROMPTS:-30}"
N="${N:-8}"
TP="${TP:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

mkdir -p "${DIVERSITY_OUT}"

# tag | hf_merged_path  -- pre-merged HF checkpoints required.
# These are the actor/hf_merged dirs that eval_via_verl.sh would have produced.
RUNS=(
    "grpo_4b_step60|/storage/workspace/server-1/duy/checkpoints/verl/grpo/qwen3_4b_base/global_step_60/actor/hf_merged"
    "heg_4b_step60|/storage/workspace/server-1/duy/checkpoints/verl/high-ent-grpo/qwen3_4b_base/global_step_60/actor/hf_merged"
    "v18c_4b_step60|/storage/workspace/server-1/duy/checkpoints/verl/pivot-v18c/qwen3_4b_base/global_step_60/actor/hf_merged"
    "grpo_1p7b_step100|/storage/workspace/server-1/duy/checkpoints/verl/grpo/qwen3_1p7b_base/global_step_100/actor/hf_merged"
    "heg_1p7b_step100|/storage/workspace/server-1/duy/checkpoints/verl/high-ent-grpo/qwen3_1p7b_base/global_step_100/actor/hf_merged"
    "v18b_1p7b_step100|/storage/workspace/server-1/duy/checkpoints/verl/pivot-v18b/qwen3_1p7b_base/global_step_100/actor/hf_merged"
)

source /storage/workspace/server-1/duy/venv-vault/miniconda3/etc/profile.d/conda.sh
conda activate verl

for spec in "${RUNS[@]}"; do
    IFS='|' read -r tag ckpt <<<"${spec}"
    if [[ ! -d "${ckpt}" ]]; then
        echo "[skip] ${tag}: ${ckpt} not merged yet (run eval_via_verl.sh's merge step or merge_ckpt.sh first)"
        continue
    fi
    out="${DIVERSITY_OUT}/diversity__${tag}.json"
    if [[ -f "${out}" ]]; then
        echo "[have] ${tag}"
        continue
    fi
    echo "[run]  ${tag}"
    python3 "${SCRIPT_DIR}/rollout_diversity.py" \
        --ckpt "${ckpt}" \
        --prompts "${PROMPTS}" \
        --n_prompts "${N_PROMPTS}" \
        --n "${N}" \
        --tp "${TP}" \
        --gpu_mem_util "${GPU_MEM_UTIL}" \
        --tag "${tag}" \
        --out_dir "${DIVERSITY_OUT}"
done

echo
echo "Done.  Aggregate / plot:"
echo "    ls ${DIVERSITY_OUT}"
