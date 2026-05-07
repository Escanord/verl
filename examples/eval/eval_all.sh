#!/usr/bin/env bash
# End-to-end eval driver.  Calls each per-method script in sequence; each
# script saturates 8 GPUs so we cannot parallelize across methods.
#
# Override step lists via env:
#     STEPS_1P7B="100 120 200" STEPS_4B="60 100 200" bash eval_all.sh
#
# Once all jobs complete, write a markdown summary for the doc with:
#     python3 aggregate_eval.py --root ${EVAL_OUT_ROOT} --format md
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EVAL_OUT_ROOT="${EVAL_OUT_ROOT:-/storage/workspace/server-1/duy/checkpoints/verl/eval_out}"
mkdir -p "${EVAL_OUT_ROOT}"
export EVAL_OUT_ROOT

# Run order: 1.7B first (smaller, faster), then 4B.
SCRIPTS=(
    eval_v18b_1p7b.sh
    eval_grpo_1p7b.sh
    eval_heg_1p7b.sh
    eval_v18c_4b.sh
    eval_grpo_4b.sh
    eval_heg_4b.sh
)

for s in "${SCRIPTS[@]}"; do
    echo
    echo "########## ${s} ##########"
    bash "${SCRIPT_DIR}/${s}"
done

echo
echo "All eval jobs complete.  Aggregate results:"
echo "    python3 ${SCRIPT_DIR}/aggregate_eval.py --root ${EVAL_OUT_ROOT} --format md"
