#!/usr/bin/env bash
# Prepare AIME25 + OlympiadBench + Minerva eval parquets for verl.
#
# All three use the math__* data_source convention so they plug directly into
# data.val_files=['...'] without changes to guru_rl_reward.py.
#
# Usage:
#     bash prepare_all_eval.sh                        # save under ~/data/guru_rl
#     LOCAL_SAVE_DIR=/path/to/data bash prepare_all_eval.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_SAVE_DIR="${LOCAL_SAVE_DIR:-$HOME/data/guru_rl}"

echo "==> Saving eval parquets to: ${LOCAL_SAVE_DIR}"
mkdir -p "${LOCAL_SAVE_DIR}"

echo
echo "==> [1/3] AIME25"
python3 "${SCRIPT_DIR}/prepare_eval_aime25.py" --local_save_dir "${LOCAL_SAVE_DIR}" --aime25_only

echo
echo "==> [2/3] OlympiadBench (math, English, competition)"
python3 "${SCRIPT_DIR}/prepare_eval_olympiadbench.py" --local_save_dir "${LOCAL_SAVE_DIR}"

echo
echo "==> [3/3] Minerva-Math"
python3 "${SCRIPT_DIR}/prepare_eval_minerva.py" --local_save_dir "${LOCAL_SAVE_DIR}"

echo
echo "==> Done.  New parquets:"
ls -lh "${LOCAL_SAVE_DIR}"/test_aime25.parquet \
       "${LOCAL_SAVE_DIR}"/test_olympiadbench_math_en.parquet \
       "${LOCAL_SAVE_DIR}"/test_minerva_math.parquet
