# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prepare AIME24 + AIME25 evaluation parquets for verl, using math-ai/aime24 and
math-ai/aime25 from HuggingFace.  Based on:
  Beyond-the-80-20-Rule-RLVR/recipe/open_math_reasoning/prepare_eval_dataset.py

Usage:
    python prepare_eval_aime25.py --local_save_dir ~/data/guru_rl
    python prepare_eval_aime25.py --local_save_dir ~/data/guru_rl --aime25_only

Outputs:
    test_aime24.parquet
    test_aime25.parquet

Both files share the same schema as the existing test_aime.parquet so they
plug directly into data.val_files=['...'] without further changes.  The
data_source values are 'math__aime24' and 'math__aime25' so the existing
guru_rl_reward.compute_score (which routes math__* to math_dapo.compute_score)
handles them unchanged.
"""

import argparse
import os

import datasets

INSTRUCTION_FOLLOWING = "Please reason step by step, and put your final answer within \\boxed{}."


def _strip_boxed(answer_str: str) -> str:
    """Remove a single outermost \\boxed{...} wrapper if present."""
    s = answer_str.strip()
    prefix = r"\boxed{"
    if s.startswith(prefix) and s.endswith("}"):
        return s[len(prefix) : -1]
    return s


def make_map_fn(data_source: str):
    def process_fn(example, idx):
        question_raw = example["problem"]
        question = question_raw + " " + INSTRUCTION_FOLLOWING

        # AIME splits sometimes carry "answer", sometimes "solution"
        if "solution" in example and example["solution"]:
            answer_raw = example["solution"]
        else:
            answer_raw = str(example["answer"])

        try:
            ground_truth = _strip_boxed(answer_raw)
        except Exception:
            ground_truth = answer_raw

        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {"split": "test", "index": idx, "answer_raw": answer_raw, "question": question_raw},
        }

    return process_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_save_dir",
        default="~/data/guru_rl",
        help="Directory to save processed parquet files",
    )
    parser.add_argument(
        "--aime25_only",
        action="store_true",
        help="Only prepare AIME25 (skip AIME24, which we already have as test_aime.parquet)",
    )
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if not args.aime25_only:
        print("Loading math-ai/aime24 ...", flush=True)
        ds24 = datasets.load_dataset("math-ai/aime24", split="test")
        ds24 = ds24.map(
            make_map_fn("math__aime24"),
            with_indices=True,
            remove_columns=ds24.column_names,
        )
        out24 = os.path.join(save_dir, "test_aime24.parquet")
        ds24.to_parquet(out24)
        print(f"  wrote {out24}  ({len(ds24)} rows)")

    print("Loading math-ai/aime25 ...", flush=True)
    ds25 = datasets.load_dataset("math-ai/aime25", split="test")
    ds25 = ds25.map(
        make_map_fn("math__aime25"),
        with_indices=True,
        remove_columns=ds25.column_names,
    )
    out25 = os.path.join(save_dir, "test_aime25.parquet")
    ds25.to_parquet(out25)
    print(f"  wrote {out25}  ({len(ds25)} rows)")


if __name__ == "__main__":
    main()
