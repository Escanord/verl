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
Prepare GPQA-Diamond evaluation parquet for verl, using Idavidrein/gpqa from
HuggingFace.  Based on:
  Beyond-the-80-20-Rule-RLVR/recipe/r1/data_process.py  (build_gpqa_dimond_dataset)

Usage:
    python prepare_eval_gpqa.py --local_save_dir ~/data/guru_rl

Output:
    test_gpqa_diamond.parquet
    (data_source = 'mcq__gpqa_diamond')

NOTE on scoring.  GPQA is multiple-choice (A/B/C/D), unlike the math__* eval
files in this directory which use math-style numeric answer extraction.  This
script writes data_source='mcq__gpqa_diamond' so it does NOT route to the
math_dapo scorer in guru_rl_reward.py.  To actually use this parquet for
evaluation you must extend guru_rl_reward.compute_score to dispatch
mcq__* to a multiple-choice scorer (a 4-line regex match on
r'(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?' is sufficient and matches the
upstream simple-evals convention).
"""

import argparse
import os
import random

import datasets

GPQA_QUERY_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your response should be of the following "
    "format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before "
    "answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)


def make_map_fn(seed: int = 0):
    rng = random.Random(seed)

    def process_fn(example, idx):
        # Shuffle distractors then insert the correct answer at a random index.
        # Seeded so the parquet is reproducible across runs.
        choices = [
            example["Incorrect Answer 1"],
            example["Incorrect Answer 2"],
            example["Incorrect Answer 3"],
        ]
        rng.shuffle(choices)
        gold_index = rng.randint(0, 3)
        choices.insert(gold_index, example["Correct Answer"])

        question = GPQA_QUERY_TEMPLATE.format(
            A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=example["Question"]
        )
        gold_letter = "ABCD"[gold_index]

        return {
            "data_source": "mcq__gpqa_diamond",
            "prompt": [{"role": "user", "content": question}],
            "ability": "science",
            "reward_model": {"style": "rule", "ground_truth": gold_letter},
            "extra_info": {"split": "test", "index": idx, "subject": example.get("Subject", "")},
        }

    return process_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="~/data/guru_rl")
    parser.add_argument("--seed", type=int, default=0, help="Seed for distractor shuffling")
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print("Loading Idavidrein/gpqa (gpqa_diamond) ...", flush=True)
    ds = datasets.load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    ds = ds.map(make_map_fn(seed=args.seed), with_indices=True, remove_columns=ds.column_names)

    out = os.path.join(save_dir, "test_gpqa_diamond.parquet")
    ds.to_parquet(out)
    print(f"  wrote {out}  ({len(ds)} rows)")
    print()
    print("Reminder: extend guru_rl_reward.compute_score with an 'mcq__' branch")
    print("  before adding this file to data.val_files.  See script docstring.")


if __name__ == "__main__":
    main()
