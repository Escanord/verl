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
Prepare OlympiadBench evaluation parquet for verl.

OlympiadBench (He et al., 2024) is a competition-level math/physics benchmark.
We use the text-only English math competition split, which is the standard
"OlympiadBench" column in recent RL-for-reasoning papers (e.g. the entropy
mechanism paper, Cui et al. 2025).

HuggingFace source: Hothan/OlympiadBench, config OE_TO_maths_en_COMP
  - OE: open-ended (free-form numeric/symbolic answer)
  - TO: text-only (no images required)
  - maths: math (vs physics)
  - en: English (vs Chinese zh)
  - COMP: competition

Usage:
    python prepare_eval_olympiadbench.py --local_save_dir ~/data/guru_rl

Output:
    test_olympiadbench_math_en.parquet
    (data_source = 'math__olympiadbench_en')
"""

import argparse
import os

import datasets

INSTRUCTION_FOLLOWING = "Please reason step by step, and put your final answer within \\boxed{}."


def _normalize_answer(answer_field) -> str:
    """OlympiadBench's `final_answer` field is a list of strings (one per
    sub-question). We use the canonical one-answer subset, so the list has a
    single element which we extract."""
    if isinstance(answer_field, list):
        if len(answer_field) == 0:
            return ""
        return str(answer_field[0]).strip()
    return str(answer_field).strip()


def process_fn(example, idx):
    question_raw = example["question"]
    question = question_raw + " " + INSTRUCTION_FOLLOWING

    answer_raw = example.get("final_answer", "")
    ground_truth = _normalize_answer(answer_raw)

    return {
        "data_source": "math__olympiadbench_en",
        "prompt": [{"role": "user", "content": question}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "split": "test",
            "index": idx,
            "subject": example.get("subject", ""),
            "language": example.get("language", ""),
            "subfield": example.get("subfield", ""),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="~/data/guru_rl")
    parser.add_argument(
        "--config",
        default="OE_TO_maths_en_COMP",
        help="HF config from Hothan/OlympiadBench. Default = open-ended text-only English math competition.",
    )
    parser.add_argument(
        "--max_per_subject",
        type=int,
        default=None,
        help="Optional cap on examples retained per subject (for sanity testing)",
    )
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading Hothan/OlympiadBench config={args.config} ...", flush=True)
    # OlympiadBench uses 'train' as its split name even though it's an eval set
    ds = datasets.load_dataset("Hothan/OlympiadBench", args.config, split="train")

    # Drop multi-answer items (the final_answer list having >1 element indicates
    # multi-part problems where strict numeric matching would be ill-defined).
    # The simple-answer subset matches what most paper tables report.
    def is_single_answer(ex):
        fa = ex.get("final_answer")
        return isinstance(fa, list) and len(fa) == 1

    n_before = len(ds)
    ds = ds.filter(is_single_answer)
    print(f"  filtered to single-answer items: {len(ds)} / {n_before}")

    if args.max_per_subject:
        # Optional: subsample per subject for fast smoke tests
        from collections import defaultdict

        seen = defaultdict(int)
        keep_idx = []
        for i, ex in enumerate(ds):
            s = ex.get("subject", "")
            if seen[s] < args.max_per_subject:
                keep_idx.append(i)
                seen[s] += 1
        ds = ds.select(keep_idx)
        print(f"  capped at {args.max_per_subject}/subject -> {len(ds)} rows")

    ds = ds.map(process_fn, with_indices=True, remove_columns=ds.column_names)

    out = os.path.join(save_dir, "test_olympiadbench_math_en.parquet")
    ds.to_parquet(out)
    print(f"  wrote {out}  ({len(ds)} rows)")


if __name__ == "__main__":
    main()
