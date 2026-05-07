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
Prepare Minerva-Math evaluation parquet for verl.

"Minerva" in recent RL-for-reasoning paper tables (Cui et al. 2025, etc.)
refers to the Minerva-style evaluation subset of MATH used by Lewkowycz et al.
2022.  The standard implementation in lm-eval-harness / lighteval is the
`minerva_math_*` set of tasks defined over the MATH dataset, partitioned by
subject (algebra, counting_and_probability, geometry, intermediate_algebra,
number_theory, prealgebra, precalculus) and evaluated with strict
boxed-answer extraction.

We build a single Minerva-Math parquet by concatenating these seven subjects'
test splits from EleutherAI/hendrycks_math.

HuggingFace source: EleutherAI/hendrycks_math (one config per subject)

Usage:
    python prepare_eval_minerva.py --local_save_dir ~/data/guru_rl

Output:
    test_minerva_math.parquet
    (data_source = 'math__minerva')
"""

import argparse
import os

import datasets

INSTRUCTION_FOLLOWING = "Please reason step by step, and put your final answer within \\boxed{}."

# Subjects evaluated by the lm-eval-harness `minerva_math` task family.
MINERVA_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def _strip_boxed(s: str) -> str:
    s = (s or "").strip()
    if s.startswith(r"\boxed{") and s.endswith("}"):
        return s[len(r"\boxed{") : -1]
    return s


def make_map_fn(subject: str):
    def process_fn(example, idx):
        question_raw = example["problem"]
        question = question_raw + " " + INSTRUCTION_FOLLOWING

        # MATH dataset stores the canonical answer inside \\boxed{...} in the
        # 'solution' field. Extract it for ground_truth; the verifier will
        # handle equivalent forms.
        solution_raw = example.get("solution", "")
        # The ground-truth is the boxed expression at the end of the solution.
        # We do a simple right-most boxed extraction; math_dapo's verifier is
        # tolerant of formatting variation so a perfect parse isn't required.
        gt = solution_raw
        marker = r"\boxed{"
        i = solution_raw.rfind(marker)
        if i != -1:
            j = i + len(marker)
            depth = 1
            k = j
            while k < len(solution_raw) and depth > 0:
                if solution_raw[k] == "{":
                    depth += 1
                elif solution_raw[k] == "}":
                    depth -= 1
                k += 1
            gt = solution_raw[j : k - 1] if depth == 0 else solution_raw[j:]
        gt = gt.strip()

        return {
            "data_source": "math__minerva",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": {
                "split": "test",
                "index": idx,
                "subject": subject,
                "level": example.get("level", ""),
                "type": example.get("type", ""),
            },
        }

    return process_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="~/data/guru_rl")
    parser.add_argument(
        "--source",
        default="EleutherAI/hendrycks_math",
        help="HF dataset providing per-subject MATH configs. Default matches lm-eval-harness convention.",
    )
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    parts = []
    for subject in MINERVA_SUBJECTS:
        print(f"Loading {args.source} / {subject} ...", flush=True)
        ds = datasets.load_dataset(args.source, subject, split="test", trust_remote_code=True)
        ds = ds.map(make_map_fn(subject), with_indices=True, remove_columns=ds.column_names)
        parts.append(ds)
        print(f"  {subject}: {len(ds)} rows")

    full = datasets.concatenate_datasets(parts)
    out = os.path.join(save_dir, "test_minerva_math.parquet")
    full.to_parquet(out)
    print(f"  wrote {out}  ({len(full)} rows total)")


if __name__ == "__main__":
    main()
