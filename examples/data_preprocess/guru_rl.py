# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Download and preprocess the guru-RL-92k dataset (LLM360/guru-RL-92k) to parquet format.

The dataset is already in verl format. This script downloads the train and eval splits,
filters to the required columns, and saves them locally.

Also produces a 4x-duplicated AIME eval file (32x_960) mirroring the duplication step in
the reference prepare_train_test_datasets.sh from Beyond-the-80-20-Rule-RLVR.

Usage:
    python guru_rl.py --local_save_dir ~/data/guru_rl
    python guru_rl.py --local_save_dir ~/data/guru_rl --subset math
    python guru_rl.py --local_save_dir ~/data/guru_rl --overwrite
"""

import argparse
import os

import datasets

REQUIRED_KEYS = ["data_source", "prompt", "reward_model", "extra_info"]

HF_REPO = "LLM360/guru-RL-92k"

TRAIN_FILES = {
    "math": "train/math__combined_54.4k.parquet",
}

EVAL_FILES = {
    "aime": "offline_eval/math__aime_repeated_8x_240.parquet",
    "math500": "offline_eval/math__math_500.parquet",
}

# The reference script duplicates the AIME set 4x (8x*4=32x, 240*4=960 rows)
AIME_DUPLICATE_TIMES = 4


def load_and_filter(data_files: str | list[str]) -> datasets.Dataset:
    ds = datasets.load_dataset(HF_REPO, data_files=data_files, split="train")
    available = [k for k in REQUIRED_KEYS if k in ds.column_names]
    missing = [k for k in REQUIRED_KEYS if k not in ds.column_names]
    if missing:
        print(f"Warning: columns not found and will be skipped: {missing}")
    if not available:
        raise ValueError(f"None of the required keys {REQUIRED_KEYS} found in dataset columns: {ds.column_names}")
    return ds.select_columns(available)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess guru-RL-92k for verl training")
    parser.add_argument(
        "--local_save_dir",
        default="~/data/guru_rl",
        help="Directory to save processed parquet files",
    )
    parser.add_argument(
        "--subset",
        default="math",
        choices=list(TRAIN_FILES.keys()),
        help="Which training subset to download",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download and overwrite existing files (default: skip if file exists)",
    )
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Train split
    out_train = os.path.join(local_dir, "train.parquet")
    if not os.path.exists(out_train) or args.overwrite:
        train_file = TRAIN_FILES[args.subset]
        print(f"Loading train split: {train_file}")
        train_ds = load_and_filter(train_file)
        print(f"Train size: {len(train_ds)}, columns: {train_ds.column_names}")
        train_ds.to_parquet(out_train)
        print(f"Saved train -> {out_train}")
    else:
        print(f"Skipping train (already exists): {out_train}")

    # Eval splits
    for name, eval_file in EVAL_FILES.items():
        out = os.path.join(local_dir, f"test_{name}.parquet")
        if not os.path.exists(out) or args.overwrite:
            print(f"Loading eval split: {eval_file}")
            eval_ds = load_and_filter(eval_file)
            print(f"  {name} size: {len(eval_ds)}, columns: {eval_ds.column_names}")
            eval_ds.to_parquet(out)
            print(f"  Saved -> {out}")
        else:
            print(f"  Skipping {name} (already exists): {out}")

    # Duplicate AIME eval 4x to produce the 32x_960 variant
    aime_out = os.path.join(local_dir, "test_aime.parquet")
    aime_32x_out = os.path.join(local_dir, "test_aime_32x.parquet")
    if not os.path.exists(aime_32x_out) or args.overwrite:
        if os.path.exists(aime_out):
            print(f"Duplicating AIME eval {AIME_DUPLICATE_TIMES}x...")
            aime_ds = datasets.load_dataset("parquet", data_files=aime_out, split="train")
            aime_32x_ds = datasets.concatenate_datasets([aime_ds] * AIME_DUPLICATE_TIMES)
            aime_32x_ds.to_parquet(aime_32x_out)
            print(f"  AIME 32x size: {len(aime_32x_ds)}, saved -> {aime_32x_out}")
        else:
            print(f"  Skipping AIME duplication: {aime_out} not found")
    else:
        print(f"Skipping AIME 32x (already exists): {aime_32x_out}")