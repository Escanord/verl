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
Evaluate a single (already merged) HF checkpoint on a single benchmark parquet.

Outputs a JSON file containing per-problem scores and aggregate mean@n / best@n.

Usage:
    python eval_checkpoint.py \
        --ckpt /path/to/hf_merged \
        --benchmark aime25 \
        --data_dir /storage/workspace/server-1/duy/data/guru_rl \
        --out_dir ./eval_out \
        --n 16

Benchmarks (each maps to a parquet under --data_dir and a scorer):
    aime24          test_aime.parquet                math_dapo
    aime25          test_aime25.parquet              math_dapo
    olympiadbench   test_olympiadbench_math_en.parquet  math_dapo
    minerva         test_minerva_math.parquet        math_dapo
    gpqa            test_gpqa_diamond.parquet        mcq (regex on 'Answer: $LETTER')

Environment: must run inside the `verl` conda env (vllm + verl available).
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import datasets
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from verl.utils.reward_score import math_dapo

MCQ_PATTERN = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?")


def score_math(solution_str: str, ground_truth: str) -> float:
    try:
        return float(math_dapo.compute_score(solution_str, ground_truth))
    except Exception:
        return 0.0


def score_mcq(solution_str: str, ground_truth: str) -> float:
    m = MCQ_PATTERN.search(solution_str or "")
    return 1.0 if (m is not None and m.group(1).upper() == str(ground_truth).upper()) else 0.0


BENCHMARKS = {
    "aime24":        {"file": "test_aime.parquet",                   "scorer": score_math},
    "aime25":        {"file": "test_aime25.parquet",                 "scorer": score_math},
    "olympiadbench": {"file": "test_olympiadbench_math_en.parquet",  "scorer": score_math},
    "minerva":       {"file": "test_minerva_math.parquet",           "scorer": score_math},
    "gpqa":          {"file": "test_gpqa_diamond.parquet",           "scorer": score_mcq},
}


def build_prompts(ds, tokenizer):
    """Apply chat template if available, else fall back to raw user content."""
    out = []
    for ex in ds:
        msgs = ex["prompt"]  # [{"role": "user", "content": str}]
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            prompt = msgs[0]["content"]
        out.append(prompt)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="HF model directory")
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARKS.keys()))
    parser.add_argument("--data_dir", default="/storage/workspace/server-1/duy/data/guru_rl")
    parser.add_argument("--out_dir", default="./eval_out")
    parser.add_argument("--n", type=int, default=16, help="Samples per problem")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--tp", type=int, default=1, help="vLLM tensor-parallel size")
    parser.add_argument("--gpu_mem_util", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=None, help="Cap problems (for debug)")
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional run/checkpoint tag used in the output filename (e.g. v18b_step120)",
    )
    args = parser.parse_args()

    bench = BENCHMARKS[args.benchmark]
    data_dir = Path(args.data_dir).expanduser()
    parquet_path = data_dir / bench["file"]
    if not parquet_path.exists():
        raise FileNotFoundError(f"Benchmark parquet not found: {parquet_path}")

    print(f"[eval] benchmark={args.benchmark}  parquet={parquet_path}")
    ds = datasets.Dataset.from_parquet(str(parquet_path))
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"[eval] loaded {len(ds)} problems")

    print(f"[eval] loading tokenizer from {args.ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)

    prompts = build_prompts(ds, tokenizer)
    ground_truths = [ex["reward_model"]["ground_truth"] for ex in ds]

    print(f"[eval] loading model into vLLM (tp={args.tp})")
    llm = LLM(
        model=args.ckpt,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=True,
        enforce_eager=False,
    )

    sp = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    print(f"[eval] generating n={args.n} samples per problem ...")
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    print(f"[eval] generation finished in {time.time() - t0:.1f}s")

    per_problem = []
    for i, out in enumerate(outputs):
        gt = ground_truths[i]
        sample_texts = [o.text for o in out.outputs]
        scores = [bench["scorer"](t, gt) for t in sample_texts]
        per_problem.append(
            {
                "idx": i,
                "ground_truth": gt,
                "scores": scores,
                "mean": sum(scores) / len(scores),
                "best": float(max(scores) if scores else 0.0),
            }
        )

    mean_at_n = sum(r["mean"] for r in per_problem) / len(per_problem)
    best_at_n = sum(r["best"] for r in per_problem) / len(per_problem)

    print(f"[eval] {args.benchmark}@{args.n}:  mean={mean_at_n:.4f}  best={best_at_n:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or Path(args.ckpt).name
    out_path = Path(args.out_dir) / f"eval__{tag}__{args.benchmark}.json"
    summary = {
        "benchmark": args.benchmark,
        "ckpt": args.ckpt,
        "tag": tag,
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "num_problems": len(per_problem),
        f"mean@{args.n}": mean_at_n,
        f"best@{args.n}": best_at_n,
        "per_problem": per_problem,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
