#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""
Rollout-level diversity within a GRPO group.

For each prompt, sample n rollouts at temperature 1.0 from one checkpoint, then
compute pairwise diversity between the n rollouts (the variance the model
actually produces *as a group*).  We report two metrics per prompt:

    1. Token-level Levenshtein distance, normalized by max-length pair.
    2. 1 - Jaccard similarity over token sets.

To distinguish "useful" diversity from "all-wrong" diversity, each rollout is
scored with the math_dapo verifier against the parquet's ground_truth, and we
additionally report:

    - correct_only_*: pairwise diversity AMONG correct rollouts (groups with
      <2 correct rollouts contribute no pairs to this aggregate)
    - mixed_correctness_fraction: fraction of n-rollout groups that contain
      both a correct and an incorrect rollout (the GRPO-relevant signal:
      these are the groups with non-zero within-group reward variance)
    - solve_rate: per-prompt mean correctness across the n rollouts

Run once per checkpoint (GRPO/HEG/DRIFT).  Compare per-prompt diversity
distributions side-by-side to defend §3.2's "the diffusion term spreads the n
rollouts into distinct continuations" claim.

Usage:
    python rollout_diversity.py \
        --ckpt /storage/.../grpo/qwen3_4b_base/global_step_60/actor/hf_merged \
        --prompts /storage/.../data/guru_rl/test_aime.parquet \
        --n_prompts 30 \
        --n 8 \
        --tag grpo_4b_step60 \
        --out_dir ./diversity_out

Inputs:
    --ckpt        path to merged HF checkpoint dir (or HF model id)
    --prompts     path to a verl-format eval parquet (uses the 'prompt' column)
    --n_prompts   subset of the parquet to use (default: 30)
    --n           samples per prompt (default: 8)
    --tag         label written into the output JSON (e.g. grpo_4b_step60)

Output:
    {out_dir}/diversity__{tag}.json
        {
          "tag": ...,
          "n": 8,
          "n_prompts": 30,
          "per_prompt": [
              {"idx": 0, "lev_norm_pairs": [...], "jaccard_dist_pairs": [...]},
              ...
          ],
          "summary": {"lev_norm/mean": ..., "jaccard/mean": ..., ...}
        }
"""

import argparse
import json
import os
import statistics
import sys
import time
from itertools import combinations
from pathlib import Path

import datasets
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from verl.utils.reward_score import math_dapo


def levenshtein(a, b):
    """Iterative Levenshtein on token lists.  O(len(a)*len(b)); fine for our sizes."""
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompts", required=True, help="parquet path; uses 'prompt' column")
    p.add_argument("--n_prompts", type=int, default=30)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu_mem_util", type=float, default=0.85)
    p.add_argument("--tag", required=True)
    p.add_argument("--out_dir", default="./diversity_out")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = Path(args.out_dir) / f"diversity__{args.tag}.json"
    if out_path.exists():
        print(f"[have] {out_path} (delete to re-run)", file=sys.stderr)
        sys.exit(0)

    print(f"[diversity] tag={args.tag}  ckpt={args.ckpt}", file=sys.stderr)
    ds = datasets.Dataset.from_parquet(args.prompts)
    ds = ds.select(range(min(args.n_prompts, len(ds))))
    print(f"[diversity] using {len(ds)} prompts × n={args.n} samples", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    prompts = []
    ground_truths = []
    for ex in ds:
        msgs = ex["prompt"]
        if tokenizer.chat_template:
            prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        else:
            prompts.append(msgs[0]["content"])
        ground_truths.append(ex["reward_model"]["ground_truth"])

    print(f"[diversity] loading vLLM (tp={args.tp})", file=sys.stderr)
    llm = LLM(
        model=args.ckpt,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=True,
        seed=args.seed,
    )

    sp = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    print(f"[diversity] generation done in {time.time() - t0:.1f}s", file=sys.stderr)

    per_prompt = []
    all_lev = []
    all_jac = []
    all_lev_correct = []
    all_jac_correct = []
    n_mixed = 0
    n_groups_with_pairs = 0
    for i, out in enumerate(outputs):
        # Token IDs per sample (from vLLM's RequestOutput)
        token_seqs = [list(s.token_ids) for s in out.outputs]
        # Decoded text + math_dapo correctness per rollout
        texts = [s.text for s in out.outputs]
        gt = ground_truths[i]
        scores = []
        for txt in texts:
            try:
                r = math_dapo.compute_score(txt, gt)
                # math_dapo returns dict with 'score' or 'acc'; both are 0/1 here
                acc = r.get("acc", r.get("score", 0))
                scores.append(int(acc == 1 or acc == 1.0))
            except Exception:
                scores.append(0)
        n_correct = sum(scores)
        solve_rate = n_correct / max(len(scores), 1)
        if 0 < n_correct < len(scores):
            n_mixed += 1

        lev_norm_pairs = []
        jac_dist_pairs = []
        lev_norm_correct_pairs = []
        jac_dist_correct_pairs = []
        for (idx_a, a), (idx_b, b) in combinations(enumerate(token_seqs), 2):
            lv = levenshtein(a, b)
            denom = max(len(a), len(b), 1)
            lv_n = lv / denom
            sa, sb = set(a), set(b)
            inter = len(sa & sb)
            union = len(sa | sb) or 1
            jc = 1.0 - inter / union
            lev_norm_pairs.append(lv_n)
            jac_dist_pairs.append(jc)
            if scores[idx_a] == 1 and scores[idx_b] == 1:
                lev_norm_correct_pairs.append(lv_n)
                jac_dist_correct_pairs.append(jc)
        if lev_norm_pairs:
            n_groups_with_pairs += 1

        per_prompt.append(
            {
                "idx": i,
                "n_pairs": len(lev_norm_pairs),
                "n_correct": n_correct,
                "solve_rate": solve_rate,
                "scores": scores,
                "lev_norm_pairs": lev_norm_pairs,
                "jaccard_dist_pairs": jac_dist_pairs,
                "lev_norm_correct_pairs": lev_norm_correct_pairs,
                "jaccard_dist_correct_pairs": jac_dist_correct_pairs,
                "mean_lev_norm": statistics.mean(lev_norm_pairs) if lev_norm_pairs else 0.0,
                "mean_jaccard_dist": statistics.mean(jac_dist_pairs) if jac_dist_pairs else 0.0,
                "mean_lev_norm_correct": (statistics.mean(lev_norm_correct_pairs)
                                          if lev_norm_correct_pairs else None),
                "mean_jaccard_dist_correct": (statistics.mean(jac_dist_correct_pairs)
                                              if jac_dist_correct_pairs else None),
                "response_lens": [len(s) for s in token_seqs],
            }
        )
        all_lev.extend(lev_norm_pairs)
        all_jac.extend(jac_dist_pairs)
        all_lev_correct.extend(lev_norm_correct_pairs)
        all_jac_correct.extend(jac_dist_correct_pairs)

    def _summary(xs):
        if not xs:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
        return {
            "mean": statistics.mean(xs),
            "median": statistics.median(xs),
            "std": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs),
            "max": max(xs),
            "n": len(xs),
        }

    solve_rates = [pp["solve_rate"] for pp in per_prompt]
    summary = {
        "lev_norm": _summary(all_lev),
        "jaccard_dist": _summary(all_jac),
        "lev_norm_correct": _summary(all_lev_correct),
        "jaccard_dist_correct": _summary(all_jac_correct),
        "solve_rate": _summary(solve_rates),
        "mixed_correctness_fraction": (n_mixed / n_groups_with_pairs) if n_groups_with_pairs else 0.0,
        "n_groups_with_pairs": n_groups_with_pairs,
        "n_mixed_groups": n_mixed,
        "n_correct_pairs": len(all_lev_correct),
    }
    print(
        f"[diversity] {args.tag}  "
        f"lev.all={summary['lev_norm']['mean']:.4f} ({summary['lev_norm']['n']} pairs)  "
        f"lev.correct={summary['lev_norm_correct']['mean']:.4f} ({summary['lev_norm_correct']['n']} pairs)  "
        f"mixed_frac={summary['mixed_correctness_fraction']:.4f}  "
        f"solve={summary['solve_rate']['mean']:.4f}",
        file=sys.stderr,
    )

    with open(out_path, "w") as f:
        json.dump(
            {
                "tag": args.tag,
                "ckpt": args.ckpt,
                "n": args.n,
                "n_prompts": len(ds),
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "summary": summary,
                "per_prompt": per_prompt,
            },
            f,
            indent=2,
        )
    print(f"[diversity] wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
