#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""
Reward-variance per group over training — pure JSONL parsing, no compute.

Empirically defends the §1 dead-zone framing: how often does the within-group
reward distribution have any spread at all under each method?

For each training step we extract from the JSONL:
    - critic/score/std       (within-batch score std; near 0 ⇒ degenerate batch)
    - critic/score/min, /max (range; gap=0 ⇒ all rollouts in batch identical reward)
    - critic/advantages/std  (the variable that GRPO actually uses)
    - actor/pg_loss          (gradient magnitude; near 0 ⇒ no learning)
    - actor/entropy          (policy entropy; for the entropy-stability claim)
    - response_length/mean   (for length-collapse diagnostics)

Then aggregate: fraction of steps with score_std > threshold (per method).

Usage:
    python reward_variance.py \
        --jsonls method=path \
                 GRPO=/storage/.../grpo/.../qwen3_4b_base_grpo.jsonl \
                 HEG=/storage/.../high-ent-grpo/.../qwen3_4b_base_high_ent_grpo.jsonl \
                 DRIFT=/storage/.../pivot-v18c/.../qwen3_4b_base_pivot_v18c.jsonl \
        --out reward_variance_4b.json

Output schema:
    {
      "GRPO":  [{"step": 1, "score_std": 0.0, ..., "pg_loss": 0.0, ...}, ...],
      "HEG":   [...],
      "DRIFT": [...]
    }

Plot from Python with anything (matplotlib/seaborn) — script is plot-format-agnostic.
"""

import argparse
import json
import os
import sys


KEYS = [
    "critic/score/mean",
    "critic/score/min",
    "critic/score/max",
    "critic/score/std",
    "critic/advantages/mean",
    "critic/advantages/std",
    "critic/rewards/mean",
    "actor/pg_loss",
    "actor/entropy",
    "actor/grad_norm",
    "response_length/mean",
    "pivot/lan_grpo_trig_frac",
]


def parse(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            d = row.get("data", row)
            step = row.get("step")
            if "critic/score/mean" not in d:
                continue
            entry = {"step": step}
            for k in KEYS:
                v = d.get(k)
                if v is None:
                    continue
                entry[k.replace("/", ".")] = v
            # Derived: gap = max - min, useful as a coarse "any variance" indicator
            mx = d.get("critic/score/max")
            mn = d.get("critic/score/min")
            if mx is not None and mn is not None:
                entry["score.gap"] = float(mx) - float(mn)
            rows.append(entry)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--jsonls",
        nargs="+",
        required=True,
        help='Pairs of method=path, e.g. GRPO=/path/to/grpo.jsonl HEG=... DRIFT=...',
    )
    p.add_argument("--out", default="reward_variance.json")
    p.add_argument(
        "--var_threshold",
        type=float,
        default=1e-6,
        help="threshold above which a step counts as 'has variance'",
    )
    args = p.parse_args()

    by_method = {}
    for spec in args.jsonls:
        if "=" not in spec:
            print(f"bad spec: {spec}", file=sys.stderr)
            sys.exit(1)
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"missing: {path}", file=sys.stderr)
            continue
        rows = parse(path)
        by_method[name] = rows
        with_var = sum(1 for r in rows if r.get("critic.score.std", 0.0) > args.var_threshold)
        print(
            f"[{name}] {len(rows)} steps,  "
            f"{with_var} ({100 * with_var / max(len(rows), 1):.1f}%) "
            f"with score.std>{args.var_threshold}",
            file=sys.stderr,
        )

    with open(args.out, "w") as f:
        json.dump(by_method, f, indent=2)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
